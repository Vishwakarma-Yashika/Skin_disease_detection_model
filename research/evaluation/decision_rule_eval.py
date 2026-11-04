"""
Freezes the app's decision system on HAM10000 validation, then tests the full rule.

    python -m research.evaluation.decision_rule_eval --config research/configs/decision_rule.yaml \
        --ham-root <HAM10000 root> --isic-images <folder> --isic-ground-truth <csv> --cifar-dir <folder>

Validation only: fusion features become the out-of-scope reference and its threshold is set;
manifest.json and feature_bank.npz are written before any test image is scored. The rule then
runs unchanged on HAM10000 test, ISIC 2018 test and images that should be rejected. Features
are cached, because they take minutes per set on CPU.
"""
import argparse
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

from research.data.data import build_image_index, build_transforms
from research.evaluation.calibration_eval import verify_run
from research.evaluation.ensemble_eval import scores_from_probs
from research.evaluation.isic_eval import read_ground_truth
from research.evaluation.metrics import log_softmax64
from research.evaluation.selective_eval import load_temperature
from research.evaluation.selective_policies import threshold_for_coverage
from research.models.fusion_model import assert_eval_transform_matches_backend, build_model
from research.utils.config import REPO_ROOT, load_config, repo_path
from research.utils.environment import collect_environment
from research.utils.io import read_json, write_json


class DecisionError(RuntimeError):
    pass


def _check(cond, msg):
    if not cond:
        raise DecisionError(msg)


class ImageList(Dataset):
    """Items are file paths or PIL images, in a fixed order."""

    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        if isinstance(item, Image.Image):
            return self.transform(item.convert("RGB"))
        with Image.open(item) as img:
            return self.transform(img.convert("RGB"))


@torch.no_grad()
def extract(models, items, transform, device, cache: Path, batch_size: int) -> dict:
    """Per-member L2-normalised fusion features and raw logits, cached by set name."""
    if cache.is_file():
        z = np.load(cache)
        if z["features"].shape[1] == len(items):
            return {"features": z["features"], "logits": z["logits"]}
    loader = DataLoader(ImageList(items, transform), batch_size=batch_size, shuffle=False, num_workers=0)
    all_f, all_z = [], []
    for model in models:
        fs, zs = [], []
        for x in loader:
            f = model.backbone(x.to(device))
            zs.append(model.classifier(f).cpu().numpy())
            fs.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
        all_f.append(np.concatenate(fs))
        all_z.append(np.concatenate(zs))
    out = {"features": np.stack(all_f).astype(np.float32), "logits": np.stack(all_z).astype(np.float32)}
    np.savez(cache, **out)
    print(f"extracted {cache.stem}: {len(items)} images", flush=True)
    return out


def knn_distance(features, bank, k, exclude_self=False):
    """Mean over members of the mean cosine distance to the k nearest reference images."""
    scores = []
    for f, b in zip(features, bank):
        sim = f @ b.T
        if exclude_self:
            np.fill_diagonal(sim, -np.inf)
        top = np.sort(sim, axis=1)[:, -k:]
        scores.append(1.0 - top.mean(axis=1))
    return np.mean(scores, axis=0)


def ensemble_probs(logits, temps):
    members = [np.exp(log_softmax64(z.astype(np.float64) / t)) for z, t in zip(logits, temps)]
    return np.mean(members, axis=0), members


def apply_rule(s, oos, thresholds):
    """Rules in order. Returns one label per case: out_of_scope, refer_malignancy, refer_uncertain or automated."""
    decision = np.full(len(oos), "automated", dtype=object)
    decision[s["msp"] > thresholds["msp"]] = "refer_uncertain"
    decision[s["p_mal"] >= thresholds["p_mal"]] = "refer_malignancy"
    decision[oos > thresholds["out_of_scope"]] = "out_of_scope"
    return decision


def rule_report(decision, s, y, classes, mal_idx) -> dict:
    auto = decision == "automated"
    in_scope = decision != "out_of_scope"
    is_mal, is_mel = np.isin(y, mal_idx), y == classes.index("mel")
    benign_label = ~np.isin(s["pred"], mal_idx)
    wrong = s["pred"] != y
    return {"n": int(len(y)), "shares": {d: float((decision == d).mean()) for d in ("automated", "refer_malignancy", "refer_uncertain", "out_of_scope")},
            "n_automated": int(auto.sum()), "automated_error": float(wrong[auto].mean()) if auto.any() else float("nan"),
            "n_malignant": int(is_mal.sum()), "n_melanoma": int(is_mel.sum()),
            "malignant_automated": int((auto & is_mal).sum()),
            "malignant_cleared_as_benign": int((auto & is_mal & benign_label).sum()),
            "melanoma_cleared_as_benign": int((auto & is_mel & benign_label).sum()),
            "malignant_rejected_out_of_scope": int((~in_scope & is_mal).sum()),
            "melanoma_rejected_out_of_scope": int((~in_scope & is_mel).sum()),
            "share_of_errors_not_automated": float((wrong & ~auto).sum() / wrong.sum()) if wrong.any() else float("nan")}


def rejection_sets(cfg, cifar_dir):
    t = cfg["out_of_scope_test"]
    rng = np.random.default_rng(0)
    n = t["synthetic_per_kind"]
    size = (450, 600, 3)
    synthetic = {
        "white": [Image.fromarray(np.full(size, 255 - i, dtype=np.uint8)) for i in range(n)],
        "black": [Image.fromarray(np.full(size, i, dtype=np.uint8)) for i in range(n)],
        "flat_skin": [Image.fromarray(np.full(size, (225 - 8 * i, 185 - 8 * i, 160 - 8 * i), dtype=np.uint8)) for i in range(n)],
        "noise": [Image.fromarray(rng.integers(0, 256, size, dtype=np.uint8)) for _ in range(n)],
        "gradient": [Image.fromarray(np.tile(np.linspace(i * 8, 255, 600, dtype=np.uint8)[None, :, None], (450, 1, 3))) for i in range(n)],
    }
    charts = sorted(p for p in repo_path("research/outputs").rglob("*.png")
                    if "figures" in p.parts and "decision_system" not in p.parts and not any(w in p.stem.lower() for w in ("error", "gallery", "montage", "example", "overlay", "gradcam")))
    step = max(1, len(charts) // t["project_charts_max"])
    cifar = CIFAR10(cifar_dir, train=False, download=True)
    return {**synthetic, "project_charts": charts[::step][: t["project_charts_max"]],
            "cifar10": [Image.fromarray(cifar.data[i]) for i in range(t["cifar10_images"])]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ham-root", required=True)
    ap.add_argument("--isic-images", required=True)
    ap.add_argument("--isic-ground-truth", required=True)
    ap.add_argument("--cifar-dir", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    command = shlex.join(["python", "-m", "research.evaluation.decision_rule_eval", *(argv if argv is not None else sys.argv[1:])])

    cfg = load_config(args.config)
    ens_cfg = load_config(repo_path(cfg["ensemble_config"]))
    e0_cfg = load_config(repo_path("research/configs/e0.yaml"))
    classes = e0_cfg["data"]["classes"]
    mal_idx = [classes.index(c) for c in ens_cfg["referral"]["malignant_classes"]]
    seeds = ens_cfg["seeds"]
    out = repo_path(cfg["experiment"]["output_dir"])
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    frozen = read_json(repo_path(ens_cfg["experiment"]["output_dir"]) / "thresholds.json")
    ens_t = frozen["models"]["ensemble"]

    device = torch.device(args.device)
    _, eval_tf = build_transforms(e0_cfg)
    assert_eval_transform_matches_backend(eval_tf)
    base = repo_path("research/outputs/e0_baseline")
    models, temps, member_info = [], [], []
    for seed in seeds:
        run_dir = base / f"seed_{seed}"
        run = verify_run(run_dir, e0_cfg)
        T, _ = load_temperature(repo_path("research/outputs/calibration") / f"seed_{seed}", run)
        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
        model = build_model(e0_cfg, pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)
        temps.append(T)
        member_info.append({"seed": seed, "checkpoint": (run_dir / "best.pt").relative_to(REPO_ROOT).as_posix(),
                            "sha256": run["meta"]["checkpoint_sha256"], "temperature": T})

    # ===================== validation only =====================
    batch = e0_cfg["training"]["batch_size"]
    image_index = build_image_index(args.ham_root, e0_cfg)
    saved = {s: np.load(base / f"seed_{s}" / "predictions" / "val.npz") for s in seeds}
    val_ids = saved[seeds[0]]["image_id"]
    val = extract(models, [image_index[str(i)] for i in val_ids], eval_tf, device, cache / "val.npz", batch)
    parity = {}
    for m, s in enumerate(seeds):
        _check(np.array_equal(saved[s]["image_id"], val_ids), "validation order differs across seeds")
        diff = np.abs(val["logits"][m] - saved[s]["logits"])
        parity[str(s)] = {"max_abs_logit_difference": float(diff.max()),
                          "same_prediction": float((val["logits"][m].argmax(1) == saved[s]["logits"].argmax(1)).mean())}
        _check(parity[str(s)]["same_prediction"] > 0.99, f"seed {s}: re-extracted logits disagree with the saved run")
    k = cfg["out_of_scope"]["k"]
    val_scores = knn_distance(val["features"], val["features"], k, exclude_self=True)
    oos = threshold_for_coverage(val_scores, cfg["out_of_scope"]["validation_retained"])
    thresholds = {"p_mal": ens_t["referral"][cfg["referral_target"]]["threshold"],
                  "msp": ens_t["workload"][cfg["uncertainty"]["workload"]][cfg["uncertainty"]["score"]]["threshold"],
                  "out_of_scope": oos["threshold"]}
    bank_path = out / "feature_bank.npz"
    np.savez_compressed(bank_path, features=val["features"], image_id=val_ids)
    write_json(out / "manifest.json", {
        "name": "E0 3-seed ensemble", "classes": classes, "malignant_classes": ens_cfg["referral"]["malignant_classes"],
        "members": member_info, "thresholds": thresholds, "out_of_scope": {"method": cfg["out_of_scope"]["method"], "k": k},
        "feature_bank": bank_path.relative_to(REPO_ROOT).as_posix(),
        "selected_on": "HAM10000 validation only"})

    # ===================== rule fixed; test sets =====================
    results = {"thresholds": thresholds, "validation_parity": parity}
    split = pd.read_csv(repo_path(e0_cfg["split"]["file"]))
    test_ids = np.load(base / f"seed_{seeds[0]}" / "predictions" / "test.npz")["image_id"]
    y_test = np.load(base / f"seed_{seeds[0]}" / "predictions" / "test.npz")["labels"]
    _check(set(map(str, test_ids)) == set(split.loc[split["split"] == "test", "image_id"]), "test images differ from the frozen split")
    meta = pd.read_csv(Path(args.ham_root) / e0_cfg["data"]["metadata_file"]).set_index("image_id")
    dx_type = meta.loc[[str(i) for i in test_ids], "dx_type"].to_numpy()

    def run_set(name, items, y=None, mask=None):
        ex = extract(models, items, eval_tf, device, cache / f"{name}.npz", batch)
        probs, members = ensemble_probs(ex["logits"], temps)
        s = scores_from_probs(probs, mal_idx, member_probs=members)
        scores = knn_distance(ex["features"], val["features"], k)
        decision = apply_rule(s, scores, thresholds)
        return ex, s, scores, decision

    _, s, scores, d = run_set("ham_test", [image_index[str(i)] for i in test_ids])
    results["ham10000_test"] = {"all": rule_report(d, s, y_test, classes, mal_idx)}
    biopsy = dx_type == "histo"
    results["ham10000_test"]["biopsy_confirmed"] = rule_report(d[biopsy], {k_: v[biopsy] for k_, v in s.items()}, y_test[biopsy], classes, mal_idx)

    frame = read_ground_truth(args.isic_ground_truth, classes)
    y_isic = np.array([classes.index(c) for c in frame["dx"]])
    isic_items = [Path(args.isic_images) / f"{i}.jpg" for i in frame["image_id"]]
    ex, s, scores, d = run_set("isic_test", isic_items)
    saved_isic = np.load(repo_path("research/outputs/isic2018_test") / f"logits_seed_{seeds[0]}.npz")
    _check(np.abs(ex["logits"][0] - saved_isic["logits"]).max() < 1e-3, "ISIC logits differ from the external test run")
    results["isic2018_test"] = {"all": rule_report(d, s, y_isic, classes, mal_idx)}

    rejection = {}
    for name, items in rejection_sets(cfg, args.cifar_dir).items():
        _, s, scores, d = run_set(f"reject_{name}", items)
        rejection[name] = {"n": len(items), "rejected": float((d == "out_of_scope").mean()), "referred": float(np.isin(d, ["refer_malignancy", "refer_uncertain"]).mean()),
                           "automated": float((d == "automated").mean()), "median_score": float(np.median(scores)), "mean_top_confidence": float(1 - s["msp"].mean())}
    results["should_be_rejected"] = rejection

    write_json(out / "summary.json", results)
    write_json(out / "provenance.json", {"command": command, "environment": collect_environment(REPO_ROOT)})
    print({k_: v for k_, v in results.items() if k_ != "validation_parity"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
