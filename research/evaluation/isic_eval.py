"""
One external test of the E0 seeds and their ensemble on the ISIC 2018 Task 3 test set.

    python -m research.evaluation.isic_eval --config research/configs/ensemble_e0.yaml \
        --images <folder of ISIC_*.jpg> --ground-truth <ISIC2018_Task3_Test_GroundTruth.csv> \
        --ham-root <folder with HAM10000_metadata.csv and the image folders>

Temperatures and thresholds are read from the HAM10000 validation-only runs and used
unchanged; nothing is fitted or selected on ISIC. Any image shared with HAM10000, by ID or
by file content, stops the run before a prediction is made. The output folder must be
empty, so a finished test cannot be repeated by accident. ISIC 2018 has no lesion IDs
here, so the bootstrap resamples images.
"""
import argparse
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research.data.data import HAM10000Dataset, build_image_index, build_transforms, make_loader
from research.evaluation.calibration_eval import verify_run
from research.evaluation.ensemble_eval import balanced_accuracy, calibration_block, scores_from_probs
from research.evaluation.evaluate import verify_malignant_mapping
from research.evaluation.metrics import cluster_bootstrap, log_softmax64
from research.evaluation.selective_eval import load_temperature
from research.evaluation.selective_policies import aurc_for_uncertainty, evaluate_referral
from research.models.fusion_model import assert_eval_transform_matches_backend, build_model
from research.training.predict import collect_logits
from research.utils.config import REPO_ROOT, load_config, repo_path
from research.utils.environment import collect_environment
from research.utils.hashing import canonical_json, sha256_file, sha256_text
from research.utils.io import read_json, write_json


class IsicError(RuntimeError):
    pass


def _check(cond, msg):
    if not cond:
        raise IsicError(msg)


def read_ground_truth(path, classes) -> pd.DataFrame:
    gt = pd.read_csv(path)
    columns = [c.upper() for c in classes]
    _check(set(columns) <= set(gt.columns), f"ground truth lacks a class column: {list(gt.columns)}")
    onehot = gt[columns].to_numpy(dtype=float)
    _check((onehot.sum(axis=1) == 1).all(), "a ground truth row does not have exactly one class")
    return pd.DataFrame({"image_id": gt["image"], "lesion_id": gt["image"], "dx": [classes[i] for i in onehot.argmax(axis=1)]})


def check_overlap(frame, image_paths, ham_root, cfg) -> dict:
    meta = pd.read_csv(Path(ham_root) / cfg["data"]["metadata_file"])
    shared_ids = sorted(set(frame["image_id"]) & set(meta["image_id"]))
    _check(not shared_ids, f"{len(shared_ids)} image IDs are also in HAM10000, e.g. {shared_ids[:5]}")
    ham_hashes = {sha256_file(p) for p in build_image_index(ham_root, cfg).values()}
    isic_hashes = {i: sha256_file(image_paths[i]) for i in frame["image_id"]}
    shared_files = sorted(i for i, h in isic_hashes.items() if h in ham_hashes)
    _check(not shared_files, f"{len(shared_files)} files are byte-identical to HAM10000 files, e.g. {shared_files[:5]}")
    return {"shared_image_ids": 0, "byte_identical_files": 0, "n_isic": len(isic_hashes), "n_ham10000_files": len(ham_hashes),
            "not_checked": "visually similar images and the same lesion photographed again"}


def evaluate_model(s, y, classes, mal_idx, thresholds, n_bins) -> dict:
    mel = classes.index("mel")
    correct = s["pred"] == y
    is_mal = np.isin(y, mal_idx)
    benign_label = ~np.isin(s["pred"], mal_idx)
    out = {"n": int(len(y)), "accuracy": float(correct.mean()), "balanced_accuracy": balanced_accuracy(y, s["pred"], len(classes)),
           "melanoma_recall": float(correct[y == mel].mean()), "aurc_msp": aurc_for_uncertainty(s["msp"], correct)["aurc"],
           "aurc_entropy": aurc_for_uncertainty(s["entropy"], correct)["aurc"], "calibration": calibration_block(s["probs"], y, n_bins),
           "referral": {k: evaluate_referral(s["p_mal"], y, v["threshold"], classes, [classes[i] for i in mal_idx]) for k, v in thresholds["referral"].items()},
           "workload": {}}
    for frac, by_score in thresholds["workload"].items():
        out["workload"][frac] = {}
        for score, t in by_score.items():
            auto = s[score] <= t["threshold"]
            out["workload"][frac][score] = {"test_referred": float((~auto).mean()), "automated_error": float((~correct[auto]).mean()) if auto.any() else float("nan"),
                                            "malignant_automated": float(auto[is_mal].mean()),
                                            "malignant_cleared_as_benign": int((auto & is_mal & benign_label).sum()),
                                            "melanoma_cleared_as_benign": int((auto & (y == mel) & benign_label).sum())}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--ham-root", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--output-dir", default="research/outputs/isic2018_test")
    args = ap.parse_args(argv)
    command = shlex.join(["python", "-m", "research.evaluation.isic_eval", *(argv if argv is not None else sys.argv[1:])])

    cfg = load_config(args.config)
    e0_cfg = load_config(repo_path("research/configs/e0.yaml"))
    verify_malignant_mapping(e0_cfg)
    classes = e0_cfg["data"]["classes"]
    mal_idx = [classes.index(c) for c in cfg["referral"]["malignant_classes"]]
    out = repo_path(args.output_dir)
    _check(not out.exists() or not any(out.iterdir()), f"{out} already exists and is not empty; the external test runs once")

    # Everything frozen on HAM10000 validation is checked before ISIC images are read.
    ens_dir = repo_path(cfg["experiment"]["output_dir"])
    frozen = read_json(ens_dir / "thresholds.json")
    frozen_hash = read_json(ens_dir / "provenance.json")["thresholds_sha256"]
    _check(sha256_text(canonical_json(frozen["models"])) == frozen_hash, "thresholds.json differs from the recorded hash")

    frame = read_ground_truth(args.ground_truth, classes)
    images = Path(args.images)
    image_paths = {i: images / f"{i}.jpg" for i in frame["image_id"]}
    missing = [i for i, p in image_paths.items() if not p.is_file()]
    _check(not missing, f"{len(missing)} images are missing, e.g. {missing[:5]}")
    overlap = check_overlap(frame, image_paths, args.ham_root, e0_cfg)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "overlap_check.json", overlap)
    print("overlap check passed:", overlap, flush=True)

    _, eval_tf = build_transforms(e0_cfg)
    assert_eval_transform_matches_backend(eval_tf)
    dataset = HAM10000Dataset(frame, image_paths, classes, eval_tf)
    y = np.array([classes.index(d) for d in frame["dx"]], dtype=np.int64)
    device = torch.device(args.device)
    member_probs, members = [], {}
    base = repo_path("research/outputs/e0_baseline")
    for seed in cfg["seeds"]:
        run_dir = base / f"seed_{seed}"
        run = verify_run(run_dir, e0_cfg)
        T, _ = load_temperature(repo_path("research/outputs/calibration") / f"seed_{seed}", run)
        _check(abs(T - frozen["temperatures"][str(seed)]) < 1e-12, f"seed {seed}: temperature differs from thresholds.json")
        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
        model = build_model(e0_cfg, pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        loader = make_loader(dataset, e0_cfg["training"]["batch_size"], shuffle=False, drop_last=False, num_workers=args.num_workers, seed=seed)
        logits, rows = collect_logits(model, loader, device)
        _check(np.array_equal(rows, np.arange(len(frame))), "prediction order changed")
        np.savez(out / f"logits_seed_{seed}.npz", logits=logits, labels=y, image_id=frame["image_id"].to_numpy(dtype=str))
        probs = np.exp(log_softmax64(logits.astype(np.float64) / T))
        member_probs.append(probs)
        members[f"seed_{seed}"] = scores_from_probs(probs, mal_idx)
        print(f"seed {seed}: predicted {len(y)} images", flush=True)
    members["ensemble"] = scores_from_probs(np.mean(member_probs, axis=0), mal_idx, member_probs=member_probs)

    results = {name: evaluate_model(s, y, classes, mal_idx, frozen["models"][name], cfg["ece_bins"]) for name, s in members.items()}

    ens = members["ensemble"]
    ref95 = frozen["models"]["ensemble"]["referral"]["sensitivity_0.95"]["threshold"]
    is_mal = np.isin(y, mal_idx)

    def boot_metric(idx):
        yy, pred = y[idx], ens["pred"][idx]
        correct, mal, ref = pred == yy, is_mal[idx], ens["p_mal"][idx] >= ref95
        return {"accuracy": float(correct.mean()), "balanced_accuracy": balanced_accuracy(yy, pred, len(classes)),
                "melanoma_recall": float(correct[yy == classes.index("mel")].mean()),
                "aurc_msp": aurc_for_uncertainty(ens["msp"][idx], correct)["aurc"],
                "referral95_sensitivity": float(ref[mal].mean()), "referral95_specificity": float((~ref[~mal]).mean()), "referral95_rate": float(ref.mean())}

    bs = cfg["bootstrap"]
    boot = cluster_bootstrap(boot_metric, frame["image_id"].to_numpy(), bs["n_resamples"], bs["seed"], bs["confidence_level"])
    boot["unit"] = "image_id (no lesion IDs for this set)"

    write_json(out / "summary.json", {"label": "External test, run once; thresholds and temperatures frozen on HAM10000 validation",
                                      "class_counts": frame["dx"].value_counts().to_dict(), "overlap": overlap, "per_model": results, "ensemble_bootstrap": boot})
    write_json(out / "provenance.json", {"command": command, "thresholds_sha256": frozen_hash, "environment": collect_environment(REPO_ROOT)})
    for name, r in results.items():
        ref = r["referral"]["sensitivity_0.95"]
        print(f"{name}: acc {r['accuracy']:.4f} bal {r['balanced_accuracy']:.4f} mel {r['melanoma_recall']:.4f} aurc {r['aurc_msp']:.4f} ece {r['calibration']['ece']:.4f} | "
              f"ref95 sens {ref['sensitivity']:.4f} spec {ref['specificity']:.4f} referred {ref['referral_rate']:.4f} cleared {ref['malignant_cleared']}/{ref['n_malignant']} mel {ref['melanoma_cleared']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
