"""
What fusion adds: EfficientNet-B0 only and ResNet-34 only against fusion, on the same two seeds.

    python -m research.evaluation.baseline_eval --isic-images <folder> --isic-ground-truth <csv>

Each model gets its own temperature and referral thresholds from validation, then runs on
HAM10000 test and ISIC 2018 test. Differences are fusion minus the arm, for the 2-seed
ensembles, with paired bootstrap intervals (lesions on HAM10000, images on ISIC 2018).
The output folder must be empty, so the ISIC comparison cannot be repeated by accident.
"""
import argparse
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research.calibration.temperature_scaling import fit_temperature
from research.data.data import HAM10000Dataset, build_transforms, load_split, make_loader
from research.evaluation.ensemble_eval import balanced_accuracy, calibration_block, scores_from_probs, stratum_metrics
from research.evaluation.evaluate import load_predictions
from research.evaluation.isic_eval import read_ground_truth
from research.evaluation.metrics import cluster_bootstrap, log_softmax64
from research.evaluation.selective_policies import aurc_for_uncertainty, threshold_for_sensitivity
from research.models.fusion_model import assert_eval_transform_matches_backend, build_model
from research.training.predict import collect_logits
from research.utils.config import REPO_ROOT, load_config, repo_path
from research.utils.environment import collect_environment
from research.utils.hashing import config_hash, sha256_file
from research.utils.io import read_json, write_json

SEEDS = [123, 2024]
ARMS = {"fusion": "research/configs/e0.yaml", "effnet": "research/configs/baseline_effnet.yaml", "resnet": "research/configs/baseline_resnet.yaml"}
RUN_DIRS = {"fusion": "research/outputs/e0_baseline", "effnet": "research/outputs/baseline_effnet", "resnet": "research/outputs/baseline_resnet"}
TARGET = 0.95


class BaselineError(RuntimeError):
    pass


def _check(cond, msg):
    if not cond:
        raise BaselineError(msg)


def verify_arm_run(run_dir: Path, cfg: dict, split_sha: str) -> dict:
    meta = read_json(run_dir / "run_metadata.json")
    _check(meta.get("status") == "complete" and not meta.get("smoke_test"), f"{run_dir} is not a complete run")
    _check(meta["config_hash"] == config_hash(cfg), f"{run_dir} config differs from its yaml")
    _check(meta["split_sha256"] == split_sha, f"{run_dir} used another split")
    _check(sha256_file(run_dir / "best.pt") == meta["checkpoint_sha256"], f"{run_dir}/best.pt differs from its record")
    return meta


def probs_from(logits, temperature):
    return np.exp(log_softmax64(np.asarray(logits, dtype=np.float64) / temperature))


def predict_isic(cfg, run_dir, frame, images, classes, seed, cache: Path):
    """Logits for one arm member, saved so the comparison never needs a second prediction pass."""
    if cache.is_file():
        return np.load(cache)["logits"]
    _, eval_tf = build_transforms(cfg)
    assert_eval_transform_matches_backend(eval_tf)
    dataset = HAM10000Dataset(frame, {i: images / f"{i}.jpg" for i in frame["image_id"]}, classes, eval_tf)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = make_loader(dataset, cfg["training"]["batch_size"], shuffle=False, drop_last=False, num_workers=0, seed=seed)
    logits, rows = collect_logits(model, loader, torch.device("cpu"))
    _check(np.array_equal(rows, np.arange(len(frame))), "prediction order changed")
    np.savez(cache, logits=logits)
    return logits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isic-images", required=True)
    ap.add_argument("--isic-ground-truth", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--output-dir", default="research/outputs/baseline_comparison")
    ap.add_argument("--effnet-dir", default=RUN_DIRS["effnet"])
    ap.add_argument("--resnet-dir", default=RUN_DIRS["resnet"])
    args = ap.parse_args(argv)
    command = shlex.join(["python", "-m", "research.evaluation.baseline_eval", *(argv if argv is not None else sys.argv[1:])])

    out = repo_path(args.output_dir)
    _check(not out.exists() or not any(p for p in out.iterdir() if p.name != "cache"), f"{out} already has results; the comparison runs once")
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    fit_cfg = load_config(repo_path("research/configs/calibration_ts.yaml"))["fit"]
    configs = {k: load_config(repo_path(v)) for k, v in ARMS.items()}
    run_dirs = {"fusion": repo_path(RUN_DIRS["fusion"]), "effnet": repo_path(args.effnet_dir), "resnet": repo_path(args.resnet_dir)}
    e0 = configs["fusion"]
    classes = e0["data"]["classes"]
    mal_idx = [classes.index(c) for c in e0["evaluation"]["malignant_classes"]]
    split_df, split_sha = load_split(e0, verify_hash=True)

    frame = read_ground_truth(args.isic_ground_truth, classes)
    y_isic = np.array([classes.index(d) for d in frame["dx"]])
    images = Path(args.isic_images)
    _check(all((images / f"{i}.jpg").is_file() for i in frame["image_id"]), "an ISIC image is missing")
    _check(repo_path("research/outputs/isic2018_test/overlap_check.json").is_file(), "run isic_eval first: it checks overlap with HAM10000")
    meta = pd.read_csv(args.metadata).set_index("image_id")

    results, ensembles = {}, {}
    y_test = lesions = biopsy = None
    for arm, cfg in configs.items():
        val_probs, test_probs, isic_probs, per_seed = [], [], [], {}
        for seed in SEEDS:
            run_dir = run_dirs[arm] / f"seed_{seed}"
            run = verify_arm_run(run_dir, cfg, split_sha)
            val = load_predictions(run_dir, "val", seed, classes, split_df)
            test = load_predictions(run_dir, "test", seed, classes, split_df)
            T = fit_temperature(val["logits"], val["y"], fit_cfg)["temperature"]
            if arm == "fusion":
                saved = read_json(repo_path("research/outputs/calibration") / f"seed_{seed}" / "temperature.json")["temperature"]
                _check(abs(T - saved) < 1e-6, f"fusion seed {seed}: temperature differs from its calibration run")
                isic_logits = np.load(repo_path("research/outputs/isic2018_test") / f"logits_seed_{seed}.npz")["logits"]
            else:
                isic_logits = predict_isic(cfg, run_dir, frame, images, classes, seed, cache / f"isic_{arm}_{seed}.npz")
            if y_test is None:
                y_test, lesions = test["y"], np.asarray(test["lesion_id"])
                test_ids = np.asarray(test["image_id"])
                y_val, val_ids = val["y"], np.asarray(val["image_id"])
                biopsy = meta.loc[test_ids, "dx_type"].to_numpy() == "histo"
            _check(np.array_equal(np.asarray(test["image_id"]), test_ids) and np.array_equal(np.asarray(val["image_id"]), val_ids), f"{arm} seed {seed}: row order differs")
            pv, pt, pi = probs_from(val["logits"], T), probs_from(test["logits"], T), probs_from(isic_logits, T)
            val_probs.append(pv)
            test_probs.append(pt)
            isic_probs.append(pi)
            per_seed[str(seed)] = {"temperature": T, "selected_epoch": run["selected_epoch"], "device": run["runtime"]["device"],
                                   "git_commit": run["environment"]["git_commit"], "git_dirty": run["environment"]["git_dirty"],
                                   **evaluate_model(pv, pt, pi, y_val, y_test, y_isic, biopsy, classes, mal_idx, e0["evaluation"]["ece_bins"])}
        pv, pt, pi = np.mean(val_probs, axis=0), np.mean(test_probs, axis=0), np.mean(isic_probs, axis=0)
        ensembles[arm] = {"t": threshold_for_sensitivity(pv[:, mal_idx].sum(axis=1), np.isin(y_val, mal_idx), TARGET)["threshold"], "test": pt, "isic": pi}
        results[arm] = {"seeds": per_seed, "ensemble_2_seeds": evaluate_model(pv, pt, pi, y_val, y_test, y_isic, biopsy, classes, mal_idx, e0["evaluation"]["ece_bins"])}

    # ---- paired differences: fusion minus arm, 2-seed ensembles ----
    def metrics(p, t, y, idx, mask=None):
        yy, pred = y[idx], p[idx].argmax(axis=1)
        correct = pred == yy
        s_mal = p[idx][:, mal_idx].sum(axis=1) >= t
        mal = np.isin(yy, mal_idx)
        msp = 1 - p[idx].max(axis=1)
        out_ = {"accuracy": float(correct.mean()), "balanced_accuracy": balanced_accuracy(yy, pred, len(classes)),
                "melanoma_recall": float(correct[yy == classes.index("mel")].mean()),
                "aurc_msp": aurc_for_uncertainty(msp, correct)["aurc"],
                "referral_sensitivity": float(s_mal[mal].mean()), "referral_specificity": float((~s_mal[~mal]).mean()), "referral_rate": float(s_mal.mean())}
        if mask is not None:
            b = mask[idx]
            out_["aurc_msp_biopsy"] = aurc_for_uncertainty(msp[b], correct[b])["aurc"]
        return out_

    bs = {"n_resamples": 2000, "seed": 0, "confidence_level": 0.95}
    diffs = {}
    for arm in ("effnet", "resnet"):
        def on_test(idx, a=arm):
            f, o = metrics(ensembles["fusion"]["test"], ensembles["fusion"]["t"], y_test, idx, biopsy), metrics(ensembles[a]["test"], ensembles[a]["t"], y_test, idx, biopsy)
            return {k: f[k] - o[k] for k in f}

        def on_isic(idx, a=arm):
            f, o = metrics(ensembles["fusion"]["isic"], ensembles["fusion"]["t"], y_isic, idx), metrics(ensembles[a]["isic"], ensembles[a]["t"], y_isic, idx)
            return {k: f[k] - o[k] for k in f}

        diffs[arm] = {"ham10000_test": cluster_bootstrap(on_test, lesions, bs["n_resamples"], bs["seed"], bs["confidence_level"]),
                      "isic2018_test": cluster_bootstrap(on_isic, frame["image_id"].to_numpy(), bs["n_resamples"], bs["seed"], bs["confidence_level"])}

    write_json(out / "summary.json", {"label": "Comparison added after E0's test split was viewed; ISIC run once", "seeds": SEEDS, "results": results, "fusion_minus_arm": diffs})
    write_json(out / "provenance.json", {"command": command, "environment": collect_environment(REPO_ROOT)})
    (out / "report.md").write_text(report(results, diffs), encoding="utf-8")
    print((out / "report.md").read_text(encoding="utf-8"))
    return 0


def evaluate_model(pv, pt, pi, y_val, y_test, y_isic, biopsy, classes, mal_idx, n_bins) -> dict:
    t = threshold_for_sensitivity(pv[:, mal_idx].sum(axis=1), np.isin(y_val, mal_idx), TARGET)["threshold"]
    refs = {"sensitivity_0.95": t}
    st, si = scores_from_probs(pt, mal_idx), scores_from_probs(pi, mal_idx)
    return {"threshold": t,
            "ham10000_test": {"all": stratum_metrics(st, y_test, np.ones(len(y_test), dtype=bool), classes, mal_idx, refs),
                              "biopsy_confirmed": stratum_metrics(st, y_test, biopsy, classes, mal_idx, refs), "calibration": calibration_block(pt, y_test, n_bins)},
            "isic2018_test": {"all": stratum_metrics(si, y_isic, np.ones(len(y_isic), dtype=bool), classes, mal_idx, refs), "calibration": calibration_block(pi, y_isic, n_bins)}}


def report(results, diffs) -> str:
    def row(arm, r, dataset):
        a = r[dataset]["all"]
        ref = a["referral"]["sensitivity_0.95"]
        return (f"| {arm} | {100 * a['accuracy']:.1f}% | {100 * a['balanced_accuracy']:.1f}% | {100 * a['melanoma_recall']:.1f}% | {a['aurc_msp']:.4f} | {r[dataset]['calibration']['ece']:.4f} | "
                f"{100 * ref['sensitivity']:.1f}% | {100 * ref['specificity']:.1f}% | {100 * ref['referral_rate']:.1f}% | {ref['malignant_cleared']} of {ref['n_malignant']} |")

    lines = ["# Fusion against single backbones, seeds 123 and 2024", "", "Added after the HAM10000 test split had been viewed; ISIC 2018 run once.", ""]
    for dataset in ("ham10000_test", "isic2018_test"):
        lines += [f"## {dataset}: 2-seed ensembles", "", "| model | accuracy | balanced | melanoma recall | AURC | ECE | referral sensitivity | specificity | referred | malignant cleared |", "|---|---|---|---|---|---|---|---|---|---|"]
        lines += [row(arm, results[arm]["ensemble_2_seeds"], dataset) for arm in results]
        lines += ["", f"## {dataset}: single seeds", "", "| model, seed | accuracy | balanced | melanoma recall | AURC | ECE | referral sensitivity | specificity | referred | malignant cleared |", "|---|---|---|---|---|---|---|---|---|---|"]
        lines += [row(f"{arm} {seed}", r, dataset) for arm in results for seed, r in results[arm]["seeds"].items()]
        lines += ["", f"## {dataset}: fusion minus arm, 95% CI", "", "| arm | metric | difference [CI] |", "|---|---|---|"]
        for arm, by_set in diffs.items():
            for name, b in by_set[dataset]["metrics"].items():
                lines.append(f"| {arm} | {name} | {b['point']:+.4f} [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] |")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
