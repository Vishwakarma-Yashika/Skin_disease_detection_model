"""
Evaluate E0 from saved raw logits (no model, no images needed).

    python -m research.evaluation.evaluate --config research/configs/e0.yaml

Per seed  -> research/outputs/E0/seed_<s>/evaluation/
    metrics_val.json, metrics_test.json      classification, reliability, selective, binary, melanoma
    bootstrap_test.json                      lesion-cluster bootstrap 95% CIs on the test predictions
    reliability_{val,test}.png, risk_coverage_{val,test}.png
Across seeds -> research/outputs/E0/summary/
    e0_summary.json, e0_summary.md, risk_coverage_test_all_seeds.png

Two different uncertainties are reported and never mixed:
    * seed variation  = mean +/- SD (ddof=1) of the per-seed point estimates
    * bootstrap CI    = sampling uncertainty of the test set, per seed
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.data.data import load_split
from research.evaluation.calibration_metrics import (
    brier,
    classwise_ece,
    nll,
    plot_reliability_diagram,
    top_label_calibration,
)
from research.evaluation.metrics import (
    binary_malignant_metrics,
    classification_metrics,
    cluster_bootstrap,
    melanoma_operating_point,
    sens_spec_at_threshold,
    softmax64,
)
from research.evaluation.selective_prediction import (
    aurc,
    optimal_aurc,
    plot_risk_coverage,
    risk_coverage_curve,
    selective_metrics,
)
from research.utils.config import load_config, repo_path
from research.utils.hashing import config_hash
from research.utils.io import read_json, write_json

# backend/app/class_info.py is the project's declared clinical mapping.
from research.models import fusion_model as _fm  # noqa: F401  (puts backend/ on sys.path)
from app.class_info import CLASS_INFO  # noqa: E402

SUMMARY_METRICS = [
    "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
    "macro_roc_auc_ovr", "nll", "brier", "ece", "mce", "classwise_ece", "aurc", "e_aurc",
    "malignant_sensitivity", "malignant_specificity", "malignant_auroc",
    "mel_sensitivity_at_val_threshold", "mel_specificity_at_val_threshold",
]


class EvaluationError(RuntimeError):
    pass


def verify_malignant_mapping(cfg: dict) -> None:
    ev, classes = cfg["evaluation"], cfg["data"]["classes"]
    declared = {c for c in classes if CLASS_INFO[c]["category"] == "malignant"}
    if set(ev["malignant_classes"]) != declared:
        raise EvaluationError(f"Config malignant_classes {ev['malignant_classes']} != class_info.py {sorted(declared)}")
    if set(ev["malignant_classes"]) | set(ev["benign_classes"]) != set(classes) or \
            set(ev["malignant_classes"]) & set(ev["benign_classes"]):
        raise EvaluationError("malignant_classes/benign_classes must partition data.classes")


def load_predictions(run_dir: Path, split: str, seed: int, classes: list[str], split_df: pd.DataFrame) -> dict:
    path = run_dir / "predictions" / f"{split}.npz"
    if not path.is_file():
        raise EvaluationError(f"Missing predictions: {path}")
    d = np.load(path, allow_pickle=False)
    if int(d["seed"]) != seed:
        raise EvaluationError(f"{path} was produced by seed {int(d['seed'])}, expected {seed}")
    if list(d["class_names"]) != classes:
        raise EvaluationError(f"{path} class order {list(d['class_names'])} != config {classes}")

    expected = split_df[split_df["split"] == split].set_index("image_id")
    ids = pd.Index(d["image_id"])
    if ids.duplicated().any():
        raise EvaluationError(f"{path} contains duplicate image_ids")
    if set(ids) != set(expected.index):
        raise EvaluationError(f"{path} image_ids do not match the '{split}' split ({len(ids)} vs {len(expected)})")
    true = np.array([classes.index(dx) for dx in expected.loc[ids, "dx"]])
    if not np.array_equal(true, d["labels"]):
        raise EvaluationError(f"{path} labels disagree with the split file")
    # Probabilities are recomputed from the raw logits so the logits are the single source of truth.
    return {"logits": d["logits"], "probs": softmax64(d["logits"]), "y": true,
            "lesion_id": d["lesion_id"], "image_id": d["image_id"]}


def evaluate_split(p: dict, cfg: dict, classes: list[str]) -> dict:
    ev = cfg["evaluation"]
    cls = classification_metrics(p["y"], p["probs"], classes)
    top = top_label_calibration(p["probs"], p["y"], ev["ece_bins"])
    return {
        "classification": cls,
        "reliability": {
            "nll": nll(p["logits"], p["y"]),
            "brier": brier(p["probs"], p["y"]),
            "brier_definition": "mean over samples of sum_k (p_k - y_k)^2, range [0, 2]",
            "ece": top["ece"],
            "mce": top["mce"],
            "ece_bins": top["bins"],
            "classwise": classwise_ece(p["probs"], p["y"], ev["ece_bins"], classes),
            "n_bins": ev["ece_bins"],
            "binning": ev["ece_binning"],
            "probabilities": "raw softmax of logits, NOT calibrated",
        },
        "selective": selective_metrics(p["probs"], p["y"], ev["selective_coverages"]),
        "malignant_vs_benign": binary_malignant_metrics(p["y"], p["probs"], classes, ev["malignant_classes"]),
    }


def scalar_metrics(logits, probs, y, classes, malignant, n_bins, mel_threshold) -> dict:
    """Flat scalar view used for both the per-seed summary and every bootstrap resample."""
    cls = classification_metrics(y, probs, classes)
    top = top_label_calibration(probs, y, n_bins)
    conf, correct = probs.max(axis=1), probs.argmax(axis=1) == y
    a = aurc(conf, correct)
    binary = binary_malignant_metrics(y, probs, classes, malignant)
    mel = classes.index("mel")
    mel_sens, mel_spec = sens_spec_at_threshold(probs[:, mel], y == mel, mel_threshold)
    return {
        **{k: cls[k] for k in ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
                               "macro_f1", "weighted_f1", "macro_roc_auc_ovr"]},
        "nll": nll(logits, y),
        "brier": brier(probs, y),
        "ece": top["ece"],
        "mce": top["mce"],
        "classwise_ece": classwise_ece(probs, y, n_bins, classes)["classwise_ece"],
        "aurc": a,
        "e_aurc": a - optimal_aurc(correct),
        "malignant_sensitivity": binary["sensitivity"],
        "malignant_specificity": binary["specificity"],
        "malignant_auroc": binary["auroc"],
        "mel_sensitivity_at_val_threshold": mel_sens,
        "mel_specificity_at_val_threshold": mel_spec,
    }


def evaluate_seed(run_dir: Path, seed: int, cfg: dict, split_df: pd.DataFrame, cfg_hash: str) -> dict:
    classes, ev = cfg["data"]["classes"], cfg["evaluation"]
    meta = read_json(run_dir / "run_metadata.json")
    if meta.get("config_hash") != cfg_hash:
        raise EvaluationError(f"{run_dir} was trained with config_hash {meta.get('config_hash')}, not {cfg_hash}")
    if meta.get("status") != "complete":
        raise EvaluationError(f"{run_dir} did not finish (status={meta.get('status')})")

    out_dir = run_dir / "evaluation"
    out_dir.mkdir(exist_ok=True)
    preds = {s: load_predictions(run_dir, s, seed, classes, split_df) for s in ("val", "test")}

    mel_cfg = ev["melanoma_operating_point"]
    mel_op = melanoma_operating_point(
        preds["val"]["y"], preds["val"]["probs"], preds["test"]["y"], preds["test"]["probs"],
        preds["test"]["lesion_id"], classes, mel_cfg["target_specificity"], mel_cfg["min_test_positive_lesions"],
    )

    results = {}
    for split, p in preds.items():
        res = evaluate_split(p, cfg, classes)
        if split == "test":
            res["melanoma_operating_point"] = mel_op
        res.update({"seed": seed, "split": split, "selected_epoch": meta["selected_epoch"]})
        write_json(out_dir / f"metrics_{split}.json", res)
        plot_reliability_diagram(res["reliability"]["ece_bins"], out_dir / f"reliability_{split}.png",
                                 f"E0 seed {seed} - {split} (raw softmax)")
        cov, risk, _ = risk_coverage_curve(p["probs"].max(axis=1), p["probs"].argmax(axis=1) == p["y"])
        plot_risk_coverage([(f"seed {seed}", cov, risk)], out_dir / f"risk_coverage_{split}.png",
                           f"E0 seed {seed} - {split} (MSP)")
        results[split] = res

    t = preds["test"]
    thr = mel_op["threshold_p_mel"]

    def on_subset(idx):
        return scalar_metrics(t["logits"][idx], t["probs"][idx], t["y"][idx], classes,
                              ev["malignant_classes"], ev["ece_bins"], thr)

    bs = ev["bootstrap"]
    boot = cluster_bootstrap(on_subset, t["lesion_id"], bs["n_resamples"], bs["seed"], bs["confidence_level"])
    boot["note"] = "Test-set sampling uncertainty for this seed only; melanoma threshold held fixed from validation."
    write_json(out_dir / "bootstrap_test.json", boot)

    return {"seed": seed, "test_scalars": on_subset(np.arange(len(t["y"]))), "bootstrap": boot,
            "test": results["test"], "curve": risk_coverage_curve(t["probs"].max(axis=1),
                                                                   t["probs"].argmax(axis=1) == t["y"])}


def summarize(per_seed: list[dict], cfg: dict, declared_seeds: list[int]) -> dict:
    classes = cfg["data"]["classes"]
    seeds = [r["seed"] for r in per_seed]
    table = pd.DataFrame([r["test_scalars"] for r in per_seed], index=seeds)[SUMMARY_METRICS]
    across = {m: {"mean": table[m].mean(), "sd": table[m].std(ddof=1) if len(seeds) > 1 else float("nan"),
                  "min": table[m].min(), "max": table[m].max(), "per_seed": table[m].to_dict()}
              for m in SUMMARY_METRICS}

    per_class = {}
    for c in classes:
        per_class[c] = {}
        for k in ["precision", "recall", "f1", "specificity", "roc_auc_ovr"]:
            vals = np.array([r["test"]["classification"]["per_class"][c][k] for r in per_seed])
            per_class[c][k] = {"mean": vals.mean(), "sd": vals.std(ddof=1) if len(vals) > 1 else float("nan")}

    cms = np.array([r["test"]["classification"]["confusion_matrix"] for r in per_seed], dtype=float)
    row_norm = cms / cms.sum(axis=2, keepdims=True)

    complete = sorted(seeds) == sorted(declared_seeds)
    return {
        "experiment": cfg["experiment"]["name"],
        "status": "complete" if complete else "INCOMPLETE - not all predeclared seeds evaluated",
        "declared_seeds": declared_seeds,
        "evaluated_seeds": seeds,
        "split": "test",
        "seed_variation": {"description": "mean and SD (ddof=1) of per-seed test point estimates",
                           "metrics": across, "per_class": per_class,
                           "mean_row_normalized_confusion_matrix": row_norm.mean(axis=0).tolist()},
        "bootstrap_per_seed": {
            "description": "lesion-cluster percentile bootstrap CIs on each seed's test predictions",
            "seeds": {r["seed"]: {m: r["bootstrap"]["metrics"][m] for m in SUMMARY_METRICS} for r in per_seed},
        },
        "melanoma_operating_point_per_seed": {r["seed"]: r["test"]["melanoma_operating_point"] for r in per_seed},
        "class_names": classes,
    }


def summary_markdown(summary: dict) -> str:
    lines = [
        f"# E0 test-set summary ({summary['status']})",
        "",
        f"Seeds evaluated: {summary['evaluated_seeds']} (declared: {summary['declared_seeds']})",
        "",
        "Seed variation = mean ± SD over seeds. Bootstrap = lesion-cluster 95% CI per seed (range of CI bounds over seeds shown).",
        "Probabilities are raw softmax outputs (uncalibrated).",
        "",
        "| metric | mean ± SD (seeds) | min – max (seeds) | bootstrap CI low (min over seeds) | bootstrap CI high (max over seeds) |",
        "|---|---|---|---|---|",
    ]
    boots = summary["bootstrap_per_seed"]["seeds"]
    for m, v in summary["seed_variation"]["metrics"].items():
        lows = [b[m]["ci_low"] for b in boots.values()]
        highs = [b[m]["ci_high"] for b in boots.values()]
        lines.append(f"| {m} | {v['mean']:.4f} ± {v['sd']:.4f} | {v['min']:.4f} – {v['max']:.4f} | "
                     f"{np.nanmin(lows):.4f} | {np.nanmax(highs):.4f} |")
    lines += ["", "## Per-class (test, mean ± SD over seeds)", "",
              "| class | precision | recall | F1 | specificity | ROC-AUC (OvR) |", "|---|---|---|---|---|---|"]
    for c, d in summary["seed_variation"]["per_class"].items():
        lines.append(f"| {c} | " + " | ".join(f"{d[k]['mean']:.3f} ± {d[k]['sd']:.3f}"
                                             for k in ["precision", "recall", "f1", "specificity", "roc_auc_ovr"]) + " |")
    lines += ["", "## Melanoma operating point (threshold from validation, applied to test)", "",
              "| seed | target spec | p(mel) threshold | test sensitivity | test specificity | positive lesions | sufficient |",
              "|---|---|---|---|---|---|---|"]
    for s, d in summary["melanoma_operating_point_per_seed"].items():
        lines.append(f"| {s} | {d['target_specificity']} | {d['threshold_p_mel']:.4f} | {d['test_sensitivity']:.4f} | "
                     f"{d['test_specificity']:.4f} | {d['test_positive_lesions']} | {d['sufficient_sample']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="default: all predeclared seeds")
    parser.add_argument("--output-dir", default=None, help="override experiment.output_dir (e.g. the smoke dir)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg_hash = config_hash(cfg)
    verify_malignant_mapping(cfg)
    split_df, _ = load_split(cfg, verify_hash=True)
    base = repo_path(args.output_dir or cfg["experiment"]["output_dir"])
    declared = cfg["experiment"]["seeds"]
    seeds = args.seeds or declared

    per_seed = []
    for seed in seeds:
        print(f"evaluating seed {seed} ...", flush=True)
        per_seed.append(evaluate_seed(base / f"seed_{seed}", seed, cfg, split_df, cfg_hash))

    summary_dir = base / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(per_seed, cfg, declared)
    write_json(summary_dir / "e0_summary.json", summary)
    (summary_dir / "e0_summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    plot_risk_coverage([(f"seed {r['seed']}", r["curve"][0], r["curve"][1]) for r in per_seed],
                       summary_dir / "risk_coverage_test_all_seeds.png", "E0 test risk-coverage (MSP)")
    print(summary_markdown(summary))
    print(f"wrote {summary_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
