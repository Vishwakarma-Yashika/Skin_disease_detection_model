"""
STEP 1 -- Baseline reliability evaluation of the EXISTING checkpoint (evaluation only).

Two explicitly labelled evaluation modes:

  # Existing checkpoint on the E0 split (cross-protocol; see report caveats)
  python -m research.evaluation.step1_reliability --e0-split test --data-root <HAM10000 root>

  # Existing checkpoint on an explicit manifest (columns: image_id, lesion_id, label, image_path)
  python -m research.evaluation.step1_reliability --manifest evaluation_manifest.csv

No retraining, calibration, temperature scaling, dropout sampling, ensembling, OOD
detection, conformal prediction or test-time augmentation. The checkpoint is read-only
(its SHA-256 is verified before and after the run).
"""
import argparse
import io
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from research.data.data import load_validated_split
from research.evaluation import reliability_figures as figs
from research.evaluation.calibration_metrics import brier, classwise_ece, nll, top_label_calibration
from research.evaluation.metrics import (
    binary_malignant_metrics,
    classification_metrics,
    cluster_bootstrap,
    log_softmax64,
    softmax64,
)
from research.evaluation.reliability_analysis import (
    compare_correct_incorrect,
    confidence_bin_table,
    error_detection,
    high_confidence_errors,
    operating_points,
    predictive_entropy,
    top2_margin,
)
from research.evaluation.selective_prediction import optimal_aurc, risk_coverage_curve, selective_metrics
from research.models.fusion_model import FusionClassifier, backend_inference_transform, verify_architecture
from research.utils.config import REPO_ROOT, load_config, repo_path, resolve_data_root
from research.utils.environment import collect_environment
from research.utils.hashing import canonical_json, sha256_file, sha256_text
from research.utils.io import write_json
from research.utils.seed import set_seed

# Backend modules (on sys.path via research.models.fusion_model).
from app import config as backend_config  # noqa: E402
from app.class_info import CLASS_INFO  # noqa: E402

E0_CONFIG = "research/configs/e0.yaml"
MANIFEST_COLUMNS = ["image_id", "lesion_id", "label", "image_path"]


class Step1Error(RuntimeError):
    """A STEP 1 validation check failed. Never caught: the run stops."""


def _check(condition: bool, message: str):
    if not condition:
        raise Step1Error(message)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path, expected_classes: list[str], e0_cfg: dict, device):
    _check(path.is_file(), f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise Step1Error(f"Checkpoint failed to load: {exc!r}") from exc
    _check({"model_state_dict", "diagnosis_classes"} <= set(ckpt), f"Checkpoint keys: {sorted(ckpt)}")
    classes = list(ckpt["diagnosis_classes"])
    _check(classes == expected_classes, f"Checkpoint class order {classes} != expected {expected_classes}")

    # Same construction as backend/app/model.py ModelRegistry.ensure_loaded.
    model = FusionClassifier(num_classes=len(classes), feature_dim=512, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    verify_architecture(model, e0_cfg)
    model.to(device).eval()
    _check(not any(m.training for m in model.modules()), "Model is not fully in eval mode")
    info = {k: ckpt[k] for k in ("val_acc", "epoch") if k in ckpt}
    return model, classes, info


def validate_manifest(frame: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    missing = set(MANIFEST_COLUMNS) - set(frame.columns)
    _check(not missing, f"Manifest is missing columns: {sorted(missing)}")
    frame = frame[MANIFEST_COLUMNS].copy()
    nulls = frame.isna().sum()
    _check(not nulls.any(), f"Manifest has empty values (every image needs a label): {nulls[nulls > 0].to_dict()}")
    dup = frame["image_id"][frame["image_id"].duplicated()]
    _check(dup.empty, f"Duplicate image_ids: {dup.head().tolist()}")
    bad = sorted(set(frame["label"]) - set(classes))
    _check(not bad, f"Labels outside the checkpoint classes: {bad}")
    paths = frame["image_path"].map(Path)
    absent = [str(p) for p in paths if not p.is_file()]
    _check(not absent, f"{len(absent)} missing image file(s), e.g. {absent[:5]}")
    resolved = paths.map(lambda p: str(p.resolve()).lower())
    _check(not resolved.duplicated().any(), "Two image_ids point to the same file")
    return frame


def e0_manifest(split: str, data_root: Path, e0_cfg: dict) -> tuple[pd.DataFrame, dict]:
    split_df, image_index, _, split_hash = load_validated_split(data_root, e0_cfg)  # hash + leakage checks
    part = split_df[split_df["split"] == split]
    frame = pd.DataFrame({"image_id": part["image_id"], "lesion_id": part["lesion_id"], "label": part["dx"],
                          "image_path": [str(image_index[i]) for i in part["image_id"]]})
    return frame.reset_index(drop=True), {"e0_split_file": e0_cfg["split"]["file"], "e0_split_sha256": split_hash,
                                          "e0_split_name": split}


class ManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.paths = frame["image_path"].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            img = img.convert("RGB")
        return backend_inference_transform(img), i


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, frame, batch_size, num_workers, device) -> np.ndarray:
    loader = DataLoader(ManifestDataset(frame), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logits, rows = [], []
    for x, idx in loader:
        logits.append(model(x.to(device)).float().cpu().numpy())
        rows.append(idx.numpy())
    rows = np.concatenate(rows)
    _check(np.array_equal(rows, np.arange(len(frame))), "Inference order does not match the manifest order")
    return np.concatenate(logits)


@torch.no_grad()
def backend_parity(model, frame, logits, n, tol, device) -> dict:
    """Re-run the first n images exactly as backend/app/main.py /api/predict does (one image at a time)."""
    diffs = []
    for i in range(min(n, len(frame))):
        raw = Path(frame["image_path"].iloc[i]).read_bytes()
        image = Image.open(io.BytesIO(raw))
        image.load()
        x = backend_inference_transform(image.convert("RGB")).unsqueeze(0).to(device)
        diffs.append(float(np.abs(model(x).float().cpu().numpy()[0] - logits[i]).max()))
    worst = max(diffs)
    _check(worst <= tol, f"Batched logits differ from the backend single-image path by {worst} > {tol}")
    return {"n_images": len(diffs), "max_abs_logit_difference": worst, "tolerance": tol}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def bootstrap_scalars(logits, probs, y, lesions, classes, malignant, ece_bins, bs_cfg):
    mel = classes.index("mel")

    def fn(idx):
        lg, p, yy = logits[idx], probs[idx], y[idx]
        cls = classification_metrics(yy, p, classes)
        conf, ok = p.max(axis=1), p.argmax(axis=1) == yy
        ent, marg = predictive_entropy(lg), top2_margin(p)
        det = error_detection({"msp": -conf, "entropy": ent, "margin": -marg}, ok)
        sel = selective_metrics(p, yy, [1.0])
        binary = binary_malignant_metrics(yy, p, classes, malignant)
        return {
            "accuracy": cls["accuracy"], "balanced_accuracy": cls["balanced_accuracy"], "macro_f1": cls["macro_f1"],
            "weighted_f1": cls["weighted_f1"], "macro_roc_auc_ovr": cls["macro_roc_auc_ovr"],
            "mel_recall": cls["per_class"]["mel"]["recall"] if (yy == mel).any() else float("nan"),
            "nll": nll(lg, yy), "brier": brier(p, yy), "ece": top_label_calibration(p, yy, ece_bins)["ece"],
            "classwise_ece": classwise_ece(p, yy, ece_bins, classes)["classwise_ece"],
            "aurc": sel["aurc"], "e_aurc": sel["e_aurc"],
            "error_auroc_msp": det["msp"]["auroc_error_detection"],
            "error_auroc_entropy": det["entropy"]["auroc_error_detection"],
            "error_auroc_margin": det["margin"]["auroc_error_detection"],
            "error_auroc_entropy_minus_msp": det["entropy"]["auroc_error_detection"] - det["msp"]["auroc_error_detection"],
            "error_auroc_margin_minus_msp": det["margin"]["auroc_error_detection"] - det["msp"]["auroc_error_detection"],
            "malignant_sensitivity": binary["sensitivity"], "malignant_specificity": binary["specificity"],
            "malignant_auroc": binary["auroc"],
        }

    return cluster_bootstrap(fn, lesions, bs_cfg["n_resamples"], bs_cfg["seed"], bs_cfg["confidence_level"])


def binary_confusion(y, probs, classes, malignant) -> dict:
    idx = [classes.index(c) for c in malignant]
    t, p = np.isin(y, idx), np.isin(probs.argmax(axis=1), idx)
    return {"rows": "true [malignant, benign]", "cols": "predicted [malignant, benign]",
            "matrix": [[int((t & p).sum()), int((t & ~p).sum())], [int((~t & p).sum()), int((~t & ~p).sum())]]}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _ci(boot, key, fmt="{:.4f}"):
    m = boot["metrics"][key]
    return f"{fmt.format(m['point'])} [{fmt.format(m['ci_low'])}, {fmt.format(m['ci_high'])}]"


def build_report(r: dict) -> str:
    e, cls, cal, sel, conf, boot = r["experiment"], r["classification"], r["calibration"], r["selective"], r["confidence"], r["bootstrap"]
    classes = e["class_order"]
    L = [f"# STEP 1 — Baseline reliability evaluation: {e['evaluation_label']}", ""]
    L += ["> **Scope.** Evaluation of the existing (historical) checkpoint only. No retraining, calibration, temperature",
          "> scaling, dropout sampling, ensembling, OOD detection, conformal prediction or test-time augmentation.",
          "> STEP 1 does not complete the research; the next step (calibration) follows only after these results are reviewed.",
          "", "## 1–2. Evaluation set and provenance", ""]
    L += [f"- **Evaluation type:** {e['evaluation_type']}", f"- **Label:** {e['evaluation_label']}"]
    for line in e["provenance_notes"]:
        L.append(f"- {line}")
    L += [f"- Checkpoint: `{e['checkpoint_path_relative']}` (SHA-256 `{e['checkpoint_sha256']}`), stored val_acc "
          f"{e['checkpoint_info'].get('val_acc')}, epoch {e['checkpoint_info'].get('epoch')}",
          f"- Evaluation set content hash: `{e['evaluation_set_content_sha256']}`",
          f"- Preprocessing: {e['preprocessing']}",
          f"- Backend parity: max |Δlogit| = {e['backend_parity']['max_abs_logit_difference']:.2e} over "
          f"{e['backend_parity']['n_images']} images (batched vs backend single-image path)", ""]

    L += ["## 3–4. Samples and class distribution", "",
          f"N = {e['n_samples']} images from {e['n_lesions']} lesions, {e['n_classes']} classes.", "",
          "| class | images | % |", "|---|---|---|"]
    for c in classes:
        s = cls["per_class"][c]["support"]
        L.append(f"| {c} | {s} | {100 * s / e['n_samples']:.1f} |")

    L += ["", "## 5. Classification (point estimate [lesion-bootstrap 95% CI])", "",
          f"- Accuracy: {_ci(boot, 'accuracy')}", f"- Balanced accuracy: {_ci(boot, 'balanced_accuracy')}",
          f"- Macro precision / recall: {cls['macro_precision']:.4f} / {cls['macro_recall']:.4f}",
          f"- Macro F1: {_ci(boot, 'macro_f1')}", f"- Weighted F1: {_ci(boot, 'weighted_f1')}",
          f"- Macro OvR AUROC: {_ci(boot, 'macro_roc_auc_ovr')}", "",
          "| class | support | precision | recall | F1 | specificity | OvR AUROC |", "|---|---|---|---|---|---|---|"]
    for c in classes:
        d = cls["per_class"][c]
        L.append(f"| {c} | {d['support']} | {d['precision']:.3f} | {d['recall']:.3f} | {d['f1']:.3f} | "
                 f"{d['specificity']:.3f} | {d['roc_auc_ovr']:.3f} |")
    L += ["", "Confusion matrix (rows = true, cols = predicted, order " + ", ".join(classes) + "):", "", "```"]
    L += ["        " + " ".join(f"{c:>6}" for c in classes)]
    L += [f"{c:>6}  " + " ".join(f"{v:>6}" for v in row) for c, row in zip(classes, cls["confusion_matrix"])]
    L += ["```", ""]

    tb = cal["confidence_bins_10"]
    L += ["## 6. Calibration (measured only — probabilities not modified)", "",
          f"- NLL: {_ci(boot, 'nll')}", f"- Multiclass Brier (Σ_k (p_k − y_k)², range 0–2): {_ci(boot, 'brier')}",
          f"- ECE (15 equal-width bins, E0 convention): {_ci(boot, 'ece')}", f"- MCE (15 bins): {cal['mce']:.4f}",
          f"- Classwise ECE (15 bins, mean over classes): {_ci(boot, 'classwise_ece')}",
          f"- ECE / MCE on the 10-bin table below: {tb['ece_on_these_bins']:.4f} / {tb['mce_on_these_bins']:.4f}", "",
          "ECE = Σ_b (n_b / N) · |acc_b − conf_b| with conf = max softmax probability; MCE = max_b |acc_b − conf_b| over non-empty bins.",
          "", "| bin | n | mean confidence | accuracy | |gap| |", "|---|---|---|---|---|"]
    for b in tb["bins"]:
        if b["count"]:
            L.append(f"| {b['bin']} | {b['count']} | {b['mean_confidence']:.4f} | {b['accuracy']:.4f} | {b['abs_gap']:.4f} |")
        else:
            L.append(f"| {b['bin']} | 0 | – | – | – |")
    L += ["", "Bins: [i/10, (i+1)/10), last bin closed; empty bins are listed but not drawn. With 7 classes, max softmax ≥ 1/7, so [0.0, 0.1) is always empty.", ""]

    L += ["## 7. Confidence (max softmax) — correct vs incorrect", "", _dist_table(conf["msp"]),
          f"Mann–Whitney U two-sided p = {conf['msp'].get('mann_whitney_p_value', float('nan')):.3g} "
          f"(anti-conservative: images within a lesion are not independent). Effect size AUROC(correct > incorrect) = "
          f"{conf['msp'].get('auroc_correct_vs_incorrect', float('nan')):.4f}. Statistical significance is not clinical significance.", ""]

    L += ["## 8. High-confidence errors", "", "| confidence ≥ | predictions | % of all | incorrect | error rate | incorrect as % of all errors |",
          "|---|---|---|---|---|---|"]
    for k, h in r["high_confidence"].items():
        L.append(f"| {k} | {h['n_predictions']} | {h['percent_of_all_predictions']:.1f}% | {h['n_incorrect']} | "
                 f"{h['error_rate']:.4f} | {h['incorrect_as_percent_of_all_errors']:.1f}% |")
    L += ["", "Image-level list: `metrics/high_confidence_errors.csv` (dataset IDs only, no patient data).", ""]

    L += ["## 9. Entropy and top-2 margin", "",
          f"Predictive entropy H(p) = −Σ p_i ln p_i (nats; maximum ln 7 = {np.log(7):.4f}). This is predictive/distributional "
          "uncertainty of the output vector, not epistemic uncertainty.", "", _dist_table(conf["entropy"]),
          "Top-2 margin (p_top1 − p_top2), exploratory:", "", _dist_table(conf["margin"]),
          "Error detection (positive = incorrect prediction; point [lesion-bootstrap 95% CI]):", "",
          f"- max softmax: AUROC {_ci(boot, 'error_auroc_msp')}",
          f"- entropy: AUROC {_ci(boot, 'error_auroc_entropy')}; difference vs MSP {_ci(boot, 'error_auroc_entropy_minus_msp')}",
          f"- margin: AUROC {_ci(boot, 'error_auroc_margin')}; difference vs MSP {_ci(boot, 'error_auroc_margin_minus_msp')}", ""]
    det = conf["error_detection"]
    L += ["| score | AUROC | AUPR (error) | error prevalence | AURC if ranked by it |", "|---|---|---|---|---|"]
    for name, d in det.items():
        L.append(f"| {name} | {d['auroc_error_detection']:.4f} | {d['aupr_error']:.4f} | "
                 f"{d['aupr_error_baseline_prevalence']:.4f} | {d['aurc_when_ranked_by_this_score']:.4f} |")

    L += ["", "## 10. Selective prediction (raw max softmax only; descriptive)", "",
          f"- AURC: {_ci(boot, 'aurc')} (oracle {sel['optimal_aurc']:.4f})", f"- E-AURC: {_ci(boot, 'e_aurc')}", "",
          "| target coverage | threshold | accepted | coverage | risk | accuracy |", "|---|---|---|---|---|---|"]
    for op in sel["operating_points"]:
        L.append(f"| {op['target_coverage']:.0%} | {op['threshold']:.4f} | {op['n_accepted']} | {op['coverage']:.4f} | "
                 f"{op['risk']:.4f} | {op['accuracy']:.4f} |")
    L += ["", "Thresholds are read off this evaluation set to describe the curve; they are **not** deployment thresholds.", ""]

    mb = r["malignant_benign"]
    L += ["## 11. Malignant vs benign (project-level mapping)", "",
          f"- Mapping (from `backend/app/class_info.py`): malignant = {', '.join(mb['malignant_classes'])}; benign = "
          f"{', '.join(r['config']['analysis']['benign_classes'])}.",
          "- **akiec (actinic keratosis / Bowen's disease) is clinically pre-malignant / in-situ, not equivalent to invasive "
          "malignancy; it is grouped as malignant only because that is the project's declared mapping.**",
          f"- Decision rule: argmax class ∈ malignant set (no threshold tuned). n malignant = {mb['n_malignant']}, n benign = {mb['n_benign']}.",
          f"- Sensitivity: {_ci(boot, 'malignant_sensitivity')}", f"- Specificity: {_ci(boot, 'malignant_specificity')}",
          f"- AUROC (score = Σ malignant probabilities): {_ci(boot, 'malignant_auroc')}",
          f"- Confusion (rows true mal/ben, cols pred mal/ben): {mb['confusion']['matrix']}",
          f"- Melanoma (mel) sensitivity / recall, argmax rule: {_ci(boot, 'mel_recall')} (n mel = {cls['per_class']['mel']['support']})", ""]

    L += ["## 12. Main observations (auto-generated, descriptive)", ""] + [f"- {o}" for o in r["observations"]] + [""]
    L += ["## 13. Limitations", ""] + [f"- {x}" for x in r["limitations"]] + [""]
    L += ["## 14. Files generated", ""] + [f"- `{f}`" for f in r["files"]] + [""]
    L += ["## 15. Command", "", "```", r["command"], "```", ""]
    return "\n".join(L)


def _dist_table(d: dict) -> str:
    rows = ["| group | n | mean | median | SD | Q1 | Q3 | min | max |", "|---|---|---|---|---|---|---|---|---|"]
    for g in ("correct", "incorrect"):
        s = d[g]
        if s["n"]:
            rows.append(f"| {g} | {s['n']} | {s['mean']:.4f} | {s['median']:.4f} | {s['std']:.4f} | {s['q1']:.4f} | "
                        f"{s['q3']:.4f} | {s['min']:.4f} | {s['max']:.4f} |")
    return "\n".join(rows) + "\n"


def observations(r) -> list[str]:
    cls, conf, boot, hc = r["classification"], r["confidence"], r["bootstrap"]["metrics"], r["high_confidence"]
    acc, bal = cls["accuracy"], cls["balanced_accuracy"]
    worst = min(cls["per_class"].items(), key=lambda kv: kv[1]["recall"])
    obs = [
        f"Accuracy {acc:.3f} vs balanced accuracy {bal:.3f}; lowest class recall: {worst[0]} ({worst[1]['recall']:.3f}, n={worst[1]['support']}).",
        f"Mean confidence {conf['msp']['correct']['mean']:.3f} for correct vs {conf['msp']['incorrect']['mean']:.3f} for incorrect predictions; "
        f"confidence separated correct from incorrect predictions with AUROC {conf['msp'].get('auroc_correct_vs_incorrect', float('nan')):.3f}.",
        f"Top-label ECE {r['calibration']['ece']:.4f} (15 bins); mean confidence {r['calibration']['mean_confidence']:.4f} vs accuracy {acc:.4f} "
        f"({'over' if r['calibration']['mean_confidence'] > acc else 'under'}-confident on average).",
        f"{hc['0.90']['n_incorrect']} incorrect predictions had confidence ≥ 0.90 "
        f"({hc['0.90']['incorrect_as_percent_of_all_errors']:.1f}% of all errors); {hc['0.99']['n_incorrect']} had confidence ≥ 0.99.",
        f"Error-detection AUROC: MSP {boot['error_auroc_msp']['point']:.3f}, entropy {boot['error_auroc_entropy']['point']:.3f}, "
        f"margin {boot['error_auroc_margin']['point']:.3f}; entropy − MSP 95% CI "
        f"[{boot['error_auroc_entropy_minus_msp']['ci_low']:.3f}, {boot['error_auroc_entropy_minus_msp']['ci_high']:.3f}].",
    ]
    if r["experiment"]["evaluation_type"] == "e0_split_cross_protocol":
        obs.append("These numbers are very likely optimistic: most evaluated images were probably in the historical "
                   "model's training data (see provenance). They describe behaviour on largely seen data, not generalisation.")
    return obs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="research/configs/step1_reliability.yaml")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--e0-split", choices=["train", "val", "test"])
    mode.add_argument("--manifest")
    ap.add_argument("--data-root", default=None, help="HAM10000 root (required for --e0-split; optional metadata hash otherwise)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cfg, e0_cfg = load_config(args.config), load_config(repo_path(E0_CONFIG))
    a = cfg["analysis"]
    set_seed(0, deterministic=True)  # inference has no randomness; this pins the bootstrap environment too
    device = torch.device(args.device)

    out = repo_path(args.output_dir or cfg["experiment"]["output_dir"])
    if (out / "predictions" / "predictions.npz").exists() and not args.overwrite:
        raise SystemExit(f"{out} already has results. Pass --overwrite to replace them.")
    dirs = {k: out / k for k in ("metadata", "predictions", "metrics", "figures", "reports")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # --- checkpoint (read-only) ---
    ckpt_path = Path(args.checkpoint or cfg["model"]["checkpoint"] or backend_config.MODEL_PATH).resolve()
    _check(ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}")
    ckpt_sha_before = sha256_file(ckpt_path)
    model, classes, ckpt_info = load_checkpoint(ckpt_path, cfg["model"]["expected_class_order"], e0_cfg, device)

    declared = sorted(c for c in classes if CLASS_INFO[c]["category"] == "malignant")
    _check(sorted(a["malignant_classes"]) == declared, f"malignant_classes {a['malignant_classes']} != class_info.py {declared}")

    # --- evaluation set ---
    data_root = None
    if args.e0_split:
        data_root = resolve_data_root(args.data_root)
        frame, split_info = e0_manifest(args.e0_split, data_root, e0_cfg)
        eval_type = "e0_split_cross_protocol"
        label = f"Existing checkpoint evaluated on E0 split ({args.e0_split}) — cross-protocol historical-model evaluation"
        provenance = [
            f"Images: E0 lesion-level split `{split_info['e0_split_file']}` (SHA-256 `{split_info['e0_split_sha256']}`), "
            f"split = `{args.e0_split}`, all integrity/leakage checks re-applied.",
            "This is NOT the historical model's validation set and NOT a held-out test set for this checkpoint. "
            "The historical training split cannot be reconstructed (no notebook, no split file; the checkpoint stores only "
            "weights, class order, val_acc and epoch).",
            "Checkpoint arithmetic (val_acc = 1250/1497 exactly; 3204 BatchNorm batches = 12 epochs × 267 batches of 32 ⇒ "
            "≈8518 training images; 8518 + 1497 = 10015 = all of HAM10000) indicates the historical model used all HAM10000 "
            "images for training+validation. If so, roughly 85% of these images were in its training data. This is an "
            "inference, not a verified fact.",
            "The E0 split is held out only with respect to E0 models, which are trained under a different protocol.",
        ]
    else:
        manifest_path = Path(args.manifest).resolve()
        _check(manifest_path.is_file(), f"Manifest not found: {manifest_path}")
        frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False).replace("", np.nan)
        base = manifest_path.parent
        frame["image_path"] = frame["image_path"].map(lambda p: str(p if Path(p).is_absolute() else base / p)
                                                      if isinstance(p, str) else p)
        split_info = {"manifest_source": str(manifest_path), "manifest_source_sha256": sha256_file(manifest_path)}
        eval_type = "explicit_manifest"
        label = f"Historical-model evaluation on manifest {manifest_path.name}"
        provenance = [f"Images: explicitly provided manifest `{manifest_path}` (SHA-256 `{split_info['manifest_source_sha256']}`).",
                      "Overlap of this manifest with the historical model's (unknown) training data is not verified here."]
        if args.data_root:
            data_root = resolve_data_root(args.data_root)
    frame = validate_manifest(frame, classes)

    y = np.array([classes.index(l) for l in frame["label"]], dtype=np.int64)
    _check(((y >= 0) & (y < len(classes))).all(), "Labels outside [0, K-1]")
    manifest_out = dirs["metadata"] / "evaluation_manifest.csv"
    frame.to_csv(manifest_out, index=False, lineterminator="\n")
    content_hash = sha256_text(canonical_json(sorted(frame[["image_id", "lesion_id", "label"]].itertuples(index=False, name=None))))

    # --- inference: eval mode, no grad, deterministic, no dropout sampling, no TTA ---
    print(f"{label}\ncheckpoint {ckpt_path} ({ckpt_sha_before[:12]}), N = {len(frame)}", flush=True)
    logits = run_inference(model, frame, cfg["inference"]["batch_size"], args.num_workers, device)
    _check(logits.shape == (len(frame), len(classes)), f"Logits shape {logits.shape} != [{len(frame)}, {len(classes)}]")
    _check(np.isfinite(logits).all(), "NaN/inf in logits")
    parity = backend_parity(model, frame, logits, cfg["inference"]["backend_parity_images"],
                            cfg["inference"]["backend_parity_tolerance"], device)

    probs = softmax64(logits)
    _check(np.isfinite(probs).all(), "NaN/inf in probabilities")
    _check(np.abs(probs.sum(axis=1) - 1).max() < 1e-9, "Probabilities do not sum to 1")
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == y
    ent = predictive_entropy(logits)
    margin = top2_margin(probs)
    _check(np.isfinite(ent).all() and np.isfinite(margin).all(), "NaN/inf in entropy/margin")
    lesions, image_ids = frame["lesion_id"].to_numpy(str), frame["image_id"].to_numpy(str)

    # --- raw outputs ---
    np.savez(dirs["predictions"] / "predictions.npz", logits=logits, probs=probs, log_probs=log_softmax64(logits),
             labels=y, preds=pred, max_prob=conf, entropy=ent, top2_margin=margin, correct=correct,
             image_id=image_ids, lesion_id=lesions, class_names=np.array(classes),
             checkpoint_sha256=np.array(ckpt_sha_before), evaluation_set_content_sha256=np.array(content_hash))
    table = pd.DataFrame({"image_id": image_ids, "lesion_id": lesions, "true_class": frame["label"],
                          "true_class_index": y, "predicted_class": [classes[i] for i in pred],
                          "predicted_class_index": pred, "correct": correct.astype(int), "max_prob": conf,
                          "entropy": ent, "top2_margin": margin})
    for i, c in enumerate(classes):
        table[f"logit_{c}"] = logits[:, i]
    for i, c in enumerate(classes):
        table[f"prob_{c}"] = probs[:, i]
    table["checkpoint_sha256"] = ckpt_sha_before
    table["evaluation_set_content_sha256"] = content_hash
    table.to_csv(dirs["predictions"] / "predictions.csv", index=False, float_format="%.17g", lineterminator="\n")

    # --- metrics ---
    print("computing metrics ...", flush=True)
    cls = classification_metrics(y, probs, classes)
    top15 = top_label_calibration(probs, y, a["ece_bins"])
    bins10 = confidence_bin_table(conf, correct, a["confidence_bins"])
    calibration = {"probabilities": "raw softmax of logits (float64); NOT calibrated", "nll": nll(logits, y),
                   "brier": brier(probs, y), "brier_definition": "mean over samples of sum_k (p_k - y_k)^2, range [0, 2]",
                   "ece": top15["ece"], "mce": top15["mce"], "ece_mce_bins": top15["bins"],
                   "ece_definition": "sum_b (n_b/N)|acc_b - conf_b|, conf = max softmax; 15 equal-width bins (i/15,(i+1)/15]",
                   "classwise_ece": classwise_ece(probs, y, a["ece_bins"], classes),
                   "confidence_bins_10": bins10, "mean_confidence": float(conf.mean()), "accuracy": float(correct.mean())}
    confidence = {
        "msp": compare_correct_incorrect(conf, correct, "more confident"),
        "entropy": compare_correct_incorrect(ent, correct, "more uncertain"),
        "margin": compare_correct_incorrect(margin, correct, "more separated top-2"),
        "error_detection": error_detection({"msp": -conf, "entropy": ent, "margin": -margin}, correct),
        "entropy_units": "nats", "entropy_max": float(np.log(len(classes))),
    }
    hc, hc_rows = high_confidence_errors(conf, correct, a["high_confidence_thresholds"], image_ids, lesions, y, pred, classes)
    confidence["high_confidence_errors"] = hc
    cov, risk, thr = risk_coverage_curve(conf, correct)
    selective = {**selective_metrics(probs, y, a["selective_coverages"]),
                 "operating_points": operating_points(conf, correct, a["selective_coverages"]),
                 "note": "Descriptive only; thresholds read off this evaluation set are not deployment thresholds."}
    pd.DataFrame({"coverage": cov, "risk": risk, "confidence_threshold": thr,
                  "oracle_risk": np.maximum(0, np.arange(1, len(y) + 1) - correct.sum()) / np.arange(1, len(y) + 1)}
                 ).to_csv(dirs["metrics"] / "risk_coverage_curve.csv", index=False, float_format="%.17g")
    binary = binary_malignant_metrics(y, probs, classes, a["malignant_classes"])
    binary["confusion"] = binary_confusion(y, probs, classes, a["malignant_classes"])
    binary["melanoma_sensitivity_argmax"] = cls["per_class"]["mel"]["recall"]
    binary["mapping_note"] = ("Project-level mapping from backend/app/class_info.py. akiec is clinically pre-malignant/"
                              "in-situ, not equivalent to invasive malignancy.")
    print("bootstrapping (lesion clusters) ...", flush=True)
    boot = bootstrap_scalars(logits, probs, y, lesions, classes, a["malignant_classes"], a["ece_bins"], a["bootstrap"])

    write_json(dirs["metrics"] / "classification_metrics.json", {**cls, "malignant_vs_benign": binary})
    write_json(dirs["metrics"] / "calibration_metrics.json", calibration)
    write_json(dirs["metrics"] / "selective_metrics.json", selective)
    write_json(dirs["metrics"] / "confidence_analysis.json", confidence)
    write_json(dirs["metrics"] / "bootstrap_ci.json", boot)
    pd.DataFrame(hc_rows, columns=["threshold", "image_id", "lesion_id", "true_class", "predicted_class", "confidence"]
                 ).to_csv(dirs["metrics"] / "high_confidence_errors.csv", index=False, float_format="%.17g")

    # --- figures ---
    print("drawing figures ...", flush=True)
    short = "existing checkpoint, " + (f"E0 {args.e0_split} (cross-protocol)" if args.e0_split else "manifest")
    files = []
    files += figs.confusion_matrix_figure(cls["confusion_matrix"], classes, dirs["figures"], f"Confusion matrix — {short}")
    files += figs.per_class_f1_figure(cls["per_class"], classes, dirs["figures"], f"Per-class metrics — {short}")
    files += figs.reliability_diagram_figure(bins10, dirs["figures"], f"Reliability (raw softmax) — {short}")
    files += figs.correct_vs_incorrect_figure(conf, correct, dirs["figures"], "confidence_correct_vs_incorrect",
                                              "max softmax confidence", f"Confidence — {short}", (1 / len(classes), 1))
    files += figs.correct_vs_incorrect_figure(ent, correct, dirs["figures"], "entropy_correct_vs_incorrect",
                                              "predictive entropy (nats)", f"Entropy — {short}", (0, np.log(len(classes))))
    files += figs.risk_coverage_figure(cov, risk, correct, dirs["figures"], f"Risk–coverage (MSP) — {short}",
                                       selective["operating_points"])
    files += figs.confidence_bins_figure(bins10, dirs["figures"], f"Confidence vs correctness — {short}")

    # --- metadata ---
    ckpt_sha_after = sha256_file(ckpt_path)
    _check(ckpt_sha_after == ckpt_sha_before, "Checkpoint file changed during evaluation")
    env = {**collect_environment(REPO_ROOT), "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    metadata_hash = None
    if data_root is not None and (data_root / e0_cfg["data"]["metadata_file"]).is_file():
        metadata_hash = sha256_file(data_root / e0_cfg["data"]["metadata_file"])
    try:
        ckpt_rel = str(ckpt_path.relative_to(REPO_ROOT))
    except ValueError:
        ckpt_rel = str(ckpt_path)
    experiment = {
        "experiment": cfg["experiment"]["name"], "evaluation_type": eval_type, "evaluation_label": label,
        "provenance_notes": provenance, "checkpoint_path": str(ckpt_path), "checkpoint_path_relative": ckpt_rel,
        "checkpoint_sha256": ckpt_sha_before, "checkpoint_sha256_after": ckpt_sha_after, "checkpoint_info": ckpt_info,
        "git_commit": env["git_commit"], "git_dirty": env["git_dirty"],
        "dataset_metadata_sha256": metadata_hash, "evaluation_manifest_file": str(manifest_out),
        "evaluation_manifest_sha256": sha256_file(manifest_out), "evaluation_set_content_sha256": content_hash,
        **split_info, "class_order": classes, "n_samples": int(len(y)), "n_lesions": int(len(np.unique(lesions))),
        "n_classes": len(classes), "preprocessing": f"PIL convert('RGB') -> {backend_inference_transform!r}".replace("\n", " "),
        "inference": {"mode": "model.eval() + torch.no_grad()", "dropout_sampling": False, "tta": False,
                      "batch_size": cfg["inference"]["batch_size"], "device": str(device)},
        "backend_parity": parity, "config": cfg, "timestamp_utc": env["timestamp_utc"],
    }
    write_json(dirs["metadata"] / "experiment.json", experiment)
    write_json(dirs["metadata"] / "environment.json", env)

    command = shlex.join(["python", "-m", "research.evaluation.step1_reliability", *(argv if argv is not None else sys.argv[1:])])
    files = [str(p.relative_to(REPO_ROOT)) for d in dirs.values() for p in sorted(d.iterdir())] + \
            [str((dirs["reports"] / "step1_report.md").relative_to(REPO_ROOT))]
    results = {"experiment": experiment, "classification": cls, "calibration": calibration, "selective": selective,
               "confidence": confidence, "high_confidence": hc, "malignant_benign": binary, "bootstrap": boot,
               "config": cfg, "files": sorted(set(files)), "command": command}
    results["observations"] = observations(results)
    results["limitations"] = [
        "The historical model's training/validation split is unknown, so overlap between evaluated images and its training data cannot be measured."
        + (" On the E0 split, most images were probably seen during training." if args.e0_split else ""),
        "Single checkpoint, single training run: no estimate of training-seed variation is possible.",
        "Lesion-bootstrap CIs capture sampling variation of the evaluation set only, not model-training variation.",
        "HAM10000 has no patient identifier; images are dermoscopic and from a limited set of sources and skin types.",
        "Rare classes (df, vasc, akiec) have small supports; their per-class estimates are imprecise.",
        "Mann–Whitney p-values assume independent images, which multi-image lesions violate.",
        "Malignant/benign uses the project mapping, in which akiec (pre-malignant) counts as malignant.",
    ]
    (dirs["reports"] / "step1_report.md").write_text(build_report(results), encoding="utf-8")
    print(f"done. outputs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
