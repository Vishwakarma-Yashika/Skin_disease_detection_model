"""
Single-run E0 report: machine-readable metrics, figures and provenance for ONE trained
E0 seed, computed from its saved raw logits (no model, no images, no test-time choices).

    python -m research.evaluation.e0_run_report --config research/configs/e0.yaml \
        --run-dir research/outputs/e0_baseline/seed_42

Writes <run-dir>/report/:
    metrics.json, per_class_metrics.csv, confusion_matrix.csv, test_predictions.csv,
    training_history.csv, provenance.json, e0_run_report.md, figures/*.png|pdf

A single seed is NOT the E0 result; E0 is defined over the five predeclared seeds.
"""
import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from research.data.data import load_split  # noqa: E402
from research.evaluation.calibration_metrics import brier, nll, top_label_calibration  # noqa: E402
from research.evaluation.evaluate import load_predictions, verify_malignant_mapping  # noqa: E402
from research.evaluation.metrics import binary_malignant_metrics, classification_metrics  # noqa: E402
from research.utils.config import load_config  # noqa: E402
from research.utils.hashing import config_hash, sha256_file  # noqa: E402
from research.utils.io import read_json, write_json  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})


class ReportError(RuntimeError):
    pass


def full_metrics(p: dict, classes: list[str], cfg: dict) -> dict:
    y, probs = p["y"], p["probs"]
    cls = classification_metrics(y, probs, classes)
    support = np.array([cls["per_class"][c]["support"] for c in classes], dtype=float)
    w = support / support.sum()
    top = top_label_calibration(probs, y, cfg["evaluation"]["ece_bins"])
    binary = binary_malignant_metrics(y, probs, classes, cfg["evaluation"]["malignant_classes"])
    mal_idx = [classes.index(c) for c in cfg["evaluation"]["malignant_classes"]]
    t, pr = np.isin(y, mal_idx), np.isin(probs.argmax(1), mal_idx)
    binary["accuracy"] = float((t == pr).mean())
    binary["confusion_matrix"] = {"rows": "true [malignant, benign]", "cols": "predicted [malignant, benign]",
                                  "matrix": [[int((t & pr).sum()), int((t & ~pr).sum())],
                                             [int((~t & pr).sum()), int((~t & ~pr).sum())]]}
    mel = cls["per_class"]["mel"]
    return {
        "n": cls["n"],
        "classification": {
            "accuracy": cls["accuracy"], "balanced_accuracy": cls["balanced_accuracy"],
            "macro_precision": cls["macro_precision"], "macro_recall": cls["macro_recall"], "macro_f1": cls["macro_f1"],
            "weighted_precision": float(sum(w[i] * cls["per_class"][c]["precision"] for i, c in enumerate(classes))),
            "weighted_recall": float(sum(w[i] * cls["per_class"][c]["recall"] for i, c in enumerate(classes))),
            "weighted_f1": cls["weighted_f1"], "never_predicted": cls["never_predicted"],
        },
        "probabilistic": {
            "nll": nll(p["logits"], y), "brier": brier(probs, y), "ece": top["ece"],
            "ece_definition": f"top-label, {cfg['evaluation']['ece_bins']} equal-width bins (i/M,(i+1)/M]",
            "brier_definition": "mean over samples of sum_k (p_k - y_k)^2, range [0, 2]",
            "mean_max_softmax_probability": float(probs.max(axis=1).mean()),
            "probabilities": "raw softmax of logits, uncalibrated",
        },
        "ranking": {"macro_roc_auc_ovr": cls["macro_roc_auc_ovr"],
                    "per_class_roc_auc_ovr": {c: cls["per_class"][c]["roc_auc_ovr"] for c in classes}},
        "per_class": cls["per_class"],
        "confusion_matrix": cls["confusion_matrix"],
        "malignant_vs_benign": binary,
        "melanoma": {k: mel[k] for k in ("precision", "recall", "f1", "support", "specificity", "roc_auc_ovr")},
    }


def _save(fig, out: Path, stem: str):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def cm_figure(cm, classes, out, normalized, title):
    cm = np.asarray(cm, dtype=float)
    shown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalized else cm
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    thresh = shown.max() * 0.55
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{shown[i, j]:.2f}" if normalized else f"{int(cm[i, j])}", ha="center", va="center",
                    fontsize=7, color="white" if shown[i, j] > thresh else "black")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), [f"{c} (n={int(cm[i].sum())})" for i, c in enumerate(classes)])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised" if normalized else "count")
    _save(fig, out, "confusion_matrix_normalized" if normalized else "confusion_matrix")


def roc_figure(p, classes, mal_classes, metrics, out, title):
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(9, 4))
    for k, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(p["y"] == k, p["probs"][:, k])
        ax.plot(fpr, tpr, linewidth=1.1, label=f"{c} (AUC {metrics['ranking']['per_class_roc_auc_ovr'][c]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"{title}: one-vs-rest")
    ax.legend(fontsize=7, loc="lower right")
    idx = [classes.index(c) for c in mal_classes]
    fpr, tpr, _ = roc_curve(np.isin(p["y"], idx), p["probs"][:, idx].sum(axis=1))
    axb.plot(fpr, tpr, color="#d03b3b", linewidth=1.3,
             label=f"malignant vs benign (AUC {metrics['malignant_vs_benign']['auroc']:.3f})")
    axb.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axb.set_xlabel("false positive rate")
    axb.set_title(f"{title}: {'+'.join(mal_classes)} vs rest")
    axb.legend(fontsize=7, loc="lower right")
    _save(fig, out, "roc_curves")


def curve_figure(hist, out, stem, train_col, val_col, ylabel, selected_epoch, note):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(hist["epoch"], hist[train_col], "o-", markersize=2.5, linewidth=1.2, label=f"train ({note})")
    ax.plot(hist["epoch"], hist[val_col], "o-", markersize=2.5, linewidth=1.2, label="validation (eval mode)")
    ax.axvline(selected_epoch, color="#898781", linestyle="--", linewidth=1, label=f"selected epoch {selected_epoch} (min val NLL)")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    _save(fig, out, stem)


def report_markdown(r: dict) -> str:
    t, v, prov, classes = r["test"], r["validation_at_selected_epoch"], r["provenance"], r["classes"]
    c, pr, rk, mb, mel = t["classification"], t["probabilistic"], t["ranking"], t["malignant_vs_benign"], t["melanoma"]
    L = [f"# E0 single-seed report: seed {prov['seed']}", "",
         "> One seed of the five predeclared E0 seeds. It is **not** the E0 result (mean ± SD over seeds). It is unrelated to "
         "the historical checkpoint (`backend/models/best_fusion_model.pt`), whose results are a separate exploratory artifact.",
         "", "## Provenance", "",
         f"- git commit `{prov['git_commit']}` (dirty working tree: {prov['git_dirty']})",
         f"- config `research/configs/e0.yaml`, config hash `{prov['config_hash']}`",
         f"- split `{prov['split_file']}` SHA-256 `{prov['split_sha256']}`",
         f"- checkpoint `{prov['checkpoint']}` SHA-256 `{prov['checkpoint_sha256']}`",
         f"- selected epoch {prov['selected_epoch']} of {prov['epochs_run']} by minimum validation NLL",
         f"- device {prov['device']}; test-access audit: selection finalized {prov['test_access_audit']['selection_finalized_utc']}, "
         "test Dataset built only after that", "",
         "| split | images | lesions | " + " | ".join(classes) + " |", "|---|---|---|" + "---|" * len(classes)]
    for s, d in prov["splits"].items():
        L.append(f"| {s} | {d['n_images']} | {d['n_lesions']} | " + " | ".join(str(d["class_image_counts"][k]) for k in classes) + " |")
    L += ["", "## Validation at the selected epoch", "",
          f"NLL {v['probabilistic']['nll']:.4f} · accuracy {v['classification']['accuracy']:.4f} · balanced accuracy "
          f"{v['classification']['balanced_accuracy']:.4f} · macro F1 {v['classification']['macro_f1']:.4f} · macro AUROC "
          f"{v['ranking']['macro_roc_auc_ovr']:.4f}", "", "## Test (untouched until selection was final)", "",
          "| metric | value |", "|---|---|"]
    for k in ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision",
              "weighted_recall", "weighted_f1"):
        L.append(f"| {k} | {c[k]:.4f} |")
    for k in ("nll", "brier", "ece", "mean_max_softmax_probability"):
        L.append(f"| {k} | {pr[k]:.4f} |")
    L.append(f"| macro_roc_auc_ovr | {rk['macro_roc_auc_ovr']:.4f} |")
    L += ["", "| class | support | precision | recall | F1 | specificity | OvR AUROC |", "|---|---|---|---|---|---|---|"]
    for k in classes:
        d = t["per_class"][k]
        L.append(f"| {k} | {d['support']} | {d['precision']:.3f} | {d['recall']:.3f} | {d['f1']:.3f} | {d['specificity']:.3f} | {d['roc_auc_ovr']:.3f} |")
    L += ["", "Confusion matrix (rows true, cols predicted): `confusion_matrix.csv`.", "",
          "## Malignant vs benign (project mapping)", "",
          f"malignant = {', '.join(r['malignant'])}; benign = {', '.join(r['benign'])} (from `backend/app/class_info.py`). "
          "akiec (actinic keratosis / Bowen's disease) is clinically pre-malignant / in-situ, grouped as malignant only by the project mapping.", "",
          f"Sensitivity {mb['sensitivity']:.4f} · specificity {mb['specificity']:.4f} · accuracy {mb['accuracy']:.4f} · AUROC {mb['auroc']:.4f} "
          f"(n malignant {mb['n_malignant']}, benign {mb['n_benign']}); confusion {mb['confusion_matrix']['matrix']} "
          "(rows true mal/ben, cols predicted mal/ben; decision = argmax class).", "",
          "## Melanoma", "", f"precision {mel['precision']:.4f} · recall/sensitivity {mel['recall']:.4f} · F1 {mel['f1']:.4f} · support {mel['support']}", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    verify_malignant_mapping(cfg)
    classes = cfg["data"]["classes"]
    run_dir = Path(args.run_dir)
    meta = read_json(run_dir / "run_metadata.json")
    if meta.get("status") != "complete":
        raise ReportError(f"Run not complete: {meta.get('status')}")
    if meta.get("smoke_test"):
        raise ReportError("Refusing to report a smoke-test run as E0.")
    if meta["config_hash"] != config_hash(cfg):
        raise ReportError("Run config_hash differs from the given config")
    split_df, split_hash = load_split(cfg, verify_hash=True)
    if meta["split_sha256"] != split_hash:
        raise ReportError("Run was trained on a different split file")
    ckpt_sha = sha256_file(run_dir / "best.pt")
    if ckpt_sha != meta["checkpoint_sha256"]:
        raise ReportError("best.pt changed after training")

    seed = meta["seed"]
    preds = {s: load_predictions(run_dir, s, seed, classes, split_df) for s in ("val", "test")}
    out = run_dir / "report"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    test = full_metrics(preds["test"], classes, cfg)
    val = full_metrics(preds["val"], classes, cfg)
    hist = pd.read_csv(run_dir / "history.csv")

    prov = {
        "experiment": "E0", "seed": seed, "git_commit": meta["environment"]["git_commit"],
        "git_dirty": meta["environment"]["git_dirty"], "config_hash": meta["config_hash"], "config": cfg,
        "split_file": cfg["split"]["file"], "split_sha256": split_hash,
        "checkpoint": str(run_dir / "best.pt"), "checkpoint_sha256": ckpt_sha,
        "selected_epoch": meta["selected_epoch"], "epochs_run": meta["total_epochs_run"],
        "selection_rule": meta["checkpoint_rule"], "device": meta["runtime"]["device"],
        "test_access_audit": meta.get("test_access_audit"),
        "splits": {s: {k: d[k] for k in ("n_images", "n_lesions", "class_image_counts", "class_lesion_counts")}
                   for s, d in meta["split_summary"]["splits"].items()},
        "environment": meta["environment"], "determinism": meta["determinism"],
    }
    result = {"classes": classes, "malignant": cfg["evaluation"]["malignant_classes"],
              "benign": cfg["evaluation"]["benign_classes"], "test": test,
              "validation_at_selected_epoch": val, "provenance": prov}

    write_json(out / "metrics.json", {k: result[k] for k in ("test", "validation_at_selected_epoch", "classes", "malignant", "benign")})
    write_json(out / "provenance.json", prov)
    pd.DataFrame([{"class": k, **test["per_class"][k]} for k in classes]).to_csv(out / "per_class_metrics.csv", index=False)
    pd.DataFrame(test["confusion_matrix"], index=pd.Index(classes, name="true\\predicted"), columns=classes).to_csv(out / "confusion_matrix.csv")
    shutil.copyfile(run_dir / "predictions" / "test.csv", out / "test_predictions.csv")
    shutil.copyfile(run_dir / "history.csv", out / "training_history.csv")

    title = f"E0 seed {seed}, test"
    cm_figure(test["confusion_matrix"], classes, figs, False, f"{title} (counts)")
    cm_figure(test["confusion_matrix"], classes, figs, True, f"{title} (row-normalised)")
    roc_figure(preds["test"], classes, cfg["evaluation"]["malignant_classes"], test, figs, title)
    curve_figure(hist, figs, "loss_curve", "train_loss", "val_nll", "cross-entropy (NLL)", meta["selected_epoch"],
                 "running mean, augmented, train mode")
    curve_figure(hist, figs, "accuracy_curve", "train_accuracy", "val_accuracy", "accuracy", meta["selected_epoch"],
                 "running mean, augmented, train mode")

    (out / "e0_run_report.md").write_text(report_markdown(result), encoding="utf-8")
    print(report_markdown(result))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
