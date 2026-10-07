"""STEP 1 figures. Each figure is written as 300-dpi PNG and vector PDF."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "figure.dpi": 100})
CORRECT, INCORRECT = "#2a78d6", "#d03b3b"


def _save(fig, out_dir: Path, stem: str) -> list[str]:
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=300)
        paths.append(str(p))
    plt.close(fig)
    return paths


def confusion_matrix_figure(cm, classes, out_dir, title):
    cm = np.asarray(cm, dtype=float)
    row = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{int(cm[i, j])}\n{row[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if row[i, j] > 0.55 else "black")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), [f"{c} (n={int(cm[i].sum())})" for i, c in enumerate(classes)])
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised (recall)")
    return _save(fig, out_dir, "confusion_matrix")


def per_class_f1_figure(per_class, classes, out_dir, title):
    f1 = [per_class[c]["f1"] for c in classes]
    rec = [per_class[c]["recall"] for c in classes]
    prec = [per_class[c]["precision"] for c in classes]
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(x - 0.27, prec, 0.27, label="precision", color="#9ec5ef")
    ax.bar(x, rec, 0.27, label="recall", color="#5b9be0")
    ax.bar(x + 0.27, f1, 0.27, label="F1", color="#1c4f8f")
    ax.set_xticks(x, [f"{c}\nn={per_class[c]['support']}" for c in classes])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=8, loc="lower right")
    return _save(fig, out_dir, "per_class_f1")


def reliability_diagram_figure(bin_table, out_dir, title):
    bins = [b for b in bin_table["bins"] if b["count"]]
    width = bin_table["bins"][0]["upper"] - bin_table["bins"][0]["lower"]
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(4.6, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal (accuracy = confidence)")
    ax.bar([b["lower"] + width / 2 for b in bins], [b["accuracy"] for b in bins], width=width * 0.95,
           color="#5b9be0", edgecolor="black", linewidth=0.4, label="empirical accuracy per bin")
    ax.plot([b["mean_confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", color="black",
            markersize=3, linewidth=1, label="(mean confidence, accuracy)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("empirical accuracy")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")
    axh.bar([b["lower"] + width / 2 for b in bin_table["bins"]], [b["count"] for b in bin_table["bins"]],
            width=width * 0.95, color="#9a9a9a")
    axh.set_yscale("symlog")
    axh.set_xlabel("max softmax confidence (10 equal-width bins; empty bins drawn empty)")
    axh.set_ylabel("count")
    return _save(fig, out_dir, "reliability_diagram")


def correct_vs_incorrect_figure(values, correct, out_dir, stem, xlabel, title, value_range):
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [3, 1.2]})
    edges = np.linspace(value_range[0], value_range[1], 31)
    ax.hist(values[correct], bins=edges, density=True, alpha=0.6, color=CORRECT, label=f"correct (n={correct.sum()})")
    ax.hist(values[~correct], bins=edges, density=True, alpha=0.6, color=INCORRECT,
            label=f"incorrect (n={(~correct).sum()})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    bp = axb.boxplot([values[correct], values[~correct]], widths=0.6, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [CORRECT, INCORRECT]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axb.set_xticks([1, 2], ["correct", "incorrect"])
    axb.set_ylabel(xlabel)
    return _save(fig, out_dir, stem)


def risk_coverage_figure(coverage, risk, correct, out_dir, title, operating):
    n, err = len(correct), (~correct).mean()
    k = np.arange(1, n + 1)
    oracle = np.maximum(0, k - correct.sum()) / k
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.plot(coverage, risk, color="#1c4f8f", linewidth=1.4, label="max softmax confidence")
    ax.plot(k / n, oracle, color="#0ca30c", linewidth=1, linestyle=":", label="oracle ranking")
    ax.axhline(err, color="#898781", linewidth=1, linestyle="--", label=f"random ranking (= overall error {err:.3f})")
    for op in operating:
        if op["target_coverage"] < 1:
            ax.plot(op["coverage"], op["risk"], "o", color="black", markersize=3)
            ax.annotate(f"{op['coverage']:.0%}: risk {op['risk']:.3f}", (op["coverage"], op["risk"]),
                        textcoords="offset points", xytext=(-4, 6), fontsize=7, ha="right")
    ax.set_xlabel("coverage (fraction accepted)")
    ax.set_ylabel("selective risk (error rate of accepted)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.legend(fontsize=7)
    return _save(fig, out_dir, "risk_coverage")


def confidence_bins_figure(bin_table, out_dir, title):
    """Confidence vs correctness: per-bin counts split into correct/incorrect, with accuracy."""
    bins = bin_table["bins"]
    labels = [b["bin"] for b in bins]
    counts = np.array([b["count"] for b in bins])
    acc = np.array([b["accuracy"] if b["accuracy"] is not None else np.nan for b in bins])
    n_ok = np.nan_to_num(acc * counts).round().astype(int)
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(x, n_ok, color=CORRECT, alpha=0.75, label="correct")
    ax.bar(x, counts - n_ok, bottom=n_ok, color=INCORRECT, alpha=0.75, label="incorrect")
    ax.set_yscale("symlog")
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("number of predictions (symlog)")
    ax.set_xlabel("max softmax confidence bin")
    ax2 = ax.twinx()
    ax2.plot(x, acc, "ko-", markersize=3, linewidth=1, label="accuracy")
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("empirical accuracy")
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_title(title)
    return _save(fig, out_dir, "confidence_bins")
