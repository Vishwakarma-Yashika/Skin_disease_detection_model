"""
Reliability measurements of the RAW (uncalibrated) softmax outputs.
Measurement only -- nothing in this module changes the probabilities.
"""
import numpy as np

from research.evaluation.metrics import log_softmax64


def nll(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Mean negative log-likelihood, computed from logits in float64 for precision."""
    lsm = log_softmax64(logits)
    return float(-lsm[np.arange(len(y_true)), y_true].mean())


def brier(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Multi-class Brier score: mean over samples of sum_k (p_k - 1[y=k])^2. Range [0, 2]."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def _bin_index(conf: np.ndarray, n_bins: int) -> np.ndarray:
    # Equal-width bins (i/M, (i+1)/M]; conf == 0 goes to the first bin.
    return np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)


def calibration_bins(conf: np.ndarray, hit: np.ndarray, n_bins: int) -> list[dict]:
    ids = _bin_index(conf, n_bins)
    bins = []
    for b in range(n_bins):
        mask = ids == b
        n = int(mask.sum())
        bins.append({
            "lower": b / n_bins,
            "upper": (b + 1) / n_bins,
            "count": n,
            "mean_confidence": float(conf[mask].mean()) if n else None,
            "accuracy": float(hit[mask].mean()) if n else None,
        })
    return bins


def _ece_mce_from_bins(bins: list[dict], total: int) -> tuple[float, float]:
    gaps = [(b["count"], abs(b["accuracy"] - b["mean_confidence"])) for b in bins if b["count"]]
    ece = sum(n * g for n, g in gaps) / total
    mce = max(g for _, g in gaps)
    return float(ece), float(mce)


def top_label_calibration(probs: np.ndarray, y_true: np.ndarray, n_bins: int) -> dict:
    """ECE / MCE of the top-label confidence (max softmax) vs. correctness."""
    conf = probs.max(axis=1)
    hit = (probs.argmax(axis=1) == y_true).astype(np.float64)
    bins = calibration_bins(conf, hit, n_bins)
    ece, mce = _ece_mce_from_bins(bins, len(y_true))
    return {"ece": ece, "mce": mce, "n_bins": n_bins, "binning": "equal_width", "bins": bins}


def classwise_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int, class_names: list[str]) -> dict:
    """Classwise ECE (Kull et al., 2019): per-class ECE of p_k vs 1[y=k], averaged over classes."""
    per_class = {}
    for k, name in enumerate(class_names):
        bins = calibration_bins(probs[:, k], (y_true == k).astype(np.float64), n_bins)
        per_class[name] = _ece_mce_from_bins(bins, len(y_true))[0]
    return {"classwise_ece": float(np.mean(list(per_class.values()))), "per_class": per_class, "n_bins": n_bins}


def plot_reliability_diagram(bins: list[dict], path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width = bins[0]["upper"] - bins[0]["lower"]
    centers = [b["lower"] + width / 2 for b in bins]
    acc = [b["accuracy"] or 0.0 for b in bins]
    conf = [b["mean_confidence"] or 0.0 for b in bins]
    counts = [b["count"] for b in bins]

    fig, (ax, axh) = plt.subplots(2, 1, figsize=(5, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.bar(centers, acc, width=width, edgecolor="black", linewidth=0.5, label="accuracy")
    ax.bar(centers, [c - a for a, c in zip(acc, conf)], bottom=acc, width=width, color="none",
           edgecolor="red", hatch="//", linewidth=0.5, label="gap to confidence")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    axh.bar(centers, counts, width=width, edgecolor="black", linewidth=0.5)
    axh.set_xlabel("confidence (max softmax)")
    axh.set_ylabel("count")
    axh.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
