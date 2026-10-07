"""
Descriptive reliability analysis of RAW softmax outputs (STEP 1).

Nothing here changes logits or probabilities. Entropy and the top-2 margin are
summaries of the output probability vector (predictive/distributional
uncertainty), not epistemic uncertainty.
"""
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score

from research.evaluation.metrics import log_softmax64
from research.evaluation.selective_prediction import aurc


def predictive_entropy(logits: np.ndarray) -> np.ndarray:
    """H(p) = -sum_i p_i log p_i in nats, via log-softmax (stable, exact 0*log0 = 0)."""
    lsm = log_softmax64(logits)
    return -(np.exp(lsm) * lsm).sum(axis=1)


def top2_margin(probs: np.ndarray) -> np.ndarray:
    top2 = np.sort(probs, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def describe(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return {"n": 0}
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "median": float(med),
        "std": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def compare_correct_incorrect(values: np.ndarray, correct: np.ndarray, higher_means: str) -> dict:
    """
    Distribution summary for correct vs incorrect, a two-sided Mann-Whitney U test, and
    the effect size AUROC = P(score of a random correct > score of a random incorrect).
    Images of one lesion are not independent, so the p-value is anti-conservative; the
    lesion-cluster bootstrap CI reported elsewhere is the primary uncertainty statement.
    """
    c, w = values[correct], values[~correct]
    out = {"correct": describe(c), "incorrect": describe(w), "higher_value_means": higher_means}
    if len(c) and len(w):
        test = mannwhitneyu(c, w, alternative="two-sided")
        out["mann_whitney_u"] = float(test.statistic)
        out["mann_whitney_p_value"] = float(test.pvalue)
        out["auroc_correct_vs_incorrect"] = float(test.statistic / (len(c) * len(w)))
        out["test_note"] = ("Two-sided Mann-Whitney U; assumes independent images (violated by "
                            "multi-image lesions), so p is anti-conservative. Statistical, not clinical, significance.")
    return out


def error_detection(scores_for_error: dict[str, np.ndarray], correct: np.ndarray) -> dict:
    """
    How well each score ranks errors above correct predictions (higher score = more suspect).
    AUROC/AUPR with 'error' as the positive class, plus AURC when samples are accepted in
    order of increasing suspicion.
    """
    is_error = ~correct
    out = {}
    for name, s in scores_for_error.items():
        both = is_error.any() and correct.any()
        out[name] = {
            "auroc_error_detection": float(roc_auc_score(is_error, s)) if both else float("nan"),
            "aupr_error": float(average_precision_score(is_error, s)) if both else float("nan"),
            "aupr_error_baseline_prevalence": float(is_error.mean()),
            "aurc_when_ranked_by_this_score": aurc(-s, correct),
        }
    return out


def high_confidence_errors(conf, correct, thresholds, image_id, lesion_id, y_true, y_pred, classes) -> tuple[dict, list]:
    n, n_err = len(conf), int((~correct).sum())
    summary, rows = {}, []
    for t in thresholds:
        sel = conf >= t
        wrong = sel & ~correct
        n_sel, n_wrong = int(sel.sum()), int(wrong.sum())
        summary[f"{t:.2f}"] = {
            "threshold": t,
            "rule": "confidence >= threshold",
            "n_predictions": n_sel,
            "n_incorrect": n_wrong,
            "error_rate": n_wrong / n_sel if n_sel else float("nan"),
            "percent_of_all_predictions": 100 * n_sel / n,
            "incorrect_as_percent_of_all_predictions": 100 * n_wrong / n,
            "incorrect_as_percent_of_all_errors": 100 * n_wrong / n_err if n_err else float("nan"),
        }
        for i in np.flatnonzero(wrong):
            rows.append({"threshold": t, "image_id": image_id[i], "lesion_id": lesion_id[i],
                         "true_class": classes[y_true[i]], "predicted_class": classes[y_pred[i]],
                         "confidence": float(conf[i])})
    return summary, rows


def confidence_bin_table(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> dict:
    """
    Equal-width bins [0,0.1), [0.1,0.2), ..., [0.9,1.0] (last bin closed). Empty bins are
    kept in the table with count 0 and null statistics; plots draw nothing for them.
    """
    ids = np.minimum(np.floor(conf * n_bins).astype(int), n_bins - 1)
    bins, gaps = [], []
    for b in range(n_bins):
        m = ids == b
        k = int(m.sum())
        lo, hi = b / n_bins, (b + 1) / n_bins
        row = {"bin": f"[{lo:.1f}, {hi:.1f}{']' if b == n_bins - 1 else ')'}", "lower": lo, "upper": hi, "count": k,
               "mean_confidence": None, "accuracy": None, "abs_gap": None}
        if k:
            mc, acc = float(conf[m].mean()), float(correct[m].mean())
            row.update(mean_confidence=mc, accuracy=acc, abs_gap=abs(acc - mc))
            gaps.append((k, abs(acc - mc)))
        bins.append(row)
    n = len(conf)
    return {
        "n_bins": n_bins,
        "definition": "equal-width [i/M, (i+1)/M), last bin [ (M-1)/M, 1 ]; empty bins kept with count 0",
        "ece_on_these_bins": sum(k * g for k, g in gaps) / n,
        "mce_on_these_bins": max(g for _, g in gaps),
        "bins": bins,
    }


def operating_points(conf: np.ndarray, correct: np.ndarray, coverages) -> list[dict]:
    """Descriptive only: the confidence threshold that yields each coverage on THIS data."""
    order = np.argsort(-conf, kind="stable")
    c_sorted, ok_sorted = conf[order], correct[order]
    n = len(conf)
    out = []
    for cov in coverages:
        k = max(1, int(np.floor(cov * n)))
        acc = float(ok_sorted[:k].mean())
        out.append({"target_coverage": cov, "threshold": float(c_sorted[k - 1]), "n_accepted": k,
                    "coverage": k / n, "risk": 1 - acc, "accuracy": acc})
    return out
