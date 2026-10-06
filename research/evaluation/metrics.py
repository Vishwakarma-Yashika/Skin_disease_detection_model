"""
Classification, malignant-vs-benign and bootstrap utilities for E0.

All inputs are full-precision arrays: y_true (N,) int, probs (N, K) float64.
Probabilities come from raw logits via a float64 softmax -- nothing here
calibrates or otherwise transforms them.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def softmax64(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def log_softmax64(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)


def _safe_div(num, den):
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def safe_auc(is_positive: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC, or NaN when only one class is present (e.g. in a bootstrap resample)."""
    if is_positive.all() or not is_positive.any():
        return float("nan")
    return float(roc_auc_score(is_positive, score))


def classification_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    k = len(class_names)
    y_pred = probs.argmax(axis=1)
    cm = confusion(y_true, y_pred, k)
    tp = np.diag(cm).astype(np.float64)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    fp, fn = predicted - tp, support - tp
    tn = cm.sum() - tp - fp - fn

    # zero_division -> 0 (sklearn's zero_division=0): a class that is never
    # predicted gets precision 0; such classes are listed in `never_predicted`.
    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    specificity = _safe_div(tn, tn + fp)
    aucs = np.array([safe_auc(y_true == i, probs[:, i]) for i in range(k)])
    present = support > 0

    return {
        "n": int(len(y_true)),
        "accuracy": float(tp.sum() / len(y_true)),
        # Balanced accuracy = mean recall over classes present in y_true (sklearn definition).
        "balanced_accuracy": float(recall[present].mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / support.sum()),
        "macro_roc_auc_ovr": float(aucs.mean()) if np.isfinite(aucs).all() else float("nan"),
        "per_class": {
            name: {
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "specificity": float(specificity[i]),
                "roc_auc_ovr": float(aucs[i]),
            }
            for i, name in enumerate(class_names)
        },
        "never_predicted": [class_names[i] for i in range(k) if predicted[i] == 0],
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_axes": "rows=true, cols=predicted, order=class_names",
        "class_names": list(class_names),
    }


def binary_malignant_metrics(y_true, probs, class_names, malignant_classes) -> dict:
    idx = [class_names.index(c) for c in malignant_classes]
    is_mal = np.isin(y_true, idx)
    pred_mal = np.isin(probs.argmax(axis=1), idx)
    score = probs[:, idx].sum(axis=1)
    n_pos, n_neg = int(is_mal.sum()), int((~is_mal).sum())
    return {
        "malignant_classes": list(malignant_classes),
        "decision_rule": "predicted malignant iff argmax class is in malignant_classes",
        "score": "sum of malignant-class softmax probabilities",
        "n_malignant": n_pos,
        "n_benign": n_neg,
        "sensitivity": float((pred_mal & is_mal).sum() / n_pos) if n_pos else float("nan"),
        "specificity": float((~pred_mal & ~is_mal).sum() / n_neg) if n_neg else float("nan"),
        "auroc": safe_auc(is_mal, score),
    }


def threshold_for_specificity(score: np.ndarray, is_positive: np.ndarray, target: float) -> float:
    """Smallest threshold t (positive iff score >= t) whose specificity on these data is >= target."""
    neg = np.sort(score[~is_positive])
    candidates = np.unique(np.concatenate([score, [np.inf]]))
    spec = np.searchsorted(neg, candidates, side="left") / len(neg)
    return float(candidates[spec >= target].min())


def sens_spec_at_threshold(score, is_positive, threshold) -> tuple[float, float]:
    pred = score >= threshold
    sens = float((pred & is_positive).sum() / is_positive.sum()) if is_positive.any() else float("nan")
    spec = float((~pred & ~is_positive).sum() / (~is_positive).sum()) if (~is_positive).any() else float("nan")
    return sens, spec


def melanoma_operating_point(
    val_y, val_probs, test_y, test_probs, test_lesions, class_names, target_specificity, min_positive_lesions
) -> dict:
    """mel-vs-rest, score p(mel); threshold fixed on validation and applied unchanged to test."""
    k = class_names.index("mel")
    threshold = threshold_for_specificity(val_probs[:, k], val_y == k, target_specificity)
    val_sens, val_spec = sens_spec_at_threshold(val_probs[:, k], val_y == k, threshold)
    test_pos = test_y == k
    test_sens, test_spec = sens_spec_at_threshold(test_probs[:, k], test_pos, threshold)
    n_pos_lesions = int(len(np.unique(np.asarray(test_lesions)[test_pos])))
    return {
        "target_specificity": target_specificity,
        "threshold_source": "validation",
        "threshold_p_mel": threshold,
        "val_sensitivity": val_sens,
        "val_specificity": val_spec,
        "test_sensitivity": test_sens,
        "test_specificity": test_spec,
        "test_positive_images": int(test_pos.sum()),
        "test_positive_lesions": n_pos_lesions,
        "min_test_positive_lesions": min_positive_lesions,
        "sufficient_sample": n_pos_lesions >= min_positive_lesions,
    }


def cluster_bootstrap(metric_fn, groups, n_resamples: int, seed: int, confidence_level: float) -> dict:
    """
    Percentile bootstrap that resamples whole clusters (lesions) with replacement,
    because images of one lesion are not independent observations.
    metric_fn(index_array) -> dict[str, float].
    """
    groups = np.asarray(groups)
    _, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
    n_groups = len(members)

    rng = np.random.default_rng(seed)
    point = metric_fn(np.arange(len(groups)))
    draws = {name: np.empty(n_resamples) for name in point}
    for b in range(n_resamples):
        idx = np.concatenate([members[g] for g in rng.integers(0, n_groups, n_groups)])
        for name, value in metric_fn(idx).items():
            draws[name][b] = value

    alpha = (1 - confidence_level) / 2
    out = {}
    for name, values in draws.items():
        finite = values[np.isfinite(values)]
        out[name] = {
            "point": point[name],
            "ci_low": float(np.quantile(finite, alpha)) if len(finite) else float("nan"),
            "ci_high": float(np.quantile(finite, 1 - alpha)) if len(finite) else float("nan"),
            "n_valid_resamples": int(len(finite)),
        }
    return {
        "unit": "lesion_id",
        "n_clusters": n_groups,
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
        "seed": seed,
        "method": "percentile",
        "metrics": out,
    }
