"""
Uncertainty scores, validation-only threshold selection, and fixed-threshold policy
evaluation for selective prediction and malignant referral.

Conventions
-----------
* Uncertainty u: higher = more uncertain. A multiclass prediction is ACCEPTED iff u <= tau.
* Referral score p_mal: a case is REFERRED iff p_mal >= t.
* Thresholds are always one of the observed validation values, and the inequality is
  inclusive, so tied samples are accepted/referred together (deterministic, no tie-breaking).
* Selection functions take validation arrays only. They never receive test data.
"""
import math

import numpy as np

from research.evaluation.metrics import log_softmax64
from research.evaluation.selective_prediction import aurc, optimal_aurc


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def scores_from_logits(logits: np.ndarray, temperature: float, malignant_idx: list[int]) -> dict:
    lsm = log_softmax64(np.asarray(logits, dtype=np.float64) / temperature)
    p = np.exp(lsm)
    top2 = np.sort(p, axis=1)[:, -2:]
    return {
        "probs": p,
        "pred": p.argmax(axis=1),
        "msp": 1.0 - p.max(axis=1),
        "entropy": -(p * lsm).sum(axis=1),
        "margin": 1.0 - (top2[:, 1] - top2[:, 0]),
        "p_mal": p[:, malignant_idx].sum(axis=1),
    }


def aurc_for_uncertainty(u: np.ndarray, correct: np.ndarray) -> dict:
    a = aurc(-u, correct)  # aurc() accepts in order of decreasing confidence = increasing u
    return {"aurc": a, "e_aurc": a - optimal_aurc(correct)}


# ---------------------------------------------------------------------------
# Validation-only threshold selection
# ---------------------------------------------------------------------------

def threshold_for_coverage(u_val: np.ndarray, target: float) -> dict:
    """Smallest observed tau whose validation coverage P(u <= tau) is >= target."""
    n = len(u_val)
    k = math.ceil(target * n - 1e-12)
    tau = float(np.sort(u_val)[k - 1])
    cov = float((u_val <= tau).mean())
    return {"feasible": True, "threshold": tau, "val_coverage": cov, "target_coverage": target,
            "tie_excess": cov - k / n}


def threshold_for_risk(u_val: np.ndarray, correct_val: np.ndarray, target: float) -> dict:
    """Largest observed tau (max validation coverage) whose validation retained-case error <= target."""
    order = np.argsort(u_val, kind="stable")
    u_s, err_s = u_val[order], (~correct_val[order]).astype(np.int64)
    cum_err = np.cumsum(err_s)
    # only positions at the end of a tie group are valid cut points
    last_of_group = np.r_[u_s[1:] != u_s[:-1], True]
    k = np.arange(1, len(u_s) + 1)
    risk = cum_err / k
    ok = last_of_group & (risk <= target + 1e-12)
    if not ok.any():
        return {"feasible": False, "target_risk": target,
                "reason": f"no validation threshold reaches retained-case error <= {target}; "
                          f"lowest achievable is {float(risk[last_of_group].min()):.4f}"}
    i = int(np.flatnonzero(ok).max())
    return {"feasible": True, "threshold": float(u_s[i]), "target_risk": target,
            "val_coverage": float(k[i] / len(u_s)), "val_risk": float(risk[i]), "val_n_accepted": int(k[i])}


def threshold_for_sensitivity(p_mal_val: np.ndarray, is_mal_val: np.ndarray, target: float) -> dict:
    """Largest observed t (fewest referrals) whose validation malignant sensitivity P(p_mal >= t | mal) >= target."""
    pos = np.sort(p_mal_val[is_mal_val])[::-1]
    if len(pos) == 0:
        return {"feasible": False, "target_sensitivity": target, "reason": "no malignant validation cases"}
    k = math.ceil(target * len(pos) - 1e-12)
    t = float(pos[k - 1])
    return {"feasible": True, "threshold": t, "target_sensitivity": target,
            "val_sensitivity": float((p_mal_val[is_mal_val] >= t).mean()),
            "val_specificity": float((p_mal_val[~is_mal_val] < t).mean()),
            "val_referral_rate": float((p_mal_val >= t).mean())}


# ---------------------------------------------------------------------------
# Fixed-threshold evaluation (any split)
# ---------------------------------------------------------------------------

def evaluate_selective(u, y, pred, tau, classes) -> dict:
    acc = u <= tau
    correct = pred == y
    n_acc = int(acc.sum())
    out = {"threshold": tau, "n": int(len(y)), "n_accepted": n_acc, "n_deferred": int(len(y) - n_acc),
           "coverage": n_acc / len(y),
           "selective_risk": float((~correct[acc]).mean()) if n_acc else float("nan"),
           "selective_accuracy": float(correct[acc].mean()) if n_acc else float("nan"),
           "deferred_error_rate": float((~correct[~acc]).mean()) if n_acc < len(y) else float("nan")}
    recalls, per_class = [], {}
    for k, c in enumerate(classes):
        m = y == k
        ma = m & acc
        rec = float(correct[ma].mean()) if ma.any() else float("nan")
        per_class[c] = {"n": int(m.sum()), "n_retained": int(ma.sum()),
                        "retained_fraction": float(ma.sum() / m.sum()) if m.any() else float("nan"),
                        "recall_among_retained": rec}
        if ma.any():
            recalls.append(rec)
    out["balanced_accuracy_among_retained"] = float(np.mean(recalls)) if recalls else float("nan")
    out["per_class"] = per_class
    mel = classes.index("mel")
    is_mel = y == mel
    out["melanoma"] = {
        "n_mel": int(is_mel.sum()),
        "retained": int((is_mel & acc).sum()),
        "deferred": int((is_mel & ~acc).sum()),
        "retained_and_misclassified": int((is_mel & acc & ~correct).sum()),
        "retained_fraction": float((is_mel & acc).sum() / is_mel.sum()) if is_mel.any() else float("nan"),
        "mel_share_of_deferred": float((is_mel & ~acc).sum() / (~acc).sum()) if (~acc).any() else float("nan"),
    }
    return out


def evaluate_referral(p_mal, y, t, classes, malignant) -> dict:
    mal_idx = [classes.index(c) for c in malignant]
    is_mal, ref = np.isin(y, mal_idx), p_mal >= t
    is_mel = y == classes.index("mel")
    return {
        "threshold": t, "n": int(len(y)), "n_malignant": int(is_mal.sum()), "n_benign": int((~is_mal).sum()),
        "referral_rate": float(ref.mean()), "workload_reduction": float(1 - ref.mean()),
        "sensitivity": float((ref & is_mal).sum() / is_mal.sum()) if is_mal.any() else float("nan"),
        "specificity": float((~ref & ~is_mal).sum() / (~is_mal).sum()) if (~is_mal).any() else float("nan"),
        "malignant_cleared": int((~ref & is_mal).sum()),
        "melanoma_cleared": int((~ref & is_mel).sum()),
        "melanoma_sensitivity": float((ref & is_mel).sum() / is_mel.sum()) if is_mel.any() else float("nan"),
        "per_class_referred_fraction": {c: float(ref[y == k].mean()) if (y == k).any() else float("nan")
                                        for k, c in enumerate(classes)},
    }


# ---------------------------------------------------------------------------
# Ranking comparison
# ---------------------------------------------------------------------------

def ranking_agreement(a: np.ndarray, b: np.ndarray, chunk: int = 512) -> dict:
    """Exact pair counts between two score vectors (O(N^2), chunked)."""
    n = len(a)
    conc = disc = ties_a = ties_b = 0
    for s in range(0, n, chunk):
        da = np.sign(a[s:s + chunk, None] - a[None, :])
        db = np.sign(b[s:s + chunk, None] - b[None, :])
        rows = np.arange(s, min(s + chunk, n))[:, None]
        upper = np.arange(n)[None, :] > rows
        prod = (da * db)[upper]
        conc += int((prod > 0).sum())
        disc += int((prod < 0).sum())
        ties_a += int((da[upper] == 0).sum())
        ties_b += int((db[upper] == 0).sum())
    pairs = n * (n - 1) // 2
    return {"n": n, "pairs": pairs, "concordant": conc, "discordant": disc, "tied_in_first": ties_a,
            "tied_in_second": ties_b, "kendall_tau_b": (conc - disc) / math.sqrt((pairs - ties_a) * (pairs - ties_b))
            if pairs > ties_a and pairs > ties_b else float("nan")}
