"""
Selective prediction baseline using ONLY the raw maximum softmax probability (MSP)
as the confidence score. No calibration, no conformal sets, no learned rejector.

Risk = 0/1 error among accepted samples. Samples are accepted in order of
decreasing confidence; ties keep their original order (stable sort).
"""
import numpy as np


def risk_coverage_curve(conf: np.ndarray, correct: np.ndarray):
    order = np.argsort(-conf, kind="stable")
    errors = (~correct[order]).astype(np.float64)
    k = np.arange(1, len(conf) + 1)
    return k / len(conf), np.cumsum(errors) / k, conf[order]


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (Geifman et al., 2019): mean selective risk over k = 1..N."""
    _, risk, _ = risk_coverage_curve(conf, correct)
    return float(risk.mean())


def optimal_aurc(correct: np.ndarray) -> float:
    """AURC of an oracle that ranks every correct prediction above every error."""
    n, n_err = len(correct), int((~correct).sum())
    k = np.arange(1, n + 1)
    return float((np.maximum(0, k - (n - n_err)) / k).mean())


def selective_metrics(probs: np.ndarray, y_true: np.ndarray, coverages: list[float]) -> dict:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    coverage, risk, sorted_conf = risk_coverage_curve(conf, correct)
    a, a_opt = aurc(conf, correct), optimal_aurc(correct)
    at = {}
    for c in coverages:
        k = max(1, int(np.floor(c * len(conf))))
        at[f"{c:.2f}"] = {"coverage": float(coverage[k - 1]), "selective_risk": float(risk[k - 1]),
                          "confidence_threshold": float(sorted_conf[k - 1])}
    return {
        "confidence_score": "max_softmax_probability (raw, uncalibrated)",
        "aurc": a,
        "optimal_aurc": a_opt,
        "e_aurc": a - a_opt,
        "risk_at_coverage": at,
    }


def plot_risk_coverage(curves: list[tuple[str, np.ndarray, np.ndarray]], path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    for label, coverage, risk in curves:
        ax.plot(coverage, risk, linewidth=1.2, label=label)
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk (error rate)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
