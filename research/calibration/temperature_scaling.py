"""
Single-scalar temperature scaling, fitted by minimising multiclass NLL.

Parameterised by beta = 1/T so that NLL(beta) = mean_i[logsumexp(beta z_i) - beta z_iy].
d/dbeta    = mean_i[E_p(z_i) - z_iy]
d2/dbeta2  = mean_i[Var_p(z_i)] >= 0          (p = softmax(beta z_i))
The objective is therefore convex in beta, but every fit still verifies the minimum
numerically: bounded Brent search, an independent dense grid over a wider range,
gradient/curvature at the solution, a boundary check, and the number of grid-local minima.
Any failed check raises TemperatureFitError; nothing is silently clipped.
"""
import numpy as np
from scipy.optimize import minimize_scalar


class TemperatureFitError(RuntimeError):
    pass


def _scaled_stats(beta: float, z: np.ndarray, y: np.ndarray):
    s = beta * z
    m = s.max(axis=1, keepdims=True)
    e = np.exp(s - m)
    norm = e.sum(axis=1, keepdims=True)
    p = e / norm
    lse = (m + np.log(norm))[:, 0]
    zy = z[np.arange(len(y)), y]
    nll = float((lse - beta * zy).mean())
    ez = (p * z).sum(axis=1)
    grad = float((ez - zy).mean())
    hess = float(((p * z * z).sum(axis=1) - ez ** 2).mean())
    return nll, grad, hess


def nll_at_beta(beta: float, z: np.ndarray, y: np.ndarray) -> float:
    return _scaled_stats(beta, z, y)[0]


def fit_beta_brent(z, y, bounds, xatol) -> tuple[float, object]:
    res = minimize_scalar(lambda b: nll_at_beta(b, z, y), bounds=tuple(bounds), method="bounded",
                          options={"xatol": xatol, "maxiter": 500})
    return float(res.x), res


def fit_temperature(logits: np.ndarray, labels: np.ndarray, fit_cfg: dict) -> dict:
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if z.ndim != 2 or len(z) != len(y) or not np.isfinite(z).all():
        raise TemperatureFitError("logits must be a finite [N, K] array matching labels")
    lo, hi = fit_cfg["beta_bounds"]

    beta, res = fit_beta_brent(z, y, (lo, hi), fit_cfg["brent_xatol"])
    nll, grad, hess = _scaled_stats(beta, z, y)

    grid = np.geomspace(*fit_cfg["grid_beta_range"], fit_cfg["grid_points"])
    stats = np.array([_scaled_stats(b, z, y) for b in grid])
    g_nll, g_grad, g_hess = stats[:, 0], stats[:, 1], stats[:, 2]
    k = int(g_nll.argmin())
    interior = g_nll[1:-1]
    n_local_minima = int(((interior < g_nll[:-2]) & (interior < g_nll[2:])).sum())
    grad_sign_changes = int((np.diff(np.sign(g_grad)) != 0).sum())

    margin = fit_cfg["boundary_relative_margin"]
    checks = {
        "brent_converged": bool(res.success),
        "not_at_lower_bound": beta > lo * (1 + margin),
        "not_at_upper_bound": beta < hi * (1 - margin),
        "grid_min_not_better": g_nll[k] >= nll - fit_cfg["grid_nll_tolerance"],
        "grid_argmin_interior": 0 < k < len(grid) - 1,
        "grid_argmin_brackets_solution": grid[max(k - 1, 0)] <= beta <= grid[min(k + 1, len(grid) - 1)],
        "single_grid_local_minimum": n_local_minima == 1,
        "single_gradient_sign_change": grad_sign_changes == 1,
        "nonnegative_curvature_on_grid": bool((g_hess >= -1e-12).all()),
        "gradient_near_zero": abs(grad) <= fit_cfg["gradient_tolerance"],
        "positive_curvature_at_solution": hess >= fit_cfg["hessian_min"],
    }
    diagnostics = {
        "beta": beta, "temperature": 1.0 / beta, "nll_at_solution": nll,
        "nll_at_T1": nll_at_beta(1.0, z, y), "gradient_at_solution": grad, "hessian_at_solution": hess,
        "brent": {"success": bool(res.success), "nfev": int(res.nfev), "message": str(res.message)},
        "grid": {"range": list(fit_cfg["grid_beta_range"]), "points": int(len(grid)), "argmin_beta": float(grid[k]),
                 "min_nll": float(g_nll[k]), "n_interior_local_minima": n_local_minima,
                 "gradient_sign_changes": grad_sign_changes, "min_hessian": float(g_hess.min())},
        "checks": checks, "n": int(len(y)),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise TemperatureFitError(f"Temperature fit rejected, failed checks {failed}: {diagnostics}")
    return diagnostics


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return np.asarray(logits, dtype=np.float64) / temperature
