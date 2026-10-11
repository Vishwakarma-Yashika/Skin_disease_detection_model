"""
Temperature-scaling evaluation of ONE trained E0 run (post-hoc calibration only).

    python -m research.evaluation.calibration_eval --config research/configs/calibration_ts.yaml \
        --run-dir research/outputs/e0_baseline/seed_42 --data-root <HAM10000 root>

Order of operations (enforced and time-stamped in provenance.json):
  1. compatibility checks on the run, checkpoint, config, split and saved validation logits;
  2. best.pt re-run on a few validation images to prove the saved logits came from it;
  3. T fitted on VALIDATION logits only, written to temperature.json;
  4. only then are the test logits loaded; the same T is applied, nothing is fitted on test.
No retraining; best.pt and all existing outputs are read-only.
"""
import argparse
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research.calibration.temperature_scaling import (
    TemperatureFitError,
    apply_temperature,
    fit_beta_brent,
    fit_temperature,
    nll_at_beta,
)
from research.data.data import HAM10000Dataset, build_transforms, load_split, load_validated_split
from research.evaluation.calibration_metrics import brier, classwise_ece, nll, top_label_calibration
from research.evaluation.evaluate import load_predictions, verify_malignant_mapping
from research.evaluation.metrics import binary_malignant_metrics, classification_metrics, cluster_bootstrap, softmax64
from research.models.fusion_model import (
    FusionClassifier,
    assert_eval_transform_matches_backend,
    backend_inference_transform,
    verify_architecture,
)
from research.utils.config import REPO_ROOT, load_config, repo_path, resolve_data_root
from research.utils.environment import collect_environment
from research.utils.hashing import config_hash, sha256_file
from research.utils.io import read_json, write_json
from research.utils.seed import set_seed


class CalibrationError(RuntimeError):
    pass


def _check(cond, msg):
    if not cond:
        raise CalibrationError(msg)


def _now():
    return datetime.now(timezone.utc).isoformat()


def verify_run(run_dir: Path, e0_cfg: dict) -> dict:
    meta = read_json(run_dir / "run_metadata.json")
    _check(meta.get("status") == "complete", f"run not complete: {meta.get('status')}")
    _check(not meta.get("smoke_test"), "refusing a smoke-test run")
    run_cfg = load_config(run_dir / "config.yaml")
    _check(config_hash(run_cfg) == meta["config_hash"] == config_hash(e0_cfg), "run config differs from e0.yaml")
    ckpt_sha = sha256_file(run_dir / "best.pt")
    _check(ckpt_sha == meta["checkpoint_sha256"], "best.pt hash differs from the run metadata")
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    _check(list(ckpt["class_names"]) == e0_cfg["data"]["classes"], "checkpoint class order differs from config")
    _check(ckpt["config_hash"] == meta["config_hash"] and ckpt["seed"] == meta["seed"], "checkpoint metadata mismatch")
    _, split_sha = load_split(e0_cfg, verify_hash=True)
    _check(split_sha == meta["split_sha256"] == ckpt["split_sha256"], "split hash mismatch")
    _, eval_tf = build_transforms(e0_cfg)
    assert_eval_transform_matches_backend(eval_tf)
    _check(meta["preprocessing"]["eval"] == repr(eval_tf), "recorded eval preprocessing differs from the current one")
    return {"meta": meta, "ckpt": ckpt, "checkpoint_sha256": ckpt_sha, "split_sha256": split_sha}


def verify_saved_probs(p: dict, path: Path):
    stored = np.load(path)["probs"]
    _check(np.isfinite(p["logits"]).all(), f"non-finite logits in {path}")
    _check(np.abs(stored - p["probs"]).max() < 1e-9, f"stored probabilities in {path} are not softmax(logits)")


@torch.no_grad()
def recheck_logits(ckpt, e0_cfg, data_root, val_pred, n, tol) -> dict:
    """Recompute logits for the first n saved validation images from best.pt."""
    classes = e0_cfg["data"]["classes"]
    model = FusionClassifier(num_classes=len(classes), feature_dim=e0_cfg["model"]["fusion_dim"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    verify_architecture(model, e0_cfg)
    model.eval()
    split_df, image_index, _, _ = load_validated_split(data_root, e0_cfg)
    ids = list(val_pred["image_id"][:n])
    frame = split_df.set_index("image_id").loc[ids].reset_index()
    _, eval_tf = build_transforms(e0_cfg)
    ds = HAM10000Dataset(frame, image_index, classes, eval_tf)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    fresh = model(x).float().numpy()
    diff = float(np.abs(fresh - val_pred["logits"][:n]).max())
    _check(diff <= tol, f"saved validation logits do not match best.pt (max diff {diff})")
    return {"n_images": n, "max_abs_logit_difference": diff, "tolerance": tol}


def evaluate_logits(z, y, classes, malignant, n_bins) -> dict:
    probs = softmax64(z)
    cls = classification_metrics(y, probs, classes)
    top = top_label_calibration(probs, y, n_bins)
    cw = classwise_ece(probs, y, n_bins, classes)
    return {
        "nll": nll(z, y), "brier": brier(probs, y), "ece": top["ece"], "mce": top["mce"],
        "classwise_ece": cw["classwise_ece"], "classwise_ece_per_class": cw["per_class"],
        "mean_max_probability": float(probs.max(axis=1).mean()),
        "accuracy": cls["accuracy"], "balanced_accuracy": cls["balanced_accuracy"], "macro_f1": cls["macro_f1"],
        "macro_roc_auc_ovr": cls["macro_roc_auc_ovr"],
        "malignant_auroc": binary_malignant_metrics(y, probs, classes, malignant)["auroc"],
        "per_class": cls["per_class"], "ece_bins": top["bins"],
    }


def reliability_pair(before, after, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for col, (name, bins) in enumerate([("baseline (T = 1)", before), ("temperature-scaled", after)]):
        ax, axh = axes[0, col], axes[1, col]
        w = bins["bins"][0]["upper"] - bins["bins"][0]["lower"]
        nz = [b for b in bins["bins"] if b["count"]]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.bar([b["lower"] + w / 2 for b in nz], [b["accuracy"] for b in nz], width=w * 0.95, color="#5b9be0",
               edgecolor="black", linewidth=0.4)
        ax.plot([b["mean_confidence"] for b in nz], [b["accuracy"] for b in nz], "ko-", markersize=3, linewidth=1)
        ax.set_title(f"{name}: ECE {bins['ece']:.4f}", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("accuracy")
        axh.bar([b["lower"] + w / 2 for b in bins["bins"]], [b["count"] for b in bins["bins"]], width=w * 0.95, color="#9a9a9a")
        axh.set_yscale("symlog")
        axh.set_xlabel("max softmax confidence (15 bins)")
        axh.set_xlim(0, 1)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def nll_curve(z, y, fit, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = np.geomspace(0.25, 8, 400)
    vals = [nll_at_beta(1 / t, z, y) for t in ts]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(ts, vals, color="#1c4f8f")
    ax.axvline(fit["temperature"], color="#d03b3b", linestyle="--", label=f"fitted T = {fit['temperature']:.4f}")
    ax.axvline(1.0, color="#898781", linestyle=":", label="T = 1 (uncalibrated)")
    ax.set_xscale("log")
    ax.set_xlabel("temperature T")
    ax.set_ylabel("validation NLL")
    ax.legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def confidence_hist(z, temperature, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.linspace(1 / z.shape[1], 1, 31)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(softmax64(z).max(axis=1), bins=edges, alpha=0.6, label="baseline (T = 1)")
    ax.hist(softmax64(z / temperature).max(axis=1), bins=edges, alpha=0.6, label=f"T = {temperature:.3f}")
    ax.set_xlabel("max softmax confidence")
    ax.set_ylabel("count")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def report_markdown(seed, fit, test, boot, run, recheck, command) -> str:
    b, c, m = test["baseline"], test["calibrated"], boot["metrics"]
    ci = fit["temperature_bootstrap"]["ci95"]
    L = [f"# Temperature scaling: E0 seed {seed} (single seed, preliminary)", "",
         f"- Checkpoint `{run['meta']['runtime']['output_dir']}\\best.pt` SHA-256 `{run['checkpoint_sha256']}` (unchanged), "
         f"selected epoch {run['meta']['selected_epoch']}; split SHA-256 `{run['split_sha256']}`",
         f"- Saved logits re-checked against best.pt on {recheck['n_images']} validation images: max |diff| {recheck['max_abs_logit_difference']:.2e}",
         f"- **T = {fit['temperature']:.4f}** (validation lesion-bootstrap 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]), fitted on validation logits only; "
         f"validation NLL {fit['nll_at_T1']:.4f} → {fit['nll_at_solution']:.4f}",
         "- Test logits loaded only after T was fixed; argmax (accuracy, per-class P/R/F1) unchanged by construction.", "",
         "| test metric | baseline (T=1) | temperature-scaled | difference [paired lesion-bootstrap 95% CI] |", "|---|---|---|---|"]
    for k in ("nll", "brier", "ece", "classwise_ece"):
        d = m[f"{k}_difference"]
        L.append(f"| {k} | {b[k]:.4f} | {c[k]:.4f} | {d['point']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] |")
    for k in ("mce", "mean_max_probability", "accuracy", "balanced_accuracy", "macro_roc_auc_ovr", "malignant_auroc"):
        L.append(f"| {k} | {b[k]:.4f} | {c[k]:.4f} | {c[k] - b[k]:+.4f} |")
    L += ["", "ECE/MCE: top-label, 15 equal-width bins; Brier = mean Σ_k (p_k − y_k)², range 0–2. "
          "Bootstrap CIs hold T fixed and cover test sampling only. One seed only: no seed variation.", "",
          "```", command, "```", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    command = shlex.join(["python", "-m", "research.evaluation.calibration_eval",
                          *(argv if argv is not None else sys.argv[1:])])

    set_seed(0, deterministic=True)
    cfg = load_config(args.config)
    e0_cfg = load_config(repo_path("research/configs/e0.yaml"))
    verify_malignant_mapping(e0_cfg)
    classes, malignant = e0_cfg["data"]["classes"], e0_cfg["evaluation"]["malignant_classes"]
    run_dir = repo_path(args.run_dir)
    times = {"started_utc": _now()}

    # ---- 1. compatibility checks (validation side only) ----
    run = verify_run(run_dir, e0_cfg)
    seed = run["meta"]["seed"]
    out = repo_path(cfg["experiment"]["output_dir"]) / f"seed_{seed}"
    _check(out.resolve() != run_dir.resolve() and run_dir.resolve() not in out.resolve().parents,
           "output directory must be separate from the E0 run directory")
    if (out / "temperature.json").exists() and not args.overwrite:
        raise SystemExit(f"{out} already has results; pass --overwrite to replace them.")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    split_df, _ = load_split(e0_cfg, verify_hash=True)
    val = load_predictions(run_dir, "val", seed, classes, split_df)  # verifies ids/labels/class order/seed
    verify_saved_probs(val, run_dir / "predictions" / "val.npz")

    # ---- 2. saved logits really come from best.pt ----
    v = cfg["verification"]
    recheck = recheck_logits(run["ckpt"], e0_cfg, resolve_data_root(args.data_root), val,
                             v["logit_recheck_images"], v["logit_recheck_tolerance"])

    # ---- 3. fit T on validation only ----
    try:
        fit = fit_temperature(val["logits"], val["y"], cfg["fit"])
    except TemperatureFitError as exc:
        raise CalibrationError(str(exc)) from exc
    T = fit["temperature"]
    tb = cfg["evaluation"]["temperature_bootstrap"]
    rng = np.random.default_rng(tb["seed"])
    lesions = np.asarray(val["lesion_id"])
    uniq, inv = np.unique(lesions, return_inverse=True)
    members = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    zv = val["logits"].astype(np.float64)
    t_draws = []
    for _ in range(tb["n_resamples"]):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        b, _ = fit_beta_brent(zv[idx], val["y"][idx], cfg["fit"]["beta_bounds"], cfg["fit"]["brent_xatol"])
        t_draws.append(1 / b)
    t_draws = np.array(t_draws)
    lo, hi = cfg["fit"]["beta_bounds"]
    fit["temperature_bootstrap"] = {
        "unit": "lesion_id (validation)", "n_resamples": tb["n_resamples"], "seed": tb["seed"],
        "ci95": [float(np.quantile(t_draws, 0.025)), float(np.quantile(t_draws, 0.975))],
        "n_at_bounds": int(((t_draws <= 1 / hi * (1 + 1e-6)) | (t_draws >= 1 / lo * (1 - 1e-6))).sum()),
    }
    times["temperature_fitted_utc"] = _now()
    write_json(out / "temperature.json", {**fit, "fitted_on": "validation logits only", "n_validation": int(len(val["y"]))})

    # ---- 4. only now: test logits, same T, nothing fitted ----
    times["test_loaded_utc"] = _now()
    test = load_predictions(run_dir, "test", seed, classes, split_df)
    verify_saved_probs(test, run_dir / "predictions" / "test.npz")

    n_bins = cfg["evaluation"]["ece_bins"]
    results = {}
    for name, p in (("validation_fitting_set", val), ("test", test)):
        z = p["logits"].astype(np.float64)
        zc = apply_temperature(z, T)
        _check(np.array_equal(softmax64(z).argmax(1), softmax64(zc).argmax(1)), f"{name}: argmax changed")
        results[name] = {"baseline": evaluate_logits(z, p["y"], classes, malignant, n_bins),
                         "calibrated": evaluate_logits(zc, p["y"], classes, malignant, n_bins)}
    zt, yt = test["logits"].astype(np.float64), test["y"]
    zt_c = apply_temperature(zt, T)

    def paired(idx):
        out_ = {}
        for key, fn in (("nll", lambda z_: nll(z_, yt[idx])), ("brier", lambda z_: brier(softmax64(z_), yt[idx])),
                        ("ece", lambda z_: top_label_calibration(softmax64(z_), yt[idx], n_bins)["ece"]),
                        ("classwise_ece", lambda z_: classwise_ece(softmax64(z_), yt[idx], n_bins, classes)["classwise_ece"])):
            b, c = fn(zt[idx]), fn(zt_c[idx])
            out_[f"{key}_baseline"], out_[f"{key}_calibrated"], out_[f"{key}_difference"] = b, c, c - b
        return out_

    bs = cfg["evaluation"]["bootstrap"]
    boot = cluster_bootstrap(paired, test["lesion_id"], bs["n_resamples"], bs["seed"], bs["confidence_level"])
    boot["note"] = "Paired lesion-cluster bootstrap on test; T held fixed (no refit), so CIs reflect test sampling only."

    # ---- outputs ----
    tb_ = results["test"]
    metrics = {
        "temperature": T, "beta": fit["beta"],
        "test": {k: {kk: vv for kk, vv in tb_[k].items() if kk not in ("per_class", "ece_bins")} for k in tb_},
        "validation_fitting_set": {k: {kk: vv for kk, vv in results["validation_fitting_set"][k].items()
                                       if kk not in ("per_class", "ece_bins")} for k in results["validation_fitting_set"]},
        "test_paired_bootstrap": boot,
        "argmax_unchanged": True,
        "note": "Validation metrics are in-sample for T (fitting set). Test is the only held-out evaluation.",
    }
    write_json(out / "metrics.json", metrics)
    rows = []
    for c in classes:
        b, a = tb_["baseline"], tb_["calibrated"]
        rows.append({"class": c, "support": b["per_class"][c]["support"],
                     "precision": b["per_class"][c]["precision"], "recall": b["per_class"][c]["recall"],
                     "f1": b["per_class"][c]["f1"],
                     "roc_auc_ovr_baseline": b["per_class"][c]["roc_auc_ovr"], "roc_auc_ovr_calibrated": a["per_class"][c]["roc_auc_ovr"],
                     "classwise_ece_baseline": b["classwise_ece_per_class"][c], "classwise_ece_calibrated": a["classwise_ece_per_class"][c]})
    pd.DataFrame(rows).to_csv(out / "per_class.csv", index=False, float_format="%.10g")
    np.savez(out / "test_calibrated.npz", logits=test["logits"], temperature=np.float64(T),
             probs_baseline=softmax64(zt), probs_calibrated=softmax64(zt_c), labels=yt,
             image_id=np.asarray(test["image_id"]), lesion_id=np.asarray(test["lesion_id"]), class_names=np.array(classes))

    figs = out / "figures"
    reliability_pair({"bins": tb_["baseline"]["ece_bins"], "ece": tb_["baseline"]["ece"]},
                     {"bins": tb_["calibrated"]["ece_bins"], "ece": tb_["calibrated"]["ece"]},
                     figs / "reliability_test", f"E0 seed {seed}, test: reliability before/after temperature scaling")
    vb = results["validation_fitting_set"]
    reliability_pair({"bins": vb["baseline"]["ece_bins"], "ece": vb["baseline"]["ece"]},
                     {"bins": vb["calibrated"]["ece_bins"], "ece": vb["calibrated"]["ece"]},
                     figs / "reliability_val", f"E0 seed {seed}, validation (fitting set, in-sample)")
    nll_curve(zv, val["y"], fit, figs / "val_nll_vs_temperature")
    confidence_hist(zt, T, figs / "confidence_test", f"E0 seed {seed}, test confidence before/after")

    ckpt_after = sha256_file(run_dir / "best.pt")
    _check(ckpt_after == run["checkpoint_sha256"], "best.pt changed during the experiment")
    env = collect_environment(REPO_ROOT)
    times["finished_utc"] = _now()
    write_json(out / "provenance.json", {
        "experiment": cfg["experiment"]["name"], "method": "temperature scaling (single scalar T)",
        "source_run": str(run_dir.relative_to(REPO_ROOT)), "seed": seed,
        "checkpoint": str((run_dir / "best.pt").relative_to(REPO_ROOT)), "checkpoint_sha256": run["checkpoint_sha256"],
        "checkpoint_sha256_after": ckpt_after, "checkpoint_selected_epoch": run["meta"]["selected_epoch"],
        "e0_config_hash": run["meta"]["config_hash"], "calibration_config": cfg, "calibration_config_hash": config_hash(cfg),
        "split_file": e0_cfg["split"]["file"], "split_sha256": run["split_sha256"],
        "class_order": classes, "preprocessing_eval": repr(backend_inference_transform).replace("\n", " "),
        "logit_recheck": recheck, "temperature": T, "fitted_on": "validation logits only",
        "test_usage": "loaded only after T was fitted and written; used solely for final evaluation",
        "timestamps": times, "git_commit": env["git_commit"], "git_dirty": env["git_dirty"],
        "environment": env, "command": command,
    })
    (out / "command.txt").write_text(command + "\n", encoding="utf-8")
    (out / "report.md").write_text(report_markdown(seed, fit, tb_, boot, run, recheck, command), encoding="utf-8")
    print(f"T = {T:.6f}; test NLL {tb_['baseline']['nll']:.4f} -> {tb_['calibrated']['nll']:.4f}, "
          f"ECE {tb_['baseline']['ece']:.4f} -> {tb_['calibrated']['ece']:.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
