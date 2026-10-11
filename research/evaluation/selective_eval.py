"""
Selective prediction (multiclass) and malignant referral (binary) for ONE trained E0 run,
from saved logits. PRELIMINARY single-seed analysis.

    python -m research.evaluation.selective_eval --config research/configs/selective_e0.yaml \
        --run-dir research/outputs/e0_baseline/seed_42 --calibration-dir research/outputs/calibration/seed_42

Phase 1 (validation only): scores, thresholds for every predeclared target -> thresholds.json
(hash recorded). Phase 2: test logits are loaded only after that, thresholds are applied
unchanged; at the end thresholds are recomputed from validation and must hash identically.
Operating points are empirical validation targets, not guarantees of test performance or
clinical safety. The model is never loaded; nothing existing is modified.
"""
import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.data.data import load_split
from research.evaluation.calibration_eval import verify_run, verify_saved_probs
from research.evaluation.evaluate import load_predictions, verify_malignant_mapping
from research.evaluation.metrics import cluster_bootstrap
from research.evaluation.selective_policies import (
    aurc_for_uncertainty,
    evaluate_referral,
    evaluate_selective,
    ranking_agreement,
    scores_from_logits,
    threshold_for_coverage,
    threshold_for_risk,
    threshold_for_sensitivity,
)
from research.evaluation.selective_prediction import risk_coverage_curve
from research.utils.config import REPO_ROOT, load_config, repo_path
from research.utils.environment import collect_environment
from research.utils.hashing import canonical_json, config_hash, sha256_text
from research.utils.io import read_json, to_jsonable, write_json
from research.utils.seed import set_seed

VERSIONS = {"uncalibrated": "T1", "temperature_scaled": "TS"}


class SelectiveError(RuntimeError):
    pass


def _check(cond, msg):
    if not cond:
        raise SelectiveError(msg)


def _now():
    return datetime.now(timezone.utc).isoformat()


def verify_lesions(pred: dict, split_df: pd.DataFrame, split: str):
    """lesion_id saved with the logits must equal the split file's lesion_id for every image."""
    ref = split_df[split_df["split"] == split].set_index("image_id")["lesion_id"]
    saved = pd.Series(np.asarray(pred["lesion_id"]), index=np.asarray(pred["image_id"]))
    _check(saved.index.is_unique, f"{split}: duplicate image ids")
    _check((ref.loc[saved.index].to_numpy() == saved.to_numpy()).all(), f"{split}: lesion ids disagree with the split file")


def load_temperature(cal_dir: Path, run: dict) -> tuple[float, dict]:
    t = read_json(cal_dir / "temperature.json")
    prov = read_json(cal_dir / "provenance.json")
    _check(prov["checkpoint_sha256"] == run["checkpoint_sha256"], "calibration run used a different checkpoint")
    _check(prov["split_sha256"] == run["split_sha256"], "calibration run used a different split")
    _check(t["fitted_on"] == "validation logits only" and all(t["checks"].values()), "temperature fit not validation-only/valid")
    return float(t["temperature"]), {"temperature": float(t["temperature"]), "calibration_dir": str(cal_dir),
                                     "calibration_git_commit": prov["git_commit"]}


def all_scores(logits, T, mal_idx):
    return {v: scores_from_logits(logits, 1.0 if v == "uncalibrated" else T, mal_idx) for v in VERSIONS}


def select_thresholds(val_scores: dict, y_val: np.ndarray, classes, cfg) -> dict:
    """VALIDATION ONLY. Returns every predeclared operating point, feasible or not."""
    mc, rf = cfg["multiclass"], cfg["referral"]
    mal_idx = [classes.index(c) for c in rf["malignant_classes"]]
    out = {"multiclass": {}, "referral": {}}
    for v, s in val_scores.items():
        correct = s["pred"] == y_val
        for score in cfg["scores"]["decision"]:
            key = f"{score}_{VERSIONS[v]}"
            out["multiclass"][key] = {
                **{f"coverage_{t:.2f}": threshold_for_coverage(s[score], t) for t in mc["coverage_targets"]},
                **{f"risk_{t:.2f}": threshold_for_risk(s[score], correct, t) for t in mc["risk_targets"]},
            }
        is_mal = np.isin(y_val, mal_idx)
        out["referral"][f"p_mal_{VERSIONS[v]}"] = {
            f"sensitivity_{t:.2f}": threshold_for_sensitivity(s["p_mal"], is_mal, t) for t in rf["sensitivity_targets"]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--calibration-dir", required=True)
    args = ap.parse_args(argv)
    command = shlex.join(["python", "-m", "research.evaluation.selective_eval", *(argv if argv is not None else sys.argv[1:])])

    set_seed(0, deterministic=True)
    cfg = load_config(args.config)
    e0_cfg = load_config(repo_path("research/configs/e0.yaml"))
    verify_malignant_mapping(e0_cfg)
    classes = e0_cfg["data"]["classes"]
    _check(cfg["referral"]["malignant_classes"] == e0_cfg["evaluation"]["malignant_classes"], "malignant mapping differs")
    mal_idx = [classes.index(c) for c in cfg["referral"]["malignant_classes"]]
    run_dir, cal_dir = repo_path(args.run_dir), repo_path(args.calibration_dir)
    times = {"started_utc": _now()}

    run = verify_run(run_dir, e0_cfg)
    seed = run["meta"]["seed"]
    T, t_info = load_temperature(cal_dir, run)
    out = repo_path(cfg["experiment"]["output_dir"]) / f"seed_{seed}"
    _check(not out.exists() or not any(out.iterdir()), f"{out} already exists and is not empty; refusing to overwrite")
    for d in ("figures", "curves"):
        (out / d).mkdir(parents=True, exist_ok=True)
    split_df, _ = load_split(e0_cfg, verify_hash=True)

    # ===================== PHASE 1: validation only =====================
    val = load_predictions(run_dir, "val", seed, classes, split_df)
    verify_saved_probs(val, run_dir / "predictions" / "val.npz")
    verify_lesions(val, split_df, "val")
    yv = val["y"]
    vs = all_scores(val["logits"], T, mal_idx)
    for v in VERSIONS:
        _check(np.array_equal(vs[v]["pred"], vs["uncalibrated"]["pred"]), "temperature changed an argmax")
    thresholds = select_thresholds(vs, yv, classes, cfg)
    thr_json = canonical_json(to_jsonable(thresholds))
    thr_hash = sha256_text(thr_json)
    write_json(out / "thresholds.json", {"selected_on": "validation only", "n_validation": int(len(yv)),
                                         "temperature": T, "thresholds": thresholds, "sha256_canonical": thr_hash})
    infeasible = [f"{g}/{k}/{t}" for g, grp in thresholds.items() for k, d in grp.items() for t, r in d.items()
                  if not r["feasible"]]
    times["thresholds_written_utc"] = _now()

    # ===================== PHASE 2: test, thresholds fixed =====================
    times["test_loaded_utc"] = _now()
    test = load_predictions(run_dir, "test", seed, classes, split_df)
    verify_saved_probs(test, run_dir / "predictions" / "test.npz")
    verify_lesions(test, split_df, "test")
    yt, lesions = test["y"], np.asarray(test["lesion_id"])
    ts = all_scores(test["logits"], T, mal_idx)
    correct_t = ts["uncalibrated"]["pred"] == yt
    correct_v = vs["uncalibrated"]["pred"] == yv

    # ranking invariance (T1 vs TS) on both splits
    invariance = {}
    for split, sc, corr in (("val", vs, correct_v), ("test", ts, correct_t)):
        invariance[split] = {}
        for score in ("msp", "entropy", "margin"):
            a, b = sc["uncalibrated"][score], sc["temperature_scaled"][score]
            invariance[split][score] = {**ranking_agreement(a, b),
                                        "aurc_T1": aurc_for_uncertainty(a, corr)["aurc"],
                                        "aurc_TS": aurc_for_uncertainty(b, corr)["aurc"]}
        p_a, p_b = sc["uncalibrated"]["p_mal"], sc["temperature_scaled"]["p_mal"]
        invariance[split]["p_mal"] = ranking_agreement(p_a, p_b)
    write_json(out / "ranking_invariance.json", invariance)

    # multiclass: whole-curve AURC (decision + exploratory scores) and fixed-threshold policies
    aurc_test = {f"{score}_{VERSIONS[v]}": aurc_for_uncertainty(ts[v][score], correct_t)
                 for v in VERSIONS for score in cfg["scores"]["decision"] + cfg["scores"]["exploratory"]}
    policies, rows = {}, []
    for key, ops in thresholds["multiclass"].items():
        score, ver = key.rsplit("_", 1)
        v = next(k for k, s in VERSIONS.items() if s == ver)
        policies[key] = {}
        for op_name, op in ops.items():
            if not op["feasible"]:
                policies[key][op_name] = {"feasible_on_validation": False, "reason": op["reason"]}
                continue
            res = evaluate_selective(ts[v][score], yt, ts[v]["pred"], op["threshold"], classes)
            res["validation"] = op
            policies[key][op_name] = res
            for c, d in res["per_class"].items():
                rows.append({"ranking": key, "operating_point": op_name, "class": c, **d})
    pd.DataFrame(rows).to_csv(out / "per_class_retention.csv", index=False, float_format="%.6g")

    referral = {}
    for key, ops in thresholds["referral"].items():
        v = next(k for k, s in VERSIONS.items() if s == key.rsplit("_", 1)[1])
        referral[key] = {name: {**evaluate_referral(ts[v]["p_mal"], yt, op["threshold"], classes,
                                                    cfg["referral"]["malignant_classes"]), "validation": op}
                         for name, op in ops.items() if op["feasible"]}

    # paired lesion bootstrap: one resample feeds every metric
    def boot_fn(idx):
        y, c = yt[idx], correct_t[idx]
        r = {}
        for k in ("msp_TS", "entropy_TS", "msp_T1", "entropy_T1"):
            score, ver = k.rsplit("_", 1)
            v = next(kk for kk, s in VERSIONS.items() if s == ver)
            r[f"aurc_{k}"] = aurc_for_uncertainty(ts[v][score][idx], c)["aurc"]
        r["PRIMARY_aurc_entropy_TS_minus_msp_TS"] = r["aurc_entropy_TS"] - r["aurc_msp_TS"]
        r["aurc_entropy_T1_minus_msp_T1"] = r["aurc_entropy_T1"] - r["aurc_msp_T1"]
        r["aurc_msp_TS_minus_msp_T1"] = r["aurc_msp_TS"] - r["aurc_msp_T1"]
        r["aurc_entropy_TS_minus_entropy_T1"] = r["aurc_entropy_TS"] - r["aurc_entropy_T1"]
        for key, ops in policies.items():
            score, ver = key.rsplit("_", 1)
            v = next(kk for kk, s in VERSIONS.items() if s == ver)
            for op_name, res in ops.items():
                if "threshold" in res:
                    acc = ts[v][score][idx] <= res["threshold"]
                    r[f"{key}/{op_name}/coverage"] = float(acc.mean())
                    r[f"{key}/{op_name}/selective_risk"] = float((~c[acc]).mean()) if acc.any() else float("nan")
        for key, ops in referral.items():
            v = next(kk for kk, s in VERSIONS.items() if s == key.rsplit("_", 1)[1])
            for name, res in ops.items():
                ref = ts[v]["p_mal"][idx] >= res["threshold"]
                m = np.isin(y, mal_idx)
                r[f"referral/{key}/{name}/sensitivity"] = float((ref & m).sum() / m.sum()) if m.any() else float("nan")
                r[f"referral/{key}/{name}/specificity"] = float((~ref & ~m).sum() / (~m).sum()) if (~m).any() else float("nan")
                r[f"referral/{key}/{name}/referral_rate"] = float(ref.mean())
        return r

    bs = cfg["bootstrap"]
    boot = cluster_bootstrap(boot_fn, lesions, bs["n_resamples"], bs["seed"], bs["confidence_level"])
    boot["note"] = ("Paired lesion-cluster bootstrap on test; all metrics of a replicate share one resample. "
                    "Thresholds held fixed (validation-selected), so CIs cover test sampling only.")
    write_json(out / "bootstrap_test.json", boot)

    primary = {"definition": cfg["primary"], "aurc_entropy_TS": aurc_test["entropy_TS"]["aurc"],
               "aurc_msp_TS": aurc_test["msp_TS"]["aurc"],
               "difference": boot["metrics"]["PRIMARY_aurc_entropy_TS_minus_msp_TS"],
               "interpretation_rule": "negative difference = entropy ranks errors later (better); CI excluding 0 = "
                                      "difference beyond test sampling variation for this single seed"}
    write_json(out / "multiclass_selective.json", {
        "label": "PRELIMINARY single-seed; operating points are empirical validation targets, not guarantees",
        "primary": primary, "aurc_test_secondary_and_exploratory": aurc_test,
        "exploratory_scores": cfg["scores"]["exploratory"], "policies_test": policies,
        "infeasible_on_validation": infeasible})
    write_json(out / "referral_malignant.json", {
        "label": "PRELIMINARY single-seed; separate binary referral analysis (not multiclass selective prediction)",
        "mapping": {"malignant": cfg["referral"]["malignant_classes"],
                    "note": "project mapping; akiec is clinically pre-malignant"},
        "test": referral})

    # curves + figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for k, style in (("msp_TS", "-"), ("entropy_TS", "-"), ("msp_T1", ":"), ("entropy_T1", ":"), ("margin_TS", "--")):
        score, ver = k.rsplit("_", 1)
        v = next(kk for kk, s in VERSIONS.items() if s == ver)
        cov, risk, _ = risk_coverage_curve(-ts[v][score], correct_t)
        pd.DataFrame({"coverage": cov, "risk": risk}).to_csv(out / "curves" / f"test_{k}.csv", index=False, float_format="%.10g")
        ax.plot(cov, risk, style, linewidth=1.1, label=f"{k} (AURC {aurc_test[k]['aurc']:.4f})" + (" [exploratory]" if score == "margin" else ""))
    for op_name, res in policies["entropy_TS"].items():
        if "threshold" in res:
            ax.plot(res["coverage"], res["selective_risk"], "ko", markersize=3)
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk (test)")
    ax.set_title(f"E0 seed {seed}, test (preliminary); dots: entropy_TS validation-selected points", fontsize=8)
    ax.legend(fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(out / "figures" / f"risk_coverage_test.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    for key in referral:
        v = next(kk for kk, s in VERSIONS.items() if s == key.rsplit("_", 1)[1])
        p, m = ts[v]["p_mal"], np.isin(yt, mal_idx)
        grid = np.unique(p)
        ax.plot([(p >= t).mean() for t in grid], [(p[m] >= t).mean() for t in grid], linewidth=1.1, label=key)
        for name, res in referral[key].items():
            ax.plot(res["referral_rate"], res["sensitivity"], "o", markersize=3)
    ax.set_xlabel("referral rate (test)")
    ax.set_ylabel("malignant sensitivity (test)")
    ax.set_title("Malignant referral (separate binary analysis); dots: validation-selected t", fontsize=8)
    ax.legend(fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(out / "figures" / f"referral_test.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # thresholds must be reproducible from validation alone
    _check(sha256_text(canonical_json(to_jsonable(select_thresholds(vs, yv, classes, cfg)))) == thr_hash,
           "thresholds are not reproducible from validation data")
    times["finished_utc"] = _now()
    env = collect_environment(REPO_ROOT)
    write_json(out / "provenance.json", {
        "experiment": cfg["experiment"]["name"], "label": "PRELIMINARY single-seed",
        "source_run": str(run_dir.relative_to(REPO_ROOT)), "seed": seed, "checkpoint_sha256": run["checkpoint_sha256"],
        "split_sha256": run["split_sha256"], "e0_config_hash": run["meta"]["config_hash"],
        "selective_config": cfg, "selective_config_hash": config_hash(cfg), **t_info,
        "thresholds_sha256": thr_hash, "thresholds_selected_on": "validation only",
        "test_usage": "loaded after thresholds.json was written; thresholds re-derived from validation at the end and matched",
        "infeasible_targets": infeasible, "timestamps": times, "git_commit": env["git_commit"],
        "git_dirty": env["git_dirty"], "environment": env, "command": command, "model_loaded": False})
    (out / "command.txt").write_text(command + "\n", encoding="utf-8")
    (out / "report.md").write_text(report(primary, aurc_test, policies, referral, invariance, infeasible, T, seed, command),
                                   encoding="utf-8")
    d = primary["difference"]
    print(f"PRIMARY: AURC entropy_TS {primary['aurc_entropy_TS']:.4f} vs msp_TS {primary['aurc_msp_TS']:.4f}; "
          f"diff {d['point']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")
    print(f"infeasible validation targets: {infeasible or 'none'}")
    print(f"wrote {out}")
    return 0


def report(primary, aurc_test, policies, referral, inv, infeasible, T, seed, command) -> str:
    d = primary["difference"]
    L = [f"# Selective prediction & referral: E0 seed {seed} (PRELIMINARY, single seed)", "",
         "Operating points are **empirical validation targets**; they are not guarantees of test performance or of clinical safety.",
         f"Probabilities: uncalibrated (T1) and temperature-scaled (TS, T = {T:.4f}, fitted on validation earlier).", "",
         "## Primary analysis", "",
         f"Test AURC, entropy_TS {primary['aurc_entropy_TS']:.4f} vs msp_TS {primary['aurc_msp_TS']:.4f}: difference "
         f"{d['point']:+.4f} [paired lesion-bootstrap 95% CI {d['ci_low']:+.4f}, {d['ci_high']:+.4f}] (negative favours entropy).", "",
         "## Secondary: multiclass selective prediction (validation-selected thresholds applied to test)", "",
         "| ranking | operating point | val coverage | val risk | test coverage | test risk | test bal. acc. retained | mel retained/deferred | mel retained & wrong |",
         "|---|---|---|---|---|---|---|---|---|"]
    for key, ops in policies.items():
        for op, r in ops.items():
            if "threshold" not in r:
                L.append(f"| {key} | {op} | infeasible on validation | | | | | | |")
                continue
            vr = r["validation"].get("val_risk")
            L.append(f"| {key} | {op} | {r['validation']['val_coverage']:.3f} | {'' if vr is None else f'{vr:.3f}'} | "
                     f"{r['coverage']:.3f} | {r['selective_risk']:.3f} | {r['balanced_accuracy_among_retained']:.3f} | "
                     f"{r['melanoma']['retained']}/{r['melanoma']['deferred']} | {r['melanoma']['retained_and_misclassified']} |")
    L += ["", "Per-class retention: `per_class_retention.csv`. CIs: `bootstrap_test.json`.", "",
          "AURC (test): " + ", ".join(f"{k} {v['aurc']:.4f}" for k, v in aurc_test.items()) + " (margin_* exploratory).", "",
          "## Secondary: malignant-vs-benign referral (separate binary analysis)", "",
          "| score | target | t | val sens | test sens | test spec | test referral rate | malignant cleared | melanoma cleared |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key, ops in referral.items():
        for name, r in ops.items():
            L.append(f"| {key} | {name} | {r['threshold']:.4f} | {r['validation']['val_sensitivity']:.3f} | {r['sensitivity']:.3f} | "
                     f"{r['specificity']:.3f} | {r['referral_rate']:.3f} | {r['malignant_cleared']} | {r['melanoma_cleared']} |")
    L += ["", "## Ranking invariance, T1 vs TS (verified, not assumed)", "", "| split | score | discordant pairs | Kendall tau-b | AURC T1 | AURC TS |",
          "|---|---|---|---|---|---|"]
    for split, d_ in inv.items():
        for score, r in d_.items():
            L.append(f"| {split} | {score} | {r['discordant']} / {r['pairs']} | {r['kendall_tau_b']:.6f} | "
                     f"{r.get('aurc_T1', float('nan')):.4f} | {r.get('aurc_TS', float('nan')):.4f} |")
    L += ["", f"Infeasible validation targets: {infeasible or 'none'}", "", "```", command, "```", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
