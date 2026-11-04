"""
3-seed E0 deep ensemble, deferral vs malignancy-aware referral, and results by how each
label was confirmed (dx_type), from saved logits.

    python -m research.evaluation.ensemble_eval --config research/configs/ensemble_e0.yaml \
        --metadata <folder>/HAM10000_metadata.csv

Validation only: member temperatures are read from their calibration runs and all
thresholds are selected -> thresholds.json. Test logits are loaded only after that.
The E0 test split had already been viewed, so these comparisons are exploratory.
The model is never loaded; nothing existing is modified.
"""
import argparse
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.data.data import load_split
from research.evaluation.calibration_eval import verify_run, verify_saved_probs
from research.evaluation.calibration_metrics import brier, top_label_calibration
from research.evaluation.evaluate import load_predictions, verify_malignant_mapping
from research.evaluation.metrics import cluster_bootstrap, log_softmax64
from research.evaluation.selective_eval import load_temperature
from research.evaluation.selective_policies import (
    aurc_for_uncertainty,
    evaluate_referral,
    threshold_for_coverage,
    threshold_for_sensitivity,
)
from research.utils.config import REPO_ROOT, load_config, repo_path
from research.utils.environment import collect_environment
from research.utils.hashing import canonical_json, config_hash, sha256_text
from research.utils.io import read_json, write_json


class EnsembleError(RuntimeError):
    pass


def _check(cond, msg):
    if not cond:
        raise EnsembleError(msg)


def _entropy(p):
    return -(p * np.log(np.clip(p, 1e-300, None))).sum(axis=1)


def scores_from_probs(p, mal_idx, member_probs=None) -> dict:
    out = {"probs": p, "pred": p.argmax(axis=1), "msp": 1.0 - p.max(axis=1), "entropy": _entropy(p),
           "p_mal": p[:, mal_idx].sum(axis=1)}
    if member_probs is not None:
        out["mutual_information"] = out["entropy"] - np.mean([_entropy(m) for m in member_probs], axis=0)
    return out


def balanced_accuracy(y, pred, n_classes):
    recalls = [(pred[y == k] == k).mean() for k in range(n_classes) if (y == k).any()]
    return float(np.mean(recalls)) if recalls else float("nan")


def nll_from_probs(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-300, None)).mean())


def stratum_metrics(s, y, mask, classes, mal_idx, ref_thresholds) -> dict:
    """Metrics for one stratum. Referral thresholds are fixed on validation."""
    yy, pred, msp = y[mask], s["pred"][mask], s["msp"][mask]
    mel = classes.index("mel")
    out = {"n": int(mask.sum()), "accuracy": float((pred == yy).mean()),
           "balanced_accuracy": balanced_accuracy(yy, pred, len(classes)),
           "melanoma_recall": float((pred[yy == mel] == mel).mean()) if (yy == mel).any() else float("nan"),
           "aurc_msp": aurc_for_uncertainty(msp, pred == yy)["aurc"] if (pred != yy).any() else 0.0, "referral": {}}
    for name, t in ref_thresholds.items():
        out["referral"][name] = evaluate_referral(s["p_mal"][mask], yy, t, classes, [classes[i] for i in mal_idx])
    return out


def workload_policy(s, y, follow_up, biopsy, tau, score, classes, mal_idx) -> dict:
    """Automate a case iff its score <= tau. A dangerous clearance is a malignant case automated with a benign label."""
    auto = s[score] <= tau
    is_mal = np.isin(y, mal_idx)
    benign_label = ~np.isin(s["pred"], mal_idx)
    is_mel = y == classes.index("mel")
    wrong = s["pred"] != y
    return {"threshold": tau, "test_referred": float((~auto).mean()),
            "automated_error": float(wrong[auto].mean()) if auto.any() else float("nan"),
            "automated_error_biopsy": float(wrong[auto & biopsy].mean()) if (auto & biopsy).any() else float("nan"),
            "follow_up_automated": float(auto[follow_up].mean()),
            "malignant_automated": float(auto[is_mal].mean()),
            "malignant_cleared_as_benign": int((auto & is_mal & benign_label).sum()),
            "melanoma_cleared_as_benign": int((auto & is_mel & benign_label).sum()),
            "n_malignant": int(is_mal.sum()), "n_melanoma": int(is_mel.sum())}


def calibration_block(p, y, n_bins) -> dict:
    return {"nll": nll_from_probs(p, y), "brier": float(brier(p, y)), "ece": float(top_label_calibration(p, y, n_bins)["ece"])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--metadata", required=True)
    args = ap.parse_args(argv)
    command = shlex.join(["python", "-m", "research.evaluation.ensemble_eval", *(argv if argv is not None else sys.argv[1:])])

    cfg = load_config(args.config)
    e0_cfg = load_config(repo_path("research/configs/e0.yaml"))
    verify_malignant_mapping(e0_cfg)
    classes = e0_cfg["data"]["classes"]
    _check(cfg["referral"]["malignant_classes"] == e0_cfg["evaluation"]["malignant_classes"], "malignant mapping differs")
    mal_idx = [classes.index(c) for c in cfg["referral"]["malignant_classes"]]
    seeds = cfg["seeds"]
    out = repo_path(cfg["experiment"]["output_dir"])
    _check(not out.exists() or not any(out.iterdir()), f"{out} already exists and is not empty; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    split_df, _ = load_split(e0_cfg, verify_hash=True)

    # ===================== validation only =====================
    base = repo_path("research/outputs/e0_baseline")
    temps, runs = {}, {}
    val_members = []
    for seed in seeds:
        run_dir = base / f"seed_{seed}"
        runs[seed] = verify_run(run_dir, e0_cfg)
        _check(runs[seed]["meta"]["seed"] == seed, f"{run_dir} holds another seed")
        temps[seed], _ = load_temperature(repo_path("research/outputs/calibration") / f"seed_{seed}", runs[seed])
        val = load_predictions(run_dir, "val", seed, classes, split_df)
        verify_saved_probs(val, run_dir / "predictions" / "val.npz")
        val_members.append(val)
    yv = val_members[0]["y"]
    for v in val_members:
        _check(np.array_equal(v["image_id"], val_members[0]["image_id"]) and np.array_equal(v["y"], yv), "validation rows differ across seeds")
    val_probs = [np.exp(log_softmax64(v["logits"].astype(np.float64) / temps[s])) for v, s in zip(val_members, seeds)]

    def model_scores(probs):
        models = {f"seed_{s}": scores_from_probs(p, mal_idx) for s, p in zip(seeds, probs)}
        models["ensemble"] = scores_from_probs(np.mean(probs, axis=0), mal_idx, member_probs=probs)
        return models

    vs = model_scores(val_probs)
    is_mal_v = np.isin(yv, mal_idx)
    thresholds = {}
    for name, s in vs.items():
        t = {"referral": {}, "workload": {}}
        for target in cfg["referral"]["sensitivity_targets"]:
            r = threshold_for_sensitivity(s["p_mal"], is_mal_v, target)
            _check(r["feasible"], f"{name}: sensitivity target {target} infeasible")
            t["referral"][f"sensitivity_{target:.2f}"] = r
        for frac in cfg["workload"]["referred_fractions"]:
            t["workload"][f"refer_{frac:.2f}"] = {score: threshold_for_coverage(s[score], 1 - frac)
                                                  for score in s if score not in ("probs", "pred")}
        thresholds[name] = t
    write_json(out / "thresholds.json", {"selected_on": "validation only", "temperatures": {str(k): v for k, v in temps.items()}, "models": thresholds})
    thresholds_hash = sha256_text(canonical_json(thresholds))

    # ===================== test: thresholds fixed =====================
    test_members = []
    for seed in seeds:
        run_dir = base / f"seed_{seed}"
        test = load_predictions(run_dir, "test", seed, classes, split_df)
        verify_saved_probs(test, run_dir / "predictions" / "test.npz")
        test_members.append(test)
    y = test_members[0]["y"]
    image_ids = np.asarray(test_members[0]["image_id"])
    lesions = np.asarray(test_members[0]["lesion_id"])
    for t in test_members:
        _check(np.array_equal(np.asarray(t["image_id"]), image_ids) and np.array_equal(t["y"], y), "test rows differ across seeds")
    meta = pd.read_csv(args.metadata).set_index("image_id")
    dx_type = meta.loc[image_ids, "dx_type"].to_numpy()
    _check(not pd.isna(dx_type).any(), "dx_type missing for a test image")
    masks = {"all": np.ones(len(y), dtype=bool)}
    masks.update({k: np.isin(dx_type, v) for k, v in cfg["strata"].items()})
    test_probs = [np.exp(log_softmax64(t["logits"].astype(np.float64) / temps[s])) for t, s in zip(test_members, seeds)]
    ts = model_scores(test_probs)

    results = {}
    for name, s in ts.items():
        ref = {k: v["threshold"] for k, v in thresholds[name]["referral"].items()}
        results[name] = {
            "calibration_temperature_scaled": calibration_block(s["probs"], y, cfg["ece_bins"]),
            "strata": {k: stratum_metrics(s, y, m, classes, mal_idx, ref) for k, m in masks.items()},
            "workload": {frac: {score: workload_policy(s, y, masks["follow_up_only"], masks["biopsy_confirmed"], tr["threshold"], score, classes, mal_idx)
                                for score, tr in by_score.items()}
                         for frac, by_score in thresholds[name]["workload"].items()},
        }

    members = [f"seed_{s}" for s in seeds]

    def seed_summary(get):
        vals = np.array([get(results[m]) for m in members], dtype=float)
        return {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)), "min": float(vals.min()), "max": float(vals.max()), "values": dict(zip(members, vals.tolist()))}

    summary = {}
    for stratum in masks:
        summary[stratum] = {m: seed_summary(lambda r, k=m: r["strata"][stratum][k]) for m in ("accuracy", "balanced_accuracy", "melanoma_recall", "aurc_msp")}
        for target in thresholds["ensemble"]["referral"]:
            summary[stratum][target] = {m: seed_summary(lambda r, k=m: r["strata"][stratum]["referral"][target][k])
                                        for m in ("sensitivity", "specificity", "referral_rate", "malignant_cleared", "melanoma_cleared")}
    summary["calibration"] = {m: seed_summary(lambda r, k=m: r["calibration_temperature_scaled"][k]) for m in ("nll", "brier", "ece")}
    summary["workload"] = {frac: {score: {m: seed_summary(lambda r, f=frac, c=score, k=m: r["workload"][f][c][k])
                                          for m in ("automated_error", "automated_error_biopsy", "follow_up_automated", "malignant_automated", "malignant_cleared_as_benign", "melanoma_cleared_as_benign")}
                                  for score in ("msp", "entropy", "p_mal")}
                           for frac in thresholds["ensemble"]["workload"]}

    # ---- paired lesion bootstrap: ensemble, and the comparisons that decide the paper's claims ----
    ens, biopsy = ts["ensemble"], masks["biopsy_confirmed"]
    ref95 = thresholds["ensemble"]["referral"]["sensitivity_0.95"]["threshold"]
    is_mal = np.isin(y, mal_idx)
    benign_label = ~np.isin(ens["pred"], mal_idx)
    work = {frac: {c: thresholds["ensemble"]["workload"][frac][c]["threshold"] for c in ("msp", "p_mal")} for frac in thresholds["ensemble"]["workload"]}

    def aurc_of(u, correct):
        return aurc_for_uncertainty(u, correct)["aurc"] if len(u) and not correct.all() else 0.0

    def boot_metric(idx):
        yy, pred = y[idx], ens["pred"][idx]
        correct, b = pred == yy, biopsy[idx]
        mal = is_mal[idx]
        ref = ens["p_mal"][idx] >= ref95
        out_ = {"accuracy": float(correct.mean()), "balanced_accuracy": balanced_accuracy(yy, pred, len(classes)),
                "melanoma_recall": float(correct[yy == classes.index("mel")].mean()),
                "aurc_msp_all": aurc_of(ens["msp"][idx], correct), "aurc_msp_biopsy": aurc_of(ens["msp"][idx][b], correct[b]),
                "aurc_mi_all": aurc_of(ens["mutual_information"][idx], correct),
                "referral95_sensitivity": float(ref[mal].mean()), "referral95_specificity": float((~ref[~mal]).mean()),
                "referral95_specificity_biopsy": float((~ref[~mal & b]).mean()), "referral95_rate": float(ref.mean())}
        out_["aurc_mi_minus_msp_all"] = out_["aurc_mi_all"] - out_["aurc_msp_all"]
        member_aurc = []
        member_aurc_b = []
        for m in members:
            s = ts[m]
            c_m = s["pred"][idx] == yy
            member_aurc.append(aurc_of(s["msp"][idx], c_m))
            member_aurc_b.append(aurc_of(s["msp"][idx][b], c_m[b]))
        out_["aurc_msp_all_ensemble_minus_member_mean"] = out_["aurc_msp_all"] - float(np.mean(member_aurc))
        out_["aurc_msp_biopsy_ensemble_minus_member_mean"] = out_["aurc_msp_biopsy"] - float(np.mean(member_aurc_b))
        for frac, tau in work.items():
            d = {}
            for c, t in tau.items():
                auto = ens[c][idx] <= t
                d[c] = float((auto & mal & benign_label[idx]).sum()) / max(int(mal.sum()), 1)
            out_[f"{frac}_cleared_share_msp"] = d["msp"]
            out_[f"{frac}_cleared_share_p_mal"] = d["p_mal"]
            out_[f"{frac}_cleared_share_p_mal_minus_msp"] = d["p_mal"] - d["msp"]
        return out_

    bs = cfg["bootstrap"]
    boot = cluster_bootstrap(boot_metric, lesions, bs["n_resamples"], bs["seed"], bs["confidence_level"])
    _check(sha256_text(canonical_json(thresholds)) == thresholds_hash, "thresholds changed after test was loaded")

    # Each member must reproduce its own saved referral result exactly.
    for seed in seeds:
        saved = read_json(repo_path("research/outputs/selective_prediction") / f"seed_{seed}" / "referral_malignant.json")["test"]["p_mal_TS"]
        for target, r in results[f"seed_{seed}"]["strata"]["all"]["referral"].items():
            for key in ("threshold", "sensitivity", "specificity", "referral_rate", "malignant_cleared", "melanoma_cleared"):
                _check(abs(r[key] - saved[target][key]) < 1e-9, f"seed {seed} {target} {key} differs from its saved result")

    write_json(out / "summary.json", {"label": "Exploratory: the E0 test split had already been viewed", "seeds": seeds, "per_model": results, "across_seeds": summary, "ensemble_bootstrap": boot})
    write_json(out / "provenance.json", {"command": command, "config_hash": config_hash(cfg), "thresholds_sha256": thresholds_hash,
                                         "checkpoints": {str(s): runs[s]["meta"]["checkpoint_sha256"] for s in seeds},
                                         "environment": collect_environment(REPO_ROOT)})
    np.savez(out / "ensemble_test.npz", probs=ens["probs"], labels=y, image_id=image_ids, lesion_id=lesions, dx_type=dx_type.astype(str),
             mutual_information=ens["mutual_information"], class_names=np.array(classes))
    (out / "report.md").write_text(report(results, summary, boot, seeds, temps, command), encoding="utf-8")
    print((out / "report.md").read_text(encoding="utf-8"))
    return 0


def _pct(x):
    return f"{100 * x:.1f}%"


def report(results, summary, boot, seeds, temps, command) -> str:
    def ci(name, f=_pct):
        b = boot["metrics"][name]
        return f"{f(b['point'])} [{f(b['ci_low'])}, {f(b['ci_high'])}]"

    def f4(x):
        return f"{x:.4f}"

    def ms(d, f=_pct):
        return f"{f(d['mean'])} ± {f(d['sd'])}"

    ens = results["ensemble"]
    lines = [f"# 3-seed E0 ensemble (seeds {', '.join(map(str, seeds))}): exploratory", "",
             "Every threshold comes from validation. The E0 test split had already been viewed, so these are not confirmatory results.",
             f"Temperatures: {temps}", "", f"Command: `{command}`", "", "## Classification, by label confirmation", "",
             "| stratum | n | single model accuracy | ensemble accuracy | single balanced | ensemble balanced | single melanoma recall | ensemble melanoma recall | single AURC | ensemble AURC |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k, e in ens["strata"].items():
        s = summary[k]
        lines.append(f"| {k} | {e['n']} | {ms(s['accuracy'])} | {_pct(e['accuracy'])} | {ms(s['balanced_accuracy'])} | {_pct(e['balanced_accuracy'])} | "
                     f"{ms(s['melanoma_recall'])} | {_pct(e['melanoma_recall'])} | {ms(s['aurc_msp'], f4)} | {f4(e['aurc_msp'])} |")
    lines += ["", "## Calibration (temperature-scaled)", "", "| metric | single model | ensemble |", "|---|---|---|"]
    for m in ("nll", "brier", "ece"):
        lines.append(f"| {m} | {ms(summary['calibration'][m], f4)} | {f4(ens['calibration_temperature_scaled'][m])} |")
    lines += ["", "## Malignant referral", "", "| target | stratum | single sensitivity | ensemble | single specificity | ensemble | single referred | ensemble | malignant cleared, single | ensemble | melanoma cleared, single | ensemble |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for target in ("sensitivity_0.95", "sensitivity_0.90"):
        for k in ("all", "biopsy_confirmed", "follow_up_only"):
            e = ens["strata"][k]["referral"][target]
            s = summary[k][target]
            count = lambda d: f"{d['mean']:.1f} ± {d['sd']:.1f}"
            sens = "n/a" if e["n_malignant"] == 0 else _pct(e["sensitivity"])
            single_sens = "n/a" if e["n_malignant"] == 0 else ms(s["sensitivity"])
            lines.append(f"| {target} | {k} | {single_sens} | {sens} | {ms(s['specificity'])} | {_pct(e['specificity'])} | {ms(s['referral_rate'])} | {_pct(e['referral_rate'])} | "
                         f"{count(s['malignant_cleared'])} | {e['malignant_cleared']} | {count(s['melanoma_cleared'])} | {e['melanoma_cleared']} |")
    lines += ["", "## Same workload: confidence deferral vs malignancy-aware referral", "",
              "A dangerous clearance is a malignant case automated with a benign label.", "",
              "| referred (validation) | model | score | test referred | automated error | automated error, biopsy | follow-up automated | malignant automated | malignant cleared | melanoma cleared |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for frac in ens["workload"]:
        for score in ("msp", "entropy", "mutual_information", "p_mal"):
            w = ens["workload"][frac][score]
            lines.append(f"| {frac} | ensemble | {score} | {_pct(w['test_referred'])} | {_pct(w['automated_error'])} | {_pct(w['automated_error_biopsy'])} | {_pct(w['follow_up_automated'])} | {_pct(w['malignant_automated'])} | {w['malignant_cleared_as_benign']} of {w['n_malignant']} | {w['melanoma_cleared_as_benign']} of {w['n_melanoma']} |")
        for score in ("msp", "entropy", "p_mal"):
            s = summary["workload"][frac][score]
            count = lambda d: f"{d['mean']:.1f} ± {d['sd']:.1f}"
            lines.append(f"| {frac} | single | {score} | | {ms(s['automated_error'])} | {ms(s['automated_error_biopsy'])} | {ms(s['follow_up_automated'])} | {ms(s['malignant_automated'])} | {count(s['malignant_cleared_as_benign'])} | {count(s['melanoma_cleared_as_benign'])} |")
    lines += ["", "## Ensemble, lesion-bootstrap 95% CI", "", "| metric | point [CI] |", "|---|---|"]
    for name in boot["metrics"]:
        as_pct = not name.startswith("aurc")
        lines.append(f"| {name} | {ci(name, _pct if as_pct else f4)} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
