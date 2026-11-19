"""
Post-hoc reading of the baseline comparison: is fusion's lower referral rate better ranking, or only
a different operating point? Uses saved logits and temperatures; runs no model and fits nothing.

    python -m research.evaluation.baseline_posthoc --metadata <HAM10000_metadata.csv> --isic-ground-truth <csv>

Compares the malignant score of the 2-seed ensembles by AUROC, and by sensitivity when each arm refers
the same share of images as fusion did at its own frozen threshold. Differences are fusion minus arm.
"""
import argparse

import numpy as np
import pandas as pd

from research.data.data import load_split
from research.evaluation.baseline_eval import ARMS, RUN_DIRS, SEEDS, probs_from
from research.evaluation.evaluate import load_predictions
from research.evaluation.isic_eval import read_ground_truth
from research.evaluation.metrics import cluster_bootstrap
from research.utils.config import load_config, repo_path
from research.utils.io import read_json, write_json


def auroc(score, positive):
    ranks = pd.Series(score).rank().to_numpy()
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def sensitivity_at_rate(score, positive, rate):
    return float((score[positive] >= np.quantile(score, 1 - rate)).mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--isic-ground-truth", required=True)
    args = ap.parse_args(argv)

    out = repo_path("research/outputs/baseline_comparison")
    summary = read_json(out / "summary.json")
    e0 = load_config(repo_path(ARMS["fusion"]))
    classes = e0["data"]["classes"]
    mal_idx = [classes.index(c) for c in e0["evaluation"]["malignant_classes"]]
    split_df, _ = load_split(e0, verify_hash=True)
    frame = read_ground_truth(args.isic_ground_truth, classes)
    y_isic = np.array([classes.index(d) for d in frame["dx"]])
    meta = pd.read_csv(args.metadata).set_index("image_id")

    score = {"ham": {}, "isic": {}}
    y_ham = lesions = biopsy = None
    for arm in ARMS:
        ham_probs, isic_probs = [], []
        for seed in SEEDS:
            T = summary["results"][arm]["seeds"][str(seed)]["temperature"]
            test = load_predictions(repo_path(RUN_DIRS[arm]) / f"seed_{seed}", "test", seed, classes, split_df)
            if y_ham is None:
                y_ham, lesions = test["y"], np.asarray(test["lesion_id"])
                biopsy = meta.loc[np.asarray(test["image_id"]), "dx_type"].to_numpy() == "histo"
            path = repo_path("research/outputs/isic2018_test") / f"logits_seed_{seed}.npz" if arm == "fusion" else out / "cache" / f"isic_{arm}_{seed}.npz"
            ham_probs.append(probs_from(test["logits"], T))
            isic_probs.append(probs_from(np.load(path)["logits"], T))
        score["ham"][arm] = np.mean(ham_probs, axis=0)[:, mal_idx].sum(axis=1)
        score["isic"][arm] = np.mean(isic_probs, axis=0)[:, mal_idx].sum(axis=1)

    sets = {"ham10000_test": ("ham", y_ham, lesions), "isic2018_test": ("isic", y_isic, frame["image_id"].to_numpy())}
    results = {}
    for name, (key, y, clusters) in sets.items():
        mal = np.isin(y, mal_idx)
        rate = summary["results"]["fusion"]["ensemble_2_seeds"][name]["all"]["referral"]["sensitivity_0.95"]["referral_rate"]
        block = {"referral_rate_of_fusion": rate, "arms": {}}
        for arm in ARMS:
            s = score[key][arm]
            block["arms"][arm] = {"auroc": auroc(s, mal), "sensitivity_at_same_rate": sensitivity_at_rate(s, mal, rate)}
            if key == "ham":
                block["arms"][arm]["auroc_biopsy"] = auroc(s[biopsy], mal[biopsy])
        for arm in ("effnet", "resnet"):
            def diff(idx, a=arm):
                f, o, m = score[key]["fusion"][idx], score[key][a][idx], mal[idx]
                d = {"auroc": auroc(f, m) - auroc(o, m), "sensitivity_at_same_rate": sensitivity_at_rate(f, m, rate) - sensitivity_at_rate(o, m, rate)}
                if key == "ham":
                    b = biopsy[idx]
                    d["auroc_biopsy"] = auroc(f[b], m[b]) - auroc(o[b], m[b])
                return d
            block[f"fusion_minus_{arm}"] = cluster_bootstrap(diff, clusters, 2000, 0, 0.95)["metrics"]
        results[name] = block
    write_json(out / "posthoc.json", {"label": "Post hoc, chosen after the comparison results were seen", "seeds": SEEDS, "results": results})
    for name, block in results.items():
        print(name, "fusion referral rate", round(block["referral_rate_of_fusion"], 4))
        for arm, values in block["arms"].items():
            print(" ", arm, {k: round(v, 4) for k, v in values.items()})
        for arm in ("effnet", "resnet"):
            for k, b in block[f"fusion_minus_{arm}"].items():
                print(f"  fusion-{arm} {k}: {b['point']:+.4f} [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
