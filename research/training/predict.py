"""
Export RAW logits (plus derived full-precision probabilities) for the validation
and test splits from a trained E0 run.

train.py calls this automatically after training. Standalone re-export:

    python -m research.training.predict --run-dir research/outputs/E0/seed_42 --data-root <root>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research.data.data import HAM10000Dataset, build_transforms, load_validated_split, make_loader
from research.evaluation.metrics import softmax64
from research.models.fusion_model import assert_eval_transform_matches_backend, build_model
from research.utils.config import load_config, resolve_data_root
from research.utils.hashing import config_hash


@torch.no_grad()
def collect_logits(model, loader, device, max_batches: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (logits float32 as produced by the model, dataset row indices)."""
    model.eval()
    logits, rows = [], []
    for b, (x, _, idx) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        logits.append(model(x.to(device)).float().cpu().numpy())
        rows.append(idx.numpy())
    return np.concatenate(logits), np.concatenate(rows)


def export_predictions(model, frame: pd.DataFrame, image_index, cfg, split_name, seed, device, out_dir: Path,
                       num_workers: int, max_batches: int | None = None) -> Path:
    classes = cfg["data"]["classes"]
    _, eval_tf = build_transforms(cfg)
    assert_eval_transform_matches_backend(eval_tf)
    dataset = HAM10000Dataset(frame, image_index, classes, eval_tf)
    loader = make_loader(dataset, cfg["training"]["batch_size"], shuffle=False, drop_last=False,
                         num_workers=num_workers, seed=seed)
    logits, rows = collect_logits(model, loader, device, max_batches)

    sub = dataset.frame.iloc[rows].reset_index(drop=True)
    labels = np.array([classes.index(d) for d in sub["dx"]], dtype=np.int64)
    probs = softmax64(logits)  # float64, from the raw float32 logits
    preds = probs.argmax(axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{split_name}.npz"
    np.savez(
        npz_path,
        logits=logits,
        probs=probs,
        labels=labels,
        preds=preds,
        max_prob=probs.max(axis=1),
        image_id=sub["image_id"].to_numpy(dtype=str),
        lesion_id=sub["lesion_id"].to_numpy(dtype=str),
        class_names=np.array(classes),
        seed=np.int64(seed),
        split=np.array(split_name),
    )

    table = pd.DataFrame({
        "image_id": sub["image_id"],
        "lesion_id": sub["lesion_id"],
        "seed": seed,
        "true_label": sub["dx"],
        "pred_label": [classes[i] for i in preds],
        "max_prob": probs.max(axis=1),
    })
    for i, c in enumerate(classes):
        table[f"prob_{c}"] = probs[:, i]
    for i, c in enumerate(classes):
        table[f"logit_{c}"] = logits[:, i]
    # %.17g round-trips float64 exactly; the .npz is the canonical copy.
    table.to_csv(out_dir / f"{split_name}.csv", index=False, float_format="%.17g", lineterminator="\n")
    return npz_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    if ckpt["config_hash"] != config_hash(cfg):
        raise RuntimeError("Checkpoint config_hash does not match the run's config.yaml")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    split_df, image_index, _, _ = load_validated_split(resolve_data_root(args.data_root), cfg)
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    for name in ("val", "test"):
        path = export_predictions(model, split_df[split_df["split"] == name], image_index, cfg, name,
                                  ckpt["seed"], device, run_dir / "predictions", args.num_workers)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
