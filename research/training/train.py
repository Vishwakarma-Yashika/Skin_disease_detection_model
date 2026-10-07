"""
Train one seed of Experiment E0 (fixed EfficientNet-B0 + ResNet-34 fusion).

    python -m research.training.train --config research/configs/e0.yaml --seed 42 --data-root <HAM10000 root>

Writes research/outputs/E0/seed_<seed>/:
    config.yaml            exact copy of the config used
    run_metadata.json      seed, environment, hashes, split summary, runtime settings
    history.csv / .json    per-epoch training + validation metrics
    best.pt                checkpoint selected by the predeclared validation criterion
    val_metrics_best.json  validation metrics of the selected epoch
    predictions/{val,test}.{npz,csv}  raw logits + probabilities of the selected model

The test split is not loaded until training and checkpoint selection are finished.
"""
import argparse
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from research.data.data import HAM10000Dataset, build_transforms, load_validated_split, make_loader
from research.evaluation.calibration_metrics import nll
from research.evaluation.metrics import classification_metrics, softmax64
from research.models.fusion_model import assert_eval_transform_matches_backend, build_model, verify_architecture
from research.training.predict import collect_logits, export_predictions
from research.utils.config import REPO_ROOT, load_config, repo_path, resolve_data_root
from research.utils.environment import collect_environment
from research.utils.hashing import config_hash, sha256_file
from research.utils.io import write_json
from research.utils.seed import set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data-root", default=None, help="HAM10000 root (or set HAM10000_ROOT)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true", help="allow writing into an existing run directory")
    p.add_argument("--output-dir", default=None,
                   help="override experiment.output_dir (runtime setting; recorded, not part of the config hash)")
    p.add_argument("--smoke-batches", type=int, default=None,
                   help="PIPELINE TEST ONLY: 1 epoch, N batches per loop, written to outputs/smoke/. Not an E0 result.")
    return p.parse_args(argv)


def check_config(cfg: dict) -> None:
    """Reject settings E0 does not implement, so the config cannot silently over-promise."""
    t = cfg["training"]
    expected = {
        ("training", "loss"): (t["loss"], "cross_entropy"),
        ("training", "class_weights"): (t["class_weights"], None),
        ("training", "sampler"): (t["sampler"], "uniform_shuffle"),
        ("training", "optimizer.name"): (t["optimizer"]["name"], "adamw"),
        ("training", "scheduler.name"): (t["scheduler"]["name"], "cosine"),
        ("training", "amp"): (t["amp"], False),
        ("checkpoint", "selection_metric"): (cfg["checkpoint"]["selection_metric"], "val_nll"),
        ("checkpoint", "mode"): (cfg["checkpoint"]["mode"], "min"),
        ("checkpoint", "tie_break"): (cfg["checkpoint"]["tie_break"], "earliest_epoch"),
    }
    bad = {f"{a}.{b}": v for (a, b), v in expected.items() if v[0] != v[1]}
    if bad:
        raise ValueError(f"Config values not supported by E0 (actual, supported): {bad}")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches, log_every=50):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    start = time.time()
    n_batches = len(loader) if max_batches is None else min(max_batches, len(loader))
    for b, (x, y, _) in enumerate(loader):
        if b >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at batch {b}: {loss.item()}")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        seen += len(y)
        if (b + 1) % log_every == 0 or b + 1 == n_batches:
            print(f"    batch {b + 1}/{n_batches}  loss {total_loss / seen:.4f}  {time.time() - start:.0f}s", flush=True)
    return total_loss / seen, correct / seen


def evaluate_validation(model, loader, device, classes, max_batches):
    logits, rows = collect_logits(model, loader, device, max_batches)
    labels = np.asarray(loader.dataset.labels)[rows]
    probs = softmax64(logits)
    cls = classification_metrics(labels, probs, classes)
    return {
        "val_nll": nll(logits, labels),
        "val_accuracy": cls["accuracy"],
        "val_balanced_accuracy": cls["balanced_accuracy"],
        "val_macro_f1": cls["macro_f1"],
        "val_macro_roc_auc_ovr": cls["macro_roc_auc_ovr"],
    }, cls


def save_checkpoint(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    check_config(cfg)
    smoke = args.smoke_batches is not None
    seed = args.seed
    if not smoke and seed not in cfg["experiment"]["seeds"]:
        raise SystemExit(f"Seed {seed} is not one of the predeclared E0 seeds {cfg['experiment']['seeds']}.")

    cfg_hash = config_hash(cfg)
    classes = cfg["data"]["classes"]
    t = cfg["training"]
    epochs = 1 if smoke else t["epochs"]

    base_dir = repo_path(args.output_dir or cfg["experiment"]["output_dir"])
    run_dir = (base_dir.parent / "smoke" if smoke else base_dir) / f"seed_{seed}"
    if (run_dir / "best.pt").exists() and not args.overwrite:
        raise SystemExit(f"{run_dir} already contains a trained run. Pass --overwrite to replace it.")
    run_dir.mkdir(parents=True, exist_ok=True)

    determinism = set_seed(seed, deterministic=t["deterministic"])
    device = resolve_device(args.device)
    data_root = resolve_data_root(args.data_root)

    # --- data: pinned split + every leakage check, before anything is trained ---
    split_df, image_index, split_summary, split_hash = load_validated_split(data_root, cfg)
    train_tf, eval_tf = build_transforms(cfg)
    assert_eval_transform_matches_backend(eval_tf)
    train_ds = HAM10000Dataset(split_df[split_df["split"] == "train"], image_index, classes, train_tf)
    val_ds = HAM10000Dataset(split_df[split_df["split"] == "val"], image_index, classes, eval_tf)
    train_loader = make_loader(train_ds, t["batch_size"], shuffle=True, drop_last=t["drop_last"],
                               num_workers=args.num_workers, seed=seed)
    val_loader = make_loader(val_ds, t["batch_size"], shuffle=False, drop_last=False,
                             num_workers=args.num_workers, seed=seed)

    # --- model / optimisation, exactly as declared ---
    model = build_model(cfg, pretrained=True).to(device)
    arch_report = verify_architecture(model, cfg)
    criterion = nn.CrossEntropyLoss(label_smoothing=t["label_smoothing"])
    opt = t["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt["lr"], weight_decay=opt["weight_decay"],
                                  betas=tuple(opt["betas"]))
    sch = t["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sch["t_max_epochs"], eta_min=sch["eta_min"])

    shutil.copyfile(args.config, run_dir / "config.yaml")
    metadata = {
        "experiment": cfg["experiment"]["name"],
        "status": "running",
        "smoke_test": smoke,
        "seed": seed,
        "config_hash": cfg_hash,
        "config_file": str(Path(args.config)),
        "split_file": cfg["split"]["file"],
        "split_sha256": split_hash,
        "split_summary": split_summary,
        "determinism": determinism,
        "runtime": {"device": str(device), "num_workers": args.num_workers, "data_root": str(data_root),
                    "smoke_batches": args.smoke_batches, "output_dir": str(run_dir)},
        "environment": collect_environment(REPO_ROOT),
        "architecture_verification": arch_report,
        "model": cfg["model"],
        "preprocessing": {"train": repr(train_tf), "eval": repr(eval_tf)},
        "augmentation": cfg["augmentation"],
        "training": t,
        "checkpoint_rule": cfg["checkpoint"],
        "test_split_used_for_training_or_selection": False,
    }
    write_json(run_dir / "run_metadata.json", metadata)
    print(f"E0 seed {seed} | device {device} | config {cfg_hash[:12]} | split {split_hash[:12]} | out {run_dir}")
    print(f"train {len(train_ds)} images ({len(train_loader)} batches/epoch), val {len(val_ds)} images")

    history, best = [], None
    training_started = datetime.now(timezone.utc).isoformat()
    for epoch in range(1, epochs + 1):
        start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch}/{epochs}  lr {lr:.2e}", flush=True)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, args.smoke_batches)
        val, val_cls = evaluate_validation(model, val_loader, device, classes, args.smoke_batches)
        scheduler.step()

        # Predeclared rule: minimum val NLL; strict '<' keeps the earliest epoch on ties.
        improved = best is None or val["val_nll"] < best["val_nll"]
        row = {"epoch": epoch, "lr": lr, "train_loss": train_loss, "train_accuracy": train_acc, **val,
               "epoch_seconds": time.time() - start, "selected_so_far": improved}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, float_format="%.10g")
        write_json(run_dir / "history.json", history)
        print(f"  train_loss {train_loss:.4f} acc {train_acc:.4f} | val_nll {val['val_nll']:.4f} "
              f"acc {val['val_accuracy']:.4f} bal_acc {val['val_balanced_accuracy']:.4f} "
              f"{'*' if improved else ''} ({row['epoch_seconds']:.0f}s)", flush=True)

        if improved:
            best = {"epoch": epoch, **val}
            save_checkpoint(run_dir / "best.pt", {
                "model_state_dict": model.state_dict(),
                # Same keys the backend's ModelRegistry reads, so E0 checkpoints are servable.
                "diagnosis_classes": list(classes),
                "val_acc": val["val_accuracy"],
                "epoch": epoch,
                "experiment": cfg["experiment"]["name"],
                "smoke_test": smoke,
                "architecture": dict(cfg["model"]),
                "class_names": list(classes),
                "seed": seed,
                "selection": {"metric": cfg["checkpoint"]["selection_metric"], "mode": cfg["checkpoint"]["mode"],
                              "tie_break": cfg["checkpoint"]["tie_break"], "value": val["val_nll"]},
                "config_hash": cfg_hash,
                "split_sha256": split_hash,
                "git_commit": metadata["environment"]["git_commit"],
            })
            write_json(run_dir / "val_metrics_best.json", {"epoch": epoch, **val, "classification": val_cls})

    # --- selection is final; only now are predictions exported, test included ---
    # Audit trail: the test Dataset/DataLoader is first constructed after this timestamp.
    selection_finalized = datetime.now(timezone.utc).isoformat()
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    for name in ("val", "test"):
        path = export_predictions(model, split_df[split_df["split"] == name], image_index, cfg, name, seed, device,
                                  run_dir / "predictions", args.num_workers, args.smoke_batches)
        print(f"wrote {path}")

    metadata.update({
        "status": "complete",
        "test_access_audit": {
            "training_started_utc": training_started,
            "selection_finalized_utc": selection_finalized,
            "test_export_finished_utc": datetime.now(timezone.utc).isoformat(),
            "datasets_built_during_training": ["train", "val"],
            "test_used_for_training_selection_or_thresholds": False,
        },
        "selected_epoch": best["epoch"],
        "selected_val_nll": best["val_nll"],
        "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        "total_epochs_run": epochs,
    })
    write_json(run_dir / "run_metadata.json", metadata)
    print(f"done. selected epoch {best['epoch']} (val_nll {best['val_nll']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
