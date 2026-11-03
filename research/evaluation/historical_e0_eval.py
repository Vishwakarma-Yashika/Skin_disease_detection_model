"""
Exploratory CROSS-PROTOCOL evaluation of the historical checkpoint on the E0 test split.

    python -m research.evaluation.historical_e0_eval --data-root <HAM10000 root>

NOT a clean held-out test: the historical model's train/validation split cannot be
reconstructed and some (probably most) E0 test images may have been seen during its
training. Results are written to research/outputs/historical_checkpoint_e0_eval/.
No calibration, thresholds, selective prediction or uncertainty methods.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research.evaluation.e0_run_report import cm_figure, full_metrics, roc_figure
from research.evaluation.metrics import softmax64
from research.evaluation.step1_reliability import (
    Step1Error,
    backend_parity,
    e0_manifest,
    load_checkpoint,
    run_inference,
    validate_manifest,
)
from research.models.fusion_model import backend_inference_transform
from research.utils.config import REPO_ROOT, load_config, repo_path, resolve_data_root
from research.utils.environment import collect_environment
from research.utils.hashing import sha256_file
from research.utils.io import write_json
from research.utils.seed import set_seed

CHECKPOINT = "backend/models/best_fusion_model.pt"
EXPECTED_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
EXPECTED_TEST_IMAGES = 1502


def _check(cond, msg):
    if not cond:
        raise Step1Error(msg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-dir", default="research/outputs/historical_checkpoint_e0_eval")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for inference (limits contention)")
    args = ap.parse_args(argv)

    set_seed(0, deterministic=True)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    e0_cfg = load_config(repo_path("research/configs/e0.yaml"))
    out = repo_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1-3: checkpoint loads, class order, architecture (verify_architecture inside load_checkpoint)
    ckpt_path = repo_path(CHECKPOINT)
    sha_before = sha256_file(ckpt_path)
    model, classes, ckpt_info = load_checkpoint(ckpt_path, EXPECTED_CLASSES, e0_cfg, device)

    # 5-6: E0 test split, with every split/leakage check re-applied by load_validated_split
    data_root = resolve_data_root(args.data_root)
    frame, split_info = e0_manifest("test", data_root, e0_cfg)
    frame = validate_manifest(frame, classes)
    _check(len(frame) == EXPECTED_TEST_IMAGES, f"E0 test has {len(frame)} images, expected {EXPECTED_TEST_IMAGES}")
    _check(not frame["image_id"].duplicated().any(), "duplicate image_ids in E0 test")

    # inference: model.eval() + no_grad, backend inference_transform, no TTA, no dropout sampling
    logits = run_inference(model, frame, 32, 0, device)
    _check(logits.shape == (len(frame), len(classes)) and np.isfinite(logits).all(), "bad logits")
    # 4: batched path == backend /api/predict single-image path
    parity = backend_parity(model, frame, logits, 16, 1e-4, device)
    _check(sha256_file(ckpt_path) == sha_before, "checkpoint changed during evaluation")

    y = np.array([classes.index(l) for l in frame["label"]])
    probs = softmax64(logits)
    p = {"logits": logits, "probs": probs, "y": y}
    m = full_metrics(p, classes, e0_cfg)

    pred = probs.argmax(axis=1)
    mal_idx = [classes.index(c) for c in e0_cfg["evaluation"]["malignant_classes"]]
    t, pr = np.isin(y, mal_idx), np.isin(pred, mal_idx)
    tp, fp, fn = int((t & pr).sum()), int((~t & pr).sum()), int((t & ~pr).sum())
    mb = m["malignant_vs_benign"]
    mb["precision"] = tp / (tp + fp) if tp + fp else float("nan")
    mb["f1"] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    m["counts"] = {"n_images": int(len(y)), "n_correct": int((pred == y).sum()), "n_incorrect": int((pred != y).sum())}
    m["label"] = "Historical checkpoint on the E0 test split (cross-protocol, exploratory; NOT a held-out test)"

    # cross-check against the STEP 1 run of the same checkpoint on the same split, if present
    step1 = repo_path("research/outputs/reliability_step1/predictions/predictions.npz")
    if step1.is_file():
        s1 = np.load(step1)
        same_ids = list(s1["image_id"]) == frame["image_id"].tolist()
        m["step1_crosscheck"] = {"same_image_order": same_ids,
                                 "max_abs_logit_difference": float(np.abs(s1["logits"] - logits).max()) if same_ids else None}

    # --- outputs ---
    write_json(out / "metrics.json", m)
    pd.DataFrame([{"class": c, **m["per_class"][c]} for c in classes]).to_csv(out / "per_class_metrics.csv", index=False)
    cm = np.array(m["confusion_matrix"])
    pd.DataFrame(cm, index=pd.Index(classes, name="true\\predicted"), columns=classes).to_csv(out / "confusion_matrix.csv")
    pd.DataFrame(cm / cm.sum(axis=1, keepdims=True), index=pd.Index(classes, name="true\\predicted"),
                 columns=classes).to_csv(out / "normalized_confusion_matrix.csv", float_format="%.6f")
    tbl = pd.DataFrame({"image_id": frame["image_id"], "lesion_id": frame["lesion_id"], "true_label": frame["label"],
                        "predicted_label": [classes[i] for i in pred], "correct": (pred == y).astype(int),
                        "max_confidence": probs.max(axis=1)})
    for i, c in enumerate(classes):
        tbl[f"prob_{c}"] = probs[:, i]
    for i, c in enumerate(classes):
        tbl[f"logit_{c}"] = logits[:, i]
    tbl.to_csv(out / "test_predictions.csv", index=False, float_format="%.17g", lineterminator="\n")
    np.savez(out / "test_predictions.npz", logits=logits, probs=probs, labels=y, image_id=frame["image_id"].to_numpy(str),
             lesion_id=frame["lesion_id"].to_numpy(str), class_names=np.array(classes))

    title = "Historical ckpt on E0 test (cross-protocol)"
    cm_figure(m["confusion_matrix"], classes, out, False, f"{title}, counts")
    cm_figure(m["confusion_matrix"], classes, out, True, f"{title}, row-normalised")
    for ext in ("png", "pdf"):
        shutil.move(out / f"confusion_matrix_normalized.{ext}", out / f"normalized_confusion_matrix.{ext}")
    roc_figure(p, classes, e0_cfg["evaluation"]["malignant_classes"], m, out, "Historical ckpt, E0 test")

    env = collect_environment(REPO_ROOT)
    counts = frame["label"].value_counts()
    write_json(out / "provenance.json", {
        "model": {"path": CHECKPOINT, "type": "historical_checkpoint", "sha256": sha_before,
                  "stored_val_acc": ckpt_info.get("val_acc"), "stored_epoch": ckpt_info.get("epoch"),
                  "class_order": classes, "loaded_with": "backend FusionClassifier (app.model), strict state_dict"},
        "evaluation": {
            "protocol": "cross_protocol_exploratory", "split": "E0 test split", "clean_held_out_test": False,
            "reason": "original historical train/validation split cannot be reconstructed",
            "note": "Some (likely most) E0 test images may have been seen during historical training; "
                    "NOT an independent generalization estimate.",
            "split_file": split_info["e0_split_file"], "split_sha256": split_info["e0_split_sha256"],
            "n_images": int(len(frame)), "n_lesions": int(frame["lesion_id"].nunique()),
            "class_distribution": {c: int(counts.get(c, 0)) for c in classes},
            "leakage_checks": "research.data.data.validate_split (lesion/image overlap, contamination, classes, files) passed",
        },
        "preprocessing": f"PIL convert('RGB') -> {backend_inference_transform!r}".replace("\n", " "),
        "inference": {"mode": "model.eval() + torch.no_grad()", "tta": False, "dropout_sampling": False,
                      "batch_size": 32, "device": "cpu", "torch_threads": args.threads, "backend_parity": parity},
        "git_commit": env["git_commit"], "git_dirty": env["git_dirty"], "environment": env,
    })
    print(f"accuracy on E0 test (historical checkpoint, cross-protocol): {m['classification']['accuracy']:.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
