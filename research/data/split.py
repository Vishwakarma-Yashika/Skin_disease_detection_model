"""
One-time generation of the frozen lesion-level HAM10000 split.

    python -m research.data.split --config research/configs/e0.yaml --data-root <HAM10000 root>
    python -m research.data.split --config research/configs/e0.yaml --data-root <root> --check

Generation refuses to overwrite an existing split file (use --force only if you
deliberately want a new split, which then needs a new sha256 in the config).
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from research.data.data import (
    SPLIT_COLUMNS,
    LeakageError,
    build_image_index,
    format_split_summary,
    load_metadata,
    load_split,
    validate_split,
)
from research.utils.config import load_config, repo_path, resolve_data_root
from research.utils.hashing import sha256_file


def make_split(metadata: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s = cfg["split"]
    if s["method"] != "lesion_stratified_two_stage":
        raise ValueError(f"Unsupported split method: {s['method']}")
    if not np.isclose(s["train_fraction"] + s["val_fraction"] + s["test_fraction"], 1.0):
        raise ValueError("split fractions must sum to 1")
    group, strat, rs = s["group_column"], s["stratify_column"], s["random_state"]

    # One row per lesion, sorted, so the result does not depend on metadata row order.
    lesions = metadata[[group, strat]].drop_duplicates(group).sort_values(group).reset_index(drop=True)
    holdout_fraction = s["val_fraction"] + s["test_fraction"]
    train_l, holdout_l = train_test_split(
        lesions, test_size=holdout_fraction, stratify=lesions[strat], random_state=rs
    )
    val_l, test_l = train_test_split(
        holdout_l, test_size=s["test_fraction"] / holdout_fraction, stratify=holdout_l[strat], random_state=rs
    )
    assignment = {
        **dict.fromkeys(train_l[group], "train"),
        **dict.fromkeys(val_l[group], "val"),
        **dict.fromkeys(test_l[group], "test"),
    }

    out = metadata[["image_id", "lesion_id", "dx"]].sort_values("image_id").reset_index(drop=True)
    out["split"] = out[group].map(assignment)
    assert out["split"].notna().all()
    return out[list(SPLIT_COLUMNS)]


def verify_images(split_df: pd.DataFrame, image_index) -> None:
    bad = []
    for image_id in split_df["image_id"]:
        try:
            with Image.open(image_index[image_id]) as img:
                img.verify()
        except Exception as exc:  # noqa: BLE001 -- collect every unreadable file
            bad.append((image_id, repr(exc)))
    if bad:
        raise LeakageError(f"{len(bad)} unreadable image(s), e.g. {bad[:5]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--check", action="store_true", help="validate the existing split only")
    parser.add_argument("--verify-images", action="store_true", help="also decode-check every image file")
    parser.add_argument("--force", action="store_true", help="overwrite an existing split file")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_root = resolve_data_root(args.data_root)
    classes = cfg["data"]["classes"]
    metadata = load_metadata(data_root, cfg)
    image_index = build_image_index(data_root, cfg)
    split_path = repo_path(cfg["split"]["file"])

    if args.check:
        split_df, digest = load_split(cfg, verify_hash=True)
    else:
        if split_path.exists() and not args.force:
            print(f"Refusing to regenerate: {split_path} already exists. Use --check to validate it.", file=sys.stderr)
            return 1
        split_df = make_split(metadata, cfg)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(split_path, index=False, lineterminator="\n")
        digest = sha256_file(split_path)

    summary = validate_split(split_df, metadata, image_index, classes)
    if args.verify_images:
        verify_images(split_df, image_index)

    summary.update({"split_file": str(split_path), "sha256": digest, "split_config": cfg["split"]})
    summary_path = split_path.with_name(split_path.stem + "_summary.json")
    if not args.check:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(format_split_summary(summary, classes))
    print("All split integrity checks passed (lesion overlap, image overlap, contamination, classes, files).")
    if args.verify_images:
        print(f"All {len(split_df)} image files decoded successfully.")
    print(f"sha256: {digest}")
    if not args.check and cfg["split"].get("sha256") != digest:
        print("Set split.sha256 in the config to the value above to pin this split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
