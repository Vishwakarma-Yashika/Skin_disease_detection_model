"""
HAM10000 loading, split integrity checks, transforms and Dataset for E0.

Every integrity failure raises LeakageError. Research code never catches it:
a run with a contaminated split must not produce numbers.
"""
import os
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from research.utils.config import repo_path
from research.utils.hashing import sha256_file
from research.utils.seed import make_generator, seed_worker

SPLITS = ("train", "val", "test")
SPLIT_COLUMNS = ("image_id", "lesion_id", "dx", "split")
KNOWN_TRAIN_AUGMENTATIONS = {"horizontal_flip_p", "vertical_flip_p"}


class LeakageError(RuntimeError):
    """A data/split integrity check failed."""


def _fail(message: str):
    raise LeakageError(message)


# ---------------------------------------------------------------------------
# Metadata + files
# ---------------------------------------------------------------------------

def load_metadata(data_root: Path, cfg: dict) -> pd.DataFrame:
    path = Path(data_root) / cfg["data"]["metadata_file"]
    if not path.is_file():
        _fail(f"HAM10000 metadata not found: {path}")
    df = pd.read_csv(path, dtype={"image_id": str, "lesion_id": str, "dx": str})

    missing = {"image_id", "lesion_id", "dx"} - set(df.columns)
    if missing:
        _fail(f"Metadata is missing required columns: {sorted(missing)}")
    nulls = df[["image_id", "lesion_id", "dx"]].isna().sum()
    if nulls.any():
        _fail(f"Metadata has null image_id/lesion_id/dx values: {nulls.to_dict()}")
    dup = df["image_id"][df["image_id"].duplicated()]
    if len(dup):
        _fail(f"Metadata has {len(dup)} duplicate image_id(s), e.g. {dup.iloc[:5].tolist()}")
    mixed = df.groupby("lesion_id")["dx"].nunique()
    if (mixed > 1).any():
        _fail(f"Lesions with more than one dx label: {mixed[mixed > 1].index[:5].tolist()}")

    classes = cfg["data"]["classes"]
    unknown = set(df["dx"]) - set(classes)
    if unknown:
        _fail(f"Metadata contains classes not declared in config: {sorted(unknown)}")
    absent = set(classes) - set(df["dx"])
    if absent:
        _fail(f"Declared classes missing from metadata: {sorted(absent)}")
    return df


def build_image_index(data_root: Path, cfg: dict) -> dict[str, Path]:
    """image_id -> file path. Fails on an image_id found in more than one place."""
    ext = cfg["data"]["image_extension"].lower()
    index: dict[str, Path] = {}
    for sub in cfg["data"]["image_dirs"]:
        folder = Path(data_root) / sub
        if not folder.is_dir():
            _fail(f"Image folder not found: {folder}")
        for p in sorted(folder.iterdir()):
            if not p.is_file() or p.suffix.lower() != ext:
                continue
            if p.stem in index:
                _fail(f"Duplicate file reference for {p.stem}: {index[p.stem]} and {p}")
            index[p.stem] = p
    return index


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def load_split(cfg: dict, verify_hash: bool = True) -> tuple[pd.DataFrame, str]:
    path = repo_path(cfg["split"]["file"])
    if not path.is_file():
        _fail(
            f"Split file not found: {path}. Generate it ONCE with "
            "`python -m research.data.split --config research/configs/e0.yaml --data-root <root>`."
        )
    actual = sha256_file(path)
    if verify_hash:
        expected = cfg["split"].get("sha256")
        if not expected or expected.startswith("__"):
            _fail("config split.sha256 is not set; refusing to use an unpinned split.")
        if actual != expected:
            _fail(
                f"Split file hash mismatch: config expects {expected}, file is {actual}. "
                "The split changed after it was frozen."
            )
    return pd.read_csv(path, dtype=str, keep_default_na=False), actual


def validate_split(
    split_df: pd.DataFrame,
    metadata: pd.DataFrame,
    image_index: dict[str, Path],
    classes: list[str],
) -> dict:
    """All leakage / contamination assertions. Returns the split summary on success."""
    missing_cols = set(SPLIT_COLUMNS) - set(split_df.columns)
    if missing_cols:
        _fail(f"Split CSV is missing columns: {sorted(missing_cols)}")

    bad_split = set(split_df["split"]) - set(SPLITS)
    if bad_split:
        _fail(f"Unknown split labels: {sorted(bad_split)}")

    # Duplicate image IDs / image overlap: an image may appear only once in the whole file,
    # which also rules out an image belonging to two splits.
    dup = split_df["image_id"][split_df["image_id"].duplicated(keep=False)]
    if len(dup):
        multi = split_df[split_df["image_id"].isin(dup)].groupby("image_id")["split"].unique()
        _fail(f"{dup.nunique()} image_id(s) appear more than once (image overlap), e.g. {multi.head().to_dict()}")

    # Every metadata image exactly once.
    meta_ids, split_ids = set(metadata["image_id"]), set(split_df["image_id"])
    if meta_ids - split_ids:
        _fail(f"{len(meta_ids - split_ids)} metadata image(s) missing from split, e.g. {sorted(meta_ids - split_ids)[:5]}")
    if split_ids - meta_ids:
        _fail(f"{len(split_ids - meta_ids)} split image(s) not in metadata, e.g. {sorted(split_ids - meta_ids)[:5]}")

    # Split contamination: lesion_id/dx in the split must equal the source metadata.
    merged = split_df.merge(metadata[["image_id", "lesion_id", "dx"]], on="image_id", suffixes=("", "_meta"))
    bad_lesion = merged[merged["lesion_id"] != merged["lesion_id_meta"]]
    if len(bad_lesion):
        _fail(f"{len(bad_lesion)} split row(s) disagree with metadata lesion_id, e.g. {bad_lesion['image_id'].head().tolist()}")
    bad_dx = merged[merged["dx"] != merged["dx_meta"]]
    if len(bad_dx):
        _fail(f"{len(bad_dx)} split row(s) disagree with metadata dx, e.g. {bad_dx['image_id'].head().tolist()}")

    # Lesion overlap.
    per_lesion = split_df.groupby("lesion_id")["split"].nunique()
    leaking = per_lesion[per_lesion > 1]
    if len(leaking):
        _fail(f"{len(leaking)} lesion_id(s) occur in more than one split, e.g. {leaking.index[:5].tolist()}")

    # Missing classes per split.
    for name in SPLITS:
        present = set(split_df.loc[split_df["split"] == name, "dx"])
        absent = set(classes) - present
        if absent:
            _fail(f"Split '{name}' is missing class(es): {sorted(absent)}")

    # File references: every image resolves to exactly one, unique, non-empty file.
    no_file = [i for i in split_df["image_id"] if i not in image_index]
    if no_file:
        _fail(f"{len(no_file)} split image(s) have no file, e.g. {no_file[:5]}")
    resolved = [os.path.normcase(str(image_index[i].resolve())) for i in split_df["image_id"]]
    if len(set(resolved)) != len(resolved):
        _fail("Two image_ids resolve to the same file (duplicate file reference).")
    empty = [i for i in split_df["image_id"] if image_index[i].stat().st_size == 0]
    if empty:
        _fail(f"{len(empty)} empty image file(s), e.g. {empty[:5]}")

    return split_summary(split_df, classes)


def split_summary(split_df: pd.DataFrame, classes: list[str]) -> dict:
    n_images, n_lesions = len(split_df), split_df["lesion_id"].nunique()
    summary = {"total_images": n_images, "total_lesions": n_lesions, "splits": {}}
    for name in SPLITS:
        part = split_df[split_df["split"] == name]
        counts = part["dx"].value_counts()
        lesion_counts = part.drop_duplicates("lesion_id")["dx"].value_counts()
        summary["splits"][name] = {
            "n_images": len(part),
            "n_lesions": part["lesion_id"].nunique(),
            "fraction_of_images": len(part) / n_images,
            "fraction_of_lesions": part["lesion_id"].nunique() / n_lesions,
            "class_image_counts": {c: int(counts.get(c, 0)) for c in classes},
            "class_image_fractions": {c: float(counts.get(c, 0)) / len(part) for c in classes},
            "class_lesion_counts": {c: int(lesion_counts.get(c, 0)) for c in classes},
        }
    return summary


def format_split_summary(summary: dict, classes: list[str]) -> str:
    lines = [f"total: {summary['total_images']} images, {summary['total_lesions']} lesions"]
    header = f"{'split':<6}{'images':>8}{'%img':>7}{'lesions':>9}{'%les':>7}  " + "".join(f"{c:>13}" for c in classes)
    lines.append(header)
    for name in SPLITS:
        s = summary["splits"][name]
        cells = "".join(
            f"{s['class_image_counts'][c]:>6} ({100 * s['class_image_fractions'][c]:4.1f}%)" for c in classes
        )
        lines.append(
            f"{name:<6}{s['n_images']:>8}{100 * s['fraction_of_images']:>6.1f}%{s['n_lesions']:>9}"
            f"{100 * s['fraction_of_lesions']:>6.1f}%  {cells}"
        )
    return "\n".join(lines)


def load_validated_split(data_root: Path, cfg: dict, verify_hash: bool = True):
    """Convenience: metadata + file index + pinned split, all checks applied."""
    metadata = load_metadata(data_root, cfg)
    image_index = build_image_index(data_root, cfg)
    split_df, split_hash = load_split(cfg, verify_hash=verify_hash)
    summary = validate_split(split_df, metadata, image_index, cfg["data"]["classes"])
    return split_df, image_index, summary, split_hash


# ---------------------------------------------------------------------------
# Transforms + Dataset
# ---------------------------------------------------------------------------

def build_transforms(cfg: dict):
    pre = cfg["preprocessing"]
    size = tuple(pre["resize"])
    normalize = transforms.Normalize(pre["mean"], pre["std"])

    aug = cfg["augmentation"]["train"] or {}
    unknown = set(aug) - KNOWN_TRAIN_AUGMENTATIONS
    if unknown:
        # The config must describe everything applied; reject anything not implemented.
        raise ValueError(f"Unsupported augmentation keys in config: {sorted(unknown)}")

    train_ops = [transforms.Resize(size)]
    if aug.get("horizontal_flip_p", 0) > 0:
        train_ops.append(transforms.RandomHorizontalFlip(p=aug["horizontal_flip_p"]))
    if aug.get("vertical_flip_p", 0) > 0:
        train_ops.append(transforms.RandomVerticalFlip(p=aug["vertical_flip_p"]))
    train_tf = transforms.Compose(train_ops + [transforms.ToTensor(), normalize])
    eval_tf = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), normalize])
    return train_tf, eval_tf


class HAM10000Dataset(Dataset):
    """Returns (image_tensor, label_index, row_index). row_index maps back to `frame`."""

    def __init__(self, frame: pd.DataFrame, image_index: dict[str, Path], classes: list[str], transform):
        self.frame = frame.reset_index(drop=True)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        self.paths = [str(image_index[i]) for i in self.frame["image_id"]]
        self.labels = [class_to_idx[d] for d in self.frame["dx"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            img = img.convert("RGB")
        return self.transform(img), self.labels[i], i


def make_loader(dataset, batch_size: int, shuffle: bool, drop_last: bool, num_workers: int, seed: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
        persistent_workers=num_workers > 0,
    )
