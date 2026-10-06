import json
import math
from pathlib import Path

import numpy as np


def to_jsonable(obj):
    """numpy -> python, NaN/inf -> None (strict JSON)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path, obj) -> None:
    Path(path).write_text(json.dumps(to_jsonable(obj), indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
