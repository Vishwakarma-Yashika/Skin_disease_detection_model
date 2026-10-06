import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> dict:
    """Seeds every RNG the pipeline touches and returns the settings for the run record."""
    # PYTHONHASHSEED only affects child processes (spawned DataLoader workers).
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: some CUDA backward kernels (e.g. adaptive pooling) have no
        # deterministic variant; they warn instead of aborting the run.
        torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "deterministic": deterministic,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def seed_worker(worker_id: int):
    # torch seeds each worker as base_seed + worker_id; propagate that to numpy/random.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
