import os
import platform
import subprocess
import sys
from pathlib import Path


def _git(repo_root: Path, *args) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=20, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_info(repo_root: Path) -> dict:
    commit = _git(repo_root, "rev-parse", "HEAD")
    status = _git(repo_root, "status", "--porcelain")
    return {
        "git_commit": commit,
        # A dirty tree means the commit hash alone does not identify the code that ran.
        "git_dirty": None if status is None else bool(status),
    }


def collect_environment(repo_root: Path) -> dict:
    import numpy
    import pandas
    import PIL
    import sklearn
    import torch
    import torchvision

    cuda = torch.cuda.is_available()
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": numpy.__version__,
        "pandas_version": pandas.__version__,
        "scikit_learn_version": sklearn.__version__,
        "pillow_version": PIL.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda else None,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else [],
    }
    info.update(git_info(repo_root))
    return info
