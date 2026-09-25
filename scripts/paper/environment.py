"""The software environment and the repository state, for the results manifest.

    python scripts/paper/environment.py

Writes paper_results/00_environment.json: git commit and remotes, Python, OS,
the versions of every package the pipeline imports, torch/CUDA/GPU, and the
settings file. Read only.
"""

from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess

from common import ROOT, write_json  # noqa: E402  (sets sys.path)

PACKAGES = ("numpy", "scipy", "pandas", "torch", "astropy", "scikit-learn", "matplotlib", "netCDF4", "h5py",
            "pyflakes", "ruff")


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def environment() -> dict:
    pkgs = {}
    for p in PACKAGES:
        try:
            pkgs[p] = md.version(p)
        except md.PackageNotFoundError:
            pkgs[p] = None
    gpu = {}
    try:
        import torch

        gpu = {"cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda,
               "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "memory_GB": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
               if torch.cuda.is_available() else None}
    except ImportError:
        pass
    return {
        "git": {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "uncommitted_tracked_changes": bool(_git("status", "--porcelain", "--untracked-files=no")),
                "remotes": sorted(set(_git("remote", "-v").split()[1::3]))},
        "python": platform.python_version(), "os": platform.platform(), "machine": platform.machine(),
        "packages": pkgs, "gpu": gpu,
        "settings_file": "config/project.toml",
    }


if __name__ == "__main__":
    write_json("00_environment.json", environment())
