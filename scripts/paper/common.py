"""Shared by the paper scripts: where results go, and how they are written.

Every file under paper_results/ is produced by a script in scripts/paper/ or
paper_results/plotting/ from data or from result files the pipeline wrote; no
number in it is typed by hand.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAPER = ROOT / "paper_results"


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return out.stdout.strip() + ("+uncommitted-changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def stamp() -> dict:
    """Provenance attached to every result file."""
    return {"generated_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"), "git_commit": git_commit(),
            "python": sys.version.split()[0], "script": Path(sys.argv[0]).name}


def utc(t: float | None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str | None:
    return None if t is None else datetime.fromtimestamp(float(t), UTC).strftime(fmt)


def write_json(name: str, data: dict) -> Path:
    PAPER.mkdir(parents=True, exist_ok=True)
    p = PAPER / name
    p.write_text(json.dumps({"provenance": stamp(), **data}, indent=2, default=str), encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT)}")
    return p


def write_text(name: str, text: str) -> Path:
    PAPER.mkdir(parents=True, exist_ok=True)
    p = PAPER / name
    p.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    print(f"  -> {p.relative_to(ROOT)}")
    return p


def goes_class_letter(flux_wm2: float) -> str:
    for letter, floor in (("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7)):
        if flux_wm2 >= floor:
            return letter
    return "A"
