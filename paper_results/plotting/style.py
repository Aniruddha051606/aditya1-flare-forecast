"""Shared style for the paper figures.

Springer column widths, one font scale, and experiment/line styles that stay
distinct in greyscale print (line style + marker + grey level, never colour
alone). Every figure is saved as vector PDF and 300-dpi PNG, with a JSON record
next to it: caption, the result files it was drawn from, the script and the git
commit. Figures are drawn only from result files; nothing is typed in.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_results"
FIG = PAPER / "figures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SINGLE = 84 / 25.4          # inches: Springer single column
DOUBLE = 174 / 25.4         # inches: Springer full width

#: one style per experiment, used by every figure
EXPERIMENT = {
    "E1": {"label": "E1 SoLEXS only", "color": "0.55", "ls": "--", "marker": "s", "hatch": "////"},
    "E2": {"label": "E2 HEL1OS only", "color": "0.75", "ls": ":", "marker": "^", "hatch": "...."},
    "E4": {"label": "E4 SoLEXS + HEL1OS", "color": "0.0", "ls": "-", "marker": "o", "hatch": ""},
}
REFERENCE = {"color": "0.35", "ls": "-.", "marker": "x"}
SPLIT = {"train": {"face": "0.90", "hatch": "////", "label": "training"},
         "val": {"face": "0.80", "hatch": "....", "label": "validation"},
         "test": {"face": "0.68", "hatch": "xxxx", "label": "test"}}


def grid_seconds() -> float:
    """The cache and model time step, from the final model's configuration."""
    cfg = json.loads((ROOT / "outputs" / "model" / "reports" / "config.json").read_text("utf-8"))
    return float(cfg["pre"]["dt_seconds"])


def setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
        "lines.linewidth": 1.0, "lines.markersize": 3.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
        "ps.fonttype": 42, "hatch.linewidth": 0.4,
    })


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save(fig, name: str, caption: str, sources: list[str], script: str) -> None:
    """Write <name>.pdf, <name>.png and <name>.json (caption and provenance)."""
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    (FIG / f"{name}.json").write_text(json.dumps({
        "figure": name, "caption": caption, "sources": sources, "script": f"paper_results/plotting/{script}",
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"), "git_commit": _commit()}, indent=2),
        encoding="utf-8")
    print(f"  -> paper_results/figures/{name}.pdf / .png")
