"""Box-and-arrow drawing for the two schematic figures (1 and 3), in matplotlib
so they are generated like every other figure. Coordinates are in axes units
(0-1); boxes are greyscale with an optional hatch."""

from __future__ import annotations

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, x: float, y: float, w: float, h: float, text: str, face: str = "white", hatch: str = "",
        dashed: bool = False, size: float = 6.5, bold: bool = False) -> tuple[float, float, float, float]:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01", facecolor=face,
                                edgecolor="0.0", hatch=hatch, lw=0.7, ls="--" if dashed else "-",
                                transform=ax.transAxes))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, transform=ax.transAxes,
            fontweight="bold" if bold else "normal", linespacing=1.25)
    return x, y, w, h


def arrow(ax, a: tuple[float, float], b: tuple[float, float], dashed: bool = False, text: str = "") -> None:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.7, color="0.0",
                                 ls="--" if dashed else "-", transform=ax.transAxes, shrinkA=0, shrinkB=0))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, text, fontsize=5.5, ha="center", va="bottom",
                transform=ax.transAxes)


def right(b):
    return b[0] + b[2], b[1] + b[3] / 2


def left(b):
    return b[0], b[1] + b[3] / 2


def top(b):
    return b[0] + b[2] / 2, b[1] + b[3]


def bottom(b):
    return b[0] + b[2] / 2, b[1]


def blank(fig):
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    return ax
