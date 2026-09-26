"""Figure 7: reliability of the E4 probabilities, raw and calibrated on validation.

    python paper_results/plotting/fig07_reliability.py

Source: paper_results/reliability.csv (scripts/paper/evaluate_experiments.py):
equal-width probability bins over the E4 native test windows; the calibration
map was fitted on validation only. Bins with fewer than MIN_N windows are not
drawn.
"""

from __future__ import annotations

from results_io import HEAD_LABEL, num, pick, rows
from style import BOTH, C, DOUBLE, save, setup
import matplotlib.pyplot as plt

EXP, POP, MIN_N = "E4", "native", 100
PANELS = ["in_flare", "flare_within_15min", "flare_within_60min"]


def main() -> int:
    setup()
    rel = rows("reliability.csv")
    fig, axes = plt.subplots(1, len(PANELS), figsize=(DOUBLE, 2.5), sharey=True)
    for ax, h in zip(axes, PANELS):
        ax.plot([0, 1], [0, 1], color=C["grey"], lw=0.6, ls=":", label="perfect reliability")
        for kind, st in (("raw", {"color": C["orange"], "ls": "--", "marker": "s", "mfc": "white"}),
                         ("calibrated", {"color": BOTH, "ls": "-", "marker": "o", "mfc": BOTH})):
            pts = [r for r in pick(rel, experiment=EXP, population=POP, head=h, probability=kind)
                   if num(r["n_samples"]) >= MIN_N]
            ax.plot([num(r["mean_forecast"]) for r in pts], [num(r["observed_frequency"]) for r in pts], ms=3.2, lw=0.9,
                    label=f"{kind}", **st)
        ax.set_title(HEAD_LABEL[h], loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Forecast probability")
        ax.set_aspect("equal")
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend(loc="upper left", fontsize=6)
    cap = ("Reliability of the E4 probabilities for >= C1 flares on the test period (all E4 test windows), before "
           "(open orange squares) and after (filled green circles) isotonic calibration fitted on the validation period only. "
           f"Points are equal-width probability bins with at least {MIN_N} windows; the dotted diagonal is perfect "
           "reliability. Brier scores and skill are in metrics_classification.csv.")
    save(fig, "fig07_reliability", cap, ["paper_results/reliability.csv"], "fig07_reliability.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
