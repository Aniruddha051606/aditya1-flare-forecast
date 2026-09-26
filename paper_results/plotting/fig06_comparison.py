"""Figure 6: E4 against E1 and E2, paired on the common test windows.

    python paper_results/plotting/fig06_comparison.py

Source: paper_results/paired_comparisons.csv (scripts/paper/evaluate_experiments.py):
differences E4 - E1 and E4 - E2 on the same windows, each model at its own
validation threshold and calibration, 95% intervals resampling the same whole
test days for both models.
"""

from __future__ import annotations

import numpy as np

from results_io import HEAD_ORDER, HEAD_LABEL, num, one, rows
from style import DOUBLE, HEL1OS, SOLEXS, save, setup
import matplotlib.pyplot as plt

POP = "common"
PAIRS = {"E4 - E1": ("E4 - E1 (adding HEL1OS to SoLEXS)", "s", HEL1OS),
         "E4 - E2": ("E4 - E2 (adding SoLEXS to HEL1OS)", "^", SOLEXS)}
METRICS = [("ROC_AUC", "AUC"), ("TSS", "TSS"), ("BSS_calibrated", "BSS (calibrated)")]


def _pt(r):
    v = num(r["value"])
    lo, hi = num(r["ci95_low"]), num(r["ci95_high"])
    return v, v - (lo if lo is not None else v), (hi if hi is not None else v) - v


def main() -> int:
    setup()
    pr = rows("paired_comparisons.csv")
    fig, (a, b1, b2) = plt.subplots(1, 3, figsize=(DOUBLE, 3.2), gridspec_kw={"width_ratios": [1.6, 0.7, 0.7],
                                                                              "wspace": 0.08})
    labels, y = [], 0
    for h in HEAD_ORDER:
        for m, mlabel in METRICS:
            if not any(one(pr, comparison=c, population=POP, quantity=h, metric=m) for c in PAIRS):
                continue
            for k, (c, (_, mk, col)) in enumerate(PAIRS.items()):
                r = one(pr, comparison=c, population=POP, quantity=h, metric=m)
                if r:
                    v, lo, hi = _pt(r)
                    a.errorbar(v, y + (k - 0.5) * 0.3, xerr=[[lo], [hi]], fmt=mk, color=col, ms=3.5, capsize=1.5,
                               lw=0.8, mfc=col if k == 0 else "white")
            labels.append(f"{HEAD_LABEL[h]}: {mlabel}")
            y += 1
    a.set_yticks(np.arange(len(labels)), labels, fontsize=6)
    a.invert_yaxis()
    a.axvline(0, color="#444444", lw=0.6, ls="--")
    a.set_xlabel("Difference (positive: E4 better)")
    # flux: one panel per comparison, each on its own scale (the two differ by ~30x)
    hs = sorted({int(r["horizon_min"]) for r in pr if r["quantity"] == "flux" and r["population"] == POP})
    for k, (b, (c, (label, mk, col))) in enumerate(zip((b1, b2), PAIRS.items())):
        pts = [one(pr, comparison=c, population=POP, quantity="flux", horizon_min=h, metric="MAE_dex") for h in hs]
        v = np.array([_pt(p) for p in pts])
        b.errorbar(v[:, 0], np.arange(len(hs)), xerr=v[:, 1:].T, fmt=mk, color=col, ms=3.5, capsize=1.5, lw=0.8,
                   mfc=col if k == 0 else "white", label=label)
        b.set_yticks(np.arange(len(hs)), ["flux now" if h == 0 else f"flux +{h} min" for h in hs] if k else [],
                     fontsize=6)
        b.set_ylim(len(hs) - 0.5, -0.5)
        b.axvline(0, color="#444444", lw=0.6, ls="--")
        b.set_title(c, loc="left", fontsize=7, color=col)
        b.tick_params(axis="x", labelsize=6)
    b2.yaxis.tick_right()
    fig.text(0.78, -0.02, "MAE difference, dex (negative: E4 better)", ha="center", fontsize=8)
    handles = [b1.get_legend_handles_labels(), b2.get_legend_handles_labels()]
    fig.legend([h[0][0] for h in handles], [h[1][0] for h in handles], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.04))
    r0 = one(pr, comparison="E4 - E1", population=POP, quantity="in_flare", metric="TSS")
    cap = ("Paired comparison on the common test windows (both instruments observing; "
           f"{int(num(r0['n_samples'])):,} labelled windows on {r0['n_days']} days for the in-progress head). Each "
           "point is E4 minus a single-instrument experiment on the same windows, each model at its own validation "
           "threshold and calibration; bars are 95% intervals from resampling the same whole test days for both "
           "models. Filled vermillion squares: E4 - E1 (the value of adding HEL1OS); open blue triangles: E4 - E2 (the value of "
           "adding SoLEXS). Left: flare heads (>= C1); right: flux error by horizon, one panel per comparison on its own "
           "scale.")
    save(fig, "fig06_comparison", cap, ["paper_results/paired_comparisons.csv"], "fig06_comparison.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
