"""Figure 5: performance against forecast horizon, E1 / E2 / E4 on the common test windows.

    python paper_results/plotting/fig05_horizon.py

Sources: paper_results/metrics_forecast.csv and metrics_classification.csv
(scripts/paper/evaluate_experiments.py). Error bars: 95% intervals resampling
whole test days.
"""

from __future__ import annotations

import numpy as np

from results_io import HEAD_ORDER, num, one, rows
from style import DOUBLE, EXPERIMENT, REFERENCE, save, setup
import matplotlib.pyplot as plt

POP = "common"
REFS = {"reference: GOES flux now (persistence)": ("GOES flux now (persistence)", "-.", "x"),
        "reference: SoLEXS flux now (calibrated on training)": ("SoLEXS flux now", (0, (1, 1)), "+"),
        "reference: climatology (training mean)": ("climatology", (0, (5, 2, 1, 2, 1, 2)), "_")}


def _err(r):
    v, lo, hi = num(r["value"]), num(r.get("ci95_low")), num(r.get("ci95_high"))
    return v, (v - lo if lo is not None else 0.0), (hi - v if hi is not None else 0.0)


def main() -> int:
    setup()
    fx, cl = rows("metrics_forecast.csv"), rows("metrics_classification.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.7))
    hs = sorted({int(r["horizon_min"]) for r in fx if r["population"] == POP})
    x = np.arange(len(hs))
    for k, (e, st) in enumerate(EXPERIMENT.items()):
        pts = [one(fx, experiment=e, population=POP, horizon_min=h, metric="MAE") for h in hs]
        v = np.array([_err(p) for p in pts])
        a.errorbar(x + (k - 1) * 0.08, v[:, 0], yerr=v[:, 1:].T, color=st["color"], ls=st["ls"], marker=st["marker"],
                   capsize=1.5, lw=0.9, label=st["label"])
    for name, (label, ls, mk) in REFS.items():
        pts = [one(fx, experiment=name, population=POP, horizon_min=h, metric="MAE") for h in hs]
        xs = [x[i] for i, p in enumerate(pts) if p]
        a.plot(xs, [num(p["value"]) for p in pts if p], color=REFERENCE["color"], ls=ls, marker=mk, lw=0.8,
               label=label)
    a.set_xticks(x, ["now" if h == 0 else f"+{h}" for h in hs])
    a.set_xlabel("Flux forecast horizon (min)")
    a.set_ylabel("MAE of log$_{10}$ flux (dex)")
    a.legend(loc="upper left", fontsize=6)
    heads = [h for h in HEAD_ORDER if one(cl, head=h, population=POP, metric="TSS")]
    xh = np.arange(len(heads))
    for k, (e, st) in enumerate(EXPERIMENT.items()):
        pts = [one(cl, experiment=e, population=POP, head=h, metric="TSS") for h in heads]
        v = np.array([_err(p) for p in pts])
        b.errorbar(xh + (k - 1) * 0.08, v[:, 0], yerr=v[:, 1:].T, color=st["color"], ls=st["ls"],
                   marker=st["marker"], capsize=1.5, lw=0.9, label=st["label"])
    b.set_xticks(xh, ["in progress" if h == "in_flare" else f"within {h.split('_')[-1][:-3]} min" for h in heads])
    b.set_xlabel(">= C1 flare (target)")
    b.set_ylabel("TSS (validation threshold)")
    b.legend(loc="upper right", fontsize=6)
    n = one(fx, experiment="E4", population=POP, horizon_min=hs[0], metric="MAE")
    cap = ("Performance against horizon on the common test windows (both instruments observing; "
           f"{int(num(n['n_samples'])):,} windows on {n['n_days']} days for the flux now). Left: mean absolute error of "
           "the median log10 GOES-18 XRS-B flux forecast, with the no-change and climatology references on the same "
           "windows. Right: TSS of the >= C1 flare heads at each experiment's own validation threshold. Bars: 95% "
           "intervals from resampling whole test days.")
    save(fig, "fig05_horizon", cap, ["paper_results/metrics_forecast.csv", "paper_results/metrics_classification.csv"],
         "fig05_horizon.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
