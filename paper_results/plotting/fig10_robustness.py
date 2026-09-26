"""Figure 10: how much E4 depends on having both instruments.

    python paper_results/plotting/fig10_robustness.py

Source: paper_results/metrics_classification.csv and metrics_forecast.csv
(scripts/paper/evaluate_experiments.py). On the common test windows: E4 with
both instruments; E4 with SoLEXS or HEL1OS withheld at prediction time (the
deployed model facing a missing instrument); and the models trained on one
instrument (E1, E2). 95% intervals resample whole test days.
"""

from __future__ import annotations

import numpy as np

from results_io import num, one, rows
from style import BOTH, DOUBLE, HEL1OS, SOLEXS, save, setup
import matplotlib.pyplot as plt

POP = "common"
SERIES = [("E4", "E4, both instruments", "o", BOTH, BOTH),
          ("E4, HEL1OS withheld", "E4, HEL1OS withheld", "s", BOTH, "white"),
          ("E1", "E1, trained on SoLEXS only", "s", SOLEXS, SOLEXS),
          ("E4, SoLEXS withheld", "E4, SoLEXS withheld", "^", BOTH, "white"),
          ("E2", "E2, trained on HEL1OS only", "^", HEL1OS, HEL1OS)]
PANELS = [("metrics_classification.csv", {"head": "in_flare", "metric": "ROC_AUC"}, "AUC, flare in progress"),
          ("metrics_classification.csv", {"head": "flare_within_15min", "metric": "ROC_AUC"}, "AUC, flare within 15 min"),
          ("metrics_forecast.csv", {"horizon_min": 15, "metric": "MAE"}, "MAE, flux +15 min (dex, log scale)")]


def main() -> int:
    setup()
    tables = {n: rows(n) for n in {p[0] for p in PANELS}}
    fig, axes = plt.subplots(1, len(PANELS), figsize=(DOUBLE, 2.4))
    for ax, (table, sel, title) in zip(axes, PANELS):
        for k, (e, _label, mk, col, face) in enumerate(SERIES):
            r = one(tables[table], experiment=e, population=POP, **sel)
            if not r:
                continue
            v, lo, hi = num(r["value"]), num(r.get("ci95_low")), num(r.get("ci95_high"))
            err = [[v - lo], [hi - v]] if lo is not None else None
            ax.errorbar(v, k, xerr=err, fmt=mk, color=col, mfc=face, ms=4, capsize=1.5, lw=0.8)
        ax.set_yticks(np.arange(len(SERIES)), [s[1] for s in SERIES] if ax is axes[0] else [], fontsize=6)
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.grid(True, axis="x", which="both", lw=0.3, color="#DDDDDD")
        if table == "metrics_forecast.csv":
            ax.set_xscale("log")
    r0 = one(tables["metrics_classification.csv"], experiment="E4", population=POP, head="in_flare", metric="ROC_AUC")
    cap = ("Dependence of E4 on both instruments, on the common test windows "
           f"({int(num(r0['n_samples'])):,} labelled windows, {r0['n_days']} days). Rows: E4 as deployed; E4 with one "
           "instrument withheld at prediction time (its thresholds and calibration unchanged); and the experiments "
           "trained on one instrument only. Bars: 95% intervals from resampling whole test days. MAE is of the median "
           "log10 GOES-18 XRS-B flux forecast.")
    save(fig, "fig10_robustness", cap, ["paper_results/metrics_classification.csv", "paper_results/metrics_forecast.csv"],
         "fig10_robustness.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
