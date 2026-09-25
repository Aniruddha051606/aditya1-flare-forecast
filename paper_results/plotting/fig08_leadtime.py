"""Figure 8: lead time before the GOES peak, per test flare, for the E4 alerts and
simple SoLEXS references at the same validation-chosen false-alarm rate.

    python paper_results/plotting/fig08_leadtime.py

Source: paper_results/leadtime_events.csv (scripts/paper/leadtime_tables.py, from
the alerts stage). Each curve is the fraction of all test flares of the alert
type that were warned at least x minutes before the GOES peak; flares never
warned count as not warned at any lead, so a curve's value at 0 is the
detection rate.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from style import DOUBLE, PAPER, REFERENCE, save, setup
import matplotlib.pyplot as plt

CURVES = {
    "GOES >= C1 flare within 15 min": [("model", "E4 network", {"color": "0.0", "ls": "-"}),
                                       ("trend", "SoLEXS trend", {"color": "0.4", "ls": "--"}),
                                       ("current", "SoLEXS now", {"color": REFERENCE["color"], "ls": ":"})],
    "GOES flux reaches M1 within 30 min": [("combined", "E4 network or SoLEXS now", {"color": "0.0", "ls": "-"}),
                                           ("model", "E4 network", {"color": "0.25", "ls": "-."}),
                                           ("trend", "SoLEXS trend", {"color": "0.4", "ls": "--"}),
                                           ("current", "SoLEXS now", {"color": REFERENCE["color"], "ls": ":"})],
}
XMAX = 60


def main() -> int:
    setup()
    with open(PAPER / "leadtime_events.csv", encoding="utf-8") as fh:
        ev = list(csv.DictReader(fh))
    src = json.loads((PAPER / "leadtime_source.json").read_text("utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), sharey=True)
    x = np.arange(0, XMAX + 1)
    notes = []
    for ax, (alert, curves) in zip(axes, CURVES.items()):
        rows = [r for r in ev if r["alert"] == alert]
        for key, label, st in curves:
            lead = np.array([float(r[f"lead_before_peak_min_{key}"]) if r[f"lead_before_peak_min_{key}"] else -np.inf
                             for r in rows])
            frac = [(lead >= v).mean() for v in x]
            ax.step(x, frac, where="post", label=f"{label} ({int(np.isfinite(lead).sum())}/{lead.size})", **st)
        ax.set_title(f"{alert} ({len(rows)} test flares)", loc="left")
        ax.set_xlabel("Lead before the GOES peak (min)")
        ax.set_xlim(0, XMAX)
        ax.set_ylim(0, 1.02)
        ax.grid(True, lw=0.3, color="0.85")
        ax.legend(loc="upper right", handlelength=2.4)
        notes.append(f"{alert}: {rows[0]['operating_point']}")
    axes[0].set_ylabel("Fraction of flares warned\nat least this early")
    fa = src["primary_false_alarms_per_day"]
    cap = ("Lead time of the E4 alerts on the test period, per flare. 'E4 network or SoLEXS now' is the M1 alert "
           "that fires on the higher of the network's M1 signal and the calibrated SoLEXS flux now; 'SoLEXS trend' "
           "extrapolates that flux 15 min along its last-5-min rise. Each curve is the fraction of all test flares "
           "warned at least the given time before the GOES peak (unwarned flares never count), so its value at "
           "zero is the detection rate; legend: flares warned / flares. Every method runs at the threshold giving "
           f"{fa['C']:g} (>= C1 alerts) and {fa['M']:g} (M1 alerts) false alarms per day on the validation period "
           f"({src['test_days_with_data']} test days with data, {src['test_period'][0]} to {src['test_period'][1]} "
           "UTC). Definitions: solarflare/products/leadtime.py.")
    save(fig, "fig08_leadtime", cap, ["paper_results/leadtime_events.csv", "paper_results/leadtime_source.json"],
         "fig08_leadtime.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
