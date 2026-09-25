"""Figure 9: the E4 forecasts through the example flare of Figure 2.

    python paper_results/plotting/fig09_example_forecast.py

Sources: outputs/alerts/pred_final.npz (the frozen E4 model run minute by minute,
written by ``python -m solarflare alerts --predict``: calibrated probabilities and
validation-widened flux quantiles), paper_results/operating_points.csv and
leadtime_events.csv (scripts/paper/leadtime_tables.py), GOES-18 XRS as truth.
Each value is plotted when it became available (origin + 1 min, as the alerts
stage does); the +15 min flux forecast is plotted at the time it forecasts.
"""

from __future__ import annotations

import csv

import numpy as np

from example_event import pick
from style import DOUBLE, PAPER, save, setup
import matplotlib.pyplot as plt

from solarflare.settings import load_settings
from solarflare.util import utc

H_MIN = 15


def _op(alert: str, method: str, op: str) -> float:
    with open(PAPER / "operating_points.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["alert"] == alert and r["method"] == method and r["operating_point"] == op:
                return float(r["threshold"])
    raise SystemExit(f"no operating point {alert} / {method} / {op}")


def main() -> int:
    from solarflare.io.goes import load_goes

    setup()
    S = load_settings()
    f, t0, t1 = pick()
    g = load_goes(S.goes_dir)
    P = np.load(S.alerts / "pred_final.npz")
    avail = P["origin"] + 60.0
    m = (avail >= t0) & (avail < t1)
    hs = [int(h / 60) for h in P["horizons_s"]]
    qs = list(P["quantiles"])
    i50, k15 = qs.index(0.5), hs.index(H_MIN)
    fc = P["fc"][m]
    thr_c = _op("GOES >= C1 flare within 15 min", "network (E4)", "fa_2")
    thr_m = _op("GOES flux reaches M1 within 30 min", "network (E4)", "fa_0.5")
    m_signal = np.max(fc[:, [hs.index(h) for h in (5, 15, 30)], i50], axis=1)
    with open(PAPER / "leadtime_events.csv", encoding="utf-8") as fh:
        ev = {(r["alert"], r["peak_utc"]): r for r in csv.DictReader(fh)}
    key_peak = utc(f.peak_unix, "%Y-%m-%d %H:%M")
    lead_c = ev.get(("GOES >= C1 flare within 15 min", key_peak), {}).get("lead_before_peak_min_model")
    lead_m = ev.get(("GOES flux reaches M1 within 30 min", key_peak), {}).get("lead_before_peak_min_model")
    ref = f.start_unix

    def mins(t):
        return (np.asarray(t) - ref) / 60.0

    gm = (g.time_unix >= t0) & (g.time_unix < t1)
    fig, axes = plt.subplots(3, 1, figsize=(DOUBLE, 4.6), sharex=True, gridspec_kw={"hspace": 0.08})
    ax = axes[0]
    ax.plot(mins(g.time_unix[gm]), np.log10(g.xrsb[gm]), color="0.0", lw=1.1, label="GOES-18 XRS-B (truth)")
    tgt = P["origin"][m] + 60.0 * H_MIN
    ax.fill_between(mins(tgt), fc[:, k15, 0], fc[:, k15, -1], color="0.8", lw=0,
                    label=f"E4 +{H_MIN} min forecast, q{int(qs[0] * 100)}-q{int(qs[-1] * 100)}")
    ax.plot(mins(tgt), fc[:, k15, i50], color="0.25", ls="--", lw=1.0, label=f"E4 +{H_MIN} min forecast, median")
    ax.set_ylabel("log$_{10}$ flux (W m$^{-2}$)")
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.plot(mins(avail[m]), P["p_occ"][m, 0], color="0.0", lw=1.0, label="E4 P(>= C1 flare within 15 min), calibrated")
    ax.axhline(thr_c, color="0.3", ls=":", lw=0.9, label="alert threshold (2 false alarms/day on validation)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Probability")
    ax.legend(loc="upper left")
    ax = axes[2]
    ax.plot(mins(avail[m]), m_signal, color="0.0", lw=1.0, label="E4 highest median forecast, +5/+15/+30 min")
    ax.axhline(thr_m, color="0.3", ls=":", lw=0.9, label="M1 alert threshold (0.5 false alarms/day on validation)")
    ax.set_ylabel("log$_{10}$ flux (W m$^{-2}$)")
    ax.legend(loc="center left")
    for ax in axes:
        for t in (f.start_unix, f.peak_unix, f.end_unix):
            ax.axvline(mins(t), color="0.45", lw=0.6, ls="-.")
    for t, name in ((f.start_unix, "GOES start"), (f.peak_unix, "peak"), (f.end_unix, "end")):
        axes[0].text(mins(t), 1.02, name, transform=axes[0].get_xaxis_transform(), ha="center", fontsize=6)
    axes[-1].set_xlabel(f"Minutes from the GOES start ({utc(f.start_unix, '%Y-%m-%d %H:%M')} UTC)")
    # the +15 min median forecast whose target is the GOES peak minute, against the observed peak
    j = int(np.argmin(np.abs(tgt - f.peak_unix)))
    peak_fc = (f"The +{H_MIN} min median forecast issued at {utc(P['origin'][m][j], '%H:%M')} UTC for the peak minute "
               f"was 10^{fc[j, k15, i50]:.2f} W m^-2 (10-90%: 10^{fc[j, k15, 0]:.2f}-10^{fc[j, k15, -1]:.2f}); the "
               f"observed GOES peak was 10^{np.log10(f.peak_flux):.2f} W m^-2. ")
    leads = "; ".join(x for x in (
        f">= C1 alert first on {lead_c} min before the GOES peak" if lead_c else ">= C1 alert: not warned",
        f"M1 alert (network) first on {lead_m} min before the GOES peak" if lead_m else "M1 alert (network): not warned"))
    cap = (f"The frozen E4 model through the {f.goes_class} flare of Figure 2, minute by minute, each value shown "
           "when it became available. Top: GOES-18 XRS-B flux and the E4 flux forecast for 15 min ahead, plotted at "
           "the time it forecasts (median and validation-widened 10-90% interval). Middle: calibrated probability of "
           "a >= C1 flare within 15 min and the alert threshold chosen on validation. Bottom: the M1 alert signal and "
           f"its validation threshold. {peak_fc}{leads} (paper_results/leadtime_events.csv). One illustrative flare chosen by "
           "a fixed rule (Figure 2); the statistics over all test flares are in Figure 8 and the tables.")
    save(fig, "fig09_example_forecast", cap, ["outputs/alerts/pred_final.npz", "paper_results/operating_points.csv",
                                               "paper_results/leadtime_events.csv", "GOES-18 XRS avg1m (goes_dir)"],
         "fig09_example_forecast.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
