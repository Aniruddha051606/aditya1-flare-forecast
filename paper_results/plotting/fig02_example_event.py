"""Figure 2: one flare seen by GOES-18 XRS, SoLEXS and HEL1OS, synchronised.

    python paper_results/plotting/fig02_example_event.py

The flare is chosen by the fixed rule in example_event.py. Sources: the GOES-18
XRS 1-min flux (goes_dir) and the frozen study cache (the 20 s rates the model is
built from, before normalisation). HEL1OS bands are the sum of CZT1 and CZT2.
"""

from __future__ import annotations

import numpy as np

from example_event import DT, MIN_COVERAGE, POST_MIN, PRE_MIN, pick, series
from style import C, DOUBLE, GOES, save, setup
import matplotlib.pyplot as plt

from solarflare.settings import load_settings
from solarflare.util import utc

SOFT = [("slx_2_3keV", "2-3 keV", "-", C["blue"]), ("slx_4_6keV", "4-6 keV", "--", C["sky"]),
        ("slx_8_12keV", "8-12 keV", ":", "#003B64")]
HARD = [("20_40keV", "20-40 keV", "-", C["vermillion"]), ("40_60keV", "40-60 keV", "--", C["orange"]),
        ("60_80keV", "60-80 keV", ":", "#8C2D04")]


def main() -> int:
    from solarflare.io.goes import load_goes

    setup()
    S = load_settings()
    f, t0, t1 = pick()
    g = load_goes(S.goes_dir)
    m = (g.time_unix >= t0) & (g.time_unix < t1)
    ts, soft, _ = series(S.cache, "solexs", t0, t1)
    th, hard, _ = series(S.cache, "hel1os", t0, t1)
    ref = f.start_unix

    def mins(t):
        return (np.asarray(t) - ref) / 60.0

    fig, axes = plt.subplots(3, 1, figsize=(DOUBLE, 4.6), sharex=True, gridspec_kw={"hspace": 0.08})
    ax = axes[0]
    ax.semilogy(mins(g.time_unix[m]), g.xrsb[m], color=GOES, lw=1.1, label="GOES-18 XRS-B (0.1-0.8 nm)")
    for lvl, name in ((1e-6, "C1"), (1e-5, "M1"), (1e-4, "X1")):
        ax.axhline(lvl, color="0.6", lw=0.5, ls=(0, (2, 3)))
        ax.text(mins(t1), lvl, f" {name}", va="center", fontsize=6)
    ax.set_ylabel("Flux (W m$^{-2}$)")
    ax.legend(loc="upper right")
    ax = axes[1]
    for key, label, ls, colour in SOFT:
        ax.semilogy(mins(ts), soft[key], color=colour, ls=ls, lw=1.0, label=f"SoLEXS {label}")
    ax.set_ylabel("Count rate (s$^{-1}$)")
    ax.legend(loc="upper right")
    ax = axes[2]
    for key, label, ls, colour in HARD:
        rate = np.nansum([hard.get(f"hls_czt1_{key}"), hard.get(f"hls_czt2_{key}")], axis=0)
        rate[np.isnan(hard[f"hls_czt1_{key}"]) & np.isnan(hard[f"hls_czt2_{key}"])] = np.nan
        ax.semilogy(mins(th), np.where(rate > 0, rate, np.nan), color=colour, ls=ls, lw=1.0,
                    label=f"HEL1OS CZT {label}")
    ax.set_ylabel("Count rate (s$^{-1}$)")
    ax.legend(loc="upper right")
    for ax in axes:
        for t in (f.start_unix, f.peak_unix, f.end_unix):
            ax.axvline(mins(t), color=C["grey"], lw=0.6, ls="-.")
    for t, name in ((f.start_unix, "GOES start"), (f.peak_unix, "peak"), (f.end_unix, "end")):
        axes[0].text(mins(t), 1.02, name, transform=axes[0].get_xaxis_transform(), ha="center", fontsize=6)
    axes[-1].set_xlabel(f"Minutes from the GOES start ({utc(f.start_unix, '%Y-%m-%d %H:%M')} UTC)")
    cap = (f"{'An' if f.goes_class[0] in 'MX' else 'A'} {f.goes_class} flare on {utc(f.peak_unix, '%Y-%m-%d')} (GOES start {utc(f.start_unix, '%H:%M')}, "
           f"peak {utc(f.peak_unix, '%H:%M')}, end {utc(f.end_unix, '%H:%M')} UTC) seen by GOES-18 XRS-B (top, 1-min "
           "flux; reference only, never a model input), SoLEXS (middle) and HEL1OS CZT1+CZT2 (bottom), both as the "
           f"{DT:g} s rates the model is built from. Chosen by a fixed rule: the highest-peak GOES flare of the test "
           f"period with both instruments observing at least {100 * MIN_COVERAGE:.0f}% of the interval from "
           f"{PRE_MIN} min before its start to {POST_MIN} min after its end.")
    save(fig, "fig02_example_event", cap, ["GOES-18 XRS avg1m (goes_dir)", "cache/ (frozen study cache)",
                                            "paper_results/02_experiment_split.json"], "fig02_example_event.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
