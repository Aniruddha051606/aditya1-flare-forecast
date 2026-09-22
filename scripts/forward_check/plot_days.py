"""Model vs actual, one PNG per UTC day (13-19 Sep 2026) plus an overview.

Reads forecast.npz (predict_after_test.py), GOES-18 from D:/Data/goes_2026-09-22
and HEL1OS minute light curves from this workspace's cache.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone, UTC
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "outputs" / "_dev" / "forward_check"   # working folder: data links, cache, PNGs, seals
HERE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from solarflare.catalog.build import hel1os_minutes  # noqa: E402
from solarflare.io.goes import load_goes  # noqa: E402

OUT = HERE / "png"
UTC = UTC
IST = timezone(timedelta(hours=5, minutes=30))
THRESHOLD_C = 0.6389                     # outputs/alerts/alert_rules.json

# reference palette (dataviz skill), light surface
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9994", "#e7e6e1"
GOES_C, MODEL_C, NOW_C, HXR_C = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
ALERT_C = "#ec835a"                      # status: serious

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 9,
})


def gaps_to_nan(t: np.ndarray, y: np.ndarray, max_gap: float = 180.0):
    """Break a line where samples are more than max_gap seconds apart."""
    if t.size == 0:
        return t, y
    br = np.flatnonzero(np.diff(t) > max_gap)
    return np.insert(t.astype(float), br + 1, np.nan), np.insert(y.astype(float), br + 1, np.nan)


def fresh(path: Path) -> Path:
    """Never overwrite a generated image: add _v2, _v3, ... when the name is taken."""
    k = 2
    out = path
    while out.exists():
        out = path.with_name(f"{path.stem}_v{k}{path.suffix}")
        k += 1
    return out


def to_dt(t):
    return [datetime.fromtimestamp(float(x), UTC) if np.isfinite(x) else None for x in t]


def plot_line(ax, t, y, **kw):
    t2, y2 = gaps_to_nan(t, y)
    x = np.array([np.datetime64(int(v), "s") if np.isfinite(v) else np.datetime64("NaT", "s") for v in t2])
    return ax.plot(x, y2, **kw)


def npdt(t):
    return np.array([np.datetime64(int(v), "s") for v in np.atleast_1d(t)])


def episodes(on: np.ndarray):
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def fmt_class(flux: float) -> str:
    for letter, base in (("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7), ("A", 1e-8)):
        if flux >= base:
            return f"{letter}{flux / base:.1f}"
    return "A0.0"


def time_axis(ax, t0: float, t1: float, step_h: int):
    ticks = np.arange(t0, t1 + 1, step_h * 3600.0)
    ax.set_xticks(npdt(ticks))
    ax.set_xticklabels([f"{datetime.fromtimestamp(x, UTC):%H:%M} UTC\n{datetime.fromtimestamp(x, IST):%H:%M} IST"
                        if step_h < 24 else
                        f"{datetime.fromtimestamp(x, UTC):%d %b}\n00:00 UTC" for x in ticks], fontsize=8.5)
    ax.set_xlim(npdt(t0)[0], npdt(t1)[0])


def main() -> int:
    OUT.mkdir(exist_ok=True)
    F = dict(np.load(HERE / "forecast.npz"))
    g = load_goes(Path("D:/Data/goes_2026-09-22"))
    hx = hel1os_minutes(HERE / "cache")
    o = F["origin"]
    known = o + 60.0
    hs = [60.0, 300.0, 900.0, 1800.0, 3600.0]
    h15 = hs.index(900.0)
    q10, q50, q90 = F["fc"][:, h15, 0], F["fc"][:, h15, 1], F["fc"][:, h15, 2]
    t15 = o + 900.0                                   # the time each +15 min forecast aimed at
    p15 = F["p_occ"][:, 0]
    gt, gx = g.time_unix + 30.0, g.xrsb
    ht, hr = hx["cdte_5_20"]

    days = [datetime(2026, 9, d, tzinfo=UTC).timestamp() for d in range(13, 20)]
    written = []
    for d0 in days + [None]:
        overview = d0 is None
        t0, t1 = (days[0], days[-1] + 86400.0) if overview else (d0, d0 + 86400.0)
        fig, axes = plt.subplots(3, 1, figsize=(13, 9.2), sharex=True,
                                 gridspec_kw={"height_ratios": [3.2, 1.6, 1.4], "hspace": 0.28})
        a1, a2, a3 = axes
        sel = (known >= t0) & (known < t1)
        gsel = (gt >= t0) & (gt < t1)
        flares = [x for x in g.flares if t0 <= x.peak_unix < t1]
        c_eps = [(known[a], known[b - 1]) for a, b in episodes(F["c_on"] & sel)]

        # alert shading on every panel, GOES flare peaks as faint guides
        for ax in axes:
            for s, e in c_eps:
                ax.axvspan(npdt(s - 30)[0], npdt(e + 30)[0], color=ALERT_C, alpha=0.28, lw=0, zorder=0)
            for x in flares:
                ax.axvline(npdt(x.peak_unix)[0], color=MUTED, lw=0.8, ls=":", zorder=1)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)

        # 1. flux
        plot_line(a1, gt[gsel], np.log10(np.where(gx[gsel] > 0, gx[gsel], np.nan)), color=GOES_C, lw=1.6,
                  label="Actual: GOES-18 X-ray flux", zorder=4)
        if sel.any():
            s15 = (t15 >= t0) & (t15 < t1)
            tt, lo = gaps_to_nan(t15[s15], q10[s15])
            _, hi = gaps_to_nan(t15[s15], q90[s15])
            xx = np.array([np.datetime64(int(v), "s") if np.isfinite(v) else np.datetime64("NaT", "s") for v in tt])
            a1.fill_between(xx, lo, hi, color=MODEL_C, alpha=0.16, lw=0, zorder=2,
                            label="Model forecast, 80% range")
            plot_line(a1, t15[s15], q50[s15], color=MODEL_C, lw=1.4, zorder=3,
                      label="Model forecast made 15 min earlier")
            plot_line(a1, known[sel], F["now"][sel], color=NOW_C, lw=1.0, alpha=0.9, zorder=3,
                      label="Model estimate of flux now (from Aditya)")
        for lvl, name in ((-6.0, "C1"), (-5.0, "M1")):
            a1.axhline(lvl, color=MUTED, lw=0.9, ls="--", zorder=1)
            a1.text(1.003, lvl, name, transform=a1.get_yaxis_transform(), va="center", fontsize=9, color=INK2)
        for x in flares:
            y = np.log10(x.peak_flux)
            a1.plot(npdt(x.peak_unix), [y], marker="D", ms=5, color=INK, zorder=5)
            if not overview or x.goes_class[:1] in "CMX" or x.peak_flux >= 9e-7:
                a1.annotate(x.goes_class, (npdt(x.peak_unix)[0], y), xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=8.5, color=INK2)
        a1.set_ylim(-7.3, -4.8)
        a1.set_yticks([-7, -6, -5])
        a1.set_yticklabels(["B1 (10⁻⁷)", "C1 (10⁻⁶)", "M1 (10⁻⁵)"])
        a1.set_ylabel("X-ray flux, W/m²")
        a1.set_title("X-ray flux: actual (GOES) vs what the model predicted")
        a1.legend(loc="upper left", ncol=2)

        # 2. probability
        if sel.any():
            plot_line(a2, known[sel], p15[sel], color=MODEL_C, lw=1.3, label="Model: P(≥C1 flare within 15 min)")
        a2.axhline(THRESHOLD_C, color=INK2, lw=0.9, ls="--")
        a2.text(1.003, THRESHOLD_C, "alert\nline", transform=a2.get_yaxis_transform(), va="center",
                fontsize=8.5, color=INK2)
        a2.set_ylim(0, 1.02)
        a2.set_ylabel("probability")
        a2.set_title("Flare warning: the model's probability, and when it raised an alert (shaded)")
        if sel.any():
            a2.legend(loc="upper left")

        # 3. HEL1OS
        hsel = (ht + 60.0 >= t0) & (ht + 60.0 < t1) & (hr > 0)
        plot_line(a3, ht[hsel] + 60.0, np.log10(hr[hsel]), color=HXR_C, lw=1.0,
                  label="HEL1OS CdTe 5-20 keV (both detectors)")
        a3.set_ylabel("log₁₀ counts/s")
        a3.set_title("HEL1OS hard X-rays (model input)")
        a3.legend(loc="upper left")

        time_axis(a3, t0, t1, 24 if overview else 3)

        # no-input notes
        if not overview:
            n_fc = int(sel.sum())
            if n_fc < 60:  # under an hour of input: say so rather than show fragments
                for ax in (a1, a2):
                    ax.text(0.5, 0.55, "No SoLEXS file for this day: the model had no input and made no forecast",
                            transform=ax.transAxes, ha="center", fontsize=10.5, color=INK2,
                            bbox={"fc": SURFACE, "ec": GRID, "boxstyle": "round,pad=0.4"})
        # header
        big = max(flares, key=lambda x: x.peak_flux) if flares else None
        n_c = sum(x.goes_class[:1] in "CMX" for x in flares)
        # each alert: on a >=C flare, on a B flare (within 15 min), or on nothing
        kinds = {"C": 0, "B": 0, "none": 0}
        for s, e in c_eps:
            near = [x for x in flares if s - 900 <= x.peak_unix <= e + 900]
            kinds["C" if any(x.goes_class[:1] in "CMX" for x in near) else "B" if near else "none"] += 1
        warned = []
        for x in flares:
            if x.goes_class[:1] in "CMX":
                w = F["c_on"] & (known >= x.start_unix - 900) & (known <= x.peak_unix)
                warned.append(f"{x.goes_class} warned {(x.peak_unix - known[w][0]) / 60:.0f} min before peak"
                              if w.any() else f"{x.goes_class} not warned")
        alerts = (f"{len(c_eps)} ≥C1 alerts ({kinds['C']} on C flares, {kinds['B']} on B flares, "
                  f"{kinds['none']} on no flare)" if c_eps else "no alerts")
        if overview:
            first, lastd = datetime.fromtimestamp(t0, UTC), datetime.fromtimestamp(t1 - 1, UTC)
            head = f"{first:%d}–{lastd:%d %B %Y}: blind forecast vs GOES"
            sub = (f"{sel.sum() / 60:.0f} h of forecasts · GOES flares ≥C1: {n_c} · biggest {big.goes_class if big else '-'}"
                   f" · {alerts}" + (" · " + "; ".join(warned) if warned else ""))
        else:
            day = datetime.fromtimestamp(t0, UTC)
            head = f"{day:%d %B %Y} (UTC day): model vs actual"
            sub = (f"forecasts for {sel.sum() / 60:.1f} h · GOES flares ≥C1: {n_c} · biggest GOES flare "
                   f"{big.goes_class if big else 'none'} · {alerts}" + (" · " + "; ".join(warned) if warned else ""))
            sealed = HERE / f"forecast_{day:%Y-%m-%d}.json"
            if sealed.exists():
                sd = json.loads(sealed.read_text("utf-8"))
                nm = sum(x.goes_class[:1] in "MX" for x in flares)
                sub += (f"\nSealed day-ahead forecast ({sd['sha256'][:8]}): ≥C1 {sd['C']['probability'] * 100:.0f}% → "
                        f"{'happened (' + big.goes_class + ')' if n_c else 'did not happen'} · "
                        f"≥M1 {sd['M']['probability'] * 100:.0f}% → {'happened' if nm else 'did not happen'}")
        fig.suptitle(head, x=0.06, y=0.985, ha="left", fontsize=14, fontweight="bold")
        fig.text(0.06, 0.95, sub, ha="left", va="top", fontsize=10, color=INK2, linespacing=1.5)
        fig.text(0.06, 0.012, "Forecast lines are drawn at the time they were aiming at. Diamonds = GOES flare list "
                 "peaks (dotted guides). The model never saw GOES; it read only SoLEXS and HEL1OS.",
                 fontsize=8.5, color=MUTED)
        fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.1)
        name = "overview_13-19_sep.png" if overview else f"{datetime.fromtimestamp(t0, UTC):%Y-%m-%d}.png"
        dest = fresh(OUT / name)
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        written.append(dest)
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
