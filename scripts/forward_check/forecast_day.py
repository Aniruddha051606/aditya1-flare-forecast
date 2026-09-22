"""Sealed day-ahead forecast for one UTC day, issued before its data exist here.

    python forecast_day.py --day 2026-09-19

Same features and model families as solarflare.products.dayahead (SoLEXS
activity, hourly origins), but the target is the whole of 20 Sep: a GOES flare
>= C1 / >= M1 peaking 25-49 h after the origin (origin = 18 Sep 23:00 UTC, the
last full hour of SoLEXS on disk).

Blindness: GOES truth comes only from the pipeline's GOES folder (ends
2026-09-13). The newer GOES download (D:/Data/goes_2026-09-22), which holds
20 Sep, is not opened here.

Writes forecast_20sep.json (with a SHA-256 of its content) and
png/forecast_2026-09-20.png.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone, UTC
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "outputs" / "_dev" / "forward_check"   # working folder: data links, cache, PNGs, seals
HERE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from solarflare.catalog.build import load_solexs  # noqa: E402
from solarflare.catalog.detect import PiecewiseCalibration, noaa_events, to_minutes  # noqa: E402
from solarflare.io.goes import load_goes  # noqa: E402
from solarflare.products import dayahead as da  # noqa: E402
from solarflare.settings import load_settings  # noqa: E402
from solarflare.util import read_rows, ts  # noqa: E402

UTC = UTC
IST = timezone(timedelta(hours=5, minutes=30))
DAY0 = float("nan")                              # set in main() from --day
LEAD0 = LEAD1 = float("nan")                     # hours from the origin to the start / end of that day
NEW_FROM = datetime(2026, 9, 13, tzinfo=UTC).timestamp()


def features(tm, lf, hard, known, pf, origins):
    """solarflare.products.dayahead.xray_features, feature part, on given series."""
    idx = pd.to_datetime(tm + 60.0, unit="s", utc=True)
    s = pd.DataFrame({"f": lf, "hard": hard}, index=idx)
    feat = pd.DataFrame(index=origins)

    def roll(col, win, how):
        r_ = getattr(s[col].rolling(win, min_periods=max(1, int(pd.Timedelta(win).total_seconds() / 60 * 0.3))), how)()
        return r_.reindex(origins, method="ffill", tolerance=pd.Timedelta("2min"))

    feat["flux_now"] = s["f"].reindex(origins, method="ffill", tolerance=pd.Timedelta("5min"))
    for w in ("1h", "6h", "24h"):
        feat[f"flux_min_{w}"] = roll("f", w, "min")
        feat[f"flux_max_{w}"] = roll("f", w, "max")
        feat[f"flux_mean_{w}"] = roll("f", w, "mean")
    feat["change_1h"] = feat["flux_now"] - s["f"].reindex(origins - pd.Timedelta("1h"), method="ffill",
                                                          tolerance=pd.Timedelta("5min")).to_numpy()
    feat["change_6h"] = feat["flux_mean_1h"] - s["f"].rolling("1h", min_periods=10).mean().reindex(
        origins - pd.Timedelta("6h"), method="ffill", tolerance=pd.Timedelta("5min")).to_numpy()
    feat["hardness_background_1h"] = roll("hard", "1h", "median")
    cov6 = s["f"].notna().astype(float).rolling("6h").sum().reindex(origins, method="ffill",
                                                                     tolerance=pd.Timedelta("2min")) / 360.0
    o = ((origins - pd.Timestamp(0, tz="UTC")) / pd.Timedelta("1s")).to_numpy(dtype=np.float64)
    for lab, lo in (("B", 1e-7), ("C", 1e-6), ("M", 1e-5)):
        kk = known[pf >= lo]
        for h in (6, 24, 72):
            feat[f"n_{lab}_{h}h"] = np.searchsorted(kk, o, side="right") - np.searchsorted(kk, o - 3600.0 * h, side="right")
        if lab in ("C", "M"):
            j = np.searchsorted(kk, o, side="right") - 1
            feat[f"hours_since_{lab}"] = np.where(j >= 0, np.minimum((o - kk[np.maximum(j, 0)]) / 3600.0, 168.0), 168.0)
    for h in (24, 72):
        mx = np.full(o.size, -8.0)
        for i, t_ in enumerate(o):
            a, b = np.searchsorted(known, [t_ - 3600.0 * h, t_], side="right")
            if b > a:
                mx[i] = np.log10(pf[a:b].max())
        feat[f"max_flare_{h}h"] = mx
    return feat, cov6.to_numpy() >= 0.5, o


def cls(f: float) -> str:
    for letter, base in (("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7), ("A", 1e-8)):
        if f >= base:
            return f"{letter}{f / base:.1f}"
    return "A0.0"


def main() -> int:
    import argparse

    global DAY0, LEAD0, LEAD1
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True, help="UTC day to forecast, YYYY-MM-DD")
    day = ap.parse_args().day
    DAY0 = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC).timestamp()
    sealed_files = [HERE / f"forecast_{day}.json", HERE / "png" / f"forecast_{day}_sealed.png"]
    if day == "2026-09-20":
        sealed_files.append(HERE / "forecast_20sep.json")
    if any(p.exists() for p in sealed_files):
        print(f"{day} is already sealed ({', '.join(p.name for p in sealed_files if p.exists())}); "
              "a sealed forecast is never re-issued.")
        return 1
    S = load_settings()
    split = S.split_dates(None)
    da.TRAIN_END, da.TEST_START = split["train_end"], split["test_start"]
    train_end, test_start = da.TRAIN_END, da.TEST_START
    goes_dir = Path(S.goes_dir)                       # ends 2026-09-13: never sees 20 Sep
    truth = load_goes(goes_dir)
    # yearly files are padded to 31 Dec: check the last real measurement and flare
    last_real = truth.time_unix[np.isfinite(truth.xrsb)][-1]
    assert last_real < DAY0 - 86400 and max(f.peak_unix for f in truth.flares) < DAY0 - 86400, \
        "GOES truth must end before the forecast day"

    # SoLEXS minutes: main cache (to 12 Sep) + forecast workspace (13-18 Sep)
    _, t, r, hi, ok = load_solexs(Path(S.cache))
    tm, rm, cm, vm = to_minutes(t, r, ok, 20.0)
    _, hm, _, vh = to_minutes(t, hi, ok, 20.0)
    gmin = truth.flux_on_grid(tm)
    fit_on = vm & (tm <= train_end) & np.isfinite(gmin) & (gmin > 0) & (rm > 0)
    cal = PiecewiseCalibration.fit(np.log10(rm[fit_on]), np.log10(gmin[fit_on]))
    _, t2, r2, hi2, ok2 = load_solexs(HERE / "cache")
    tn, rn, cn, vn = to_minutes(t2, r2, ok2, 20.0)
    _, hn, _, vhn = to_minutes(t2, hi2, ok2, 20.0)
    keep_main, keep_new = tm < NEW_FROM, tn >= NEW_FROM
    # one contiguous minute grid
    gap = np.arange(tm[keep_main][-1] + 60.0, tn[keep_new][0], 60.0)
    T = np.concatenate([tm[keep_main], gap, tn[keep_new]])
    R = np.concatenate([rm[keep_main], np.full(gap.size, np.nan), rn[keep_new]])
    Hh = np.concatenate([hm[keep_main], np.full(gap.size, np.nan), hn[keep_new]])
    V = np.concatenate([vm[keep_main], np.zeros(gap.size, bool), vn[keep_new]])
    VH = np.concatenate([vh[keep_main], np.zeros(gap.size, bool), vhn[keep_new]])
    with np.errstate(divide="ignore", invalid="ignore"):
        lf = np.log10(cal(np.where(V, R, np.nan)))
        hard = np.where(V & VH & (R > 0), Hh / R, np.nan)

    # flares: the master catalogue to 12 Sep, the same SoLEXS rule on 13-18 Sep
    cat = [r_ for r_ in read_rows(S.catalog / "master_catalog.csv")
           if r_["origin"] in ("soft", "soft+hard") and r_["peak_flux_solexs_Wm2"]]
    known = [ts(r_["peak_utc"]) + 300.0 for r_ in cat if ts(r_["peak_utc"]) < NEW_FROM]
    pf = [float(r_["peak_flux_solexs_Wm2"]) for r_ in cat if ts(r_["peak_utc"]) < NEW_FROM]
    with np.errstate(divide="ignore", invalid="ignore"):
        new_events = [e for e in noaa_events(tn, cal(np.where(vn, rn, np.nan)), vn, counts=cn)
                      if e.peak_unix >= NEW_FROM]
    known += [e.peak_unix + 300.0 for e in new_events]
    pf += [e.peak_flux for e in new_events]
    order = np.argsort(known)
    known, pf = np.asarray(known)[order], np.asarray(pf)[order]

    idx = pd.to_datetime(T + 60.0, unit="s", utc=True)
    origins = pd.date_range(idx[0].ceil("h"), idx[-1].floor("h"), freq="h")
    feat, have, o = features(T, lf, hard, known, pf, origins)

    # check: identical to the dayahead product's features where both exist
    ref, _, o_ref, _ = da.xray_features(SimpleNamespace(goes_dir=str(goes_dir), cache_dir=str(S.cache),
                                                         catalog=str(S.catalog / "master_catalog.csv")))
    common = np.intersect1d(o_ref[o_ref < NEW_FROM - 4 * 86400], o)
    a_ = ref.loc[pd.to_datetime(common, unit="s", utc=True)].to_numpy(float)
    b_ = feat.loc[pd.to_datetime(common, unit="s", utc=True)].to_numpy(float)
    same = np.allclose(np.nan_to_num(a_, nan=-99), np.nan_to_num(b_, nan=-99), atol=1e-9)
    print(f"feature check vs dayahead product on {common.size} origins: {'identical' if same else 'DIFFERENT'}")
    if not same:
        return 1

    # targets: GOES flare peaking 25-49 h after the origin
    fl = [f for f in truth.flares if f.goes_class[:1] in "CMX"]
    gp, gf = np.array([f.peak_unix for f in fl]), np.array([f.peak_flux for f in fl])
    gv = np.isfinite(truth.xrsb).astype(float)
    gcs = np.concatenate([[0.0], np.cumsum(gv)])
    gt = truth.time_unix
    X = feat.to_numpy(float)
    finite = np.isfinite(X).all(1)
    weeks = np.floor(o / (7 * 86400)).astype(int)
    last = int(np.flatnonzero(have & finite & (o < DAY0))[-1])
    origin = o[last]
    LEAD0 = (DAY0 - origin) / 3600.0
    LEAD1 = LEAD0 + 24.0
    print(f"origin {datetime.fromtimestamp(origin, UTC):%Y-%m-%d %H:%M} UTC; {day} is {LEAD0:.0f}-{LEAD1:.0f} h ahead")
    result = {}
    for c, lo in (("C", 1e-6), ("M", 1e-5)):
        pk = np.sort(gp[gf >= lo])
        n = np.searchsorted(pk, o + 3600.0 * LEAD1, side="right") - np.searchsorted(pk, o + 3600.0 * LEAD0, side="right")
        a, b = np.searchsorted(gt, o + 3600.0 * LEAD0), np.searchsorted(gt, o + 3600.0 * LEAD1)
        gok = (gcs[b] - gcs[a]) / (60.0 * (LEAD1 - LEAD0)) >= 0.8
        y = (n > 0).astype(float)
        base = have & gok & finite
        tr = base & (o + 3600.0 * LEAD1 <= train_end)
        va = base & (o > train_end) & (o + 3600.0 * LEAD1 <= test_start)
        te = base & (o >= test_start)
        clim = float(y[tr].mean())
        res, (_, pt) = da.fit_score(X, y, tr, va, te, weeks[te], clim)
        # persistence reference on the same hours
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        pj = list(feat.columns).index(f"n_{c}_24h")
        pm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr][:, [pj]], y[tr])
        pers_auc = float(roc_auc_score(y[te], pm.predict_proba(X[te][:, [pj]])[:, 1]))
        # the operational fit: the family chosen on validation, refitted on every labelled hour
        m = da._models()[res["selected"]]
        allh = tr | va | te
        m.fit(X[allh], y[allh])
        p = float(m.predict_proba(X[[last]])[0, 1])
        recent = base & (o >= o[allh].max() - 30 * 86400)
        result[c] = {"probability": round(p, 3), "threshold_TSS": res["threshold"],
                     "alert": bool(p >= res["threshold"]), "model": res["selected"],
                     "test_AUC": res["AUC"], "test_AUC_ci": res["AUC_ci"], "test_TSS": res["TSS"],
                     "test_BSS_vs_climatology": res["BSS_vs_training_climatology"],
                     "persistence_test_AUC": round(pers_auc, 3),
                     "climatology_train": round(clim, 3), "rate_last_30_days": round(float(y[recent].mean()), 3),
                     "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
                     "n_fit_operational": int(allh.sum())}
        print(f">= {c}1 on {day}: P = {p:.3f} ({res['selected']}; test AUC {res['AUC']} {res['AUC_ci']}, "
              f"TSS {res['TSS']}, persistence AUC {pers_auc:.3f}; climatology {clim:.2f}, last 30 d "
              f"{y[recent].mean():.2f})")

    f_last = feat.iloc[last]
    sealed = {
        "forecast_for": f"{day} 00:00-24:00 UTC (05:30 IST that day to 05:30 IST the next)",
        "issued_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        "origin_utc": datetime.fromtimestamp(origin, UTC).strftime("%Y-%m-%d %H:%M"),
        "data_used": "SoLEXS to 2026-09-18 23:59 UTC; GOES labels to 2026-09-13 (pipeline folder only)",
        "lead_hours": [LEAD0, LEAD1],
        "features_at_origin": {k: (None if not np.isfinite(v) else round(float(v), 4)) for k, v in f_last.items()},
        "new_solexs_flares_13_18_sep": [[datetime.fromtimestamp(e.peak_unix, UTC).strftime("%m-%d %H:%M"),
                                         cls(e.peak_flux)] for e in new_events],
        "C": result["C"], "M": result["M"],
    }
    body = json.dumps(sealed, indent=2, sort_keys=True)
    digest = hashlib.sha256(body.encode()).hexdigest()
    (HERE / f"forecast_{day}.json").write_text(json.dumps({"sha256": digest, **sealed}, indent=2), encoding="utf-8")
    print("sealed sha256", digest[:16])
    figure(sealed, digest, T, lf, new_events, origin, day)
    return 0


def figure(s, digest, T, lf, events, origin, day):
    dname = datetime.fromtimestamp(DAY0, UTC).strftime("%d %b").lstrip("0")
    SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9994", "#e7e6e1"
    ADITYA, FC = "#1baf7a", "#eb6834"
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.labelcolor": INK2, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig = plt.figure(figsize=(13, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25], hspace=0.42, wspace=0.28,
                          left=0.07, right=0.95, top=0.83, bottom=0.08)
    t0 = datetime(2026, 9, 13, tzinfo=UTC).timestamp()
    t1 = DAY0 + 86400.0

    # top: the probabilities, as the headline
    for k, (c, name) in enumerate((("C", f"≥C1 flare on {dname}"), ("M", f"≥M1 flare on {dname}"))):
        ax = fig.add_subplot(gs[0, k])
        ax.axis("off")
        r = s[c]
        ax.text(0, 0.95, name, fontsize=12, fontweight="bold", va="top", transform=ax.transAxes)
        ax.text(0, 0.62, f"{r['probability'] * 100:.0f}%", fontsize=40, color=FC, va="center",
                transform=ax.transAxes, family="monospace")
        ax.text(0, 0.30, f"usual rate (training period): {r['climatology_train'] * 100:.0f}%\n"
                         f"last 30 days with GOES: {r['rate_last_30_days'] * 100:.0f}%", fontsize=9.5,
                color=INK2, va="top", transform=ax.transAxes)
        ax.text(0, -0.04, f"skill on the test period: AUC {r['test_AUC']} "
                          f"(persistence {r['persistence_test_AUC']}), TSS {r['test_TSS']}", fontsize=8.8,
                color=MUTED, va="top", transform=ax.transAxes)
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.text(0, 0.95, "How this was made", fontsize=12, fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(0, 0.78, "Day-ahead model (SoLEXS activity, logistic /\n"
                     "boosted trees chosen on validation), fitted\n"
                     f"for flares {s['lead_hours'][0]:.0f}–{s['lead_hours'][1]:.0f} h after the last data hour.\n"
                     f"Last data: {s['origin_utc']} UTC.\n"
                     f"GOES for {dname} was not opened.\n"
                     f"Sealed {s['issued_utc']} UTC\nsha256 {digest[:16]}",
            fontsize=9.2, color=INK2, va="top", transform=ax.transAxes, linespacing=1.45)

    # bottom: the week before, and the forecast day
    ax = fig.add_subplot(gs[1, :])
    m = (t0 <= T) & (origin + 3600 > T)
    tt = T[m] + 60.0
    br = np.flatnonzero(np.diff(tt) > 180.0)
    x = np.insert(tt, br + 1, np.nan)
    y = np.insert(lf[m], br + 1, np.nan)
    xd = np.array([np.datetime64(int(v), "s") if np.isfinite(v) else np.datetime64("NaT", "s") for v in x])
    ax.plot(xd, y, color=ADITYA, lw=0.9, label="Aditya SoLEXS flux (GOES scale): what the forecast was made from")
    for e in events:
        if e.peak_flux >= 5e-7:
            ax.annotate(cls(e.peak_flux), (np.datetime64(int(e.peak_unix), "s"), np.log10(e.peak_flux)),
                        xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    gap0, d20 = np.ceil(origin / 86400.0) * 86400.0, DAY0      # days between the data and the forecast day
    if d20 > gap0:
        ax.axvspan(np.datetime64(int(gap0), "s"), np.datetime64(int(d20), "s"), color=GRID, alpha=0.7, lw=0)
        ax.text(np.datetime64(int((gap0 + d20) / 2), "s"), -5.45, "no data\nyet", ha="center", va="center",
                fontsize=9, color=INK2)
    ax.axvspan(np.datetime64(int(d20), "s"), np.datetime64(int(d20 + 86400), "s"), color=FC, alpha=0.12, lw=0)
    ax.text(np.datetime64(int(d20 + 43200), "s"), -5.45,
            f"{dname}\nforecast\n≥C1 {s['C']['probability'] * 100:.0f}%\n≥M1 {s['M']['probability'] * 100:.0f}%",
            ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold")
    for lvl, name in ((-6.0, "C1"), (-5.0, "M1")):
        ax.axhline(lvl, color=MUTED, lw=0.9, ls="--")
        ax.text(1.003, lvl, name, transform=ax.get_yaxis_transform(), va="center", fontsize=9, color=INK2)
    ax.set_ylim(-7.3, -4.8)
    ax.set_yticks([-7, -6, -5])
    ax.set_yticklabels(["B1", "C1", "M1"])
    ax.set_ylabel("X-ray flux")
    ticks = np.arange(t0, t1 + 1, 86400.0)
    ax.set_xticks([np.datetime64(int(v), "s") for v in ticks])
    ax.set_xticklabels([f"{datetime.fromtimestamp(v, UTC):%d %b}" for v in ticks])
    ax.set_xlim(np.datetime64(int(t0), "s"), np.datetime64(int(t1), "s"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("The week before (UTC days), and the forecast day", loc="left", fontsize=11, fontweight="bold")

    nxt = datetime.fromtimestamp(DAY0 + 86400, UTC).strftime("%d %b").lstrip("0")
    fig.suptitle(f"Forecast for {datetime.fromtimestamp(DAY0, UTC):%d %B %Y}: will the Sun flare?".lstrip("0"),
                 x=0.07, y=0.975, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.07, 0.905, f"Issued before any {dname} data were loaded. {dname} (UTC) = 05:30 IST on {dname} to "
                          f"05:30 IST on {nxt}.\nThe minute-by-minute warnings follow once that day's SoLEXS and "
                          "HEL1OS are uploaded.", fontsize=9.8, color=INK2, linespacing=1.4)
    out = HERE / "png" / f"forecast_{day}_sealed.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    raise SystemExit(main())
