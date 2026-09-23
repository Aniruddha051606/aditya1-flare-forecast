"""The two X-ray feature blocks of the E1-E6 matrix, at the day-ahead system's origins.

SoLEXS is the day-ahead system's own "xray" feature set, built by the functions
of solarflare.products.dayahead, imported and called unchanged: E1 is therefore
exactly what that system scores.

The day-ahead system has no HEL1OS features, so the HEL1OS block is new. It is
built the same way, from the catalogue builder's HEL1OS minute light curves
(solarflare.catalog.build.hel1os_minutes: CdTe 5-20 keV and CZT 20-40 keV, each
summed over its two detectors on minutes both observed) and the HEL1OS bursts of
the master catalogue:

* log10 rate of each band now; its 1/6/24 h minimum, maximum and mean; its 1 h
  and 6 h change -- the SoLEXS flux features, band by band;
* background hardness: the 1 h median of log10(CZT 20-40 / CdTe 5-20);
* catalogue HEL1OS bursts, known 5 min after their hard X-ray peak (the SoLEXS
  convention): how many in the last 6/24/72 h, how many with a CZT 20-40 keV
  excess (non-thermal) in the last 24/72 h, hours since the last one, and the
  largest CZT 20-40 keV peak of the last 24/72 h.

Nothing uses GOES: HEL1OS rates stay in counts/s, and the catalogue's
GOES-matched columns are never read. Every value is known at its origin: a
minute counts from its end, a burst from 5 min after its peak.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LOG_FLOOR = 1e-2          # counts/s: a zero-rate minute stays finite in log
BANDS = ("cdte_5_20", "czt_20_40")


def _minutes_frame(minutes: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    cols = {}
    for band in BANDS:
        t, r = minutes.get(band, (np.zeros(0), np.zeros(0)))
        idx = pd.to_datetime(np.asarray(t, dtype=np.float64) + 60.0, unit="s", utc=True)   # known when it ends
        ser = pd.Series(np.log10(np.maximum(np.asarray(r, dtype=np.float64), LOG_FLOOR)), index=idx).sort_index()
        # a minute seen twice (overlapping products carry the same telemetry): once
        cols[band] = ser[~ser.index.duplicated(keep="last")]
    return pd.DataFrame(cols).sort_index()


def hel1os_block(minutes: dict[str, tuple[np.ndarray, np.ndarray]], bursts: dict[str, np.ndarray],
                 o: np.ndarray, min_coverage: float = 0.5) -> tuple[pd.DataFrame, np.ndarray]:
    """(features, observed) at origins ``o`` (unix s). ``bursts``: arrays ``known``
    (unix s the burst became known) and ``czt`` (its CZT 20-40 keV peak, counts/s,
    0 when it had none), sorted by ``known``. ``observed``: HEL1OS covered at
    least ``min_coverage`` of the previous 6 h in either band."""
    s = _minutes_frame(minutes)
    origins = pd.to_datetime(o, unit="s", utc=True)
    feat = pd.DataFrame(index=origins)
    cov = np.zeros(o.size)

    def roll(col, win, how):
        r_ = getattr(s[col].rolling(win, min_periods=max(1, int(pd.Timedelta(win).total_seconds() / 60 * 0.3))), how)()
        return r_.reindex(origins, method="ffill", tolerance=pd.Timedelta("2min")).to_numpy()

    for band in BANDS:
        x = s[band].dropna()
        if x.empty:
            for k in ("now", "min_1h", "max_1h", "mean_1h", "min_6h", "max_6h", "mean_6h",
                      "min_24h", "max_24h", "mean_24h", "change_1h", "change_6h"):
                feat[f"{band}_{k}"] = np.nan
            continue
        now = x.reindex(origins, method="ffill", tolerance=pd.Timedelta("5min")).to_numpy()
        feat[f"{band}_now"] = now
        for w in ("1h", "6h", "24h"):
            for how in ("min", "max", "mean"):
                feat[f"{band}_{how}_{w}"] = roll(band, w, how)
        feat[f"{band}_change_1h"] = now - x.reindex(origins - pd.Timedelta("1h"), method="ffill",
                                                    tolerance=pd.Timedelta("5min")).to_numpy()
        feat[f"{band}_change_6h"] = feat[f"{band}_mean_1h"].to_numpy() - x.rolling("1h", min_periods=10).mean().reindex(
            origins - pd.Timedelta("6h"), method="ffill", tolerance=pd.Timedelta("5min")).to_numpy()
        c6 = x.notna().astype(float).rolling("6h").sum().reindex(origins, method="ffill",
                                                                  tolerance=pd.Timedelta("2min")).to_numpy() / 360.0
        cov = np.fmax(cov, np.nan_to_num(c6))
    hard = (s["czt_20_40"] - s["cdte_5_20"]).rolling("1h", min_periods=18).median()
    feat["hardness_background_1h"] = hard.reindex(origins, method="ffill", tolerance=pd.Timedelta("2min")).to_numpy()

    known = np.asarray(bursts.get("known", np.zeros(0)), dtype=np.float64)
    czt = np.asarray(bursts.get("czt", np.zeros(0)), dtype=np.float64)
    nt = known[czt > 0]
    for h in (6, 24, 72):
        feat[f"n_bursts_{h}h"] = np.searchsorted(known, o, side="right") - np.searchsorted(known, o - 3600.0 * h, side="right")
    for h in (24, 72):
        feat[f"n_nonthermal_{h}h"] = np.searchsorted(nt, o, side="right") - np.searchsorted(nt, o - 3600.0 * h, side="right")
    j = np.searchsorted(known, o, side="right") - 1
    feat["hours_since_burst"] = np.where(j >= 0, np.minimum((o - known[np.maximum(j, 0)]) / 3600.0, 168.0), 168.0)
    for h in (24, 72):
        mx = np.full(o.size, -1.0)
        for i, t_ in enumerate(o):
            a, b = np.searchsorted(known, [t_ - 3600.0 * h, t_], side="right")
            if b > a and czt[a:b].max() > 0:
                mx[i] = np.log10(czt[a:b].max())
        feat[f"max_burst_czt_{h}h"] = mx
    return feat, cov >= min_coverage


def catalogue_bursts(path: Path) -> dict[str, np.ndarray]:
    """HEL1OS bursts of the master catalogue: known 5 min after the hard X-ray
    peak, with their CZT 20-40 keV peak (0 when none). GOES columns are not read."""
    from solarflare.util import read_rows, ts

    rows = [r for r in read_rows(path) if r["origin"] in ("hard", "soft+hard") and r["hard_peak_utc"]]
    known = np.array([ts(r["hard_peak_utc"]) + 300.0 for r in rows], dtype=np.float64)
    czt = np.array([float(r["hard_peak_czt_20_40_cps"] or 0.0) for r in rows], dtype=np.float64)
    order = np.argsort(known)
    return {"known": known[order], "czt": czt[order]}


def solexs_block(cache_dir: Path, catalog: Path, truth, train_end: float):
    """The day-ahead system's SoLEXS features, by its own functions: (features,
    observed, origins). The flux scale is fitted on training-period minutes only,
    as there."""
    from solarflare.products import dayahead as D

    tm, rm, _, vm, hm, vh = D.solexs_minutes(cache_dir)
    cal = D.fit_calibration(tm, rm, vm, truth, train_end)
    lf, hard = D.flux_series(cal, rm, vm, hm, vh)
    known, pf = D.catalogue_flares(catalog)
    return D.activity_features(tm, lf, hard, known, pf)
