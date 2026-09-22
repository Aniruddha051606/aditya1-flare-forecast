"""Blind forecast over the days after the scored test period (no GOES answers yet).

Runs the frozen model minute by minute, exactly as solarflare.products.leadtime
does, but keeps segments that GOES does not cover, then applies the frozen
alert rules (outputs/alerts/alert_rules.json). Writes forecast.npz and
warnings.csv next to this file.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timedelta, timezone, UTC
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "outputs" / "_dev" / "forward_check"   # working folder: data links, cache, PNGs, seals
HERE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402

from solarflare import probcal  # noqa: E402
from solarflare.catalog.build import load_solexs  # noqa: E402
from solarflare.catalog.detect import PiecewiseCalibration, to_minutes  # noqa: E402
from solarflare.config import Config  # noqa: E402
from solarflare.forward import _segments, load_frozen  # noqa: E402
from solarflare.io.goes import load_goes  # noqa: E402
from solarflare.settings import load_settings  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
T_FROM = datetime(2026, 9, 13, tzinfo=UTC).timestamp()


def utc(x: float) -> str:
    return datetime.fromtimestamp(float(x), UTC).strftime("%Y-%m-%d %H:%M")


def ist(x: float) -> str:
    return datetime.fromtimestamp(float(x), IST).strftime("%d %b %H:%M")


def goes_class(logf: float) -> str:
    if not np.isfinite(logf):
        return "?"
    f = 10.0 ** logf
    for letter, base in (("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7), ("A", 1e-8)):
        if f >= base:
            return f"{letter}{f / base:.1f}"
    return "A0.0"


def episodes(on: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def main() -> int:
    S = load_settings()
    rules = json.loads((S.alerts / "alert_rules.json").read_text("utf-8"))
    train_end = float(S.split_dates(Path(S.model_dir))["train_end"])

    # A small workspace of hard links (data/) with its own cache, so the
    # finished run's cache and data root are left exactly as they were.
    cfg = Config(data_root=HERE / "data", out_dir=Path(S.model_dir))
    cfg.cache_dir = HERE / "cache"
    frozen, model, norm, device = load_frozen(Path(S.frozen_dir), cfg)
    cfg.pre.goes_dir = str(S.goes_dir)
    dt, L = cfg.pre.dt_seconds, cfg.steps_per_window
    stride = max(int(round(60.0 / dt)), 1)
    t_to = np.inf      # days after the freeze cutoff (18 Sep) are the forward test itself
    t0 = time.time()
    segments, _ = _segments(cfg, verbose=False)

    rec = {k: [] for k in ("origin", "p_now", "p_occ", "now", "fc", "hard_frac")}
    with torch.no_grad():
        for seg in segments:
            if len(seg) < L or seg.time_unix[-1] < T_FROM or seg.time_unix[0] > t_to:
                continue
            avail = np.maximum(seg.soft_mask, seg.hard_mask)
            csum = np.concatenate([[0.0], np.cumsum(avail)])
            hsum = np.concatenate([[0.0], np.cumsum(seg.hard_mask)])
            ends = np.arange(L, len(seg) + 1)
            origin = seg.time_unix[ends - 1]
            keep = ((origin >= T_FROM) & (origin <= t_to)
                    & (np.rint(origin / dt).astype(np.int64) % stride == 0)
                    & (seg.soft_mask[ends - 1] > 0)
                    & ((csum[ends] - csum[ends - L]) / L >= cfg.win.min_observed_fraction))
            ends = ends[keep]
            if ends.size == 0:
                continue
            soft_n = np.nan_to_num(norm.apply_soft(seg.soft), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            hard_n = np.nan_to_num(norm.apply_hard(seg.hard), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            views = {"soft": sliding_window_view(soft_n, L, axis=0),
                     "hard": sliding_window_view(hard_n, L, axis=0),
                     "clock": sliding_window_view(seg.clock.astype(np.float32), L, axis=0),
                     "ms": sliding_window_view(seg.soft_mask.astype(np.float32), L),
                     "mh": sliding_window_view(seg.hard_mask.astype(np.float32), L)}
            for b0 in range(0, ends.size, 512):
                s = ends[b0:b0 + 512] - L

                def tens(k, _s=s, _v=views):
                    a = _v[k][_s]
                    if a.ndim == 3:
                        a = a.transpose(0, 2, 1)
                    return torch.from_numpy(np.ascontiguousarray(a)).to(device)

                o = model(tens("soft"), tens("ms"), tens("hard"), tens("mh"), tens("clock"))
                rec["p_now"].append(torch.sigmoid(o["in_flare"]).cpu().numpy())
                p_occ = torch.sigmoid(o["occurrence"]).cpu().numpy()
                cal = frozen.get("calibration")
                if cal:
                    p_occ = np.column_stack([probcal.apply(cal["occurrence"][h], p_occ[:, h])
                                             for h in range(p_occ.shape[1])])
                rec["p_occ"].append(p_occ)
                rec["now"].append(o["nowcast"].cpu().numpy())
                rec["fc"].append(o["forecast"].cpu().numpy())
            rec["origin"].append(seg.time_unix[ends - 1])
            rec["hard_frac"].append(((hsum[ends] - hsum[ends - L]) / L).astype(np.float32))
    P = {k: np.concatenate(v) for k, v in rec.items()}
    order = np.argsort(P["origin"], kind="stable")
    P = {k: v[order] for k, v in P.items()}
    print(f"{P['origin'].size} minute forecasts {utc(P['origin'][0])} -> {utc(P['origin'][-1])} UTC "
          f"on {device} in {time.time() - t0:.0f} s")

    # SoLEXS flux now, GOES-equivalent: the same piecewise calibration the alert
    # scorer fits on training minutes only.
    truth = load_goes(Path(S.goes_dir))
    _, t, r, _, ok = load_solexs(Path(S.cache))      # main cache, read only: training minutes
    tm, rm, _, vm = to_minutes(t, r, ok, 20.0)
    gmin = truth.flux_on_grid(tm)
    fit_on = vm & (tm <= train_end) & np.isfinite(gmin) & (gmin > 0) & (rm > 0)
    calib = PiecewiseCalibration.fit(np.log10(rm[fit_on]), np.log10(gmin[fit_on]))
    _, t, r, _, ok = load_solexs(cfg.cache_dir)       # the forecast days
    tm, rm, _, vm = to_minutes(t, r, ok, 20.0)
    known = P["origin"] + 60.0                       # a forecast is usable a minute after its origin
    idx = np.searchsorted(tm + 60.0, known)
    idx = np.clip(idx, 0, tm.size - 1)
    same = np.abs(tm[idx] + 60.0 - known) < 30.0
    with np.errstate(divide="ignore", invalid="ignore"):
        f_now = np.where(same & vm[idx], np.log10(calib(np.where(rm[idx] > 0, rm[idx], np.nan))), np.nan)

    hs = list(np.asarray(cfg.win.forecast_horizons_s))
    q50 = int(np.argmin(np.abs(np.asarray(cfg.win.quantiles) - 0.5)))
    m_net = np.max(P["fc"][:, [hs.index(h) for h in (300.0, 900.0, 1800.0)], q50], axis=1)
    m_sig = np.fmax(m_net, f_now)
    c_on = P["p_occ"][:, 0] >= rules["C"]["threshold"]
    m_on = m_sig >= rules["M"]["threshold"]

    np.savez_compressed(HERE / "forecast.npz", **P, f_now=f_now, m_signal=m_sig, c_on=c_on, m_on=m_on)

    # Warnings: merge alert minutes into episodes (gaps of <= 2 min joined).
    rows = []
    for kind, on in (("C", c_on), ("M", m_on)):
        for a, b in episodes(on):
            if rows and rows[-1]["kind"] == kind and known[a] - rows[-1]["_end"] <= 180.0:
                rows[-1]["_end"] = known[b - 1]
                rows[-1]["_b"] = b
                continue
            rows.append({"kind": kind, "_a": a, "_b": b, "_start": known[a], "_end": known[b - 1]})
    out = []
    for w in rows:
        a, b = w["_a"], w["_b"]
        after = (known >= w["_start"]) & (known <= w["_start"] + 3600.0)
        peak_after = np.nanmax(np.where(after, f_now, -np.inf)) if after.any() else np.nan
        out.append({
            "alert": ">=C1 within 15 min" if w["kind"] == "C" else "flux reaches M1 within 30 min",
            "start_utc": utc(w["_start"]), "start_ist": ist(w["_start"]),
            "minutes_on": int(round((w["_end"] - w["_start"]) / 60.0)) + 1,
            "max_p_c": round(float(np.max(P["p_occ"][a:b, 0])), 3),
            "max_m_signal_class": goes_class(float(np.nanmax(m_sig[a:b]))),
            "aditya_flux_at_start": goes_class(float(f_now[a])),
            "aditya_peak_next_60min": goes_class(float(peak_after)),
        })
    with open(HERE / "warnings.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0]) if out else ["alert"])
        wr.writeheader()
        wr.writerows(out)

    days = (P["origin"][-1] - P["origin"][0]) / 86400.0
    print(f"coverage: {P['origin'].size / 1440:.1f} observed days over {days:.1f} calendar days; "
          f"HEL1OS in {np.mean(P['hard_frac'] > 0) * 100:.0f}% of windows")
    print(f"C warnings: {sum(r['alert'].startswith('>=C1') for r in out)}, "
          f"M warnings: {sum(not r['alert'].startswith('>=C1') for r in out)}")
    for r in out:
        print(f"  {r['alert']:32s} {r['start_utc']} UTC ({r['start_ist']} IST)  on {r['minutes_on']:4d} min  "
              f"P(C)max {r['max_p_c']:.2f}  flux then {r['aditya_flux_at_start']:>5s}  "
              f"Aditya peak next hour {r['aditya_peak_next_60min']:>5s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
