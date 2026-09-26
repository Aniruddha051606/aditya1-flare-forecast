"""Paper step 16: which flares gain from HEL1OS? (pre-registered physics analysis)

    python scripts/paper/physics_interpretation.py

Implements paper_results/physics/00_preregistration.md exactly; that file was
committed before any per-flare E4-vs-E1 difference existed, and its SHA-256 and
commit are recorded in physics_provenance.json. Per test-period GOES flare
(>= C1.0) it measures how much earlier E4 (SoLEXS + HEL1OS) warns than E1
(SoLEXS only), dL = L_E4 - L_E1, from the saved predictions at the validation
thresholds, and relates dL to hard-X-ray timing, strength, spectrum and Neupert
behaviour with day-block bootstrap intervals and permutation tests.

Reads outputs/paper/predictions/, the GOES flare list, outputs/catalog/
master_catalog.csv and outputs/physics/{hxr_spectra,flare_catalog_dec120s}.csv.
Nothing is trained or tuned. Writes paper_results/physics/:
flare_level_comparison.csv, hxr_timing_features.csv, hxr_spectral_features.csv,
neupert_features.csv, physics_gain_association.csv, physics_provenance.json,
README.md.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import UTC, datetime

import numpy as np
from scipy.stats import rankdata

from common import PAPER, ROOT, stamp, utc  # noqa: E402  (sets sys.path)
from paper_metrics import fit_on_validation
from solarflare import probcal
from solarflare.io.goes import class_flux, load_goes
from solarflare.metrics import day_block_ci
from solarflare.settings import load_settings

S = load_settings()
PRED = S.outputs / "paper" / "predictions"
OUT = PAPER / "physics"
PREREG = OUT / "00_preregistration.md"

# --- fixed by the pre-registration -------------------------------------------
MIN_CLASS = "C1.0"
PRIMARY_HEAD, HEADS = 60, (60, 30, 15)
LOOKBACK_S = 3600.0          # opportunity window starts at most 60 min before the GOES start
MIN_LEAD_IN_S = 600.0        # ... and at least 10 min before it
MIN_COVERAGE = 0.8           # of the expected origins in [a, p], in the common population
CONF_S, MIN_CONF_ORIGINS = 1800.0, 5
EARLIER_MIN = 2.0            # "E4 earlier" = dL >= one origin step
N_BOOT, N_PERM, SEED, ALPHA, MIN_N = 2000, 10000, 0, 0.05, 30
#: Neupert features follow the onset study's own rule (products/onset_study.neupert_stats)
NEUPERT_MIN_SIGMA = 5.0
PRIMARY = {"P1_hard_minus_soft_alert_min": -1, "P2_log10_czt_20_40_peak_cps": +1}
SECONDARY = ["gamma", "max_energy_keV", "neupert_r_best", "neupert_lag_s", "hxr_peak_minus_impulsive_s",
             "dec_hxr_sigma", "hxr_rise_min", "onset_to_impulsive_min"]
#: added after the first run, labelled post hoc everywhere (see the pre-registration's Amendments)
POST_HOC = ["P2b_log10_cdte_5_20_peak_cps"]
#: post hoc: warning times compared at equal false-positive rates on the test origins
FPR_LEVELS = (0.05, 0.10, 0.15, 0.20)
EQUAL_FPR = "post hoc: equal test false-alarm rate (ROC-style, descriptive)"


def _load(name: str, split: str) -> dict:
    with np.load(PRED / f"{name}_{split}.npz") as z:
        return {k: z[k] for k in z.files}


def _t(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return float("nan")
    fmt = "%Y-%m-%d %H:%M:%S" if s.count(":") == 2 else "%Y-%m-%d %H:%M"
    return datetime.strptime(s, fmt).replace(tzinfo=UTC).timestamp()


def _f(s) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _csv(path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write(name: str, rows: list[dict]) -> None:
    cols = list(dict.fromkeys(c for r in rows for c in r))
    with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) and np.isfinite(v) else
                            "" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  -> {(OUT / name).relative_to(ROOT)} ({len(rows)} rows)")


# --- per-flare warning times --------------------------------------------------

class Signal:
    """One model's test predictions on the common origins, time-ordered."""

    def __init__(self, pt: dict, fitted: dict, order: np.ndarray, occ: list[int]):
        self.above, self.pcal = {}, {}
        for k, h in enumerate(occ):
            p = pt["p_occurrence"][order, k]
            f = fitted[f"flare_within_{h}min"]
            self.above[h] = p >= f["threshold"]
            self.pcal[h] = probcal.apply(f["calibration"], p)


def lead(tc: np.ndarray, above: np.ndarray, a: float, p: float) -> tuple[float, bool]:
    """(minutes from the first above-threshold origin in [a, p] to the peak -- 0
    without one, alert already on at the first origin of the window)."""
    lo, hi = np.searchsorted(tc, a, "left"), np.searchsorted(tc, p, "right")
    on = np.flatnonzero(above[lo:hi])
    if hi <= lo or on.size == 0:
        return 0.0, False
    return (p - tc[lo + on[0]]) / 60.0, bool(on[0] == 0)


def flare_table(occ: list[int]) -> tuple[list[dict], dict]:
    meta = json.loads((PRED / "predictions_meta.json").read_text("utf-8"))
    t1, t4, tw = _load("E1", "test"), _load("E4", "test"), _load("E4__HEL1OS_withheld", "test")
    v1, v4 = _load("E1", "val"), _load("E4", "val")
    for other in (t1, tw):
        assert np.array_equal(other["y_t_unix"], t4["y_t_unix"]), "prediction files are not aligned"
    fit1 = fit_on_validation(v1, v1["pop_soft"], occ)                   # E1 on its native windows
    fit4 = fit_on_validation(v4, np.ones(v4["pop_soft"].size, bool), occ)
    common = t4["pop_soft"] & t4["pop_hard"]
    order = np.flatnonzero(common)[np.argsort(t4["y_t_unix"][common], kind="stable")]
    tc = t4["y_t_unix"][order]
    stride = float(np.median(np.diff(tc)))
    sig = {"E1": Signal(t1, fit1, order, occ), "E4": Signal(t4, fit4, order, occ),
           "E4noHXR": Signal(tw, fit4, order, occ)}
    eq = {f"{m}@fpr{q:.2f}": equal_fpr(pt, fit, order, occ, q)
          for q in FPR_LEVELS for m, pt, fit in (("E1", t1, fit1), ("E4", t4, fit4))}
    sig.update({m: Signal(t1 if m.startswith("E1") else t4, f, order, occ) for m, f in eq.items()})
    t_lo, t_hi = float(t4["y_t_unix"].min()), float(t4["y_t_unix"].max())

    floor = class_flux(MIN_CLASS)
    flares = sorted((f for f in load_goes(S.goes_dir).flares if f.peak_flux >= floor), key=lambda f: f.start_unix)
    rows, prev_end = [], -np.inf
    i = 0
    for f in flares:
        while i < len(flares) and flares[i].start_unix < f.start_unix:   # ends of flares that started earlier
            prev_end = max(prev_end, flares[i].end_unix)
            i += 1
        if not (t_lo <= f.peak_unix <= t_hi):
            continue
        s, p = f.start_unix, f.peak_unix
        a = max(s - LOOKBACK_S, prev_end)
        lo, hi = np.searchsorted(tc, a, "left"), np.searchsorted(tc, p, "right")
        expected = max(np.floor((p - a) / stride) + 1, 1)
        r = {"goes_flare_id": f.flare_id, "goes_class": f.goes_class, "class_letter": f.goes_class[:1],
             "log10_goes_peak_flux": float(np.log10(f.peak_flux)), "start_utc": utc(s), "peak_utc": utc(p),
             "window_start_utc": utc(a) if np.isfinite(a) else "", "lead_in_min": (s - a) / 60.0,
             "isolated": bool(prev_end <= s - LOOKBACK_S), "common_coverage": min((hi - lo) / expected, 1.0)}
        for h in occ:
            for m in sig:
                r[f"L_{m}_{h}min"], r[f"censored_{m}_{h}min"] = lead(tc, sig[m].above[h], a, p)
            r[f"dL_{h}min"] = r[f"L_E4_{h}min"] - r[f"L_E1_{h}min"]
            r[f"dL_placebo_{h}min"] = r[f"L_E4noHXR_{h}min"] - r[f"L_E1_{h}min"]
            for q in FPR_LEVELS:
                e1, e4 = f"E1@fpr{q:.2f}", f"E4@fpr{q:.2f}"
                r[f"dL_fpr{q:.2f}_{h}min"] = r.pop(f"L_{e4}_{h}min") - r.pop(f"L_{e1}_{h}min")
                c4, c1_ = r.pop(f"censored_{e4}_{h}min"), r.pop(f"censored_{e1}_{h}min")
                r[f"censored_fpr{q:.2f}_{h}min"] = c4 or c1_
        c0, c1 = np.searchsorted(tc, max(a, s - CONF_S), "left"), np.searchsorted(tc, s, "left")
        r["n_confidence_origins"] = int(c1 - c0)
        r["dC_60min"] = (float(np.mean(sig["E4"].pcal[PRIMARY_HEAD][c0:c1] - sig["E1"].pcal[PRIMARY_HEAD][c0:c1]))
                         if c1 - c0 >= MIN_CONF_ORIGINS else float("nan"))
        cens = r[f"censored_E1_{PRIMARY_HEAD}min"] or r[f"censored_E4_{PRIMARY_HEAD}min"]
        r["exclusion"] = ("lead-in under 10 min" if s - a < MIN_LEAD_IN_S else
                          "common coverage under 80%" if r["common_coverage"] < MIN_COVERAGE else
                          "alert already on (left-censored)" if cens else "")
        r["primary_sample"] = r["exclusion"] == ""
        r["t_peak_unix"] = p
        rows.append(r)
    duty = []
    for k, h in enumerate(occ):
        y, lab = t4["y_occurrence"][order, k], t4["y_occurrence_mask"][order, k] > 0
        for m in ("E1", "E4"):
            on = sig[m].above[h]
            duty += [dict(analysis="context (post hoc): alert duty cycle, common test origins", outcome=f"alert_{m}",
                          head_min=h, feature="", sample="common test origins", statistic=st, value=float(v),
                          n_origins=int(n))
                     for st, v, n in (("alert_on_fraction", on.mean(), on.size),
                                      ("alert_on_when_no_flare_within_horizon", on[lab & (y == 0)].mean(),
                                       (lab & (y == 0)).sum()),
                                      ("alert_on_when_flare_within_horizon", on[lab & (y == 1)].mean(),
                                       (lab & (y == 1)).sum()))]
    info = {"origin_stride_s": stride, "n_common_origins": int(order.size), "duty": duty,
            "thresholds": {m: {h: fit[f"flare_within_{h}min"]["threshold"] for h in occ}
                           for m, fit in (("E1", fit1), ("E4", fit4), *eq.items())},
            "test_period_utc": [utc(t_lo), utc(t_hi)], "window": meta["window"]}
    return rows, info


def equal_fpr(pt: dict, fitted: dict, order: np.ndarray, occ: list[int], q: float) -> dict:
    """Post hoc, descriptive like an ROC curve: the threshold at which the model's alert
    is on at a fraction ``q`` of the common test origins with no flare within the
    horizon. Nothing is selected from it; it only lets E1 and E4 be compared at
    equal false-alarm rates (their own validation thresholds give different ones)."""
    out = {}
    for k, h in enumerate(occ):
        name = f"flare_within_{h}min"
        neg = (pt["y_occurrence_mask"][order, k] > 0) & (pt["y_occurrence"][order, k] == 0)
        out[name] = {"threshold": float(np.quantile(pt["p_occurrence"][order, k][neg], 1.0 - q)),
                     "calibration": fitted[name]["calibration"]}
    return out


# --- hard X-ray features ------------------------------------------------------

def features(flares: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    cat: dict[int, list[dict]] = {}
    for r in _csv(S.catalog / "master_catalog.csv"):
        if r["goes_peak_utc"]:
            cat.setdefault(int(round(_t(r["goes_peak_utc"]) / 60.0)), []).append(r)
    spectra = {r["id"]: r for r in _csv(S.physics / "hxr_spectra.csv")}
    onset = {r["start_utc"][:16]: r for r in _csv(S.physics / "flare_catalog_dec120s.csv")}
    timing, spectral, neupert = [], [], []
    for f in flares:
        key = int(round(f["t_peak_unix"] / 60.0))
        rows = cat.get(key, [])
        # one GOES flare can match several detector events: prefer one both instruments
        # detected, then the earliest alert
        rows = sorted(rows, key=lambda r: (not (r["soft_alert_utc"] and r["hard_alert_utc"]),
                                           np.nan_to_num(_t(r["alert_utc"]), nan=np.inf)))
        c = rows[0] if rows else {}
        ha, sa, hp = _t(c.get("hard_alert_utc")), _t(c.get("soft_alert_utc")), _t(c.get("hard_peak_czt_20_40_utc"))
        cps, cdte = _f(c.get("hard_peak_czt_20_40_cps")), _f(c.get("hard_peak_cdte_5_20_cps"))
        timing.append({"goes_flare_id": f["goes_flare_id"], "catalogue_id": c.get("id", ""),
                       "n_catalogue_events": len(rows), "catalogue_origin": c.get("origin", ""),
                       "hel1os_detected": any(r["hard_alert_utc"] for r in rows),
                       "soft_alert_utc": c.get("soft_alert_utc", ""), "hard_alert_utc": c.get("hard_alert_utc", ""),
                       "hard_peak_czt_20_40_utc": c.get("hard_peak_czt_20_40_utc", ""),
                       "P1_hard_minus_soft_alert_min": (ha - sa) / 60.0,
                       "P2_log10_czt_20_40_peak_cps": float(np.log10(cps)) if cps > 0 else float("nan"),
                       "P2b_log10_cdte_5_20_peak_cps": float(np.log10(cdte)) if cdte > 0 else float("nan"),
                       "hxr_rise_min": (hp - ha) / 60.0})
        sp = next((spectra[r["id"]] for r in rows if r["id"] in spectra and spectra[r["id"]]["status"] == "ok"),
                  spectra.get(c.get("id", ""), {}))
        ok = sp.get("status") == "ok"                    # the spectra stage's own reliability flag
        spectral.append({"goes_flare_id": f["goes_flare_id"], "spectrum_id": sp.get("id", ""),
                         "spectrum_status": sp.get("status", "no spectrum"),
                         "gamma": _f(sp.get("gamma")) if ok else float("nan"),
                         "gamma_err": _f(sp.get("gamma_err")) if ok else float("nan"),
                         "chi2_dof": _f(sp.get("chi2_dof")) if ok else float("nan"),
                         "max_energy_keV": _f(c.get("max_energy_keV"))})
        o = onset.get(f["start_utc"][:16], {})
        good = o.get("status") == "ok"
        hxr = good and o.get("hel1os") in ("True", "true", "1")
        valid = hxr and _f(o.get("hxr_peak_sigma")) >= NEUPERT_MIN_SIGMA
        neupert.append({"goes_flare_id": f["goes_flare_id"], "onset_status": o.get("status", "not in onset study"),
                        "hel1os_in_onset_study": hxr, "hxr_peak_sigma": _f(o.get("hxr_peak_sigma")) if hxr else float("nan"),
                        "neupert_valid": valid,
                        **{k: _f(o.get(k)) if valid else float("nan")
                           for k in ("neupert_r0", "neupert_r_best", "neupert_lag_s", "hxr_peak_minus_impulsive_s")},
                        "dec_hxr_sigma": _f(o.get("dec_hxr_sigma")) if hxr else float("nan"),
                        "onset_to_impulsive_min": _f(o.get("onset_to_impulsive_min")) if good else float("nan")})
    return timing, spectral, neupert


# --- statistics ---------------------------------------------------------------

def _finite(*a):
    m = np.ones(len(a[0]), bool)
    for x in a:
        m &= np.isfinite(x)
    return m


def spearman(x, y) -> float:
    m = _finite(x, y)
    if m.sum() < 3:
        return float("nan")
    rx, ry = rankdata(x[m]), rankdata(y[m])
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else float("nan")


def partial_spearman(x, y, z) -> float:
    m = _finite(x, y, z)
    if m.sum() < 4:
        return float("nan")
    rx, ry, rz = rankdata(x[m]), rankdata(y[m]), rankdata(z[m])
    if rz.std() == 0:
        return spearman(x[m], y[m])

    def resid(r):
        return r - np.polyval(np.polyfit(rz, r, 1), rz)

    ex, ey = resid(rx), resid(ry)
    return float(np.corrcoef(ex, ey)[0, 1]) if ex.std() > 0 and ey.std() > 0 else float("nan")


def perm_p(x, y) -> float:
    m = _finite(x, y)
    if m.sum() < 3:
        return float("nan")
    rx, ry = rankdata(x[m]), rankdata(y[m])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    rx, ry = (rx - rx.mean()) / rx.std(), (ry - ry.mean()) / ry.std()
    obs = abs(float(np.mean(rx * ry)))
    rng = np.random.default_rng(SEED)
    perm = rng.permuted(np.tile(ry, (N_PERM, 1)), axis=1)
    null = np.abs(perm @ rx / rx.size)
    return float((1 + np.sum(null >= obs - 1e-12)) / (1 + N_PERM))


def holm(ps: list[float]) -> list[float]:
    idx = [i for i, p in enumerate(ps) if np.isfinite(p)]
    out = [float("nan")] * len(ps)
    running = 0.0
    for k, i in enumerate(sorted(idx, key=lambda i: ps[i])):
        running = max(running, min(1.0, (len(idx) - k) * ps[i]))
        out[i] = running
    return out


def _ci(t, stat):
    return day_block_ci(t, stat, n_boot=N_BOOT, seed=SEED) or [float("nan"), float("nan")]


def correlation_rows(base: dict, x, y, t, z=None, sign: int | None = None) -> list[dict]:
    m = _finite(x, y)
    xm, ym, tm = x[m], y[m], t[m]
    n, days = int(m.sum()), int(np.unique(np.floor(tm / 86400.0)).size)
    common = dict(base, n_flares=n, n_days=days, predicted_sign={1: "+", -1: "-"}.get(sign, ""))
    rows = [dict(common, statistic="spearman_rho", value=spearman(xm, ym),
                 **dict(zip(("ci95_low", "ci95_high"), _ci(tm, lambda i: spearman(xm[i], ym[i])))),
                 p_permutation=perm_p(xm, ym))]
    if z is not None:
        zm = z[m]
        rows.append(dict(common, statistic="partial_spearman_rho_given_log10_goes_peak_flux",
                         value=partial_spearman(xm, ym, zm),
                         **dict(zip(("ci95_low", "ci95_high"),
                                    _ci(tm, lambda i: partial_spearman(xm[i], ym[i], zm[i]))))))
    return rows


def group_rows(base: dict, x, dl, t) -> list[dict]:
    """Feature medians, E4 earlier (dL >= 2 min) vs not earlier (dL <= 0)."""
    m = _finite(x, dl) & ((dl >= EARLIER_MIN) | (dl <= 0))
    xm, gm, tm = x[m], dl[m] >= EARLIER_MIN, t[m]

    def diff(i):
        a, b = xm[i][gm[i]], xm[i][~gm[i]]
        return np.median(a) - np.median(b) if a.size and b.size else float("nan")

    ci = _ci(tm, diff) if gm.sum() >= 3 and (~gm).sum() >= 3 else [float("nan")] * 2
    return [dict(base, statistic="median_E4_earlier", value=float(np.median(xm[gm])) if gm.any() else float("nan"),
                 n_flares=int(gm.sum())),
            dict(base, statistic="median_E4_not_earlier",
                 value=float(np.median(xm[~gm])) if (~gm).any() else float("nan"), n_flares=int((~gm).sum())),
            dict(base, statistic="median_difference_earlier_minus_not", value=diff(np.arange(xm.size)),
                 ci95_low=ci[0], ci95_high=ci[1], n_flares=int(m.sum()),
                 n_days=int(np.unique(np.floor(tm / 86400.0)).size))]


def mean_row(base: dict, v, t, statistic="mean") -> dict:
    m = np.isfinite(v)
    vm, tm = v[m], t[m]
    return dict(base, statistic=statistic, value=float(np.mean(vm)) if vm.size else float("nan"),
                **dict(zip(("ci95_low", "ci95_high"), _ci(tm, lambda i: np.mean(vm[i])))),
                n_flares=int(m.sum()), n_days=int(np.unique(np.floor(tm / 86400.0)).size))


def verdict(row: dict) -> str:
    if row["n_flares"] < MIN_N:
        return f"inconclusive (fewer than {MIN_N} flares)"
    p, v = row.get("p_holm", float("nan")), row["value"]
    if not (np.isfinite(p) and np.isfinite(v)):
        return "inconclusive (not computable)"
    if p >= ALPHA:
        return "not supported (no significant association)"
    return "supported" if np.sign(v) == (1 if row["predicted_sign"] == "+" else -1) else "contradicted (opposite sign)"


# --- main -----------------------------------------------------------------------

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    prereg_sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    prereg_commit = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h %ci", "--", str(PREREG)],
                                   cwd=ROOT, capture_output=True, text=True).stdout.strip()
    as_registered = subprocess.run(["git", "show", f"{prereg_commit.split()[0]}:{PREREG.relative_to(ROOT).as_posix()}"],
                                   cwd=ROOT, capture_output=True).stdout if prereg_commit else b""
    occ = json.loads((PRED / "predictions_meta.json").read_text("utf-8"))["window"]["occurrence_min"]
    print("per-flare warning times of E1 and E4 ...", flush=True)
    flares, info = flare_table(occ)
    timing, spectral, neupert = features(flares)
    F = {k: np.array([r[k] for r in rows], dtype=float)
         for rows in (timing, spectral, neupert) for k in rows[0]
         if k in PRIMARY or k in SECONDARY or k in POST_HOC}
    detected = np.array([r["hel1os_detected"] for r in timing])
    col = {k: np.array([r[k] for r in flares], dtype=float)
           for k in flares[0] if k.startswith(("dL", "dC", "L_", "log10_goes", "t_peak"))}
    t = col["t_peak_unix"]
    prim = np.array([r["primary_sample"] for r in flares])
    uncens = np.array([r["exclusion"] in ("", "alert already on (left-censored)") for r in flares])
    iso = np.array([r["isolated"] for r in flares])
    letter = np.array(["M+" if r["class_letter"] in "MX" else r["class_letter"] for r in flares])
    z = col["log10_goes_peak_flux"]
    dl = col[f"dL_{PRIMARY_HEAD}min"]
    rows: list[dict] = []

    def nan_out(mask, v):
        return np.where(mask, v, np.nan)

    # descriptive: the flare-level gain itself
    for h in HEADS:
        b = dict(analysis="descriptive", outcome=f"dL_{h}min", head_min=h, feature="", sample="primary")
        d = nan_out(prim, col[f"dL_{h}min"])
        rows.append(mean_row(b, d, t, "mean_dL_min"))
        for name, sel in (("fraction_E4_earlier", d >= EARLIER_MIN), ("fraction_same", d == 0),
                          ("fraction_E4_later", d < 0)):
            rows.append(mean_row(b, nan_out(np.isfinite(d), sel.astype(float)), t, name))
        for m in ("E1", "E4"):
            L = nan_out(prim, col[f"L_{m}_{h}min"])
            rows.append(mean_row(dict(b, outcome=f"L_{m}_{h}min"), L, t, "mean_lead_min"))
            rows.append(mean_row(dict(b, outcome=f"L_{m}_{h}min"), nan_out(np.isfinite(L), (L > 0).astype(float)),
                                 t, "fraction_warned_before_peak"))
    rows.append(mean_row(dict(analysis="descriptive", outcome="dC_60min", head_min=PRIMARY_HEAD, feature="",
                              sample="primary"), nan_out(prim, col["dC_60min"]), t, "mean_dC"))
    for cls in ("C", "M+"):
        rows.append(mean_row(dict(analysis="descriptive by class", outcome=f"dL_{PRIMARY_HEAD}min",
                                  head_min=PRIMARY_HEAD, feature="", sample=f"primary, class {cls}"),
                             nan_out(prim & (letter == cls), dl), t, "mean_dL_min"))

    # primary tests (Holm over the two)
    first = len(rows)
    for feat, sign in PRIMARY.items():
        rows += correlation_rows(dict(analysis="primary", outcome=f"dL_{PRIMARY_HEAD}min", head_min=PRIMARY_HEAD,
                                      feature=feat, sample="primary"), F[feat], nan_out(prim, dl), t, z, sign)
    tests = [r for r in rows[first:] if r["statistic"] == "spearman_rho"]
    for r, ph in zip(tests, holm([r["p_permutation"] for r in tests])):
        r["p_holm"] = ph
        r["verdict"] = verdict(r)

    # secondary, pre-registered
    for feat, sign in PRIMARY.items():
        rows += correlation_rows(dict(analysis="secondary: confidence outcome", outcome="dC_60min",
                                      head_min=PRIMARY_HEAD, feature=feat, sample="primary"),
                                 F[feat], nan_out(prim, col["dC_60min"]), t, z, sign)
        rows += correlation_rows(dict(analysis="secondary: placebo (E4 with HEL1OS withheld - E1)",
                                      outcome=f"dL_placebo_{PRIMARY_HEAD}min", head_min=PRIMARY_HEAD, feature=feat,
                                      sample="primary"), F[feat], nan_out(prim, col[f"dL_placebo_{PRIMARY_HEAD}min"]), t)
    b = dict(analysis="secondary: detection contrast", outcome=f"dL_{PRIMARY_HEAD}min", head_min=PRIMARY_HEAD,
             feature="hel1os_detected", sample="primary")
    rows.append(mean_row(b, nan_out(prim & detected, dl), t, "mean_dL_detected"))
    rows.append(mean_row(b, nan_out(prim & ~detected, dl), t, "mean_dL_not_detected"))
    mm = prim & np.isfinite(dl)
    dm, gm, tm = dl[mm], detected[mm], t[mm]

    def det_diff(i):
        a, c = dm[i][gm[i]], dm[i][~gm[i]]
        return np.mean(a) - np.mean(c) if a.size and c.size else float("nan")

    rows.append(dict(b, statistic="mean_dL_detected_minus_not_detected", value=det_diff(np.arange(dm.size)),
                     **dict(zip(("ci95_low", "ci95_high"), _ci(tm, det_diff))), n_flares=int(mm.sum()),
                     n_days=int(np.unique(np.floor(tm / 86400.0)).size), predicted_sign="+"))

    # sensitivity
    for feat, sign in PRIMARY.items():
        for h in HEADS[1:]:
            rows += correlation_rows(dict(analysis="sensitivity: horizon", outcome=f"dL_{h}min", head_min=h,
                                          feature=feat, sample="primary"),
                                     F[feat], nan_out(prim, col[f"dL_{h}min"]), t, sign=sign)
        rows += correlation_rows(dict(analysis="sensitivity: censored flares included",
                                      outcome=f"dL_{PRIMARY_HEAD}min", head_min=PRIMARY_HEAD, feature=feat,
                                      sample="primary + left-censored"), F[feat], nan_out(uncens, dl), t, sign=sign)
        rows += correlation_rows(dict(analysis="sensitivity: isolated flares", outcome=f"dL_{PRIMARY_HEAD}min",
                                      head_min=PRIMARY_HEAD, feature=feat, sample="primary, isolated"),
                                 F[feat], nan_out(prim & iso, dl), t, sign=sign)

    # exploratory: every secondary feature, and group medians for all features
    for feat in SECONDARY:
        rows += correlation_rows(dict(analysis="exploratory", outcome=f"dL_{PRIMARY_HEAD}min", head_min=PRIMARY_HEAD,
                                      feature=feat, sample="primary"), F[feat], nan_out(prim, dl), t, z)
    for feat in [*PRIMARY, *SECONDARY]:
        rows += group_rows(dict(analysis="descriptive: E4 earlier vs not", outcome=f"dL_{PRIMARY_HEAD}min",
                                head_min=PRIMARY_HEAD, feature=feat, sample="primary"),
                           F[feat], nan_out(prim, dl), t)

    # post hoc (after the first run; see Amendments): equal test false-alarm rates, CdTe brightness
    for q in FPR_LEVELS:
        for h in HEADS:
            cq = prim & ~np.array([r[f"censored_fpr{q:.2f}_{h}min"] for r in flares])
            b = dict(analysis=EQUAL_FPR, outcome=f"dL_fpr{q:.2f}_{h}min", head_min=h, feature="",
                     sample=f"primary, not censored at false-positive rate {q:.2f}")
            d = nan_out(cq, col[f"dL_fpr{q:.2f}_{h}min"])
            rows.append(mean_row(b, d, t, "mean_dL_min"))
            for name, sel in (("fraction_E4_earlier", d >= EARLIER_MIN), ("fraction_E4_later", d < 0)):
                rows.append(mean_row(b, nan_out(np.isfinite(d), sel.astype(float)), t, name))
            if h != PRIMARY_HEAD:
                continue
            for feat in [*PRIMARY, *POST_HOC]:
                rows += correlation_rows(dict(b, feature=feat), F[feat], d, t, z)
    for feat in POST_HOC:
        rows += correlation_rows(dict(analysis="post hoc: exploratory", outcome=f"dL_{PRIMARY_HEAD}min",
                                      head_min=PRIMARY_HEAD, feature=feat, sample="primary"),
                                 F[feat], nan_out(prim, dl), t, z)
    rows += info.pop("duty")

    for r in flares:
        r.pop("t_peak_unix")
    _write("flare_level_comparison.csv", flares)
    _write("hxr_timing_features.csv", timing)
    _write("hxr_spectral_features.csv", spectral)
    _write("neupert_features.csv", neupert)
    _write("physics_gain_association.csv", rows)
    excl: dict[str, int] = {}
    for r in flares:
        excl[r["exclusion"] or "included"] = excl.get(r["exclusion"] or "included", 0) + 1
    prov = {"provenance": stamp(), "preregistration": {"file": str(PREREG.relative_to(ROOT)),
                                                       "sha256": prereg_sha, "added_in_commit": prereg_commit,
                                                       "sha256_as_registered": hashlib.sha256(as_registered).hexdigest()
                                                       if as_registered else ""},
            "n_goes_flares_in_test_period": len(flares), "sample": excl, **info,
            "fixed_settings": {"min_class": MIN_CLASS, "primary_head_min": PRIMARY_HEAD,
                               "lookback_s": LOOKBACK_S, "min_lead_in_s": MIN_LEAD_IN_S,
                               "min_coverage": MIN_COVERAGE, "n_boot": N_BOOT, "n_perm": N_PERM, "seed": SEED,
                               "alpha": ALPHA, "min_n": MIN_N, "neupert_min_sigma": NEUPERT_MIN_SIGMA}}
    (OUT / "physics_provenance.json").write_text(json.dumps(prov, indent=2, default=str), encoding="utf-8")
    readme(rows, prov)
    return 0


def _fmt(v, d=3) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:+.{d}f}"


def readme(rows: list[dict], prov: dict) -> None:
    def get(**kw):
        return next((r for r in rows if all(r.get(k) == v for k, v in kw.items())), None)

    def ci(r):
        return f"[{_fmt(r.get('ci95_low'))}, {_fmt(r.get('ci95_high'))}]"

    s = prov["sample"]
    L = [f"# Physics interpretation: which flares gain from HEL1OS? (generated by {prov['provenance']['script']})",
         "",
         f"Pre-registration: `00_preregistration.md`, added in commit {prov['preregistration']['added_in_commit']}, "
         f"sha256 `{prov['preregistration']['sha256'][:16]}...`. Every number below is read from "
         "`physics_gain_association.csv`; interpretation belongs in the paper, not here.",
         "",
         f"GOES flares >= {MIN_CLASS} in the test period: {prov['n_goes_flares_in_test_period']}. "
         + "; ".join(f"{k}: {v}" for k, v in sorted(s.items())) + ".",
         ""]
    d = get(analysis="descriptive", outcome=f"dL_{PRIMARY_HEAD}min", statistic="mean_dL_min")
    if d:
        L.append(f"- Flare-level gain, {PRIMARY_HEAD}-min head: mean dL = {_fmt(d['value'], 2)} min {ci(d)} "
                 f"over {d['n_flares']} flares on {d['n_days']} days.")
    for name in ("fraction_E4_earlier", "fraction_same", "fraction_E4_later"):
        r = get(analysis="descriptive", outcome=f"dL_{PRIMARY_HEAD}min", statistic=name)
        if r:
            L.append(f"  - {name.replace('_', ' ')}: {_fmt(r['value'], 3)} {ci(r)}")
    for feat, sign in PRIMARY.items():
        r = get(analysis="primary", feature=feat, statistic="spearman_rho")
        pr = get(analysis="primary", feature=feat, statistic="partial_spearman_rho_given_log10_goes_peak_flux")
        if r:
            L.append(f"- **Primary, {feat}** (predicted sign {'+' if sign > 0 else '-'}): rho = {_fmt(r['value'])} "
                     f"{ci(r)}, n = {r['n_flares']}, permutation p = {r['p_permutation']:.4g}, Holm p = "
                     f"{r['p_holm']:.4g} -> **{r['verdict']}**. Partial rho given GOES peak flux = "
                     f"{_fmt(pr['value']) if pr else 'n/a'} {ci(pr) if pr else ''}.")
    r = get(analysis="secondary: detection contrast", statistic="mean_dL_detected_minus_not_detected")
    if r:
        L.append(f"- Detection contrast: mean dL (HEL1OS detected - not detected) = {_fmt(r['value'], 2)} min {ci(r)}.")
    for feat in PRIMARY:
        r = get(analysis="secondary: placebo (E4 with HEL1OS withheld - E1)", feature=feat, statistic="spearman_rho")
        if r:
            L.append(f"- Placebo (HEL1OS withheld), {feat}: rho = {_fmt(r['value'])} {ci(r)}, n = {r['n_flares']}.")
    for feat in PRIMARY:
        r = get(analysis="secondary: confidence outcome", feature=feat, statistic="spearman_rho")
        if r:
            L.append(f"- Pre-flare confidence dC vs {feat}: rho = {_fmt(r['value'])} {ci(r)}, n = {r['n_flares']}.")
    L += ["", "Post hoc, not pre-registered (reasons under Amendments in the pre-registration):"]
    for q in FPR_LEVELS:
        parts = []
        for h in HEADS:
            r = get(analysis=EQUAL_FPR, outcome=f"dL_fpr{q:.2f}_{h}min", statistic="mean_dL_min", feature="")
            if r:
                parts.append(f"{h} min: {_fmt(r['value'], 2)} {ci(r)} (n = {r['n_flares']})")
        L.append(f"- Equal test false-positive rate {q:.2f}, mean dL (min) by head: " + "; ".join(parts) + ".")
        for feat in [*PRIMARY, *POST_HOC]:
            r = get(analysis=EQUAL_FPR, outcome=f"dL_fpr{q:.2f}_{PRIMARY_HEAD}min", feature=feat,
                    statistic="spearman_rho")
            if r:
                L.append(f"  - {PRIMARY_HEAD}-min dL vs {feat}: rho = {_fmt(r['value'])} {ci(r)}, "
                         f"n = {r['n_flares']}, unadjusted p = {r['p_permutation']:.4g}.")
    for feat in POST_HOC:
        r = get(analysis="post hoc: exploratory", feature=feat, statistic="spearman_rho")
        pr = get(analysis="post hoc: exploratory", feature=feat,
                 statistic="partial_spearman_rho_given_log10_goes_peak_flux")
        if r:
            L.append(f"- dL vs {feat}: rho = {_fmt(r['value'])} {ci(r)}, n = {r['n_flares']}, unadjusted p = "
                     f"{r['p_permutation']:.4g}; partial given GOES peak flux {_fmt(pr['value'])} {ci(pr)}.")
    L += ["", "Sensitivity (other horizons, censored flares, isolated flares), the exploratory features, the alert "
          "duty cycles and the E4-earlier vs not-earlier medians are in `physics_gain_association.csv`."]
    (OUT / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  -> {(OUT / 'README.md').relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
