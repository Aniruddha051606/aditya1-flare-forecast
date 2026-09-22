"""Evaluation against baselines, with operating points fitted on validation.

Nothing here touches the test set until the thresholds are already fixed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .pipeline import Prepared, make_loaders, resolve_device
from .train import collect_predictions
from .metrics import (
    skill_scores, best_threshold, roc_auc, brier_score, brier_skill_score,
    reliability, regression_scores, skill_vs_reference, multiclass_scores,
    forecast_score_mask, day_block_ci, interval_scale,
)
from .models.net import SolexHelNet


def climatology_rate(y: np.ndarray) -> float:
    return float(np.mean(y)) if y.size else float("nan")


def _ci(ci) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""


def solexs_no_change(prep: Prepared, t_query: np.ndarray) -> tuple[np.ndarray, str]:
    """GOES-scale log flux from SoLEXS alone at each query time: the SoLEXS
    GOES-long analogue at that step through a linear calibration fitted on the
    training period only. NaN where SoLEXS did not observe.

    This is the "no change" reference an Aditya-only system actually has. The
    GOES value at the origin (``y_persistence``) reads the instrument being
    predicted, which such a system never sees."""
    segs = prep.segments
    t = np.concatenate([s.time_unix for s in segs])
    gl = np.concatenate([s.goes_long if s.goes_long is not None else np.full(len(s), np.nan)
                         for s in segs]).astype(np.float64)
    soft_ok = np.concatenate([s.soft_mask for s in segs]) > 0
    goes = np.concatenate([s.log_flux for s in segs]).astype(np.float64)
    goes_ok = np.concatenate([s.target_valid for s in segs]) > 0
    order = np.argsort(t, kind="stable")
    t, gl, soft_ok, goes, goes_ok = t[order], gl[order], soft_ok[order], goes[order], goes_ok[order]
    tq = np.asarray(t_query, dtype=np.float64)
    out = np.full(tq.size, np.nan)
    if not prep.splits.get("train"):
        return out, "no training split"
    train_end = max(prep.windows[i].t_unix for i in prep.splits["train"])
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (t <= train_end) & soft_ok & goes_ok & np.isfinite(gl) & (gl > 0)
        if m.sum() < 100:
            return out, "too few training samples to calibrate SoLEXS"
        slope, icpt = np.polyfit(np.log10(gl[m]), goes[m], 1)
        sd = float(np.std(goes[m] - (icpt + slope * np.log10(gl[m]))))
        i = np.clip(np.searchsorted(t, tq), 0, t.size - 1)
        ok = np.isclose(t[i], tq) & soft_ok[i] & np.isfinite(gl[i]) & (gl[i] > 0)
        out[ok] = icpt + slope * np.log10(gl[i][ok])
    return out, f"log10 F = {icpt:.3f} + {slope:.3f} log10 rate (sd {sd:.3f} dex, training period)"


def evaluate(model: SolexHelNet, prep: Prepared, cfg: Config,
             verbose: bool = True) -> dict:
    device = resolve_device(cfg.train.device)
    _, va, te, stats = make_loaders(prep, cfg)

    val = collect_predictions(model, va, device) if len(va) else None
    test = collect_predictions(model, te, device) if len(te) else None

    # Climatology reference: the mean log flux over the TRAINING split.
    # Estimated on train, never on test -- a "climatology" fitted to the test
    # set would be using the answer.
    tr_loader, _, _, _ = make_loaders(prep, cfg)
    clim_mean = 0.0
    n_seen = 0
    for b in tr_loader:
        m = b["nowcast_mask"].cpu().numpy() > 0
        if m.any():
            v = b["nowcast"].cpu().numpy()[m]
            clim_mean += float(v.sum())
            n_seen += int(m.sum())
    clim_mean = clim_mean / max(n_seen, 1)

    report: dict = {"label_stats": stats}
    if test is None:
        report["error"] = "empty test split"
        return report

    # --- 1. Flare in progress now (nowcasting) -------------------------
    m = test["y_nowcast_mask"] > 0
    y = test["y_in_flare"][m]
    p = test["p_inflare"][m]

    thr = 0.5
    if val is not None:
        vm = val["y_nowcast_mask"] > 0
        if vm.sum() and len(np.unique(val["y_in_flare"][vm])) > 1:
            thr, _ = best_threshold(val["y_in_flare"][vm], val["p_inflare"][vm], "TSS")

    now = skill_scores(y, p >= thr) if y.size else {}
    now.update({
        "threshold": float(thr),
        "AUC": roc_auc(y, p) if y.size else float("nan"),
        "Brier": brier_score(y, p) if y.size else float("nan"),
        "BSS_vs_climatology": brier_skill_score(y, p) if y.size else float("nan"),
        "reliability": reliability(y, p) if y.size else {},
    })
    # 95% intervals resampling whole test days (a flare fills many windows)
    if y.size:
        dm = test["y_t_unix"][m]
        now["TSS_ci95"] = day_block_ci(dm, lambda i: skill_scores(y[i], p[i] >= thr)["TSS"])
        now["POD_ci95"] = day_block_ci(dm, lambda i: skill_scores(y[i], p[i] >= thr)["POD"])
        now["AUC_ci95"] = day_block_ci(dm, lambda i: roc_auc(y[i], p[i]), n_boot=200)
    report["nowcast_in_flare"] = now

    # --- 2. Flare phase ------------------------------------------------
    ph_pred = test["p_phase"].argmax(axis=1)[m]
    report["nowcast_phase"] = multiclass_scores(test["y_phase"][m], ph_pred, 4)

    # --- 3. Flare occurrence within H (forecasting) --------------------
    occ = {}
    for h, hs in enumerate(cfg.win.occurrence_horizons_s):
        mm = test["y_occurrence_mask"][:, h] > 0
        yt = test["y_occurrence"][mm, h]
        pt = test["p_occurrence"][mm, h]
        t_h = 0.5
        if val is not None:
            vmm = val["y_occurrence_mask"][:, h] > 0
            if vmm.sum() and len(np.unique(val["y_occurrence"][vmm, h])) > 1:
                t_h, _ = best_threshold(val["y_occurrence"][vmm, h],
                                        val["p_occurrence"][vmm, h], "TSS")
        s = skill_scores(yt, pt >= t_h) if yt.size else {}
        s.update({
            "threshold": float(t_h),
            "AUC": roc_auc(yt, pt) if yt.size else float("nan"),
            "Brier": brier_score(yt, pt) if yt.size else float("nan"),
            "BSS_vs_climatology": brier_skill_score(yt, pt) if yt.size else float("nan"),
            "climatology_rate": climatology_rate(yt),
        })
        if yt.size:
            dh = test["y_t_unix"][mm]
            s["TSS_ci95"] = day_block_ci(
                dh, lambda i, yt=yt, pt=pt, t_h=t_h: skill_scores(yt[i], pt[i] >= t_h)["TSS"])
            s["AUC_ci95"] = day_block_ci(dh, lambda i, yt=yt, pt=pt: roc_auc(yt[i], pt[i]), n_boot=200)
        occ[f"{int(hs / 60)}min"] = s
    report["forecast_occurrence"] = occ

    # --- 4. Nowcast regression ----------------------------------------
    # Scored where an Aditya-only system would give a flux: GOES truth at the
    # origin AND SoLEXS observing it. Windows where only HEL1OS (or nothing)
    # observed the origin are reported separately under "all_origins"; mixing
    # them in made this report disagree with fair_references.json (2026-09).
    slx, cal_text = solexs_no_change(prep, test["y_t_unix"])
    t_te = test["y_t_unix"]
    op = (test["y_nowcast_mask"] > 0) & (test["y_soft_origin"] > 0)
    nr = regression_scores(test["y_nowcast"], test["nowcast"], op)
    nr["n"] = int(op.sum())
    err_now = np.abs(test["nowcast"] - test["y_nowcast"])
    e_op = err_now[op]
    nr["MAE_ci95"] = day_block_ci(t_te[op], lambda i: float(np.mean(e_op[i]))) if op.any() else None
    opc = op & np.isfinite(slx)
    if opc.any():
        nr["solexs_calibration_MAE"] = float(np.mean(np.abs(slx[opc] - test["y_nowcast"][opc])))
        nr["model_MAE_same_windows"] = float(np.mean(err_now[opc]))
    nr["solexs_calibration"] = cal_text
    allo = test["y_nowcast_mask"] > 0
    nr["all_origins"] = {**regression_scores(test["y_nowcast"], test["nowcast"], allo), "n": int(allo.sum())}
    report["nowcast_regression"] = nr

    # --- 5. Forecast regression vs persistence -------------------------
    # Operational windows as in section 4. Two "no change" references on the
    # same windows: GOES at the origin (what the target instrument read) and
    # calibrated SoLEXS at the origin (what an Aditya-only system has).
    qs = list(cfg.win.quantiles)
    i50 = int(np.argmin(np.abs(np.array(qs) - 0.5)))
    nominal = float(qs[-1] - qs[0])
    slx_val = None
    if val is not None:
        slx_val, _ = solexs_no_change(prep, val["y_t_unix"])
    fore = {}
    scales = []
    for h, hs in enumerate(cfg.win.forecast_horizons_s):
        base = forecast_score_mask(test["y_forecast_mask"][:, h], test["y_nowcast_mask"])
        mm = base & (test["y_soft_origin"] > 0) & np.isfinite(slx)
        yt = test["y_forecast"][:, h]
        q50 = test["forecast"][:, h, i50]
        ref = test["y_persistence"]
        r = regression_scores(yt, q50, mm)
        r["n"] = int(mm.sum())
        e_mm = np.abs(q50 - yt)[mm]
        r["MAE_ci95"] = (day_block_ci(t_te[mm], lambda i, e_mm=e_mm: float(np.mean(e_mm[i])))
                         if mm.any() else None)
        r["skill_vs_persistence"] = skill_vs_reference(yt, q50, ref, mm)
        r["persistence_RMSE"] = regression_scores(yt, ref, mm)["RMSE"]
        r["skill_vs_solexs_no_change"] = skill_vs_reference(yt, q50, slx, mm)
        r["solexs_no_change_RMSE"] = regression_scores(yt, slx, mm)["RMSE"]
        r["all_origins"] = {**regression_scores(yt, q50, base), "n": int(base.sum()),
                            "skill_vs_persistence": skill_vs_reference(yt, q50, ref, base)}

        # Climatology is the reference that matters at long horizons: beating
        # persistence there is easy, because persistence extrapolates a
        # decaying flare. A model that merely predicts the mean beats
        # persistence and is still useless, so both references are reported.
        clim = np.full_like(yt, clim_mean)
        r["climatology_RMSE"] = regression_scores(yt, clim, mm)["RMSE"]
        r["skill_vs_climatology"] = skill_vs_reference(yt, q50, clim, mm)
        if mm.sum() > 2 and np.std(q50[mm]) > 0 and np.std(yt[mm]) > 0:
            r["correlation"] = float(np.corrcoef(q50[mm], yt[mm])[0, 1])
            r["spread_ratio"] = float(np.std(q50[mm]) / np.std(yt[mm]))
        else:
            r["correlation"] = float("nan")
            r["spread_ratio"] = float("nan")
        # Empirical coverage of the predicted interval, raw and after widening
        # by a factor fitted on validation (split-conformal; frozen with the model).
        k = 1.0
        if val is not None:
            vm_ = (forecast_score_mask(val["y_forecast_mask"][:, h], val["y_nowcast_mask"])
                   & (val["y_soft_origin"] > 0) & np.isfinite(slx_val))
            k = interval_scale(val["y_forecast"][vm_, h], val["forecast"][vm_, h, 0],
                               val["forecast"][vm_, h, i50], val["forecast"][vm_, h, -1], nominal)
        scales.append(round(float(k), 4))
        lo = test["forecast"][:, h, 0]
        hi = test["forecast"][:, h, -1]
        if mm.any():
            inside = (yt[mm] >= lo[mm]) & (yt[mm] <= hi[mm])
            r["interval_coverage"] = float(inside.mean())
            r["nominal_coverage"] = nominal
            lo_k, hi_k = q50 - k * (q50 - lo), q50 + k * (hi - q50)
            r["interval_scale_from_validation"] = float(k)
            r["interval_coverage_calibrated"] = float(((yt[mm] >= lo_k[mm]) & (yt[mm] <= hi_k[mm])).mean())
        fore[f"{int(hs / 60)}min"] = r
    report["forecast_regression"] = fore
    report["forecast_interval_scale"] = scales

    # --- 6. Peak prediction -------------------------------------------
    pm = test["y_peak_mask"] > 0
    if pm.any():
        report["peak"] = {
            "time_to_peak_MAE_min": float(
                np.abs(test["peak"][:, 0] - test["y_peak"][:, 0])[pm[:, 0]].mean() * 60),
            "log_peak_flux_MAE": float(
                np.abs(test["peak"][:, 1] - test["y_peak"][:, 1])[pm[:, 1]].mean()),
            "n": int(pm[:, 0].sum()),
        }
        e_pk = np.abs(test["peak"][:, 1] - test["y_peak"][:, 1])[pm[:, 1]]
        report["peak"]["log_peak_flux_MAE_ci95"] = day_block_ci(
            test["y_t_unix"][pm[:, 1]], lambda i: float(np.mean(e_pk[i])))

    # --- 7. Ablation: how much does each instrument contribute? --------
    report["modality_ablation"] = modality_ablation(model, prep, cfg, thr)
    report["modality_ablation_hel1os_observed"] = modality_ablation(
        model, prep, cfg, thr, require_hard=True)

    if verbose:
        _print_report(report, cfg)
    return report


@torch.no_grad()
def modality_ablation(model: SolexHelNet, prep: Prepared, cfg: Config,
                      thr: float, require_hard: bool = False) -> dict:
    """Re-score the test split with each modality masked off.

    This is the honest way to answer "does the hard X-ray channel help?" --
    and, with non-overlapping input files, to show plainly that it currently
    cannot.

    The ``clock_only`` row is the guard rail: it blanks **both** instruments,
    so whatever skill remains comes from non-instrument inputs alone.  Anything
    meaningfully above AUC 0.5 there means the model is memorising time rather
    than reading the Sun, which is exactly the failure that time-of-day
    features caused on this single-day dataset.

    ``require_hard`` scores only windows where HEL1OS observed the prediction
    origin. On the archive HEL1OS covers ~72 of ~170 test days, so over the
    whole test split any hard X-ray effect is diluted by windows where blanking
    HEL1OS changes nothing because it was never there.
    """
    device = resolve_device(cfg.train.device)
    _, _, te, _ = make_loaders(prep, cfg)
    model.eval()
    out: dict = {}

    for name, zero_soft, zero_hard in (("both", False, False),
                                       ("soft_only", False, True),
                                       ("hard_only", True, False),
                                       ("clock_only", True, True)):
        ys, ps, ms = [], [], []
        for batch in te:
            b = {k: v.to(device) for k, v in batch.items()}
            sm = torch.zeros_like(b["soft_mask"]) if zero_soft else b["soft_mask"]
            hm = torch.zeros_like(b["hard_mask"]) if zero_hard else b["hard_mask"]
            o = model(b["soft"], sm, b["hard"], hm, b["clock"])
            ps.append(torch.sigmoid(o["in_flare"]).cpu().numpy())
            ys.append(batch["in_flare"].cpu().numpy())
            m = batch["nowcast_mask"].cpu().numpy() > 0
            if require_hard:
                m &= batch["hard_mask"].cpu().numpy()[:, -1] > 0
            ms.append(m)
        y = np.concatenate(ys)
        p = np.concatenate(ps)
        mk = np.concatenate(ms)
        if mk.sum() and len(np.unique(y[mk])) > 1:
            s = skill_scores(y[mk], p[mk] >= thr)
            out[name] = {"TSS": s["TSS"], "HSS": s["HSS"], "POD": s["POD"],
                         "FAR": s["FAR"], "AUC": roc_auc(y[mk], p[mk]),
                         "n": int(mk.sum())}
        else:
            out[name] = {"TSS": float("nan"), "n": int(mk.sum())}
    return out


def _fmt(v) -> str:
    if isinstance(v, float):
        return "nan" if np.isnan(v) else f"{v:.4f}"
    return str(v)


def _print_report(r: dict, cfg: Config) -> None:
    print("\n" + "=" * 74)
    print("TEST-SET EVALUATION")
    print("=" * 74)

    n = r.get("nowcast_in_flare", {})
    if n:
        print(f"\n[1] Nowcast: flare in progress   (threshold {n['threshold']:.2f} "
              f"fitted on validation)")
        print(f"    base rate {n.get('base_rate', float('nan')):.3f}   "
              f"TSS {n['TSS']:.3f} {_ci(n.get('TSS_ci95'))}   HSS {n['HSS']:.3f}   "
              f"AUC {_fmt(n['AUC'])} {_ci(n.get('AUC_ci95'))}")
        print(f"    POD {n['POD']:.3f}  FAR {n['FAR']:.3f}  CSI {n['CSI']:.3f}  "
              f"F1 {n['F1']:.3f}")
        print(f"    Brier {_fmt(n['Brier'])}   BSS vs climatology "
              f"{_fmt(n['BSS_vs_climatology'])}")
        print(f"    TP {n['TP']} FP {n['FP']} FN {n['FN']} TN {n['TN']}")

    p = r.get("nowcast_phase", {})
    if p:
        names = ["quiet", "rise", "peak", "decay"]
        print(f"\n[2] Nowcast: flare phase   accuracy {p['accuracy']:.3f}   "
              f"macro-F1 {p['macro_f1']:.3f}")
        for c, name in enumerate(names):
            pc = p["per_class"][c] if c in p["per_class"] else p["per_class"].get(str(c), {})
            if pc:
                print(f"    {name:6s} P {pc['precision']:.3f} R {pc['recall']:.3f} "
                      f"F1 {pc['f1']:.3f}  n={pc['support']}")

    occ = r.get("forecast_occurrence", {})
    if occ:
        print("\n[3] Forecast: flare within horizon")
        print(f"    {'horizon':>8}  {'base':>6}  {'TSS':>6}  {'HSS':>6}  "
              f"{'POD':>6}  {'FAR':>6}  {'AUC':>6}  {'BSS':>7}")
        for k, s in occ.items():
            print(f"    {k:>8}  {s.get('climatology_rate', float('nan')):6.3f}  "
                  f"{s.get('TSS', float('nan')):6.3f}  {s.get('HSS', float('nan')):6.3f}  "
                  f"{s.get('POD', float('nan')):6.3f}  {s.get('FAR', float('nan')):6.3f}  "
                  f"{s.get('AUC', float('nan')):6.3f}  "
                  f"{s.get('BSS_vs_climatology', float('nan')):7.3f}")

    nr = r.get("nowcast_regression", {})
    if nr:
        print(f"\n[4] Nowcast regression (log flux, SoLEXS observing)   MAE {nr['MAE']:.4f} "
              f"{_ci(nr.get('MAE_ci95'))}  RMSE {nr['RMSE']:.4f}  R2 {_fmt(nr['R2'])}")
        if "solexs_calibration_MAE" in nr:
            print(f"    same windows: model MAE {nr['model_MAE_same_windows']:.4f} vs SoLEXS "
                  f"calibration {nr['solexs_calibration_MAE']:.4f}")

    fr = r.get("forecast_regression", {})
    if fr:
        print("\n[5] Forecast regression (log flux), median quantile")
        print(f"    {'horizon':>8}  {'RMSE':>7}  {'persist':>7}  {'SoLEXS':>7}  {'clim':>7}  "
              f"{'vs_pers':>7}  {'vs_SLX':>7}  {'vs_clim':>7}  {'cover':>6}  {'cover*':>6}")
        for k, s in fr.items():
            print(f"    {k:>8}  {s['RMSE']:7.4f}  "
                  f"{s.get('persistence_RMSE', float('nan')):7.4f}  "
                  f"{s.get('solexs_no_change_RMSE', float('nan')):7.4f}  "
                  f"{s.get('climatology_RMSE', float('nan')):7.4f}  "
                  f"{s.get('skill_vs_persistence', float('nan')):7.3f}  "
                  f"{s.get('skill_vs_solexs_no_change', float('nan')):7.3f}  "
                  f"{s.get('skill_vs_climatology', float('nan')):7.3f}  "
                  f"{s.get('interval_coverage', float('nan')):6.3f}  "
                  f"{s.get('interval_coverage_calibrated', float('nan')):6.3f}")
        print("    (persist = GOES no-change, SoLEXS = calibrated SoLEXS no-change; cover* after "
              "the validation-fitted widening)")
        print(f"    (skill > 0 beats that reference. Nominal interval coverage "
              f"{fr[list(fr)[0]].get('nominal_coverage', float('nan')):.2f}.)")
        print("    Beating persistence but NOT climatology, with corr ~ 0, means")
        print("    the model has fallen back to predicting the mean at that horizon.")

    pk = r.get("peak", {})
    if pk:
        print(f"\n[6] Ongoing-event peak   time-to-peak MAE "
              f"{pk['time_to_peak_MAE_min']:.1f} min   "
              f"log-peak-flux MAE {pk['log_peak_flux_MAE']:.3f}  (n={pk['n']})")

    ab = r.get("modality_ablation", {})
    if ab:
        print("\n[7] Modality ablation (flare-in-progress skill on test)")
        for k, s in ab.items():
            print(f"    {k:>10}  TSS {_fmt(s.get('TSS'))}  AUC {_fmt(s.get('AUC'))}")
        co = ab.get("clock_only", {}).get("AUC")
        if co is not None and not (isinstance(co, float) and np.isnan(co)) and co > 0.6:
            print(f"    WARNING: with both instruments blanked the model still "
                  f"reaches AUC {co:.3f}.")
            print("    It is memorising non-instrument inputs (time of day) rather "
                  "than reading")
            print("    the Sun. Set ModelConfig.use_clock = False, or add more days "
                  "of data.")


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
