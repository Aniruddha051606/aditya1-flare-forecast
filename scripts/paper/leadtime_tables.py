"""Paper steps 12-13: lead times and false-alarm operating points of the E4 alerts.

    python scripts/paper/leadtime_tables.py

Reads the alerts stage's output (outputs/alerts/lead_times.csv and
leadtime_summary.json, written by ``python -m solarflare alerts`` from the frozen
E4 model; definitions in solarflare/products/leadtime.py) and writes:

  leadtime_events.csv    one row per test flare and alert type: warned or not,
                         lead before the GOES peak (the stage's definition) and
                         relative to the GOES start, at the primary operating point
  leadtime_summary.csv   per alert type, method and operating point: detection
                         rate, chance rate, event TSS, false alarms per day on test,
                         lead statistics; per GOES class only where the class has at
                         least MIN_EVENTS flares (flagged otherwise)
  operating_points.csv   each threshold and the validation false-alarm rate it was
                         chosen at

No threshold is chosen here: all were fixed on the validation period by that
stage. A flare is one event (never a 20 s window).
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime

import numpy as np

from common import PAPER, write_json  # noqa: E402  (sets sys.path)
from solarflare.settings import load_settings

S = load_settings()
MIN_EVENTS = 20
ALERTS = {"C": "GOES >= C1 flare within 15 min", "M": "GOES flux reaches M1 within 30 min"}
METHODS = {"model": "network (E4)", "combined": "network or SoLEXS flux now (E4)",
           "trend": "SoLEXS trend (15-min extrapolation)", "current": "SoLEXS flux now",
           "hope": "hot-onset trigger (SoLEXS)", "rule": "catalogue rise rule", "rule_C_level": "rise rule at >= C1 flux"}


def _t(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=UTC).timestamp()


def _num(v):
    return None if v in ("", None) else float(v)


def events(rows: list[dict], primary: dict) -> list[dict]:
    out = []
    methods = [k.removeprefix("lead_min_") for k in rows[0] if k.startswith("lead_min_")]
    for r in rows:
        c = r["alert_type"]
        peak, start = _t(r["peak_utc"]), _t(r["start_utc"])
        e = {"alert": ALERTS[c], "operating_point": f"{primary[c]:g} false alarms/day on validation",
             "goes_class": r["goes_class"], "class": r["goes_class"][0], "start_utc": r["start_utc"],
             "peak_utc": r["peak_utc"], "rise_min": round((peak - start) / 60.0, 1),
             "window_min": int(r["window_min"]), "hel1os_coverage": float(r["hel1os_frac"])}
        for m in methods:
            lead = _num(r[f"lead_min_{m}"])
            e[f"warned_{m}"] = lead is not None
            e[f"lead_before_peak_min_{m}"] = lead
            e[f"lead_before_start_min_{m}"] = None if lead is None else round(lead - (peak - start) / 60.0, 1)
            if c == "M" and r.get("m1_reached_utc"):
                e[f"lead_before_M1_min_{m}"] = None if lead is None else round(
                    lead - (peak - _t(r["m1_reached_utc"])) / 60.0, 1)
        out.append(e)
    return out


def lead_stats(leads: np.ndarray) -> dict:
    ld = leads[np.isfinite(leads)]
    if not ld.size:
        return {}
    return {k: round(float(v), 2) for k, v in (
        ("median_lead_min", np.median(ld)), ("mean_lead_min", np.mean(ld)), ("p10_lead_min", np.percentile(ld, 10)),
        ("q25_lead_min", np.percentile(ld, 25)), ("q75_lead_min", np.percentile(ld, 75)),
        ("p90_lead_min", np.percentile(ld, 90)))}


def summary(s: dict, ev: list[dict]) -> tuple[list[dict], list[dict]]:
    rows, ops = [], []
    primary = s["primary_false_alarms_per_day"]
    for c, alert in ALERTS.items():
        res = s["results"][c]
        for m, label in METHODS.items():
            if m not in res:
                continue
            for op, d in res[m].items():
                spec = s["operating_points"][c].get(m, {}).get(op, {})
                base = {"alert": alert, "method": label, "operating_point": op, "threshold": d.get("threshold"),
                        "chosen_on": "validation", "validation_false_alarms_per_day": spec.get("val_false_per_day")}
                ops.append(base)
                row = {**base, "scope": "all flares", "n_flares": d["n"], "warned": d["warned"],
                       "detection_rate": d["TPR"], "chance_rate": d["chance"], "event_TSS": d["event_TSS"],
                       "false_alarms_per_day_test": d["false_per_day"], "alert_on_fraction_test": d.get("duty_cycle"),
                       "median_lead_min": d["median_lead_min"],
                       "q25_lead_min": d["q25_lead_min"], "q75_lead_min": d["q75_lead_min"],
                       "lead_ge_5min": d["lead_ge_5min"], "lead_ge_10min": d["lead_ge_10min"]}
                if op == f"fa_{primary[c]:g}":
                    # the per-flare table is at this operating point: add percentiles and the mean
                    lead = np.array([e[f"lead_before_peak_min_{m}"] if e[f"lead_before_peak_min_{m}"] is not None
                                     else np.nan for e in ev if e["alert"] == alert and f"warned_{m}" in e])
                    row.update({k: v for k, v in lead_stats(lead).items() if k not in row or row[k] is None})
                    row.update({f"{k}_all": v for k, v in lead_stats(lead).items()})
                rows.append(row)
                for cls in "CMX":
                    k = d.get(f"class_{cls}")
                    if not k:
                        continue
                    rows.append({**base, "scope": f"GOES class {cls}", "n_flares": k["n"],
                                 "detection_rate": k["TPR"], "median_lead_min": k["median_lead_min"],
                                 "note": "" if k["n"] >= MIN_EVENTS else f"fewer than {MIN_EVENTS} flares: indicative only"})
    return rows, ops


def _write(name: str, rows: list[dict]) -> None:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(PAPER / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> paper_results/{name} ({len(rows)} rows)")


def main() -> int:
    s = json.loads((S.alerts / "leadtime_summary.json").read_text("utf-8"))
    with open(S.alerts / "lead_times.csv", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    ev = events(raw, s["primary_false_alarms_per_day"])
    rows, ops = summary(s, ev)
    PAPER.mkdir(parents=True, exist_ok=True)
    _write("leadtime_events.csv", ev)
    _write("leadtime_summary.csv", rows)
    _write("operating_points.csv", ops)
    write_json("leadtime_source.json", {
        "source": ["outputs/alerts/lead_times.csv", "outputs/alerts/leadtime_summary.json"],
        "model": s["model"], "model_sha256": s["model_sha256"], "test_period": s["test_period"],
        "test_days_with_data": s["test_days_with_data"], "primary_false_alarms_per_day": s["primary_false_alarms_per_day"],
        "definitions": "solarflare/products/leadtime.py module docstring: window = 30 min before the GOES start "
                       "(or the previous flare's peak) to the GOES peak; warned = alert ON at some minute of it; "
                       "lead = GOES peak minus the first ON minute; chance = the same windows moved 2 h where no "
                       "flare of the class occurs; event TSS = warned - chance; false alarms = ON episodes with no "
                       "flare of the class up to 15 min (C) / 30 min (M) after, per day of data",
        "min_events_for_class_statistics": MIN_EVENTS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
