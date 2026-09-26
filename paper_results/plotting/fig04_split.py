"""Figure 4: daily observing time of each instrument, with the chronological
train / validation / test split and the embargo gaps.

    python paper_results/plotting/fig04_split.py

Sources: paper_results/02_experiment_split.json (split, from
scripts/paper/data_and_split.py) and the frozen study cache (cache/manifest.json
and each product's 20 s coverage record). Observing time per UTC day is the
union of 20 s bins with data, per instrument, so a product present in two
versions is not counted twice.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np

from style import DOUBLE, HEL1OS, PAPER, SOLEXS, SPLIT, grid_seconds, save, setup
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from solarflare.settings import load_settings

DT = grid_seconds()


def daily_hours(cache, kind: str) -> tuple[np.ndarray, np.ndarray]:
    entries = [e for e in json.loads((cache / "manifest.json").read_text("utf-8"))
               if e.get("status") == "ok" and e["source"]["kind"] == kind]
    bins = set()
    for e in entries:
        with np.load(cache / f"{e['key']}.npz") as z:
            t, c = z["time_unix"], z["coverage"]
        bins.update(np.floor(t[c > 0] / DT).astype(np.int64).tolist())
    b = np.fromiter(bins, np.int64)
    day = np.floor(b * DT / 86400).astype(np.int64)
    days, n = np.unique(day, return_counts=True)
    return days, n * DT / 3600.0


def main() -> int:
    setup()
    S = load_settings()
    sp = json.loads((PAPER / "02_experiment_split.json").read_text("utf-8"))
    fig, axes = plt.subplots(3, 1, figsize=(DOUBLE, 3.1), sharex=True,
                             gridspec_kw={"hspace": 0.12, "height_ratios": [0.32, 1, 1]})

    def d(x):
        return datetime.fromtimestamp(float(x), UTC)

    to_dt = mdates.date2num
    spans = []
    for k, v in sp["splits"].items():
        a = datetime.strptime(v["first_origin_utc"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        b = datetime.strptime(v["last_origin_utc"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        spans.append((to_dt(a), to_dt(b), SPLIT[k]))
    emb = sp["checks"]["embargo_days"]
    bar = axes[0]
    for a, b, st in spans:
        bar.axvspan(a, b, facecolor=st["face"], hatch=st["hatch"], edgecolor=st["edge"], lw=0.8)
        bar.text((a + b) / 2, 0.5, st["label"], ha="center", va="center", fontsize=7,
                 bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
    for (_, b0, _), (a1, _, _) in zip(spans[:-1], spans[1:]):
        bar.text((b0 + a1) / 2, 0.5, f"{emb:g} d\nembargo", ha="center", va="center", fontsize=6)
    bar.set_yticks([])
    bar.set_ylim(0, 1)
    for s_ in ("left", "right", "top"):
        bar.spines[s_].set_visible(False)
    for ax, (kind, name, colour) in zip(axes[1:], (("solexs", "SoLEXS", SOLEXS), ("hel1os", "HEL1OS", HEL1OS))):
        days, hours = daily_hours(S.cache, kind)
        full = np.arange(days.min(), days.max() + 1)
        h = np.zeros(full.size)
        h[days - days.min()] = hours
        x = [to_dt(d(v * 86400)) for v in full] + [to_dt(d((full[-1] + 1) * 86400))]
        ax.stairs(h, x, fill=True, color=colour, alpha=0.85, lw=0)
        for a, b, st in spans:
            for edge in (a, b):
                ax.axvline(edge, color=st["edge"], ls="--", lw=0.8)
        ax.set_ylim(0, 24.5)
        ax.set_yticks([0, 12, 24])
        ax.set_ylabel(f"{name}\n(h per day)")
    ax = axes[-1]
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlabel("Date (UTC)")
    cap = (f"Top: the chronological split used by every experiment. Middle and bottom: daily observing time of "
           f"SoLEXS and HEL1OS in the frozen study data (union of {DT:g} s "
           f"bins with data); dashed lines mark the split boundaries (colours as in the top bar). Split: "
           f"training {sp['splits']['train']['first_origin_utc'][:10]} "
           f"to {sp['splits']['train']['last_origin_utc'][:10]}, validation "
           f"{sp['splits']['val']['first_origin_utc'][:10]} to {sp['splits']['val']['last_origin_utc'][:10]}, test "
           f"{sp['splits']['test']['first_origin_utc'][:10]} to {sp['splits']['test']['last_origin_utc'][:10]}. "
           f"Between the blocks are the {emb:g}-day embargoes; windows inside them are used by no split. "
           + ", ".join(f"{SPLIT[k]['label']} {v['windows']:,} windows" for k, v in sp["splits"].items()) + ".")
    save(fig, "fig04_split", cap, ["paper_results/02_experiment_split.json", "cache/manifest.json (+ cached coverage)"],
         "fig04_split.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
