"""Figure 1: the observational and modelling pipeline of Paper 1.

    python paper_results/plotting/fig01_pipeline.py

Counts are read from paper_results/01_data_inventory.json and
02_experiment_split.json (scripts/paper/data_and_split.py); window settings from
the final model's configuration. GOES-18 XRS enters only as truth (dashed).
"""

from __future__ import annotations

import json

from diagram import arrow, blank, bottom, box, left, right, top
from style import DOUBLE, PAPER, save, setup
import matplotlib.pyplot as plt


def main() -> int:
    setup()
    inv = json.loads((PAPER / "01_data_inventory.json").read_text("utf-8"))
    sp = json.loads((PAPER / "02_experiment_split.json").read_text("utf-8"))
    s, h, g = inv["SoLEXS"], inv["HEL1OS"], inv["GOES"]
    w = sp["window"]
    n_flares = sum(g["flares_labelled_in_observed_segments_by_class"].values())
    fig = plt.figure(figsize=(DOUBLE, 2.6))
    ax = blank(fig)
    a = box(ax, 0.005, 0.60, 0.15, 0.30, f"SoLEXS L1\n{s['days_in_frozen_study']} days\n{s['first_day'][:4]}-"
            f"{s['first_day'][4:6]} onwards\n(zips read in place)", face="0.95", size=6)
    b = box(ax, 0.005, 0.18, 0.15, 0.30, f"HEL1OS L1\n{h['products_used_by_the_study']:,} products\n"
            f"(light curves cached;\nzips kept)", face="0.82", size=6)
    c = box(ax, 0.19, 0.36, 0.14, 0.34, f"per-product cache\n{w['grid_s']:g} s grid\n{inv['cache_build']['ok']:,} "
            f"products,\n{inv['cache_build']['failed']} unreadable\n(frozen)", size=6)
    d = box(ax, 0.365, 0.36, 0.14, 0.34, f"features\n{s['model_features']} SoLEXS\n{h['model_features']} HEL1OS\n"
            "trailing\nbackgrounds only", size=6)
    e = box(ax, 0.54, 0.36, 0.15, 0.34, f"windows\n{w['input_s'] / 3600:g} h input\norigin every {w['stride_s'] / 60:g} min\n"
            f"chronological split,\n{sp['checks']['embargo_days']:g}-day embargoes", size=6)
    f = box(ax, 0.725, 0.36, 0.12, 0.34, "SoLEXHEL-Net\nE1 SoLEXS\nE2 HEL1OS\nE4 both", face="0.88", size=6, bold=False)
    v = box(ax, 0.875, 0.56, 0.12, 0.34, "validation\nthresholds,\ncalibration,\ninterval width", size=6)
    t = box(ax, 0.875, 0.10, 0.12, 0.34, "test (once)\nscores, lead\ntimes, alerts,\nday-block CIs", size=6)
    gb = box(ax, 0.365, 0.02, 0.33, 0.2, f"GOES-18 XRS: truth only, never an input\nflare list >= "
             f"{g['label_threshold']}: {n_flares:,} labelled flares\n1-min flux: the flux target",
             dashed=True, size=6)
    arrow(ax, right(a), (c[0], c[1] + 0.24))
    arrow(ax, right(b), (c[0], c[1] + 0.10))
    for p, q in ((c, d), (d, e), (e, f)):
        arrow(ax, right(p), left(q))
    arrow(ax, right(f), (v[0], v[1] + 0.1))
    arrow(ax, right(f), (t[0], t[1] + 0.24))
    arrow(ax, bottom(v), top(t), text="frozen")
    arrow(ax, (gb[0] + gb[2] * 0.55, gb[1] + gb[3]), bottom(e), dashed=True, text="labels")
    arrow(ax, right(gb), (t[0], t[1] + 0.06), dashed=True, text="scoring")
    cap = (f"Pipeline of Paper 1. SoLEXS and HEL1OS Level-1 products ({s['days_in_frozen_study']} SoLEXS days, "
           f"{h['products_used_by_the_study']} HEL1OS products) are gridded to {w['grid_s']:g} s in a per-product cache, "
           f"turned into {s['model_features']} + {h['model_features']} features with trailing backgrounds, and cut into "
           f"{w['input_s'] / 3600:g} h windows split chronologically with {sp['checks']['embargo_days']:g}-day embargoes. "
           "The same network is trained on SoLEXS only (E1), HEL1OS only (E2) and both (E4); thresholds, probability "
           "calibration and interval widths are fixed on validation and applied once to the test period. GOES-18 XRS "
           f"supplies the labels (flares >= {g['label_threshold']}) and the flux target, and is never a model input.")
    save(fig, "fig01_pipeline", cap, ["paper_results/01_data_inventory.json", "paper_results/02_experiment_split.json"],
         "fig01_pipeline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
