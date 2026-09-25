"""The example flare shown in Figures 2 and 9, chosen by a fixed rule, and its
light curves from the frozen study data.

Rule (decided before looking at any prediction): the GOES-18 flare with the
highest peak flux whose peak lies in the test period and for which SoLEXS and
HEL1OS both have data for at least MIN_COVERAGE of the interval from
PRE_MIN before the GOES start to POST_MIN after the GOES end.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np

from style import PAPER, grid_seconds
from solarflare.settings import load_settings

PRE_MIN, POST_MIN, MIN_COVERAGE = 30, 30, 0.9
DT = grid_seconds()


def _entries(cache, kind: str, t0: float, t1: float) -> list[dict]:
    es = [e for e in json.loads((cache / "manifest.json").read_text("utf-8"))
          if e.get("status") == "ok" and e["source"]["kind"] == kind and e["t_stop"] > t0 and e["t_start"] < t1]
    return sorted(es, key=lambda e: str(e["source"].get("version", "")))      # a later version overwrites


def series(cache, kind: str, t0: float, t1: float) -> tuple[np.ndarray, dict, np.ndarray]:
    """Cached 20 s rates of one instrument on [t0, t1): time, {channel: values}, observed."""
    grid = np.arange(np.floor(t0 / DT) * DT, t1, DT)
    vals, cov = {}, np.zeros(grid.size, bool)
    for e in _entries(cache, kind, t0, t1):
        with np.load(cache / f"{e['key']}.npz") as z:
            t, v, c, names = z["time_unix"], z["values"], z["coverage"], [str(n) for n in z["names"]]
        k = np.rint((t - grid[0]) / DT).astype(np.int64)
        ok = (k >= 0) & (k < grid.size) & (c > 0)
        for j, n in enumerate(names):
            vals.setdefault(n, np.full(grid.size, np.nan))[k[ok]] = v[ok, j]
        cov[k[ok]] = True
    return grid, vals, cov


def pick():
    """(flare, t0, t1) for the example, following the rule in the module docstring."""
    from solarflare.io.goes import load_goes

    S = load_settings()
    sp = json.loads((PAPER / "02_experiment_split.json").read_text("utf-8"))["splits"]["test"]
    a = datetime.strptime(sp["first_origin_utc"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()
    b = datetime.strptime(sp["last_origin_utc"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()
    flares = sorted((f for f in load_goes(S.goes_dir).flares if a <= f.peak_unix <= b), key=lambda f: -f.peak_flux)
    for f in flares:
        t0, t1 = f.start_unix - 60.0 * PRE_MIN, f.end_unix + 60.0 * POST_MIN
        if all(series(S.cache, kind, t0, t1)[2].mean() >= MIN_COVERAGE for kind in ("solexs", "hel1os")):
            return f, t0, t1
    raise SystemExit("no test-period flare meets the example rule")
