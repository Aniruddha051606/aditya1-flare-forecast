"""Tests for the E1-E6 day-ahead matrix (suit/matrix.py, suit/xray.py).

SYNTHETIC DATA ONLY: these check the pipeline, never the Sun. Flares follow a
slowly varying activity level that SoLEXS and HEL1OS both see (noisily, so
HEL1OS is mostly redundant); in the "informative" world SUIT also sees a
build-up in the 18 h before each flare that the X-rays do not, in the "null"
world SUIT is pure noise. The matrix must find the first and must not
manufacture the second. Also: no look-ahead in the two X-ray blocks, duplicate
and missing minutes, timestamp alignment, training-only imputation, identical
hours for all six experiments, and the TOML settings.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarflare.io.goes import GoesFlare  # noqa: E402
from solarflare.products import dayahead as D  # noqa: E402
from suit.config import SuitConfig, load_config  # noqa: E402
from suit.matrix import EXPERIMENTS, Block, evaluate, impute, render  # noqa: E402
from suit.xray import hel1os_block  # noqa: E402

FAILURES: list[str] = []
T0 = datetime(2024, 6, 1, tzinfo=UTC).timestamp()
DAY, HOUR = 86400.0, 3600.0


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def world(seed: int, suit_informative: bool, days: int = 420):
    """Hourly origins, GOES truth, and the three blocks of a synthetic Sun."""
    rng = np.random.default_rng(seed)
    o = T0 + HOUR * np.arange(days * 24)
    a = np.zeros(o.size)                                   # activity: correlation time ~3 days
    for i in range(1, o.size):
        a[i] = 0.986 * a[i - 1] + rng.normal(0, 0.17)
    rate = 0.012 * np.exp(1.1 * a)                         # flares per hour
    hit = rng.random(o.size) < rate
    peaks = np.sort(o[hit] + rng.uniform(0, HOUR, hit.sum()))
    pre = np.zeros(o.size)                                 # build-up over the 18 h before each flare
    for p in peaks:
        m = (o > p - 18 * HOUR) & (o <= p)
        pre[m] += 1 - (p - o[m]) / (18 * HOUR)
    known = peaks + 300.0                                  # a flare is known 5 min after its peak

    def recent(h):
        return (np.searchsorted(known, o, side="right") - np.searchsorted(known, o - h * HOUR, side="right")).astype(float)

    solexs = np.column_stack([a + rng.normal(0, 0.5, o.size), a + rng.normal(0, 0.5, o.size),
                              recent(24), recent(24), recent(72)])
    hel1os = np.column_stack([a + rng.normal(0, 0.8, o.size), a + rng.normal(0, 0.8, o.size), recent(24)])
    sig = pre if suit_informative else np.zeros(o.size)
    suit = np.column_stack([sig + rng.normal(0, 0.35, o.size), sig + rng.normal(0, 0.35, o.size)])
    have_suit = rng.random(o.size) > 0.25                  # a quarter of the hours without SUIT
    suit[~have_suit] = np.nan
    have_h = rng.random(o.size) > 0.1
    blocks = {"solexs": Block(solexs, ["flux_now", "flux_mean_24h", "n_C_24h", "n_M_24h", "n_M_72h"],
                              np.ones(o.size, bool)),
              "hel1os": Block(hel1os, ["czt_now", "cdte_now", "n_bursts_24h"], have_h),
              "suit": Block(suit, ["NB03_excess", "NB08_excess"], have_suit, impute=True)}
    fl = [GoesFlare(i, p - 900, p, p + 1800, False, "M2.0", 2e-5, 1e-6) for i, p in enumerate(peaks)]
    grid = np.arange(T0, T0 + (days + 2) * DAY, 60.0)
    truth = SimpleNamespace(flares=fl, time_unix=grid, xrsb=np.ones(grid.size))
    targets = {("M", 24): D.window_targets(truth, o, 0.0, 24.0)["M"]}
    split = (T0 + 250 * DAY, T0 + 330 * DAY)
    return o, blocks, targets, split, {"M": peaks, "C": peaks}


def test_matrix_finds_real_information_and_invents_none():
    cfg = SuitConfig()
    for label, informative in (("informative SUIT", True), ("null SUIT", False)):
        o, blocks, targets, split, peaks = world(7, informative)
        s = evaluate(o, blocks, targets, split, peaks, cfg, provenance="synthetic")
        r = s["results"][">=M1 within 24 h"]
        check(f"{label}: scored", "experiments" in r, str(r)[:160])
        if "experiments" not in r:
            continue
        common = blocks["hel1os"].observed & blocks["suit"].observed
        check(f"{label}: hours without SUIT or HEL1OS are left out of all six",
              s["hours"]["all_three"] == int(common.sum()))
        n = r["n"]["test"]
        check(f"{label}: every comparison is paired on the same forecasts",
              all(c["paired_forecasts"] == n for c in r["comparisons"]))
        check(f"{label}: each experiment uses exactly its instruments",
              all(r["experiments"][E]["n_features"] == sum(len(blocks[b].names) for b in bl)
                  for E, bl in EXPERIMENTS.items()))
        check(f"{label}: FB beside every TSS", all("FB" in e and e["FB"] > 0 for e in r["experiments"].values()))
        gate = next(c for c in r["comparisons"] if c["comparison"] == "E6 - E4")
        if informative:
            check("informative SUIT: E6 - E4 gain found, interval above zero",
                  gate["AUC_gain"] > 0 and gate["AUC_gain_ci"][0] > 0, str(gate))
            check("informative SUIT: the gate passes", s["gate"]["verdict"].startswith("PASSED"), s["gate"]["verdict"])
            e51 = next(c for c in r["comparisons"] if c["comparison"] == "E5 - E1")
            check("informative SUIT: E5 - E1 gain found", e51["AUC_gain"] > 0 and e51["AUC_gain_ci"][0] > 0, str(e51))
        else:
            check("null SUIT: no gain is manufactured (interval not above zero)",
                  gate["AUC_gain_ci"][0] <= 0, str(gate))
            check("null SUIT: the gate does not pass", not s["gate"]["verdict"].startswith("PASSED"),
                  s["gate"]["verdict"])
        md = render(s)
        check(f"{label}: the report says it is synthetic", "SYNTHETIC DATA" in md and s["provenance"] == "synthetic")


def test_imputation_uses_training_hours_only():
    X = np.array([[1.0], [3.0], [np.nan], [100.0], [np.nan]])
    tr = np.array([True, True, False, False, False])
    X2 = X.copy()
    X2[3] = -50.0                                          # change a test-period value
    check("missing values are filled from training medians", impute(X, tr)[2, 0] == 2.0)
    check("test-period values never move the fill",
          np.array_equal(impute(X, tr)[[2, 4]], impute(X2, tr)[[2, 4]]))


def _minutes(t0, n, rng):
    t = t0 + 60.0 * np.arange(n)
    return {"cdte_5_20": (t, 5 + rng.random(n)), "czt_20_40": (t, 300 + 50 * rng.random(n))}


def test_hel1os_block_reads_no_future_and_aligns_on_time():
    rng = np.random.default_rng(3)
    mins = _minutes(T0, 20 * 1440, rng)
    bursts = {"known": np.sort(T0 + rng.uniform(0, 20 * DAY, 40)), "czt": rng.uniform(0, 500, 40)}
    o = T0 + HOUR * np.arange(1, 20 * 24)
    f1, ok1 = hel1os_block(mins, bursts, o)
    T = T0 + 10 * DAY
    later = {b: (t, np.where(t + 60.0 > T, r * 7.0 + 3.0, r)) for b, (t, r) in mins.items()}
    b2 = {"known": bursts["known"], "czt": np.where(bursts["known"] > T, bursts["czt"] * 9, bursts["czt"])}
    f2, _ = hel1os_block(later, b2, o)
    past = o <= T
    check("HEL1OS: no feature reads a minute or burst after its origin",
          f1[past].equals(f2[past]), str((f1[past] != f2[past]).sum().sort_values().tail(3).to_dict()))
    check("HEL1OS: later rows do change", not f1[~past].equals(f2[~past]))
    i = 50
    t, r = mins["czt_20_40"]
    last = r[np.searchsorted(t + 60.0, o[i], side="right") - 1]
    check("HEL1OS: 'now' is the last minute ended by the origin (timestamps, not indices)",
          abs(f1["czt_20_40_now"].iloc[i] - np.log10(last)) < 1e-12)
    dup = {b: (np.concatenate([t, t[:500]]), np.concatenate([r, r[:500]])) for b, (t, r) in mins.items()}
    f3, _ = hel1os_block(dup, bursts, o)
    check("HEL1OS: duplicate minutes are counted once", f3.equals(f1))
    gap = {b: (t[(t < T0 + 5 * DAY) | (t > T0 + 6 * DAY)], r[(t < T0 + 5 * DAY) | (t > T0 + 6 * DAY)])
           for b, (t, r) in mins.items()}
    _, ok4 = hel1os_block(gap, bursts, o)
    in_gap = (o > T0 + 5 * DAY + 4 * HOUR) & (o < T0 + 6 * DAY)
    check("HEL1OS: hours inside a data gap are not 'observed'", not ok4[in_gap].any() and ok1[in_gap].all())


def test_solexs_block_reads_no_future():
    """The day-ahead system's own SoLEXS features, used unchanged by E1."""
    rng = np.random.default_rng(4)
    tm = T0 + 60.0 * np.arange(15 * 1440)
    lf = -6 + 0.3 * rng.standard_normal(tm.size)
    hard = 0.1 + 0.01 * rng.random(tm.size)
    known = np.sort(T0 + rng.uniform(0, 15 * DAY, 30))
    pf = 10 ** rng.uniform(-7, -4.5, 30)
    f1, _, o = D.activity_features(tm, lf, hard, known, pf)
    T = T0 + 8 * DAY
    f2, _, _ = D.activity_features(tm, np.where(tm + 60 > T, lf + 2.0, lf), hard, known,
                                   np.where(known > T, pf * 50, pf))
    past = o <= T
    check("SoLEXS: no feature reads a minute or flare after its origin", f1[past].equals(f2[past]))


def test_settings_file():
    cfg = load_config()
    check("config/suit.toml loads, with the gate fixed in it", cfg.gate_class == "M" and cfg.gate_horizon_h == 24)
    check("settings lists become tuples", isinstance(cfg.filters, tuple) and cfg.n_required() == len(cfg.filters))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "suit.toml"
        p.write_text('[suit]\nfliters = ["NB03"]\n', encoding="utf-8")
        try:
            load_config(p)
            ok = False
        except ValueError as exc:
            ok = "fliters" in str(exc)
        check("a misspelt setting is an error, not a silent default", ok)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"\n{t.__name__}:")
        t()
    print("\n" + ("All SUIT matrix checks passed." if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
