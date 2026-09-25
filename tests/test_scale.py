"""Tests for running on a mission archive rather than a two-day sample.

Each check targets a failure that only appears at scale, and that the
correctness and robustness suites could not see:

* a flare crossing midnight split in two by per-day processing
* the cache re-processing everything, or failing to notice a changed file
* a re-processed day (v1.0 and v1.1) counted twice
* zip reading diverging from reading extracted files
* the chunked rolling percentile disagreeing with the exact one
* per-segment splitting leaking the future across hundreds of segments
* quiet-window thinning dropping flare windows, or touching val/test
* HEL1OS-only windows being trained as "quiet Sun"

    python -m tests.test_scale
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarflare.config import Config, PreprocessConfig  # noqa: E402

FAILURES: list[str] = []
DAY = 86400.0


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Synthetic PRADAN-style SoLEXS day, with flares, written as a zip
# ---------------------------------------------------------------------------

def _solexs_day(t0: float, flares=(), seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One day of 1 s, 340-channel spectra. Flares: (peak_s, rise_s, decay_s, amp)."""
    rng = np.random.default_rng(seed)
    n = 86400
    spec = rng.poisson(0.02, size=(n, 340)).astype(np.float64)
    sec = t0 + np.arange(n, dtype=float)
    chans = np.arange(45, 161)
    w = np.exp(-(chans - 45) / 30.0)
    w /= w.sum()
    for peak, rise, decay, amp in flares:
        prof = np.where(sec <= peak, np.clip(1.0 - (peak - sec) / rise, 0, None),
                        np.exp(-(sec - peak) / decay))
        spec[:, chans] += rng.poisson(amp * prof[:, None] * w[None, :])
    return sec, spec


def _write_day_zip(dirpath: Path, date: str, t0: float, flares=(), version="v1.0",
                   seed: int = 0) -> Path:
    from astropy.io import fits
    import gzip
    import io

    sec, spec = _solexs_day(t0, flares, seed)
    n = sec.size
    base = f"AL1_SLX_L1_{date}_{version}"

    def gz_fits(hdul) -> bytes:
        buf = io.BytesIO()
        hdul.writeto(buf)
        return gzip.compress(buf.getvalue(), compresslevel=1)

    pi = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(fits.ColDefs([
        fits.Column(name="TSTART", format="D", array=sec),
        fits.Column(name="TELAPSE", format="D", array=np.ones(n)),
        fits.Column(name="SPEC_NUM", format="J", array=np.arange(n)),
        fits.Column(name="CHANNEL", format="340K", array=np.tile(np.arange(340), (n, 1))),
        fits.Column(name="COUNTS", format="340D", array=spec),
        fits.Column(name="EXPOSURE", format="D", array=np.ones(n)),
    ]), name="SPECTRUM")])
    pi[0].header["OBS_DATE"] = date
    lc = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(fits.ColDefs([
        fits.Column(name="TIME", format="D", array=sec),
        fits.Column(name="COUNTS", format="D", array=spec[:, 41:].sum(axis=1)),
    ]), name="RATE")])
    gti = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(fits.ColDefs([
        fits.Column(name="START", format="D", array=np.array([sec[0]])),
        fits.Column(name="STOP", format="D", array=np.array([sec[-1]])),
    ]), name="GTI")])

    dirpath.mkdir(parents=True, exist_ok=True)
    out = dirpath / f"{base}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr(f"{base}/SDD1/AL1_SOLEXS_{date}_SDD1_L1.gti.gz", gz_fits(gti))
        z.writestr(f"{base}/SDD2/AL1_SOLEXS_{date}_SDD2_L1.pi.gz", gz_fits(pi))
        z.writestr(f"{base}/SDD2/AL1_SOLEXS_{date}_SDD2_L1.lc.gz", gz_fits(lc))
        z.writestr(f"{base}/SDD2/AL1_SOLEXS_{date}_SDD2_L1.gti.gz", gz_fits(gti))
    return out


# 2026-01-01 00:00:00 UTC, on a 20 s boundary.
T0 = 1767225600.0


def _archive(root: Path, n_days: int, flares_per_day: dict[int, tuple] | None = None):
    flares_per_day = flares_per_day or {}
    for d in range(n_days):
        t0 = T0 + d * DAY
        date = time.strftime("%Y%m%d", time.gmtime(t0))
        _write_day_zip(root, date, t0, flares_per_day.get(d, ()), seed=d)


# ---------------------------------------------------------------------------

def test_chunked_percentile_is_exact():
    import solarflare.preprocess.grid as g
    rng = np.random.default_rng(1)
    x = rng.lognormal(size=5000)
    x[rng.random(5000) < 0.2] = np.nan
    saved = g._PERCENTILE_CHUNK
    try:
        g._PERCENTILE_CHUNK = 1 << 30
        ref_t = g.trailing_percentile(x, 777, 10)
        ref_c = g.running_percentile(x, 777, 10)
        g._PERCENTILE_CHUNK = 333          # many chunk boundaries
        got_t = g.trailing_percentile(x, 777, 10)
        got_c = g.running_percentile(x, 777, 10)
    finally:
        g._PERCENTILE_CHUNK = saved
    check("chunked trailing percentile == single-chunk", np.allclose(ref_t, got_t, equal_nan=True))
    check("chunked centred percentile == single-chunk", np.allclose(ref_c, got_c, equal_nan=True))


def test_zip_reader_matches_directory_reader():
    from solarflare.io.solexs import read_solexs, read_solexs_zip, list_zip_detectors
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        z = _write_day_zip(root, "20260101", T0, ((T0 + 40000, 600, 900, 60),))
        with zipfile.ZipFile(z) as zz:
            zz.extractall(root / "x")
        ex = next((root / "x").rglob("SDD2"))
        a, b = read_solexs_zip(z, "SDD2"), read_solexs(ex)
        same = all(np.array_equal(getattr(a, f), getattr(b, f), equal_nan=True)
                   for f in ("time_unix", "spectra", "exposure", "valid", "gti", "lc_counts"))
        check("zip and extracted-directory reads are identical", same)
        check("zip detector listing ignores SDDs without spectra",
              list_zip_detectors(z) == ["SDD2"], str(list_zip_detectors(z)))


def test_index_deduplicates_versions_and_ignores_partial_downloads():
    from solarflare.preprocess.cache import index_sources
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_day_zip(root / "a", "20260101", T0, version="v1.0")
        _write_day_zip(root / "b", "20260101", T0, version="v1.1")
        part = _write_day_zip(root / "c", "20260102", T0 + DAY)
        part.rename(part.with_name(part.name + ".part"))
        src = [s for s in index_sources([root]) if s.kind == "solexs"]
        check("two versions of one day are indexed once", len(src) == 1, str(len(src)))
        check("...and the higher version wins",
              bool(src) and src[0].version == "v1.1", str([s.version for s in src]))
        check("a .zip.part (download in progress) is never indexed",
              not any(s.date == "20260102" for s in src))


def test_cache_is_incremental_and_invalidates_correctly():
    from solarflare.preprocess.cache import index_sources, build_cache
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _archive(root / "data", 3)
        cfg = PreprocessConfig()
        cdir = root / "cache"
        src = index_sources([root / "data"])

        e1 = build_cache(src, cfg, cdir, workers=1, verbose=False)
        keys1 = {e["key"] for e in e1}
        check("first build caches every day",
              sum(e["status"] == "ok" for e in e1) == 3, str([e["status"] for e in e1]))

        t = time.time()
        e2 = build_cache(src, cfg, cdir, workers=1, verbose=False)
        check("second build reuses every entry", {e["key"] for e in e2} == keys1)
        check("...without re-reading the data", time.time() - t < 1.0,
              f"{time.time() - t:.2f}s")

        z = Path(src[0].path)
        st = z.stat()
        os.utime(z, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # "re-downloaded"
        e3 = build_cache(src, cfg, cdir, workers=1, verbose=False)
        check("a changed file (new mtime) rebuilds exactly that one entry",
              len({e["key"] for e in e3} - keys1) == 1)

        cfg2 = PreprocessConfig(dt_seconds=60.0)
        e4 = build_cache(src, cfg2, cdir, workers=1, verbose=False)
        check("changing a raw-rate setting (grid) rebuilds everything",
              not ({e["key"] for e in e4} & {e["key"] for e in e3}))

        cfg3 = PreprocessConfig(background_window_s=3600.0)  # derive-stage only
        e5 = build_cache(src, cfg3, cdir, workers=1, verbose=False)
        check("changing a derive-stage setting (background) rebuilds nothing",
              {e["key"] for e in e5} == {e["key"] for e in e3})


def test_flare_across_midnight_is_one_event():
    """The reason stitching exists."""
    from solarflare.preprocess.cache import index_sources, build_cache, load_cached
    from solarflare.preprocess.dataset import build_segments_from_raw

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        # Peaks 5 minutes before midnight, decays for an hour into day 2.
        peak = T0 + DAY - 300
        _archive(root / "data", 2, {0: ((peak, 900, 1800, 150),),
                                    1: ((peak, 900, 1800, 150),)})
        cfg = Config(data_root=root / "data", out_dir=root / "out")
        src = index_sources([root / "data"])
        entries = build_cache(src, cfg.pre, root / "cache", workers=1, verbose=False)
        raws = load_cached(entries, root / "cache", "solexs")

        stitched, _ = build_segments_from_raw(raws, [], cfg)
        per_day = [build_segments_from_raw([r], [], cfg)[0] for r in raws]

        ev_stitched = [e for s in stitched for e in s.events]
        near = [e for e in ev_stitched if abs(e.peak_unix - peak) < 600]
        check("stitched days form one continuous segment", len(stitched) == 1,
              str(len(stitched)))
        check("the midnight flare is detected exactly once", len(near) == 1,
              f"{len(near)} events near the peak")
        if near:
            check("...and its decay continues past midnight",
                  near[0].end_unix > T0 + DAY + 600,
                  f"ends {near[0].end_unix - (T0 + DAY):.0f} s after midnight")
        pieces = sum(len(s[0].events) for s in per_day)
        check("processing each day separately does NOT give the same answer "
              "(this is what stitching fixes)", pieces != len(near),
              f"per-day events {pieces}")


def test_global_split_and_thinning():
    from solarflare.preprocess.cache import index_sources, build_cache, load_cached
    from solarflare.preprocess.dataset import (
        build_segments_from_raw, enumerate_windows, chronological_split,
        thin_quiet_training_windows, resolve_split_mode,
    )

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        n_days = 6
        flares = {k: ((T0 + k * DAY + 30000 + 5000 * (k % 3), 600, 1200, 80 + 20 * k),)
                  for k in range(n_days)}
        _archive(root / "data", n_days, flares)
        cfg = Config(data_root=root / "data", out_dir=root / "out")
        cfg.win.large_data_days = 3.0            # force archive behaviour
        cfg.win.stride_seconds = 120.0
        src = index_sources([root / "data"])
        entries = build_cache(src, cfg.pre, root / "cache", workers=1, verbose=False)
        segs, _ = build_segments_from_raw(load_cached(entries, root / "cache", "solexs"),
                                          [], cfg)
        win = enumerate_windows(segs, cfg)
        mode = resolve_split_mode(segs, cfg)
        check("auto split mode switches to global above the archive threshold",
              mode == "global", mode)

        tr, va, te = chronological_split(win, cfg, mode="global")
        t_tr = max(win[i].t_unix for i in tr)
        check("every validation window is after all training + embargo",
              all(win[i].t_unix > t_tr + cfg.train.embargo_s for i in va))
        check("every test window is after all validation + embargo",
              all(win[i].t_unix > max(win[j].t_unix for j in va) + cfg.train.embargo_s
                  for i in te))
        check("splits are disjoint", not (set(tr) & set(va) | set(va) & set(te)))

        thin = thin_quiet_training_windows(segs, win, tr, cfg)
        L = cfg.steps_per_window
        from solarflare.preprocess.dataset import _max_horizon_steps
        H = _max_horizon_steps(cfg)

        def active(i):
            w = win[i]
            s = segs[w.seg]
            return s.in_flare[w.end - L: min(w.end + H, len(s))].any()

        act = [i for i in tr if active(i)]
        check("thinning keeps every flare-adjacent training window",
              set(act) <= set(thin), f"dropped {len(set(act) - set(thin))}")
        quiet_all = len(tr) - len(act)
        quiet_kept = len(thin) - len(act)
        check("thinning removes most quiet training windows",
              quiet_all == 0 or quiet_kept < 0.5 * quiet_all,
              f"kept {quiet_kept}/{quiet_all}")


def test_phase_loss_ignores_windows_without_soft_truth():
    import torch
    from solarflare.models.losses import focal_ce
    logits = torch.tensor([[5.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([2, 0])            # second row: placeholder "quiet"
    mask = torch.tensor([1.0, 0.0])          # ...with no soft X-ray truth
    only_first = focal_ce(logits[:1], target[:1])
    masked = focal_ce(logits, target, mask=mask)
    check("masked phase loss equals the loss on truthful windows only",
          torch.allclose(masked, only_first), f"{masked.item()} vs {only_first.item()}")


def test_prepare_end_to_end_archive_mode():
    """prepare() on a small synthetic archive, with archive mode forced."""
    from solarflare.pipeline import prepare
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        flares = {k: ((T0 + k * DAY + 43000, 600, 1200, 100),) for k in range(5)}
        _archive(root / "data", 5, flares)
        cfg = Config(data_root=root / "data", out_dir=root / "out")
        cfg.win.large_data_days = 2.0
        prep = prepare(cfg, verbose=False)
        check("archive mode engages", prep.archive)
        check("stride raised to the archive stride",
              cfg.win.stride_seconds == cfg.win.large_stride_seconds)
        check("epoch cap applied", cfg.train.max_batches_per_epoch ==
              cfg.train.large_max_batches_per_epoch)
        check("all five days detected their flare",
              prep.meta["n_events"] >= 5, str(prep.meta["n_events"]))
        check("meta records the split mode", prep.meta["split_mode"] == "global")

        # a single-instrument run (paper E1) shares the split: same dates and
        # normaliser, and only windows of the two-instrument split
        cfg1 = Config(data_root=root / "data", out_dir=root / "out")
        cfg1.win.large_data_days = 2.0
        cfg1.model.inputs = "soft"
        soft = prepare(cfg1, verbose=False)
        check("single-instrument run: same split dates", soft.meta["split_dates"] == prep.meta["split_dates"])
        check("...same normaliser", np.array_equal(soft.norm.mean_soft, prep.norm.mean_soft)
              and np.array_equal(soft.norm.std_hard, prep.norm.std_hard))
        check("...windows a subset of each split (here all: this archive is SoLEXS only)",
              all(set(soft.splits[k]) <= set(prep.splits[k]) for k in ("train", "val", "test"))
              and soft.splits == prep.splits and soft.meta["inputs"] == "soft")
        shutil.rmtree(root / "out", ignore_errors=True)


def test_cache_refuses_a_vanished_data_folder():
    """An empty or gutted data folder must stop the cache, not train on nothing."""
    from solarflare.preprocess.cache import DataMissing, build_cache, index_sources
    pre = PreprocessConfig()
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _archive(root / "data", 5)
        build_cache(index_sources([root / "data"]), pre, root / "cache", workers=1, verbose=False)

        def raises(fn):
            try:
                fn()
            except DataMissing:
                return True
            return False

        check("an empty data folder is refused",
              raises(lambda: build_cache([], pre, root / "cache", workers=1, verbose=False)))
        for z in sorted((root / "data").glob("*.zip"))[:3]:
            z.unlink()
        check("losing 3 of 5 cached days is refused",
              raises(lambda: build_cache(index_sources([root / "data"]), pre, root / "cache",
                                         workers=1, verbose=False)))
        os.environ["SOLARFLARE_ALLOW_MISSING"] = "1"
        try:
            ok = not raises(lambda: build_cache(index_sources([root / "data"]), pre, root / "cache",
                                                workers=1, verbose=False))
        finally:
            del os.environ["SOLARFLARE_ALLOW_MISSING"]
        check("SOLARFLARE_ALLOW_MISSING=1 continues on purpose", ok)


def test_cache_survives_deleting_the_raw_files():
    """The ingest loop: cache a day, delete its raw file, and the day still counts.

    With cache_is_source the .npz is the record of what was observed, which is what
    lets scripts/ingest_batch.py free the extracted products one at a time."""
    from solarflare.preprocess.cache import build_cache, index_sources
    pre = PreprocessConfig()
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _archive(root / "data", 4)
        first = build_cache(index_sources([root / "data"]), pre, root / "cache", workers=1, verbose=False)
        n_ok = sum(e.get("status") == "ok" for e in first)
        gone = sorted((root / "data").glob("*.zip"))[0]
        day = gone.name[11:19]
        gone.unlink()                                   # as the ingest script does after caching

        pre.cache_is_source = False
        try:
            build_cache(index_sources([root / "data"]), pre, root / "cache", workers=1, verbose=False)
            kept_off = True
        except Exception:                               # noqa: BLE001 - the guard is the point
            kept_off = False
        check("without cache_is_source a deleted day is refused (guard holds)", not kept_off)

        pre.cache_is_source = True
        again = build_cache(index_sources([root / "data"]), pre, root / "cache", workers=1, verbose=False)
        ok = [e for e in again if e.get("status") == "ok"]
        check("the deleted day still counts", len(ok) == n_ok, f"{len(ok)} vs {n_ok}")
        check("it is marked as read from the cache alone",
              any(e.get("source_deleted") and day in e["source"]["path"] for e in ok))
        # a re-processed version of that same day must not make it count twice
        _write_day_zip(root / "data", day, T0 + (int(day[-2:]) - 1) * DAY, version="v1.1")
        third = build_cache(index_sources([root / "data"]), pre, root / "cache", workers=1, verbose=False)
        ok3 = [e for e in third if e.get("status") == "ok"]
        days = [e["source"]["date"] for e in ok3]
        check("a re-processed day replaces the cached one, never doubles it",
              len(days) == len(set(days)), str(sorted(days)))

        # ingest_batch reads back only what the call built: the carried products
        # come back from every build_cache call, and re-reading them each time
        # made the ingest quadratic over 2,700 HEL1OS products
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ingest_batch", Path(__file__).resolve().parents[1] / "scripts" / "ingest_batch.py")
        ib = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ib)
        carried = [e for e in again if e.get("source_deleted")]
        mine = ib.this_product(again)
        check("the ingest verifies only the product it just built, not the carried ones",
              bool(carried) and not any(e.get("source_deleted") for e in mine)
              and len(mine) == len([e for e in again if e.get("status") == "ok"]) - len(carried),
              f"{len(mine)} checked, {len(carried)} carried")


def test_a_frozen_cache_adds_nothing_new():
    """New days downloaded into data_root must not slip into the study cache: every
    command that loads the archive re-scans data_root, and one new day moves the
    chronological split under every published number."""
    from solarflare.preprocess.cache import FROZEN_MARKER, build_cache, index_sources
    pre = PreprocessConfig()
    pre.cache_is_source = True
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _archive(root / "data", 6)
        cache = root / "cache"
        first = build_cache(index_sources([root / "data"]), pre, cache, workers=1, verbose=False)
        before = sorted(e["source"]["path"] for e in first if e.get("status") == "ok")
        (cache / FROZEN_MARKER).write_text("frozen for the test", encoding="utf-8")
        t_new = T0 + 7 * DAY                                              # a later download
        _write_day_zip(root / "data", time.strftime("%Y%m%d", time.gmtime(t_new)), t_new, seed=7)
        lost = cache / f"{first[0]['key']}.npz"
        lost.unlink()                                  # and a damaged cache (1 of 6: under the 20% guard)
        after = build_cache(index_sources([root / "data"]), pre, cache, workers=1, verbose=False)
        now = sorted(e["source"]["path"] for e in after if e.get("status") == "ok")
        check("a frozen cache does not take in a newly downloaded day", now == before,
              f"{len(now)} vs {len(before)}")
        check("...but still rebuilds a missing file for a day it holds", lost.exists())
        (cache / FROZEN_MARKER).unlink()
        opened = build_cache(index_sources([root / "data"]), pre, cache, workers=1, verbose=False)
        check("without the marker the new day is cached as before",
              sum(e.get("status") == "ok" for e in opened) == len(before) + 1)


def test_ingest_finds_products_filed_under_the_next_day():
    """PRADAN files a HEL1OS product that starts at 23:59:50 under the next day's
    folder inside its zip. The ingest used to derive the folder from the name,
    so it extracted those products, then skipped them as having no light curves
    and left them on disk uncached."""
    import importlib.util
    import zipfile
    spec = importlib.util.spec_from_file_location(
        "ingest_batch", Path(__file__).resolve().parents[1] / "scripts" / "ingest_batch.py")
    ib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ib)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for name, folder in (("HLS_20240328_235950_43194sec_lev1_V211", "2024/03/29"),
                             ("HLS_20240414_000005_43180sec_lev1_V112", "2024/04/14")):
            z = root / f"{name}.zip"
            with zipfile.ZipFile(z, "w") as f:
                for member in ("czt/lightcurve_czt1.fits", "aux/gticzt1.fits", "events/evt.fits"):
                    f.writestr(f"{folder}/{name}/{member}", b"x")
            out = ib.extract_lightcurves(z, root / "ext")
            check(f"{name[4:12]} {name[13:19]}: product found in {folder}",
                  out == root / "ext" / folder / name and (out / "czt" / "lightcurve_czt1.fits").exists(),
                  str(out))
            check(f"{name[4:12]} {name[13:19]}: the event list is not extracted",
                  not (root / "ext" / folder / name / "events").exists())

        # an older version shipped without light curves: seen, not extracted, not a failure
        name = "HLS_20251017_000004_43194sec_lev1_V111"
        z = root / f"{name}.zip"
        with zipfile.ZipFile(z, "w") as f:
            for member in ("aux/gticzt1.fits", "aux/hk.fits", "events/evt.fits"):
                f.writestr(f"2025/10/17/{name}/{member}", b"x")
        check("an event-list-only version is recognised",
              ib.product_in_zip(z) == (f"2025/10/17/{name}", False), str(ib.product_in_zip(z)))
        check("...and nothing is extracted from it", ib.extract_lightcurves(z, root / "ext2") is None
              and not (root / "ext2").exists())


def test_a_gutted_cache_is_refused():
    """cache_is_source makes the .npz files the data of record, so the guard has
    to watch *them*: deleting the cache must fail as loudly as deleting the raw
    data used to, instead of quietly training on whatever is left."""
    from solarflare.preprocess.cache import DataMissing, build_cache, index_sources
    pre = PreprocessConfig()
    pre.cache_is_source = True
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _archive(root / "data", 5)
        cache = root / "cache"
        entries = build_cache(index_sources([root / "data"]), pre, cache, workers=1, verbose=False)
        keys = [e["key"] for e in entries if e.get("status") == "ok"]
        check("built a cache to gut", len(keys) >= 4, str(len(keys)))
        for z in (root / "data").glob("*.zip"):         # the ingest script's delete
            z.unlink()

        (cache / f"{keys[0]}.npz").unlink()             # one hole: under the threshold
        survived = True
        try:
            build_cache(index_sources([root / "data"]), pre, cache, workers=1, verbose=False)
        except DataMissing:
            survived = False
        check("one missing cache file is tolerated", survived)

        for k in keys[1:]:                              # most of it gone: refuse
            (cache / f"{k}.npz").unlink()
        try:
            build_cache(index_sources([root / "data"]), pre, cache, workers=1, verbose=False)
            refused = False
        except DataMissing as exc:
            refused = "cache files are missing" in str(exc)
        check("a gutted cache stops the run", refused)


def test_gather_loader_matches_dataloader():
    """GatherBatches must hand the network exactly what the per-window
    DataLoader did: same windows, values, shapes and dtypes."""
    import torch
    from solarflare.pipeline import make_loaders, prepare
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        flares = {k: ((T0 + k * DAY + 43000, 600, 1200, 100),) for k in range(4)}
        _archive(root / "data", 4, flares)
        cfg = Config(data_root=root / "data", out_dir=root / "out")
        cfg.train.device = "cpu"
        cfg.train.split_mode = "global"
        prep = prepare(cfg, verbose=False)

        def run(loader_kind):
            cfg.train.loader = loader_kind
            tr, va, te, _ = make_loaders(prep, cfg)
            out = {}
            for name, ld in (("val", va), ("test", te)):
                acc: dict = {}
                for b in ld:
                    for k, v in b.items():
                        acc.setdefault(k, []).append(v)
                out[name] = {k: torch.cat(v) for k, v in acc.items()}
            out["train_t"] = torch.cat([b["t_unix"] for b in tr]).double()
            out["train_first"] = next(iter(tr))
            return out

        old, new = run("torch"), run("gather")
        for split in ("val", "test"):
            same = all(
                old[split][k].shape == new[split][k].shape
                and (torch.equal(old[split][k], new[split][k]) if k != "t_unix" else
                     torch.equal(old[split][k], new[split][k].float()))
                for k in old[split])
            check(f"{split}: identical batches", same)
        check("dtypes match (t_unix now float64)",
              all(old["val"][k].dtype == new["val"][k].dtype for k in old["val"] if k != "t_unix")
              and new["val"]["t_unix"].dtype == torch.float64)
        check("one training epoch covers the same windows",
              torch.equal(torch.sort(old["train_t"].float()).values,
                          torch.sort(new["train_t"].float()).values))
        check("training batch keeps the configured batch size",
              len(new["train_first"]["soft"]) == min(cfg.train.batch_size, len(prep.splits["train"])))
        shutil.rmtree(root / "out", ignore_errors=True)


def test_bootstrap_is_fast_and_unchanged_at_archive_scale():
    """The per-event bootstrap must give the same interval as the direct
    implementation, and finish in seconds for thousands of events."""
    from solarflare.forecast import bootstrap_ci
    rng = np.random.default_rng(5)
    groups = rng.integers(0, 60, 3000)
    vals = rng.normal(size=3000) + groups * 0.01

    def stat(idx):
        return float(np.mean(vals[idx]))

    def direct(n_boot=200, seed=0):
        r = np.random.default_rng(seed)
        uniq = np.unique(groups)
        draws = []
        for _ in range(n_boot):
            pick = r.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.where(groups == g)[0] for g in pick])
            draws.append(stat(idx))
        return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    _, lo, hi = bootstrap_ci(None, groups, stat, n_boot=200)
    rlo, rhi = direct()
    check("bootstrap interval identical to the direct implementation",
          abs(lo - rlo) < 1e-12 and abs(hi - rhi) < 1e-12, f"{lo},{hi} vs {rlo},{rhi}")
    big = rng.integers(0, 3000, 60000)
    t = time.time()
    bootstrap_ci(None, big, lambda idx: float(np.mean(big[idx])), n_boot=100)
    check("3000 events x 60k samples bootstraps in under 20 s", time.time() - t < 20,
          f"{time.time() - t:.1f}s")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} scale test groups\n")
    for t in tests:
        print(f"{t.__name__}:")
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  ERROR {t.__name__}: {type(exc).__name__}: {exc}")
            FAILURES.append(f"{t.__name__} (exception)")
        print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("All scale checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
