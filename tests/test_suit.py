"""Tests for the SUIT forecaster (suit/).

A synthetic month of SUIT: a limb-darkened disk whose active region brightens in
the 12 h before each M flare, in plain FITS, gzipped FITS and a zip; small
region-of-interest frames and zoomed frames that must never be used; a filter
named only in the file name. Checks the reader, the full-disk test, exposure
independence, that no row reads the future, and that the whole train path finds
the planted signal.
"""

from __future__ import annotations

import gzip
import io
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarflare.io.goes import GoesFlare  # noqa: E402
from suit.config import SuitConfig, SuitPaths, normalise_filter  # noqa: E402
from suit.dataset import build_features, hourly_table, select_frames  # noqa: E402
from suit.features import feature_names, image_features  # noqa: E402
from suit.io import index_frames, read_image  # noqa: E402
from suit.train import run  # noqa: E402

FAILURES: list[str] = []
T0 = datetime(2024, 5, 1, tzinfo=UTC).timestamp()
DAY = 86400.0
NPX = 128
FLARES = [T0 + (1.7 + 2.1 * k) * DAY + 3600.0 * (k % 5) for k in range(14)]


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def cfg() -> SuitConfig:
    return SuitConfig(filters=("NB03", "BB03"), image_px=64, min_full_px=100, horizons_h=(12,),
                      classes=("M",), embargo_days=1.0)


def amplitude(t: float) -> float:
    """Active-region brightening: rises over the 12 h before each flare, fades after."""
    a = 0.3
    for f in FLARES:
        if f - 12 * 3600.0 <= t <= f:
            a += 1.2 * (1 - (f - t) / (12 * 3600.0))
        elif f < t <= f + 3 * 3600.0:
            a += 1.2 * (1 - (t - f) / (3 * 3600.0))
    return a


def disk(t: float, rng, px: int = NPX, zoom: float = 1.0, exposure: float = 1.0) -> np.ndarray:
    yy, xx = np.indices((px, px))
    cy = cx = px / 2 + rng.normal(0, 0.4)
    r = 0.33 * px * zoom
    rho = np.hypot(yy - cy, xx - cx) / r
    mu = np.sqrt(np.clip(1 - rho ** 2, 0, 1))
    img = np.where(rho <= 1, 1000 * (0.4 + 0.6 * mu), 5.0)
    blob = np.exp(-((yy - (cy - 0.3 * r)) ** 2 + (xx - (cx + 0.2 * r)) ** 2) / (2 * (0.07 * r) ** 2))
    img = img * (1 + amplitude(t) * blob * (rho <= 1)) * (1 + rng.normal(0, 0.01, img.shape))
    return (img * exposure).astype(np.float32)


def hdu(t: float, img: np.ndarray, **cards) -> fits.PrimaryHDU:
    h = fits.PrimaryHDU(img)
    h.header["DATE-OBS"] = datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%S")
    for k, v in cards.items():
        h.header[k] = v
    return h


def stamp(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y%m%dT%H%M%S")


def make_archive(root: Path) -> dict:
    rng = np.random.default_rng(0)
    n = {"roi": 0, "zoom": 0, "gz": 0, "name_only": 0}
    # NB03 every 2 h as plain FITS; a few gzipped, a few naming the filter only in the file name
    for k, t in enumerate(np.arange(T0, T0 + 30 * DAY, 2 * 3600.0)):
        cards = {"EXPTIME": 1.0, "IMG_TYPE": "SYNOPTIC"}
        if k % 50 != 7:
            cards["FTR_NAME"] = "NB03 Mg II k"
        else:
            n["name_only"] += 1
        f = root / "NB03" / f"SUT_{stamp(t)}_NB03.fits"
        f.parent.mkdir(parents=True, exist_ok=True)
        if k % 40 == 3:
            buf = f.with_suffix(".fits.gz")
            with gzip.open(buf, "wb") as g:
                hdu(t, disk(t, rng), **cards).writeto(g)
            n["gz"] += 1
        else:
            hdu(t, disk(t, rng), **cards).writeto(f)
    # zoomed frames at odd hours: full size, but the disk overflows the frame -> rejected
    for t in T0 + 3600.0 * np.array([5, 101, 333]):
        hdu(t, disk(t, rng, zoom=2.2), FTR_NAME="NB03").writeto(root / "NB03" / f"SUT_{stamp(t)}_zoom.fits")
        n["zoom"] += 1
    # small region-of-interest frames -> never used
    for t in T0 + 3600.0 * np.arange(20) * 7 + 60.0:
        hdu(t, disk(t, rng, px=48), FTR_NAME="NB03", IMG_TYPE="FLARE_ROI").writeto(
            root / "NB03" / f"SUT_{stamp(t)}_roi.fits")
        n["roi"] += 1
    # BB03 every 6 h, one zip per day, filter written as "BB3"
    for d in range(30):
        with zipfile.ZipFile(root / f"SUIT_BB03_day{d:02d}.zip", "w") as z:
            for t in T0 + d * DAY + 3600.0 * np.array([0, 6, 12, 18]):
                b = io.BytesIO()
                hdu(t, disk(t, rng), FILTER="BB3").writeto(b)
                z.writestr(f"bb03/SUT_{stamp(t)}_BB03.fits", b.getvalue())
    return n


def truth() -> SimpleNamespace:
    fl = [GoesFlare(i, f - 900, f, f + 1800, False, "M2.0", 2e-5, 1e-6) for i, f in enumerate(FLARES)]
    grid = np.arange(T0, T0 + 32 * DAY, 60.0)
    return SimpleNamespace(flares=fl, time_unix=grid, xrsb=np.ones(grid.size))


def test_filter_names():
    check("filter names are normalised", [normalise_filter(x) for x in ("NB3", "nb03", "NB03 Mg II k", "BB3", "x")]
          == ["NB03", "NB03", "NB03", "BB03", ""])


def test_suit_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        data = root / "suit"
        n = make_archive(data)
        c = cfg()

        frames = index_frames(data, root / "feat" / "frames_index.json")
        by = {}
        for f in frames:
            by.setdefault(f.filt, []).append(f)
        check("every frame indexed (plain, gzipped and zipped)",
              len(frames) == 360 + n["zoom"] + n["roi"] + 120, str(len(frames)))
        check("a filter named only in the file name is found", len(by.get("NB03", [])) == 360 + n["zoom"] + n["roi"])
        check("zipped frames are read in place", len(by.get("BB03", [])) == 120)
        img = read_image(next(f for f in frames if f.member))
        check("an image comes out of a zip", img.shape == (NPX, NPX))

        sel = select_frames(frames, c)
        check("region-of-interest frames are never selected",
              not any(min(f.nx, f.ny) < c.min_full_px for fs in sel.values() for f in fs))

        rng = np.random.default_rng(1)
        a = image_features(disk(T0, rng), c)
        b = image_features(disk(T0, np.random.default_rng(1), exposure=3.0), c)
        check("features do not depend on the exposure (flare mode changes it)",
              a is not None and b is not None and np.allclose(a, b, rtol=1e-4, atol=1e-6))
        check("a zoomed frame with no limb in view gets no features",
              image_features(disk(T0, rng, zoom=2.2), c) is None)
        quiet = image_features(disk(T0 + 0.5 * DAY, rng), c)
        hot = image_features(disk(FLARES[0] - 600.0, rng), c)
        k = feature_names(c).index("excess")
        check("the brightening before a flare raises the excess", hot[k] > 1.5 * quiet[k], f"{hot[k]:.4f} vs {quiet[k]:.4f}")

        feats = build_features(sel, c, root / "feat")
        check("zoomed frames are rejected, the rest used",
              feats["NB03"]["rejected"] == n["zoom"] and feats["NB03"]["used"] == 360, str(feats["NB03"]))
        o = np.arange(T0 + 3600.0, T0 + 30 * DAY, 3600.0)
        X, names, have = hourly_table(feats, o, c)
        T = T0 + 15 * DAY
        later = {f: {**v, "F": np.where((v["t"] > T)[:, None], v["F"] * 2.0 + 1.0, v["F"])} for f, v in feats.items()}
        X2, _, _ = hourly_table(later, o, c)
        past = o < T
        check("no row reads a frame after its origin",
              np.array_equal(np.nan_to_num(X[past], nan=-9), np.nan_to_num(X2[past], nan=-9)))
        check("rows after the change do change", not np.array_equal(np.nan_to_num(X[~past]), np.nan_to_num(X2[~past])))

        paths = SuitPaths(data=data, features=root / "feat", outputs=root / "out", goes=root / "goes",
                          split_run=root / "no_run")
        s = run(paths, c, own_split=True, verbose=False, truth=truth())
        r = s["results"].get(">=M1 within 12 h", {})
        check("training runs and scores the test period", "AUC" in r, str(r)[:200])
        if "AUC" in r:
            check("the planted pre-flare brightening is found (AUC > 0.8)", r["AUC"] > 0.8, str(r["AUC"]))
            check("frequency bias is reported beside the TSS", "FB" in r and r["FB"] > 0)
            check("the split is said to be its own, not the X-ray one", "own chronological" in r["split"])
        check("the summary is written", (root / "out" / "suit_summary.json").exists())
        check("RoI frames are counted as skipped", s["region_of_interest_frames_skipped"] == n["roi"])


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"\n{t.__name__}:")
        t()
    print("\n" + ("All SUIT checks passed." if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
