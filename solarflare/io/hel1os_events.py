"""HEL1OS L1 photon event lists (``events/evt.fits``): finding and reading them.

Measured on the real products (2026-09-18):

* One HDU per detector (``CZT1-EVENTS``, ``CZT2-EVENTS``, ``CDTE1-EVENTS``,
  ``CDTE2-EVENTS``) with ``mjd`` (UTC), ``hlsobt`` (onboard seconds), ``ener``
  (keV, already calibrated) and, for CZT, ``pix``.
* Onboard times are 10 ms ticks: ``hlsobt`` takes 100 fractional values. Bins
  that are not whole ticks alias the quantisation into a fake ~20% modulation,
  so every series here is built on the tick grid.
* The ``mjd`` column is a per-packet stamp: against ``hlsobt`` it jumps by up to
  +-1 s from one packet to the next, which turns a smooth 0.1 s light curve
  into noise 50x Poisson. Relative timing uses ``hlsobt``; UTC comes from the
  median offset over the requested interval (good to ~1 s).
* Rows are packet-ordered, not time-ordered (about 4% out of order).
* At high rates the stream has gaps: at an M6 peak, 10-80 ms with no event in
  either CZT, about every 0.65 s. ``live`` marks ticks where any detector
  recorded an event.
* CZT carries an Am-241 source: a 59.5 keV line in every file, useful as an
  energy-scale check. ``aux/cztdis/czt?dispix.txt`` lists pixels disabled onboard.

Products are found both as extracted folders and inside the PRADAN zips, which
are read in place: the archive keeps only the zips (the event lists alone would
need ~460 GB unpacked, 140 MB per product), and a flare needs a few minutes of
one product. Decompressing one event list takes about a second.

A few products from the big flare storms are far larger: HLS_20251112_000006
holds a 9.3 GB event list (1.9 GB zipped). Decompressed into memory and
converted column by column, one read of it took more than 12 GB. Event lists
above ``IN_MEMORY_MAX_BYTES`` are therefore unpacked to a scratch file and
memory-mapped (the last one is kept for the next read, since flares are read in
time order), and ``read_events`` converts only the rows near the requested
window, found by scanning the time column in chunks.
"""

from __future__ import annotations

import atexit
import io
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path, PurePosixPath

import numpy as np
from astropy.io import fits

TICK_S = 0.01
DETECTORS = ("CZT1", "CZT2", "CDTE1", "CDTE2")
_NAME = re.compile(r"HLS_(\d{8})_(\d{6})_(\d+)sec_lev1_V(\d+)")
MJD_UNIX0 = 40587.0


EVT = "events/evt.fits"
#: Event lists larger than this are memory-mapped from a scratch file, not held in RAM.
IN_MEMORY_MAX_BYTES = 512 * 1024**2
#: Folder for that scratch file (default: the system temporary folder).
SCRATCH_ENV = "SOLARFLARE_SCRATCH"
#: Free space left on the scratch drive after unpacking.
SCRATCH_RESERVE_BYTES = 10 * 1024**3
#: Rows of the time column converted at once while locating a window.
ROW_CHUNK = 4_000_000
#: Margin around the requested window when locating rows by packet UTC, which
#: jitters by up to +-1 s against the onboard clock.
MARGIN_S = 120.0


@dataclass(frozen=True)
class Product:
    path: Path | PurePosixPath    # extracted folder, or the product's folder inside ``zip``
    t_start: float
    t_stop: float
    version: int
    zip: Path | None = None       # set when the product is read from inside its zip


def _product(m: re.Match, path, zip_path: Path | None = None) -> Product:
    t0 = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=UTC).timestamp()
    return Product(path, t0, t0 + float(m.group(3)), int(m.group(4)), zip_path)


@lru_cache(maxsize=8192)
def _namelist(zip_path: str) -> frozenset[str]:
    with zipfile.ZipFile(zip_path) as z:
        return frozenset(z.namelist())


@lru_cache(maxsize=2)
def product_index(root: str) -> tuple[Product, ...]:
    """Every HEL1OS product with an event list under ``root`` (several roots may
    be joined with os.pathsep): extracted folders and products inside zips.

    A product found both ways is read from the extracted copy (memory-mapped).
    Among zips, the product's own zip wins over a neighbour's zip that happens
    to carry it too (8 of the 2,719 PRADAN zips hold two products)."""
    found: dict[str, tuple[int, Product]] = {}      # name -> (rank, product); lower rank wins
    for r in (x for x in str(root).split(os.pathsep) if x):
        base = Path(r)
        if not base.is_dir():
            continue
        for d in base.rglob("HLS_*"):
            m = _NAME.fullmatch(d.name)
            if m and d.is_dir() and (d / EVT).exists():
                found[d.name] = (0, _product(m, d))
        for z in sorted(base.rglob("HLS_*.zip")):
            try:
                names = _namelist(str(z))
            except (OSError, zipfile.BadZipFile):
                continue
            for n in names:
                if not n.endswith("/" + EVT):
                    continue
                prefix = n[: -len(EVT) - 1]
                name = prefix.rsplit("/", 1)[-1]
                m = _NAME.fullmatch(name)
                if not m:
                    continue
                rank = 1 if z.stem == name else 2
                if name not in found or rank < found[name][0]:
                    found[name] = (rank, _product(m, PurePosixPath(prefix), z))
    return tuple(sorted((p for _, p in found.values()), key=lambda p: (p.t_start, -p.version)))


def product_for(root: str | Path, t0: float, t1: float) -> Product | None:
    """The product covering the most of [t0, t1]; ties go to the higher version
    (overlapping versions carry identical telemetry, see hel1os.py)."""
    best, best_key = None, None
    for p in product_index(str(root)):
        ov = min(p.t_stop, t1) - max(p.t_start, t0)
        if ov <= 0:
            continue
        key = (round(ov, 0), p.version)
        if best_key is None or key > best_key:
            best, best_key = p, key
    return best


@lru_cache(maxsize=1)
def _zip_bytes(zip_path: str, member: str) -> bytes:
    """One decompressed event list. Flares are read in time order, so the next
    call usually wants the same product again: keep the last one."""
    with zipfile.ZipFile(zip_path) as z:
        return z.read(member)


@lru_cache(maxsize=64)
def _member_size(zip_path: str, member: str) -> int:
    with zipfile.ZipFile(zip_path) as z:
        return z.getinfo(member).file_size


_scratch: dict = {"key": None, "path": None, "stale": []}


def _drop_scratch() -> None:
    """Delete scratch files no longer needed (a file still mapped on Windows is
    retried on the next call and at exit)."""
    if _scratch["path"] is not None:
        _scratch["stale"].append(_scratch["path"])
    _scratch.update(key=None, path=None)
    keep = []
    for p in _scratch["stale"]:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            keep.append(p)
    _scratch["stale"] = keep


atexit.register(_drop_scratch)


def _unpacked(zip_path: str, member: str, size: int) -> Path:
    """A large event list unpacked to a scratch file; the previous one is deleted."""
    key = (zip_path, member)
    if _scratch["key"] == key and _scratch["path"].exists():
        return _scratch["path"]
    _drop_scratch()
    folder = Path(os.environ.get(SCRATCH_ENV) or tempfile.gettempdir())
    free = shutil.disk_usage(folder).free
    if free < size + SCRATCH_RESERVE_BYTES:
        raise OSError(f"reading {Path(zip_path).name} needs {size / 1e9:.1f} GB of scratch space in {folder}, "
                      f"which has {free / 1e9:.1f} GB free; set {SCRATCH_ENV} to a folder with room")
    fd, name = tempfile.mkstemp(prefix="hel1os_evt_", suffix=".fits", dir=folder)
    os.close(fd)
    path = Path(name)
    try:
        with zipfile.ZipFile(zip_path) as z, z.open(member) as src, open(path, "wb") as dst:
            shutil.copyfileobj(src, dst, 16 * 1024**2)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    _scratch.update(key=key, path=path)
    return path


def _open_events(product: Product):
    # "denywrite" maps the file read-only. astropy's default ("readonly") maps it
    # copy-on-write, which on Windows reserves commit for the whole file: 10 GB
    # for the largest event list, though nothing is ever written.
    if product.zip is None:
        return fits.open(product.path / EVT, mode="denywrite", memmap=True)
    zip_path, member = str(product.zip), f"{product.path.as_posix()}/{EVT}"
    size = _member_size(zip_path, member)
    if size > IN_MEMORY_MAX_BYTES:
        return fits.open(_unpacked(zip_path, member, size), mode="denywrite", memmap=True)
    return fits.open(io.BytesIO(_zip_bytes(zip_path, member)))


def disabled_pixels(product: Product, det: str) -> set[int]:
    rel = f"aux/cztdis/{det.lower()}dispix.txt"
    if product.zip is None:
        f = product.path / rel
        if not f.exists():
            return set()
        text = f.read_text()
    else:
        member = f"{product.path.as_posix()}/{rel}"
        if member not in _namelist(str(product.zip)):
            return set()
        with zipfile.ZipFile(product.zip) as z:
            text = z.read(member).decode("ascii", "replace")
    return {int(x) for x in text.split() if x.strip().lstrip("-").isdigit()}


@dataclass
class Events:
    det: str
    tick: np.ndarray       # int64 onboard 10 ms ticks, sorted
    energy: np.ndarray     # keV
    pix: np.ndarray | None
    utc_offset: float      # UTC = tick * TICK_S + utc_offset

    def utc(self) -> np.ndarray:
        return self.tick * TICK_S + self.utc_offset


def _rows_near(d, t0: float, t1: float) -> tuple[int, int]:
    """First and last+1 row whose packet UTC lies within MARGIN_S of [t0, t1),
    reading the time column ROW_CHUNK rows at a time. Rows are packet-ordered,
    and the packet stamp is within ~1 s of the onboard clock, so every event of
    the window lies inside this range."""
    i0 = i1 = None
    for a in range(0, len(d), ROW_CHUNK):
        utc = (np.asarray(d[a:a + ROW_CHUNK]["mjd"], dtype=np.float64) - MJD_UNIX0) * 86400.0
        hit = np.flatnonzero((utc >= t0 - MARGIN_S) & (utc < t1 + MARGIN_S))
        if hit.size:
            i0 = a + int(hit[0]) if i0 is None else i0
            i1 = a + int(hit[-1]) + 1
    return (0, 0) if i0 is None else (i0, i1)


def read_events(product: Product, det: str, t0: float, t1: float,
                e_lo: float = 0.0, e_hi: float = np.inf) -> Events:
    """Events of one detector with UTC in [t0, t1) and energy in [e_lo, e_hi)."""
    det = det.upper()
    with _open_events(product) as h:
        d = h[f"{det}-EVENTS"].data
        r0, r1 = _rows_near(d, t0, t1)
        if r1 <= r0:
            return Events(det, np.zeros(0, np.int64), np.zeros(0), None, np.nan)
        d = d[r0:r1]
        obt = np.asarray(d["hlsobt"], dtype=np.float64)
        utc = (np.asarray(d["mjd"], dtype=np.float64) - MJD_UNIX0) * 86400.0
        near = (utc >= t0 - MARGIN_S) & (utc < t1 + MARGIN_S)
        offset = float(np.median(utc[near] - obt[near]))
        tick = np.rint(obt / TICK_S).astype(np.int64)
        k0, k1 = int(np.floor((t0 - offset) / TICK_S)), int(np.ceil((t1 - offset) / TICK_S))
        e = np.asarray(d["ener"], dtype=np.float64)
        sel = (tick >= k0) & (tick < k1) & (e >= e_lo) & (e < e_hi)
        pix = np.asarray(d["pix"], dtype=np.int16)[sel] if "pix" in d.names else None
        tick, e = tick[sel], e[sel]
    if pix is not None:
        bad = disabled_pixels(product, det)
        if bad:
            ok = ~np.isin(pix, list(bad))
            tick, e, pix = tick[ok], e[ok], pix[ok]
    order = np.argsort(tick, kind="stable")
    return Events(det, tick[order], e[order], None if pix is None else pix[order], offset)


def tick_counts(ev: Events, k0: int, n: int, e_lo: float = 0.0, e_hi: float = np.inf) -> np.ndarray:
    """Counts per 10 ms tick on ticks k0 .. k0+n-1."""
    m = (ev.energy >= e_lo) & (ev.energy < e_hi) & (ev.tick >= k0) & (ev.tick < k0 + n)
    return np.bincount(ev.tick[m] - k0, minlength=n).astype(np.float64)
