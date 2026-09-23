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
"""

from __future__ import annotations

import io
import os
import re
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


def _open_events(product: Product):
    if product.zip is None:
        return fits.open(product.path / EVT, memmap=True)
    return fits.open(io.BytesIO(_zip_bytes(str(product.zip), f"{product.path.as_posix()}/{EVT}")))


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


def read_events(product: Product, det: str, t0: float, t1: float,
                e_lo: float = 0.0, e_hi: float = np.inf) -> Events:
    """Events of one detector with UTC in [t0, t1) and energy in [e_lo, e_hi)."""
    det = det.upper()
    with _open_events(product) as h:
        d = h[f"{det}-EVENTS"].data
        obt = np.asarray(d["hlsobt"], dtype=np.float64)
        mjd = np.asarray(d["mjd"], dtype=np.float64)
        utc = (mjd - MJD_UNIX0) * 86400.0
        near = (utc >= t0 - 120.0) & (utc < t1 + 120.0)
        if not near.any():
            return Events(det, np.zeros(0, np.int64), np.zeros(0), None, np.nan)
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
