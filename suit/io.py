"""Finding SUIT Level-1 images and reading them, from FITS files or inside zips.

SUIT (arXiv:2501.02274): 4096 x 4096 CCD at 0.7"/px, a field of view of ~1.5
solar radii, 11 filters in 200-400 nm. Level-1 images are dark, flat, scatter
and hot-pixel corrected, and their headers carry the filter, exposure, UT time
and observing mode.

The header keyword names of the PRADAN Level-1 files were not known when this
was written, so each quantity is looked up under several plausible names and,
failing that, parsed from the file name. Every frame records *where* its time
and filter came from, and ``python -m suit inventory`` reports those counts and
prints real headers: confirm them the first time real files arrive, and adjust
TIME_KEYS / FILTER_KEYS / MODE_KEYS below if needed.

Zips are read in place, as the SoLEXS reader and the HEL1OS event lists are:
the archive keeps a single copy of everything.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from astropy.io import fits

from .config import normalise_filter

TIME_KEYS = ("DATE-OBS", "DATE_OBS", "T_OBS", "DATEOBS", "OBS_TIME", "TIME-OBS")
FILTER_KEYS = ("FTR_NAME", "FILTER", "FILTNAME", "FILTER_ID", "FILT_NAME", "FTR_ID", "WAVELNTH")
MODE_KEYS = ("IMG_TYPE", "OBS_MODE", "MODE", "IMGTYPE", "OBSMODE", "OBS_TYPE")
FITS_NAME = re.compile(r"\.(fits|fit|fts)(\.gz)?$", re.I)
_STAMP = re.compile(r"(20\d{2})-?(\d{2})-?(\d{2})[T_ -]?(\d{2}):?(\d{2}):?(\d{2})")


@dataclass(frozen=True)
class SuitFrame:
    """One SUIT image, located by its container (a file or a zip) and member."""
    path: str          # the FITS file, or the zip holding it
    member: str        # "" for a plain file, else the member inside the zip
    t_unix: float      # NaN when neither header nor name gives a time
    filt: str          # "NB03", ... ("" when neither header nor name says)
    nx: int
    ny: int
    mode: str          # raw observing-mode keyword: recorded, never a feature (flare mode leaks)
    time_from: str = ""    # the header keyword the time came from, or "filename"
    filt_from: str = ""    # the same for the filter

    @property
    def key(self) -> str:
        return f"{self.path}|{self.member}"


def _lookup(header, keys) -> tuple[str, str]:
    """(keyword, value) of the first non-empty keyword of ``keys``, else ("", "")."""
    for k in keys:
        if k in header and str(header[k]).strip():
            return k, str(header[k]).strip()
    return "", ""


def _parse_time(text: str) -> float:
    m = _STAMP.search(text or "")
    if not m:
        return float("nan")
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    try:
        return datetime(y, mo, d, h, mi, s, tzinfo=UTC).timestamp()
    except ValueError:
        return float("nan")


def _open_fits(fh, name: str):
    """astropy on a file object, unzipping .gz members first (a zip stream is seekable, a gzip one barely)."""
    if name.lower().endswith(".gz"):
        fh = io.BytesIO(gzip.GzipFile(fileobj=fh).read())
    return fits.open(fh, lazy_load_hdus=True, memmap=False)


def image_hdu(hdul):
    """The first HDU holding a 2-D image, and a header merged with the primary one."""
    for h in hdul:
        if h.header.get("NAXIS", 0) == 2 or isinstance(h, fits.CompImageHDU):
            merged = fits.Header(hdul[0].header)
            merged.update(h.header)
            return h, merged
    return None, None


def _frame(path: str, member: str, hdul) -> SuitFrame | None:
    h, hdr = image_hdu(hdul)
    if h is None:
        return None
    name = Path(member or path).name
    tkey, tval = _lookup(hdr, TIME_KEYS)
    t, t_from = _parse_time(tval), tkey
    if not np.isfinite(t):
        t, t_from = _parse_time(name), "filename"
    fkey, fval = _lookup(hdr, FILTER_KEYS)
    filt, f_from = normalise_filter(fval), fkey
    if not filt:
        filt, f_from = normalise_filter(name), "filename"
    nx, ny = int(hdr.get("ZNAXIS1", hdr.get("NAXIS1", 0))), int(hdr.get("ZNAXIS2", hdr.get("NAXIS2", 0)))
    return SuitFrame(path, member, t, filt, nx, ny, _lookup(hdr, MODE_KEYS)[1],
                     t_from if np.isfinite(t) else "", f_from if filt else "")


def frames_in(container: Path, problems: list | None = None) -> list[SuitFrame]:
    """Headers only: every image in one FITS file or zip. An unreadable member is
    reported in ``problems`` and skipped; the rest of the zip is still read."""
    out = []

    def note(where: str, why: str) -> None:
        if problems is not None:
            problems.append((where, why))

    if container.suffix.lower() == ".zip":
        with zipfile.ZipFile(container) as z:
            for m in z.namelist():
                if not FITS_NAME.search(m):
                    continue
                try:
                    with z.open(m) as fh, _open_fits(fh, m) as hdul:
                        f = _frame(str(container), m, hdul)
                except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
                    note(f"{container}|{m}", f"{type(exc).__name__}: {exc}")
                    continue
                if f:
                    out.append(f)
                else:
                    note(f"{container}|{m}", "no 2-D image in the file")
    else:
        with open(container, "rb") as fh, _open_fits(fh, container.name) as hdul:
            f = _frame(str(container), "", hdul)
        if f:
            out.append(f)
        else:
            note(str(container), "no 2-D image in the file")
    return out


def read_image(frame: SuitFrame) -> np.ndarray:
    """The image as float32 (NaN where the file marks blanks)."""
    def load(fh, name):
        with _open_fits(fh, name) as hdul:
            h, _ = image_hdu(hdul)
            return np.asarray(h.data, dtype=np.float32)

    if frame.member:
        with zipfile.ZipFile(frame.path) as z, z.open(frame.member) as fh:
            return load(fh, frame.member)
    with open(frame.path, "rb") as fh:
        return load(fh, frame.path)


def index_frames(root: Path, index_path: Path | None = None, verbose: bool = False,
                 problems: list | None = None) -> list[SuitFrame]:
    """Every SUIT frame under ``root`` with a time, time-sorted. With ``index_path``
    the headers of an unchanged file are read only once (thousands a day). Files
    or members that cannot be read, and frames with no time, go to ``problems``
    as (where, why) -- they are never dropped silently."""
    probs: list = [] if problems is None else problems
    old = {}
    if index_path and index_path.exists():
        try:
            old = json.loads(index_path.read_text("utf-8"))
        except (OSError, ValueError):
            old = {}
    new, frames = {}, []
    containers = sorted(p for p in Path(root).rglob("*") if p.is_file()
                        and (p.suffix.lower() == ".zip" or FITS_NAME.search(p.name)))
    for i, c in enumerate(containers, 1):
        st = c.stat()
        sig = [st.st_size, st.st_mtime_ns]
        rec = old.get(str(c))
        if rec and rec.get("sig") == sig:
            fs = [SuitFrame(**f) for f in rec["frames"]]
            mine = [tuple(p) for p in rec.get("problems", [])]
        else:
            mine = []
            try:
                fs = frames_in(c, mine)
            except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
                probs.append((str(c), f"{type(exc).__name__}: {exc}"))
                continue
        probs.extend(mine)
        new[str(c)] = {"sig": sig, "frames": [asdict(f) for f in fs], "problems": mine}
        frames += fs
        if verbose and i % 500 == 0:
            print(f"  indexed {i}/{len(containers)} files, {len(frames)} frames", flush=True)
    if index_path:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = index_path.with_name(index_path.name + ".tmp")
        tmp.write_text(json.dumps(new), encoding="utf-8")
        os.replace(tmp, index_path)
    for f in frames:
        if not np.isfinite(f.t_unix):
            probs.append((f.key, "no time in the header or the file name"))
    return sorted((f for f in frames if np.isfinite(f.t_unix)), key=lambda f: (f.t_unix, f.filt))
