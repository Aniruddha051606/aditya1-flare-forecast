"""SUIT forecasting settings, and where its data and results live."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

#: The 11 SUIT science filters (Tripathi et al. 2025, arXiv:2501.02274, table 1):
#: centre or band in nm, and what each images.
FILTERS = {
    "NB01": (214.0, "continuum"), "NB02": (276.7, "continuum"),
    "NB03": (279.6, "Mg II k, chromosphere"), "NB04": (280.3, "Mg II h, chromosphere"),
    "NB05": (283.2, "continuum"), "NB06": (300.0, "continuum"),
    "NB07": (388.0, "CN band"), "NB08": (396.85, "Ca II H, chromosphere"),
    "BB01": ("200-242", "Herzberg continuum"), "BB02": ("242-300", "Hartley band"),
    "BB03": ("300-360", "Huggins band, photosphere"),
}


def normalise_filter(text: str) -> str:
    """"NB3", "nb03", "NB03_MgIIk", 3 in a FILTER card -> "NB03"; "" if none."""
    m = re.search(r"(NB|BB)[\s_-]*0?(\d{1,2})", str(text), re.I)
    return f"{m.group(1).upper()}{int(m.group(2)):02d}" if m else ""


@dataclass
class SuitConfig:
    # The chromospheric lines where pre-flare brightenings show (Mg II k/h,
    # Ca II H; arXiv:2607.26171 found 102 of them before 7 M/X flares), plus one
    # photospheric band for context.
    filters: tuple[str, ...] = ("NB03", "NB04", "NB08", "BB03")
    image_px: int = 256               # full disk downsampled to this before features (0.7" x 16 = 11")
    min_full_px: int = 1024           # a frame smaller than this is a region-of-interest, never used
    per_hour: int = 1                 # full-disk frames used per filter and hour (~96 a day of ~800)
    max_age_h: float = 3.0            # a filter's features older than this count as missing
    latency_h: float = 0.0            # data delay to apply for an operational forecast
    contrast_levels: tuple[float, ...] = (1.2, 1.5, 2.0)
    min_region_px: int = 4            # at image_px: a bright region smaller than this is noise
    deltas_h: tuple[int, ...] = (6, 24)
    horizons_h: tuple[int, ...] = (6, 12, 24)
    classes: tuple[str, ...] = ("C", "M")
    embargo_days: float = 27.0        # own split only: one solar rotation, as the X-ray model
    split_fractions: tuple[float, float] = (0.6, 0.2)   # own split: train, validation (rest is test)
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SuitPaths:
    data: Path          # SUIT Level-1 files (FITS, .fits.gz, or zips of them)
    features: Path      # per-frame feature tables, rebuilt from the data
    outputs: Path       # reports
    goes: Path          # GOES flare list = the truth, shared with the X-ray model
    split_run: Path     # the X-ray model run whose train/test split SUIT reuses


def default_paths() -> SuitPaths:
    from solarflare.settings import load_settings

    s = load_settings()
    return SuitPaths(data=s.data_root / "suit", features=s.cache / "suit", outputs=s.outputs / "suit",
                     goes=s.goes_dir, split_run=s.model_dir)
