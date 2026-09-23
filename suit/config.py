"""SUIT forecasting settings, and where its data and results live.

Settings come from ``config/suit.toml`` -- separate from ``config/project.toml``,
so SUIT work can never change the X-ray pipeline's settings -- on top of the
defaults below.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "config" / "suit.toml"

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
    min_filters: int = 0              # filters an hour needs to count as observed (0 = all of them)
    # Image age and the number of filters present describe SUIT's observing
    # schedule, not the Sun, and flare mode changes the schedule: never features
    # unless this is set on purpose (then they are measured, not trusted).
    schedule_features: bool = False
    # SUIT-only runs (suit/train.py); the E1-E6 matrix always uses the day-ahead system's own
    horizons_h: tuple[int, ...] = (2, 6, 12, 24)
    classes: tuple[str, ...] = ("C", "M")
    embargo_days: float = 27.0        # own split only: one solar rotation, as the X-ray model
    split_fractions: tuple[float, float] = (0.6, 0.2)   # own split: train, validation (rest is test)
    # E1-E6 matrix
    hel1os_min_coverage: float = 0.5  # as SoLEXS in the day-ahead system: >= 50% of the previous 6 h
    gate_class: str = "M"             # the decision gate, fixed before any SUIT data is seen
    gate_horizon_h: int = 24
    extra: dict = field(default_factory=dict)

    def n_required(self) -> int:
        return len(self.filters) if self.min_filters <= 0 else min(self.min_filters, len(self.filters))


def load_config(path: Path | None = None) -> SuitConfig:
    """SuitConfig from config/suit.toml ([suit] and [matrix] tables) over the
    defaults. An unknown key is an error: a typo must not silently keep a default."""
    cfg = SuitConfig()
    p = Path(path) if path else CONFIG
    if not p.exists():
        return cfg
    raw = tomllib.loads(p.read_text("utf-8"))
    for section in ("suit", "matrix"):
        for k, v in raw.get(section, {}).items():
            if k == "extra" or not hasattr(cfg, k):
                raise ValueError(f"{p.name}: unknown setting [{section}] {k}")
            setattr(cfg, k, tuple(v) if isinstance(getattr(cfg, k), tuple) else v)
    return cfg


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
