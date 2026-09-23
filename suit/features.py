"""Brightening features of one full-disk SUIT image.

Every feature is a *contrast*: the image divided by the quiet disk at the same
distance from disk centre (which also removes limb darkening). None depends on
the absolute intensity, because in flare mode SUIT sets its exposure itself: an
exposure-dependent feature would tell the model that a flare had already been
detected. For the same reason the exposure and observing-mode keywords are
never features.

A frame that does not show the whole disk -- a region-of-interest image, which
SUIT points at flares its X-ray triggers found -- gets no features at all.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .config import SuitConfig


#: Bump when what a feature *means* changes (a formula, a threshold rule), so every
#: cached table is rebuilt. Settings that change features are covered by
#: feature_fingerprint automatically.
FEATURE_VERSION = 1


def feature_fingerprint(cfg: SuitConfig) -> str:
    """Identifies how features were computed: a cache made under other settings
    is never reused (it would silently mix two definitions in one table)."""
    import hashlib
    import json

    spec = {"v": FEATURE_VERSION, "px": cfg.image_px, "levels": list(cfg.contrast_levels),
            "min_region_px": cfg.min_region_px}
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]


def feature_names(cfg: SuitConfig) -> list[str]:
    return (["c_p99", "c_p999", "c_max", "excess"]
            + [f"area_{lvl:g}" for lvl in cfg.contrast_levels]
            + ["n_regions", "max_region_area", "max_region_excess", "max_region_peak"])


def downsample(img: np.ndarray, px: int) -> np.ndarray:
    """Block mean to about px x px (NaN-aware): 4096^2 -> 256^2 keeps 11" pixels,
    enough for active regions and plage, and makes a frame cost milliseconds."""
    f = max(1, min(img.shape) // px)
    h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    blocks = img[:h, :w].reshape(h // f, f, w // f, f)
    with np.errstate(invalid="ignore"):
        return np.nanmean(blocks, axis=(1, 3)) if np.isnan(blocks).any() else blocks.mean(axis=(1, 3))


def fit_disk(small: np.ndarray) -> tuple[float, float, float] | None:
    """(row, col, radius) of the solar disk, or None unless the whole disk is in frame.

    SUIT's field of view is ~1.5 solar diameters, so a full-disk frame has a
    disk of radius ~0.33 of the frame, and dark sky all around it."""
    v = small[np.isfinite(small)]
    if v.size < 16:
        return None
    level = np.percentile(v, 85)              # well inside the disk (~35% of a full-disk frame)
    if not level > 0:
        return None
    mask = ndimage.binary_fill_holes(np.nan_to_num(small) > 0.25 * level)
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    disk = lab == (1 + int(np.argmax(sizes)))
    area = float(disk.sum())
    r = np.sqrt(area / np.pi)
    cy, cx = ndimage.center_of_mass(disk)
    ny, nx = small.shape
    inside = cy - r >= -1 and cx - r >= -1 and cy + r <= ny and cx + r <= nx
    if not inside or r < 0.15 * min(ny, nx) or area > 0.8 * ny * nx:
        return None
    return float(cy), float(cx), float(r)


def contrast_map(small: np.ndarray, cy: float, cx: float, r: float, n_rings: int = 24):
    """Image / quiet-disk intensity at the same radius, inside 0.95 radius (the limb
    itself is too steep to divide by). Returns the map and the pixels used."""
    yy, xx = np.indices(small.shape)
    rho = np.hypot(yy - cy, xx - cx) / r
    inside = (rho <= 0.95) & np.isfinite(small)
    ring = np.clip((rho / 0.95 * n_rings).astype(int), 0, n_rings - 1)
    ref = np.full(n_rings, np.nan)
    for k in range(n_rings):
        sel = inside & (ring == k)
        if sel.any():
            ref[k] = np.median(small[sel])       # the median is the quiet Sun: plage is a minority
    expected = ref[ring]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(inside & (expected > 0), small / expected, np.nan)
    return c, inside & np.isfinite(c)


def image_features(img: np.ndarray, cfg: SuitConfig) -> np.ndarray | None:
    """Contrast features of one frame, or None when it is not a full-disk view."""
    small = downsample(np.asarray(img, dtype=np.float32), cfg.image_px)
    disk = fit_disk(small)
    if disk is None:
        return None
    c, used = contrast_map(small, *disk)
    v = c[used]
    if v.size < 50:
        return None
    feats = [np.percentile(v, 99), np.percentile(v, 99.9), float(v.max()), float(np.clip(v - 1, 0, None).mean())]
    feats += [float((v > lvl).mean()) for lvl in cfg.contrast_levels]
    bright = np.nan_to_num(c) > cfg.contrast_levels[0]
    lab, n = ndimage.label(bright)
    n_reg = mx_area = mx_excess = mx_peak = 0.0
    if n:
        idx = np.arange(1, n + 1)
        size = ndimage.sum(np.ones_like(lab), lab, idx)
        keep = idx[size >= cfg.min_region_px]
        if keep.size:
            exc = ndimage.sum(np.nan_to_num(c) - 1.0, lab, keep)
            peak = ndimage.maximum(np.nan_to_num(c), lab, keep)
            k = int(np.argmax(exc))                 # the region with the most excess brightness
            n_reg = float(keep.size)
            mx_area = float(size[keep - 1][k]) / v.size
            mx_excess = float(exc[k]) / v.size
            mx_peak = float(peak[k])
    feats += [n_reg, mx_area, mx_excess, mx_peak]
    return np.asarray(feats, dtype=np.float32)
