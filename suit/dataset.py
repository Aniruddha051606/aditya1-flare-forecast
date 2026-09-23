"""From SUIT frames to an hourly feature table, one row per forecast origin.

A row at origin ``o`` sees, for each filter, the latest full-disk frame taken at
or before ``o - latency`` (and no older than ``max_age_h``), plus how its
features changed over the last 6 and 24 h. Nothing after ``o`` is ever read,
which tests/test_suit.py checks by altering the future and comparing rows.
Origins are whole UTC hours, the same grid as the X-ray day-ahead model, so the
two tables can later be joined on time.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .config import SuitConfig
from .features import feature_fingerprint, feature_names, image_features
from .io import SuitFrame, read_image


def select_frames(frames: list[SuitFrame], cfg: SuitConfig) -> dict[str, list[SuitFrame]]:
    """Per filter, the first ``per_hour`` full-disk-sized frames of each hour.
    Smaller frames are regions of interest, pointed by the X-ray flare triggers:
    never used. Of ~800 frames a day this keeps ~24 per filter."""
    out: dict[str, list[SuitFrame]] = {f: [] for f in cfg.filters}
    count: dict[tuple, int] = {}
    for fr in frames:
        if fr.filt not in out or min(fr.nx, fr.ny) < cfg.min_full_px:
            continue
        k = (fr.filt, int(fr.t_unix // 3600))
        if count.get(k, 0) < cfg.per_hour:
            out[fr.filt].append(fr)
            count[k] = count.get(k, 0) + 1
    return out


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as z:
        return {str(k): (float(t), f) for k, t, f in zip(z["key"], z["t"], z["F"])}


def _save(path: Path, table: dict, n_feat: int) -> None:
    keys = sorted(table, key=lambda k: table[k][0])
    t = np.array([table[k][0] for k in keys], dtype=np.float64)
    F = np.stack([table[k][1] for k in keys]) if keys else np.zeros((0, n_feat), np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, key=np.array(keys), t=t, F=F.astype(np.float32))
    os.replace(tmp, path)


def build_features(selected: dict[str, list[SuitFrame]], cfg: SuitConfig, feature_dir: Path,
                   verbose: bool = False) -> dict[str, dict]:
    """Features of every selected frame, cached per filter so a new day costs only
    its own frames. A frame that turns out not to show the whole disk is stored as
    a NaN row (so it is not read again) and counted as rejected."""
    n_feat = len(feature_names(cfg))
    fp = feature_fingerprint(cfg)
    sigs: dict[str, str] = {}

    def key(fr: SuitFrame) -> str:
        # the file's size and time too: a re-downloaded or re-processed file is new data
        if fr.path not in sigs:
            st = os.stat(fr.path)
            sigs[fr.path] = f"{st.st_size}|{st.st_mtime_ns}"
        return f"{fr.key}|{sigs[fr.path]}"

    out = {}
    for filt, frames in selected.items():
        path = Path(feature_dir) / f"features_{filt}_{fp}.npz"
        table = _load(path)
        todo = [f for f in frames if key(f) not in table]
        for i, fr in enumerate(todo, 1):
            try:
                v = image_features(read_image(fr), cfg)
            except (OSError, ValueError) as exc:
                if verbose:
                    print(f"  unreadable {fr.key}: {type(exc).__name__}: {exc}")
                continue
            table[key(fr)] = (fr.t_unix, v if v is not None else np.full(n_feat, np.nan, np.float32))
            if i % 200 == 0:
                _save(path, table, n_feat)
                if verbose:
                    print(f"  {filt}: {i}/{len(todo)} new frames", flush=True)
        if todo:
            _save(path, table, n_feat)
        wanted = {key(f) for f in frames}
        rows = sorted(((t, F) for k, (t, F) in table.items() if k in wanted), key=lambda r: r[0])
        t = np.array([r[0] for r in rows], dtype=np.float64)
        F = np.stack([r[1] for r in rows]) if rows else np.zeros((0, n_feat), np.float32)
        ok = np.isfinite(F).all(axis=1) if len(F) else np.zeros(0, bool)
        out[filt] = {"t": t[ok], "F": F[ok], "rejected": int((~ok).sum()), "used": int(ok.sum())}
        if verbose:
            print(f"  {filt}: {out[filt]['used']} full-disk frames, {out[filt]['rejected']} rejected")
    return out


def hourly_table(feats: dict[str, dict], origins: np.ndarray, cfg: SuitConfig):
    """(X, names, have): one row per origin; ``have`` marks rows where at least
    ``cfg.n_required()`` filters have a frame. Unless ``cfg.schedule_features``,
    the image-age and filter-count columns (SUIT's observing schedule, which flare
    mode changes) are left out."""
    X, names, n_have = _hourly_table(feats, origins, cfg)
    if not cfg.schedule_features:
        keep = [i for i, n in enumerate(names) if not (n.endswith("_age_h") or n == "n_filters")]
        X, names = X[:, keep], [names[i] for i in keep]
    return X, names, n_have >= cfg.n_required()


def _hourly_table(feats: dict[str, dict], origins: np.ndarray, cfg: SuitConfig):
    base = feature_names(cfg)
    cols, names = [], []
    n_have = np.zeros(origins.size)

    def at(t, F, when):
        q = when - 3600.0 * cfg.latency_h
        j = np.searchsorted(t, q, side="right") - 1
        jj = np.clip(j, 0, None)
        ok = (j >= 0) & (q - t[jj] <= 3600.0 * cfg.max_age_h)
        V = F[jj].astype(np.float64)
        V[~ok] = np.nan
        return V, ok, np.where(ok, (q - t[jj]) / 3600.0, np.nan)

    for filt in cfg.filters:
        d = feats.get(filt)
        k = len(base) * (1 + len(cfg.deltas_h)) + 1
        names += [f"{filt}_{n}" for n in base] + [f"{filt}_age_h"]
        names += [f"{filt}_d{h}h_{n}" for h in cfg.deltas_h for n in base]
        if d is None or not len(d["t"]):
            cols.append(np.full((origins.size, k), np.nan))
            continue
        V0, ok0, age = at(d["t"], d["F"], origins)
        block = [V0, age[:, None]]
        for h in cfg.deltas_h:
            Vd, _, _ = at(d["t"], d["F"], origins - 3600.0 * h)
            block.append(V0 - Vd)
        cols.append(np.hstack(block))
        n_have += ok0
    names.append("n_filters")
    X = np.hstack(cols + [n_have[:, None]]) if cols else np.zeros((origins.size, 0))
    return X, names, n_have
