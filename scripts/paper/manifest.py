"""Paper step 20: paper_results/RESULTS_MANIFEST.json -- everything needed to trace
and reproduce each result.

    python scripts/paper/manifest.py

Records the repository state and environment (00_environment.json), the data
(locations, the frozen study cache and a hash of its manifest, the GOES files),
the split and embargo, every experiment's run folder with the SHA-256 of its
configuration and checkpoint, the training, evaluation, calibration and
bootstrap settings, and the SHA-256 of every file under paper_results/, with the
scripts that produce them. Experiments not finished are listed as such.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from common import PAPER, ROOT, stamp, write_json  # noqa: E402  (sets sys.path)
from solarflare.preprocess.cache import cache_frozen
from solarflare.settings import load_settings

S = load_settings()
RUNS = {"E1": S.outputs / "paper" / "E1_solexs_only", "E2": S.outputs / "paper" / "E2_hel1os_only",
        "E4": S.model_dir}


def sha256(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read(p: Path) -> dict | None:
    return json.loads(p.read_text("utf-8")) if p.exists() else None


def experiment(key: str, run: Path) -> dict:
    cfg = _read(run / "reports" / "config.json")
    done = (run / "checkpoints" / "best.pt").exists() and (run / "reports" / "evaluation.json").exists()
    out = {"run": str(run), "status": "trained" if done else "not finished", "inputs": None}
    if cfg:
        out.update({"inputs": cfg["model"].get("inputs", "both"), "config_sha256": sha256(run / "reports" / "config.json"),
                    "model": cfg["model"], "training": cfg["train"], "window": cfg["win"],
                    "preprocessing": {k: cfg["pre"][k] for k in ("dt_seconds", "label_source", "goes_min_class",
                                                                 "solexs_energy_scale", "hel1os_smooth_s",
                                                                 "cache_is_source") if k in cfg["pre"]}})
    if done:
        out["checkpoint_sha256"] = sha256(run / "checkpoints" / "best.pt")
        out["evaluation_sha256"] = sha256(run / "reports" / "evaluation.json")
    return out


def main() -> int:
    env = _read(PAPER / "00_environment.json") or {}
    sp = _read(PAPER / "02_experiment_split.json") or {}
    ex = _read(PAPER / "experiments.json") or {}
    goes = sorted(p.name for p in S.goes_dir.glob("*.nc"))
    files = []
    for p in sorted(PAPER.rglob("*")):
        if p.is_file() and p.name != "RESULTS_MANIFEST.json" and "__pycache__" not in p.parts:
            files.append({"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)})
    manifest = {
        "repository": env.get("git"), "environment": {k: env.get(k) for k in ("python", "os", "machine", "packages", "gpu")},
        "settings_file": "config/project.toml", "settings_sha256": sha256(ROOT / "config" / "project.toml"),
        "scope": "Paper 1: SoLEXS + HEL1OS inputs, GOES-18 XRS truth only. SUIT excluded.",
        "data": {"data_root": str(S.data_root), "study_cache": str(S.cache), "study_cache_frozen": cache_frozen(S.cache),
                 "study_cache_manifest_sha256": sha256(S.cache / "manifest.json"), "goes_dir": str(S.goes_dir),
                 "goes_files": goes, "inventory": "paper_results/01_data_inventory.json"},
        "split": {"file": "paper_results/02_experiment_split.json", "mode": sp.get("mode"),
                  "fractions": sp.get("fractions_train_val_test"), "embargo_days": (sp.get("checks") or {}).get("embargo_days"),
                  "periods": {k: [v["first_origin_utc"], v["last_origin_utc"]] for k, v in (sp.get("splits") or {}).items()},
                  "checks": sp.get("checks")},
        "seeds": {"training": {k: (experiment(k, r).get("training") or {}).get("seed") for k, r in RUNS.items()},
                  "bootstrap": (ex.get("bootstrap") or {}).get("seed")},
        "experiments": {k: experiment(k, r) for k, r in RUNS.items()},
        "evaluation": {k: ex.get(k) for k in ("thresholds", "calibration", "interval_scale", "bootstrap",
                                              "populations_test", "robustness")} if ex else "not yet run",
        "commands": {
            "E1": "python -m solarflare train --from-run outputs/model --out-dir outputs/paper/E1_solexs_only "
                  "--set model.inputs=soft",
            "E2": "python -m solarflare train --from-run outputs/model --out-dir outputs/paper/E2_hel1os_only "
                  "--set model.inputs=hard",
            "E4": "python -m solarflare pipeline (stage train; outputs/model)",
            "all paper results": "python scripts/paper/run_all_paper_experiments.py"},
        "scripts": {"results": sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "scripts" / "paper").glob("*.py")),
                    "figures": sorted(p.relative_to(ROOT).as_posix() for p in (PAPER / "plotting").glob("fig*.py")),
                    "tables": ["scripts/paper/make_tables.py"]},
        "outputs": files,
        "manifest_generated": stamp(),
    }
    write_json("RESULTS_MANIFEST.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
