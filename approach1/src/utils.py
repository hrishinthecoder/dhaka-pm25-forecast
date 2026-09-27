"""
utils.py — shared helpers: config loading, global seeding, project paths, logging.

WHY THIS EXISTS
    Every script in this project must (a) read its settings from the ONE authoritative
    config file (`config/config.yaml`) so there are no magic numbers scattered in code,
    and (b) set the SAME random seed everywhere so results reproduce exactly. Centralising
    both here means every phase behaves consistently and a reviewer can trust the numbers.

Nothing here is ML-specific; it is plumbing the other modules import.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import yaml

# Project root = the folder that contains config/, processed/, raw/, src/.
# __file__ is src/utils.py, so root is its parent's parent.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    """Read config/config.yaml into a plain dict.

    Args:
        path: location of the YAML config (defaults to the project's config file).

    Returns:
        A nested dict mirroring the YAML. This is the single source of truth for
        every tunable in the pipeline; code should never hard-code values that live
        here.

    Why it matters: the config is declared AUTHORITATIVE in the project decision log. Loading it in
    one place guarantees all phases see identical settings.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_global_seed(seed: int) -> int:
    """Seed Python, NumPy (and, if present, common ML libs) for reproducibility.

    Args:
        seed: the integer seed from config (`project.random_seed`).

    Returns:
        The seed, so callers can log it.

    Why it matters: model training, Optuna sampling, and any subsampling are random.
    A fixed seed lets anyone re-run and get the same RMSE — a hard requirement for a
    publication-grade result.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Seed optional libs only if installed, so utils stays import-safe everywhere.
    try:  # LightGBM/XGBoost take seed per-model, but set torch here for Phase 6.
        import torch  # noqa: F401

        torch.manual_seed(seed)
    except Exception:
        pass
    return seed


def get_paths(cfg: dict) -> dict[str, Path]:
    """Resolve every directory/file path in config to an absolute Path.

    Args:
        cfg: the loaded config dict.

    Returns:
        Dict of name -> absolute Path. Creating output dirs is the caller's job
        (see `ensure_dirs`).
    """
    p = cfg["paths"]
    return {k: (PROJECT_ROOT / v) for k, v in p.items()}


def ensure_dirs(cfg: dict) -> None:
    """Create the OUTPUT directories the pipeline writes to (idempotent).

    Never touches `processed/` or `raw/` inputs except to ensure `processed/` exists
    (it already does). We only create folders we write derived artifacts into.
    """
    paths = get_paths(cfg)
    for key in ["processed_dir", "artifacts_dir", "reports_dir",
                "figures_dir", "memory_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a console logger with a consistent format (used to log the seed etc.)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger
