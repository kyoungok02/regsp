"""Filesystem layout used by ReGSP.

Nothing here is machine specific. Three locations are configurable through environment variables:

``REGSP_BACKBONE_CKPT``
    ``.../pretrained_model`` directory of the SafeDiff-VLA grounded-grasp-v2 backbone checkpoint
    (``config.json``, ``model.safetensors``, processor files, ``tokenizer/``).
``REGSP_CKPT_ROOT``
    Root of the per-task weight bundle (see ``docs/CHECKPOINTS.md``): ``<root>/<task>/<canonical file>``.
``REGSP_ROBOTWIN_ROOT``
    Checkout of RoboTwin (simulator, task code, embodiment URDF).
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
REPO_ROOT = PACKAGE_DIR.parent
CONFIG_DIR = REPO_ROOT / "configs"

#: tasks of the released four-task study
TASKS = ("click_bell", "place_container_plate", "stack_blocks_two", "handover_block")
BIMANUAL_TASKS = ("stack_blocks_two", "handover_block")


def _env_path(name: str, default: str | None = None) -> Path:
    v = os.environ.get(name, default)
    if not v:
        raise RuntimeError(f"environment variable {name} is not set (see docs/INSTALL.md)")
    return Path(v).expanduser()


def backbone_ckpt() -> Path:
    return _env_path("REGSP_BACKBONE_CKPT")


def ckpt_root() -> Path:
    return _env_path("REGSP_CKPT_ROOT")


def robotwin_root() -> Path:
    return _env_path("REGSP_ROBOTWIN_ROOT")


def data_file(*parts: str) -> Path:
    """Small JSON statistics / calibrations that ship with the package (``regsp/data``)."""
    return DATA_DIR.joinpath(*parts)


# ---- training / data-preparation layout (only needed by scripts/data and scripts/train)
def clean_root() -> Path:
    """``REGSP_CLEAN_ROOT``: working directory of the Clean-domain stage (tokens/, tokens_full/, labels/, v21cache/, views/, ...)."""
    return _env_path("REGSP_CLEAN_ROOT")


def work_root() -> Path:
    """``REGSP_WORK_ROOT``: working directory of the Randomized-domain / counterfactual / adaptive stages (rand/, ckpt_*/, da_cache/, ...)."""
    return _env_path("REGSP_WORK_ROOT")


def clean_data_root() -> Path:
    """``REGSP_CLEAN_DATA``: RoboTwin-LeRobot-v3.0 Clean-50 demonstrations root (``<root>/<task>/aloha-agilex_clean_50/{meta,data,videos}``)."""
    return _env_path("REGSP_CLEAN_DATA")
