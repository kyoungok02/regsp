"""Frozen per-task configuration of the released model (``configs/final_v21_clc.json``)."""
from __future__ import annotations

import json

from regsp import paths


def load_final_config() -> dict:
    return json.loads((paths.CONFIG_DIR / "final_v21_clc.json").read_text())


def task_env(task: str) -> dict[str, str]:
    """Environment variables of the frozen executor for one task (all other ``ACL_/CLC_/PIC_`` options are unset = default)."""
    cfg = load_final_config()
    env = {k: str(v) for k, v in cfg["common_env"].items()}
    env.update({k: str(v) for k, v in cfg["tasks"][task]["env"].items()})
    return env
