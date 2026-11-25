"""Construct the frozen ReGSP policy for one task."""
from __future__ import annotations

import os

from regsp.bundle import TaskBundle
from regsp.config import load_final_config, task_env
from regsp.execution.closed_loop import ReGSPBimanualPolicy, ReGSPSingleArmPolicy


def apply_task_env(task: str) -> dict[str, str]:
    """Set the frozen per-task executor options (clears any stale ``ACL_*/CLC_*/PIC_*`` first, as the frozen runner did)."""
    # REGSP_DATAGEN=1 (training-domain on-policy data generation only) keeps the data-recording / intervention options; never set for evaluation
    keep = ("ACL_PERTURB", "ACL_SAVE_FEATS", "ACL_TRACE", "PIC_SAVE_SG", "PIC_SG_STOP", "PIC_LANG", "PIC_INSTR") if os.environ.get("REGSP_DATAGEN") == "1" else ()
    for k in [k for k in os.environ if k.startswith(("ACL_", "CLC_", "PIC_")) and k not in keep]:
        del os.environ[k]
    env = task_env(task)
    os.environ.update(env)
    return env


def build_policy(task: str, ground: str = "v21", geom: str = "live", bundle: TaskBundle | None = None, set_env: bool = True):
    """``set_env=True`` applies ``configs/final_v21_clc.json`` before construction (the executor reads its options at init)."""
    cfg = load_final_config()["tasks"][task]
    if set_env:
        apply_task_env(task)
    bundle = bundle or TaskBundle.for_task(task)
    miss = bundle.missing()
    if miss:
        raise FileNotFoundError(f"weight bundle {bundle.root} is missing {miss} (see docs/CHECKPOINTS.md)")
    if cfg["bimanual"]:
        return ReGSPBimanualPolicy(task, tagv="b1", cycle_scorer=True, place=True, bim_dec=True, ground=ground, dec="pretrained", nview=cfg["nview"], tag="", geom=geom, bundle=bundle)
    return ReGSPSingleArmPolicy(task, ground=ground, dec="pretrained", nview=cfg["nview"], tag="", geom=geom, bundle=bundle)
