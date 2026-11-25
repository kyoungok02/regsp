"""Frozen SafeDiff-VLA backbone loading (replaces ``examples/safediff_vla/v21_common.py`` of the research tree).

The backbone is the grounded-grasp-v2 20k-step checkpoint; every parameter is frozen. The same backbone is shared by all four
tasks; per-task heads are loaded separately (``regsp.execution``).
"""
from __future__ import annotations

import numpy as np
import torch

from regsp import paths
from regsp.backbone.stageplan_v2 import SafeDiffVLAStagePlanV2Policy

#: dataset camera keys -> keys expected by the checkpoint's processors
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
CAMS = ["image", "second_image", "wrist_image"]  # head, left wrist, right wrist (RoboTwin convention, see regsp.grounding.CAMS)


def load_policy(dev: str = "cuda", ckpt: str | None = None):
    """Return ``(policy, preprocessor)`` with all parameters frozen and ``eval()`` mode."""
    from lerobot.policies.factory import make_pre_post_processors

    ckpt = str(ckpt or paths.backbone_ckpt())
    pol = SafeDiffVLAStagePlanV2Policy.from_pretrained(ckpt).to(dev).eval()
    for p in pol.parameters():
        p.requires_grad_(False)
    pre, _ = make_pre_post_processors(
        policy_cfg=pol.config,
        pretrained_path=ckpt,
        preprocessor_overrides={
            "device_processor": {"device": dev},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    return pol, pre


def make_batch(r, text):
    """One observation batch: ``r['imgs224']`` = three 224x224x3 uint8 images (head, left wrist, right wrist), ``r['state']`` = 7-D."""
    b = {
        f"observation.images.{k}": torch.from_numpy(r["imgs224"][i]).permute(2, 0, 1).float().div(255)[None]
        for i, k in enumerate(CAMS)
    }
    b["observation.state"] = torch.from_numpy(r["state"]).float()[None]
    b["task"] = [text]
    return b
