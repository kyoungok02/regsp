"""RoboTwin simulator adapter (embodiment: aloha-agilex, Large_D435 cameras, 640x480).

Merges the research tree's ``rt_env.py`` (task env/args construction exactly as RoboTwin's ``script/eval_policy.py`` does it) and
``d2_scene.py`` (silent stdout context, expert grasp hook). RoboTwin relies on its checkout being the working directory and on its own
``sys.path`` entries, so :func:`setup` must be called before any RoboTwin import.
"""
from __future__ import annotations

import importlib
import os
import sys

import yaml

from regsp.paths import robotwin_root

_READY = False


def setup() -> str:
    global _READY
    RT = str(robotwin_root())
    if not _READY:
        os.chdir(RT)
        sys.path[:0] = [RT, RT + "/policy", RT + "/description/utils"]
        _READY = True
    return RT


def make_args(task, config="demo_clean", **over):
    RT = setup()
    from envs import CONFIGS_PATH  # RoboTwin

    args = yaml.safe_load(open(f"{RT}/task_config/{config}.yml")); args["task_name"] = task; args["task_config"] = config
    emb = yaml.safe_load(open(CONFIGS_PATH + "_embodiment_config.yml")); cam = yaml.safe_load(open(CONFIGS_PATH + "_camera_config.yml"))
    f = emb[args["embodiment"][0]]["file_path"]; args["left_robot_file"] = args["right_robot_file"] = f; args["dual_arm_embodied"] = True
    ec = yaml.safe_load(open(os.path.join(f, "config.yml"))); args["left_embodiment_config"] = args["right_embodiment_config"] = ec
    h = cam[args["camera"]["head_camera_type"]]; args["head_camera_h"], args["head_camera_w"] = h["h"], h["w"]
    args["eval_mode"] = True; args["render_freq"] = 0; args["eval_video_log"] = False; args.update(over)
    return args


def make_env(task):
    setup()
    return getattr(importlib.import_module(f"envs.{task}"), task)()


class Quiet:
    def __enter__(s): s.o = sys.stdout; sys.stdout = open(os.devnull, "w"); return s
    def __exit__(s, *x): sys.stdout = s.o


def make(task, config):
    args = make_args(task, config); args["camera"]["head_camera_type"] = args["camera"]["wrist_camera_type"] = "Large_D435"; args["head_camera_h"], args["head_camera_w"] = 480, 640; args["render_freq"] = 0
    return make_env(task), args


def install_grasp_hook(env):
    """records TCP world xyz of each arm at its FIRST gripper-close during the scripted expert (= expert grasp point), used only for
    expert-validity screening / failure analysis. Returns a dict filled during ``play_once``."""
    rec = {}; orig = env.set_gripper

    def hook(left_pos=None, right_pos=None, set_tag="all", **kw):
        for side, pos in (("left", left_pos), ("right", right_pos)):
            if pos is not None and pos < 0.5 and side not in rec and set_tag in (side, "all"):
                rec[side] = list(getattr(env.robot, f"get_{side}_tcp_pose")()[:3])
        return orig(left_pos=left_pos, right_pos=right_pos, set_tag=set_tag, **kw)

    env.set_gripper = hook
    return rec
