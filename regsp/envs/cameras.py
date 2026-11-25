"""Shared RoboTwin camera geometry and the frozen SigLIP patch-feature extractor (subset of the research tree's ``rt_common.py``)."""
import json
import numpy as np
import torch
import cv2

from regsp.paths import data_file


def load_backbone(dev="cuda"):
    from regsp.backbone.loader import load_policy

    return load_policy(dev)[0]


CAL = json.load(open(data_file("camera_calibration.json"))); K = np.array(CAL["head_intrinsic"]); CAMS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
def head_w2c(): return np.linalg.inv(np.array(CAL["head_cam2world_cv"]))
def wrist_c2w(side, fk, q6): return fk.T_origin @ fk.link6(q6) @ np.array(CAL[f"{side}_wrist"]["T_link6_cam_cv"])
def project(w2c, p):  # world point -> (u,v,depth) in 640x480
    pc = w2c[:3, :3] @ p + w2c[:3, 3]; return (K @ pc)[:2] / pc[2], pc[2]
@torch.no_grad()
def patch_feats(vm, imgs_rgb, dev="cuda"):  # list of HxWx3 uint8 -> [N,768,32,32] fp16 (same preprocessing as v21p_proposals.patch_feats: bicubic 512x512, scale [-1,1])
    x = torch.from_numpy(np.stack([cv2.resize(i, (512, 512), interpolation=cv2.INTER_CUBIC) for i in imgs_rgb])).permute(0, 3, 1, 2).float().div(255).to(dev) * 2 - 1
    h = vm(pixel_values=x.to(vm.dtype)).last_hidden_state.float(); return h.reshape(len(x), 32, 32, -1).permute(0, 3, 1, 2).half().cpu()

