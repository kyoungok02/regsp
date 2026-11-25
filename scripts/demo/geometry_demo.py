#!/usr/bin/env python
"""Checkpoint-free demo of the two-view grounding geometry: a point is observed by the head camera and a wrist camera; the
class-agnostic proposals (pixels) are associated by the Hungarian two-ray gap (<= 3 cm) and triangulated back to 3-D."""
import numpy as np

import regsp.grounding as R


def look_at(c, tgt):
    z = (tgt - c) / np.linalg.norm(tgt - c); x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x); y = np.cross(z, x)
    E = np.eye(4); E[:3, :3] = np.stack([x, y, z], 1).T; E[:3, 3] = -E[:3, :3] @ c
    return E


objects = [np.array([0.05, 0.10, 0.02]), np.array([0.20, 0.25, 0.02])]
E_head, E_wrist = look_at(np.array([0.0, -0.5, 0.8]), objects[0]), look_at(np.array([0.25, 0.05, 0.45]), objects[0])
head = [(*R.project(E_head, p)[0], 0.9) for p in objects]; wrist = [(*R.project(E_wrist, p)[0], 0.8) for p in objects[::-1]]
for i, (p, j, gap) in sorted((i, v) for i, v in R.associate(head, wrist, E_head, E_wrist).items()):
    print(f"head proposal {i} <-> wrist proposal {j}: gap {gap * 100:.3f} cm, triangulated {np.round(p, 4)} (truth {objects[i]})")
