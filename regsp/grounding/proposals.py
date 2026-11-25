"""Class-agnostic 64x64 proposal heatmap head (ORIGINAL V21 ``Head(64)``) with focal loss, target generation and peak decoding.

Verbatim extraction of ``v21rt.py``. Input: frozen SigLIP patch features [B,768,32,32]; output [B,3,64,64] = (heat logit, du, dv).
``targets``/``decode`` use the anisotropic RoboTwin cell size (640/g, 480/g) (embodiment adapter).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import W, Hh, THR, KCAP

class Head(nn.Module):  # ORIGINAL v21q Head(64)
    def __init__(self, grid=64):
        super().__init__(); self.g = grid
        L = [nn.Conv2d(768, 128, 3, padding=1), nn.GELU()] + ([nn.ConvTranspose2d(128, 128, 2, stride=2), nn.GELU()] if grid == 64 else []) + [nn.Conv2d(128, 128, 3, padding=1), nn.GELU(), nn.Conv2d(128, 3, 1)]; self.net = nn.Sequential(*L)
    def forward(self, f): return self.net(f)
def focal(logit, hm):  # ORIGINAL v21p_proposals.focal
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4); pos = hm.eq(1).float(); neg = 1 - pos
    return -((1 - p) ** 2 * torch.log(p) * pos).sum() / pos.sum().clamp_min(1) - ((1 - hm) ** 4 * p ** 2 * torch.log(1 - p) * neg).sum() / pos.sum().clamp_min(1)
def targets(centres_uv, g=64):  # ORIGINAL v21q targets (sigma 1.5 cells at g=64) with anisotropic RoboTwin cell size (640/g, 480/g) = embodiment adapter
    sig = 1.5; hm = np.zeros((g, g), np.float32); off = np.zeros((2, g, g), np.float32); msk = np.zeros((g, g), np.float32); yy, xx = np.mgrid[0:g, 0:g]
    for u, v in centres_uv:
        cu, cv_ = u / (W / g), v / (Hh / g); hm = np.maximum(hm, np.exp(-((xx + 0.5 - cu) ** 2 + (yy + 0.5 - cv_) ** 2) / (2 * sig ** 2))); iu, iv = int(cu), int(cv_)
        if 0 <= iu < g and 0 <= iv < g: hm[iv, iu] = 1.0; off[:, iv, iu] = (cu - iu - 0.5, cv_ - iv - 0.5); msk[iv, iu] = 1
    return hm, off, msk
def decode(out, thr=THR, k=KCAP, g=64):  # ORIGINAL decode, anisotropic cell size; returns (u,v,score) in full-res px
    heat = torch.sigmoid(out[0]); mx = F.max_pool2d(heat[None, None], 3, 1, 1)[0, 0]; pk = (heat == mx) & (heat >= thr); ys, xs = torch.nonzero(pk, as_tuple=True); sc = heat[ys, xs]; o = torch.argsort(-sc)[:k]
    return [((xs[j].item() + 0.5 + out[1, ys[j], xs[j]].item()) * (W / g), (ys[j].item() + 0.5 + out[2, ys[j], xs[j]].item()) * (Hh / g), sc[j].item()) for j in o.tolist()]
