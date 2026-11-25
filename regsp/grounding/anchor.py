"""AnchorHead: learned centre-offset correction of the triangulated referent (ORIGINAL V21 ``AnchorHead``).

Only the input re-centering constant (VLABench (0,0.45,0.05) in the original) is an embodiment adapter: here it is the per-task training mean
of the triangulated reference in the robot-root frame (stored in the checkpoint). Output: offset in cm.
"""
import torch
import torch.nn as nn

class AnchorHead(nn.Module):  # ORIGINAL v21g_pipeline.AnchorHead; only the input re-centering constant (VLABench (0,0.45,0.05)) is an EMBODIMENT ADAPTER -> per-task training mean of the triangulated reference (robot-root frame)
    D = 128
    def __init__(self, center=(0.0, 0.45, 0.05)):
        super().__init__(); D = self.D; self.register_buffer("center", torch.tensor(center, dtype=torch.float32)); self.vis = nn.Sequential(nn.LayerNorm(960), nn.Linear(960, D)); self.pos = nn.Parameter(torch.randn(2, 64, D) * .02); self.q = nn.Parameter(torch.randn(1, 1, D) * .02); self.attn = nn.MultiheadAttention(D, 4, batch_first=True, dropout=0.1)
        self.geo = nn.Sequential(nn.Linear(3, 64), nn.GELU()); self.mlp = nn.Sequential(nn.Linear(2 * D + 64, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 3))
    def forward(self, tf, tr, ref):
        outs = []
        for i, t in enumerate((tf, tr)): v = self.vis(t) + self.pos[i]; o, _ = self.attn(self.q.expand(len(t), -1, -1), v, v, need_weights=False); outs.append(o.squeeze(1))
        return self.mlp(torch.cat(outs + [self.geo((ref - self.center) / 0.2)], -1))  # offset in cm
