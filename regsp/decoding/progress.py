"""Task-progress heads ([X]): stage-completion logits for a queried stage + arm-mode logits.

``ProgressHead`` is a light attention-pooling head on the frozen backbone tokens + both-arm proprioception + grounded target/place +
history (verbatim ``acl_common.py``). ``PicProgressHead`` adds the persistent-instruction pathway (verbatim ``pic_common.py``).
"""
import torch
import torch.nn as nn

from regsp.backbone.temporal_decoder import StateEncoder
from .adaptive import HIST, MEMD, NPROG, NMODE

class ProgressHead(nn.Module):
    def __init__(self, latent=960, h=256, nq=4, drop=0.1):
        super().__init__(); self.proj = nn.Linear(latent, h); self.q = nn.Parameter(torch.randn(1, nq, h) * 0.02); self.attn = nn.MultiheadAttention(h, 4, batch_first=True, dropout=drop)
        self.ctx = StateEncoder(10 + 10 + MEMD + HIST, h); self.trunk = nn.Sequential(nn.Linear(nq * h + h, h), nn.SiLU(), nn.Dropout(drop), nn.Linear(h, h), nn.SiLU()); self.stage = nn.Embedding(3, h)
        self.prog = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, NPROG)); self.mode = nn.Linear(h, NMODE)
    def forward(self, tok, pad, sP, sQ, mem, hist, m):
        """mem = [target xyz, place xyz] (normalised); m = queried stage index (clipped to 2). Returns (progress logits [B,5], mode logits [B,3])"""
        x = self.proj(tok); z, _ = self.attn(self.q.expand(len(tok), -1, -1), x, x, key_padding_mask=None if pad is None else ~pad, need_weights=False)
        f = self.trunk(torch.cat([z.flatten(1), self.ctx(torch.cat([sP, sQ, mem, hist], -1))], -1)); return self.prog(torch.cat([f, self.stage(m)], -1)), self.mode(f)

NIMG = 192  # image-token rows of the prefix (3 cameras x 64); language rows follow, the state row is last
def lang_mask(pad, mod=None):
    """bool [B,T]: instruction-token rows of the prefix (mod == 1 when the modality ids are available, else by position: after the 192 image rows, before the last valid (state) row)"""
    if mod is not None: return (mod == 1) & pad.bool()
    pad = pad.bool(); idx = torch.arange(pad.shape[1], device=pad.device)[None]; last = pad.long().sum(1, keepdim=True) - 1; return (idx >= NIMG) & (idx < last) & pad
class PicProgressHead(ProgressHead):
    def __init__(self, latent=960, h=256, **k):
        super().__init__(latent, h, **k); self.itxt = nn.Linear(latent, h); self.iq = nn.Parameter(torch.randn(1, 1, h) * 0.02); self.iattn = nn.MultiheadAttention(h, 4, batch_first=True, dropout=0.1)
        self.iprog = nn.Sequential(nn.Linear(3 * h, h), nn.SiLU(), nn.Linear(h, NPROG)); self.imode = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, NMODE))
        for m in (self.iprog[-1], self.imode[-1]): nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
    def instr(self, lang, lmask):
        """pooled instruction vector e [B,h] from instruction-token states lang [B,L,960] (lmask True = valid)"""
        x = self.itxt(lang); e, _ = self.iattn(self.iq.expand(len(x), -1, -1), x, x, key_padding_mask=~lmask, need_weights=False); return e.squeeze(1)
    def forward(self, tok, pad, sP, sQ, mem, hist, m, lang=None, lmask=None):
        """lang/lmask: the PERSISTENT instruction state (episode-start language rows); default = the language rows of `tok`"""
        x = self.proj(tok); z, _ = self.attn(self.q.expand(len(tok), -1, -1), x, x, key_padding_mask=None if pad is None else ~pad, need_weights=False)
        f = self.trunk(torch.cat([z.flatten(1), self.ctx(torch.cat([sP, sQ, mem, hist], -1))], -1)); s = self.stage(m)
        if lang is None: lang, lmask = tok, lang_mask(pad)
        e = self.instr(lang, lmask); return self.prog(torch.cat([f, s], -1)) + self.iprog(torch.cat([f, e, s], -1)), self.mode(f) + self.imode(torch.cat([f, e], -1))
