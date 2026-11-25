"""Bimanual extension of the TemporalActionDecoder ([X], not part of the single-effector V21).

Verbatim extraction of ``bim_common.py`` (the CycleScorer now lives in ``regsp.grounding.scorer``). Original module docstring:

[bim_common] BIMANUAL EMBODIMENT EXTENSION of the grounded-final V21 learned decoder ([X] new extension; NOT part of the historical single-effector V21).
The ORIGINAL `TemporalActionDecoder` (+ TargetQueryExtractor / VisualGroundingCrossAttention / TargetPointHead) is kept UNCHANGED as `base`; its forward is reproduced term-for-term and extended with THREE additional additive global-conditioning terms, exactly following the original pattern
(`subgoal_encoder`, `target_xyz_encoder`, `grounded_feature_encoder`, `phase_encoder` are all `StateEncoder`/`Embedding` outputs added to every action query):
  partner_encoder : StateEncoder(10 -> hidden) on the PARTNER effector's encoded state (xyz, 6-D rotation encoding, gripper)  [partner-arm state]
  role_embedding  : Embedding(4, hidden): (arm in {L,R}) x (cycle index clipped to {0,1})                                    [arm / cycle role]
  place_encoder   : StateEncoder(3 -> hidden) on the task-level PLACE target xyz of the current cycle                        [placement/receiving target]
All extra terms are zero-initialised (last Linear zero, embedding zero) => at step 0 the model equals the task-adapted original decoder exactly. Objective unchanged: (3*pos + 6*rot + 1*grip)/10 + lambda_target loss. Output layout unchanged (one 10-D effector chunk per call).
"""
import torch
import torch.nn as nn

from regsp.backbone.temporal_decoder import StateEncoder
from regsp.grounding.scorer import CycleScorer, SD  # noqa: F401  (re-exported for compatibility with the research scripts)

class BimDecoder(nn.Module):
    def __init__(self, base, hidden=None):
        super().__init__(); self.base = base; h = base.state_encoder.net[0].out_features if hidden is None else hidden
        self.partner_encoder = StateEncoder(10, h); self.role_embedding = nn.Embedding(4, h); self.place_encoder = StateEncoder(3, h)
        for m in (self.partner_encoder, self.place_encoder): nn.init.zeros_(m.net[-1].weight); nn.init.zeros_(m.net[-1].bias)
        nn.init.zeros_(self.role_embedding.weight)
    def forward(self, tok, pad, state_enc, target_xyz, mod, gf, phase, partner_enc, role, place):
        b = self.base; B = tok.shape[0]; memory = b.latent_projection(tok); kpm = None if pad is None else ~pad
        q = (b.action_queries + b.horizon_position_embedding)[None].expand(B, -1, -1) + b.state_encoder(state_enc)[:, None]
        q = q + b.target_xyz_encoder(target_xyz)[:, None] + b.grounded_feature_encoder(gf)[:, None] + b.phase_encoder(phase)[:, None]
        q = q + self.partner_encoder(partner_enc)[:, None] + self.role_embedding(role)[:, None] + self.place_encoder(place)[:, None]  # [X] bimanual terms
        return b.action_head(b.transformer(tgt=q, memory=memory, memory_key_padding_mask=kpm))
