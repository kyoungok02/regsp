"""Adaptive single/bimanual TemporalActionDecoder (joint 2 x 50-step chunk: primary 7-D + partner 7-D = 14-D) and its constants.

Verbatim extraction of ``acl_common.py``; the progress head and the proprioception helpers live in ``regsp.decoding.progress`` and
``regsp.execution.proprio``. ORIGINAL module docstring (scientific identity, conditioning terms) is kept below.

Adaptive Single/Bimanual TemporalActionDecoder + task-progress head ([X] documented extension experiment; frozen V21 checkpoints/results are never modified).
Scientific identity: Original Counterfactual Grounding + Adaptive Single/Bimanual TemporalActionDecoder + Multi-Stage Closed-Loop Execution (no RGL/KSP, no scripted expert).

AdaptiveDecoder  = the ORIGINAL `TemporalActionDecoder` (kept as `bim.base`, initialised from the task's corrected bimanual decoder ckpt_sp/<task>/bim_decoder_b2.pt) extended to a JOINT, SYNCHRONISED two-effector prediction:
  * 2 x H action queries (H=50): slot P ("primary" = the arm that owns the current manipulation stage) and slot Q ("partner"). Both slots share the original learned queries / horizon positions; slot Q adds one learned slot embedding.
    One pass of the ORIGINAL transformer decoder stack over the 2H queries (self-attention couples the two arms at every horizon step, cross-attention to the VLM latent) -> head P (= original `action_head`, 7-D pose+gripper in the original 10-D rotation encoding) and head Q (new Linear, initialised as a copy of `action_head`).
    Concatenating the two heads in fixed [L,R] order gives the synchronised 14-D bimanual action chunk; using only head P reproduces the original 7-D single-arm prediction.
  * conditioning (all additive global terms, the original pattern): own-arm state (original `state_encoder`), other-arm state (`partner_encoder`: both-arm proprioception incl. both grippers), grounded target xyz (original `target_xyz_encoder`),
    grounded visual feature, grasp phase, role (arm x cycle), place/receiving target (`place_encoder`), ARM MODE embedding (PRIMARY-only / PARTNER-only / BIMANUAL == LEFT_ARM / RIGHT_ARM / BIMANUAL given the primary arm identity),
    STAGE embedding (completed stages {0,1,2} x primary-holds x partner-holds), HISTORY encoder (both arms' TCP displacement and gripper command 1 and 2 control steps ago = observation/action history). New terms are zero-initialised.
ProgressHead = light attention-pooling head on the SAME frozen-backbone tokens (image + instruction tokens) + both-arm proprioception + grounded target / place target + history:
  * stage-completion logits for a QUERIED stage m: NOT_STARTED / IN_PROGRESS / COMPLETED / FAILED_OR_RECOVERY / TASK_COMPLETED  (FAILED only receives supervision from on-policy training-domain rollouts with training-only privileged outcome labels; never fabricated)
  * arm-mode logits (which effectors act in the next chunk) = learned arm-mode selector used by the closed-loop executor.
  The progress head deliberately does NOT see the executor's own bookkeeping (number of completed stages / hold flags), so its verdict is an independent observation-based check.
finger_opening() = MEASURED finger joint position (robot proprioception; RoboTwin's get_*_gripper_val returns the COMMANDED value, which the frozen executor used and which is always 0 after a close command).
"""
import torch
import torch.nn as nn

from regsp.backbone.temporal_decoder import StateEncoder

HIST = 16; MEMD = 6; NPROG = 5; NMODE = 3; NSTAGE = 12; PROG_NAMES = ("NOT_STARTED", "IN_PROGRESS", "COMPLETED", "FAILED_OR_RECOVERY", "TASK_COMPLETED"); MODE_NAMES = ("PRIMARY", "PARTNER", "BIMANUAL")
def mode_lr(mode, primary):  # relative mode -> LEFT_ARM / RIGHT_ARM / BIMANUAL
    return "BIMANUAL" if mode == 2 else ("LEFT_ARM" if (primary == "L") == (mode == 0) else "RIGHT_ARM")
class AdaptiveDecoder(nn.Module):
    def __init__(self, bim):
        super().__init__(); self.bim = bim; b = bim.base; h = b.state_encoder.net[0].out_features; self.H = b.horizon
        self.head_partner = nn.Linear(h, b.action_head.out_features); self.head_partner.load_state_dict(b.action_head.state_dict())
        self.slot = nn.Parameter(torch.randn(h) * 0.02); self.mode_emb = nn.Embedding(NMODE, h); self.stage_emb = nn.Embedding(NSTAGE, h); self.hist_encoder = StateEncoder(HIST, h)
        nn.init.zeros_(self.mode_emb.weight); nn.init.zeros_(self.stage_emb.weight); nn.init.zeros_(self.hist_encoder.net[-1].weight); nn.init.zeros_(self.hist_encoder.net[-1].bias)
    def forward(self, tok, pad, sP, sQ, target, mod, gf, phase, role, place, mode, stage, hist):
        """sP/sQ: encoded 10-D states of primary / partner; target, place: normalised xyz; returns (chunk_P [B,H,10], chunk_Q [B,H,10])"""
        b = self.bim.base; B = tok.shape[0]; memory = b.latent_projection(tok); kpm = None if pad is None else ~pad; q0 = (b.action_queries + b.horizon_position_embedding)[None].expand(B, -1, -1)
        g = b.target_xyz_encoder(target) + b.grounded_feature_encoder(gf) + b.phase_encoder(phase) + self.bim.role_embedding(role) + self.bim.place_encoder(place) + self.mode_emb(mode) + self.stage_emb(stage) + self.hist_encoder(hist)
        qP = q0 + (b.state_encoder(sP) + self.bim.partner_encoder(sQ) + g)[:, None]; qQ = q0 + self.slot + (b.state_encoder(sQ) + self.bim.partner_encoder(sP) + g)[:, None]
        h = b.transformer(tgt=torch.cat([qP, qQ], 1), memory=memory, memory_key_padding_mask=kpm); return b.action_head(h[:, :self.H]), self.head_partner(h[:, self.H:])
