"""Language-conditioned referent scorers.

``Scorer``        ORIGINAL V21 HR referent scorer "B" (instruction-token states x candidate crop tokens).
``CycleScorer``   Scorer + one additive cycle-index embedding on the pooled instruction query (multi-referent tasks).
``SubgoalScorer`` CycleScorer + a stage-specific offset of the learned text query (persistent instruction conditioning, used for
                  ``stack_blocks_two``). Zero-initialised extension: identical to CycleScorer at step 0.
Verbatim extraction of ``v21rt.py``, ``bim_common.py`` and ``pic_common.py``. Original ``pic_common.py`` docstring:

[V21-PIC] Persistent Instruction Conditioning - learned extensions ([X] documented, separately versioned; nothing frozen is modified; no unrelated VLA architecture).
SubgoalScorer  = the ORIGINAL HR referent Scorer("B") (+ the 10-task CycleScorer cycle embedding) with ONE additional parameter tensor: a stage-specific offset of the learned TEXT query (`tq_stage`, zero-init => identical to CycleScorer at step 0).
                 The original scorer pools the instruction-token states with a single learned query, the cycle index was only ADDED to the pooled vector, i.e. logit = f(instruction) + g(stage): the stage could not change WHICH instruction words
                 are read (measured consequence: under the role-exchanged stack instruction the first grasp switched, the second target was never found). With a stage-specific text query the SAME instruction is re-read at every stage
                 ("select the semantic subgoal of stage m from the instruction"): logit = f(instruction, stage).
PicProgressHead = acl_common.ProgressHead + an explicit INSTRUCTION pathway: attention pooling of the instruction-token states (the policy VLM's own language rows; cached at episode start = persistent instruction state, or the live rows)
                 -> residual progress logits / arm-mode logits (final layers zero-init => identical to the loaded ProgressHead at step 0). Trained with instruction-counterfactual on-policy samples (same observation, both valid instructions,
                 outcome label w.r.t. each instruction) the stage verdict becomes a verification AGAINST THE INSTRUCTION (wrong referent grasped / placed => FAILED_OR_RECOVERY).
"""
import torch
import torch.nn as nn

D = 128
class Scorer(nn.Module):  # ORIGINAL v21hr_candidate.Scorer("B")
    def __init__(self):
        super().__init__(); self.txt = nn.Linear(960, D); self.tq = nn.Parameter(torch.randn(1, 1, D) * .02); self.tattn = nn.MultiheadAttention(D, 4, batch_first=True, dropout=0.1)
        self.vis = nn.Sequential(nn.LayerNorm(960), nn.Linear(960, D)); self.drop = nn.Dropout(0.1); self.pos = nn.Parameter(torch.randn(64, D) * .02); self.vq = nn.Parameter(torch.randn(1, 1, D) * .02); self.vattn = nn.MultiheadAttention(D, 4, batch_first=True, dropout=0.1)
        self.wq, self.wv = nn.Linear(D, D), nn.Linear(D, D)
    def forward(self, txt, tmask, feats):  # txt [B,T,960]; feats [B,K,64,960]
        t = self.txt(txt); q, _ = self.tattn(self.tq.expand(len(txt), -1, -1), t, t, key_padding_mask=~tmask, need_weights=False)
        Bn, Kc = feats.shape[:2]; v = self.vis(feats.reshape(Bn * Kc, 64, 960)) + self.pos; v, _ = self.vattn(self.vq.expand(Bn * Kc, -1, -1), v, v, need_weights=False); v = v.reshape(Bn, Kc, D)
        return torch.einsum("bd,bkd->bk", self.wq(self.drop(q.squeeze(1))), self.wv(self.drop(v))) / D ** 0.5

SD = D  # alias used by the original bim_common
class CycleScorer(Scorer):  # [X] ORIGINAL HR Scorer("B") + one additive cycle-index embedding on the pooled instruction query (zero-init => identical to the original at step 0); cycle = min(number of closes so far, 1)
    def __init__(self):
        super().__init__(); self.cyc = nn.Embedding(2, SD); nn.init.zeros_(self.cyc.weight)
    def forward(self, txt, tmask, feats, cycle):
        t = self.txt(txt); q, _ = self.tattn(self.tq.expand(len(txt), -1, -1), t, t, key_padding_mask=~tmask, need_weights=False)
        Bn, Kc = feats.shape[:2]; v = self.vis(feats.reshape(Bn * Kc, 64, 960)) + self.pos; v, _ = self.vattn(self.vq.expand(Bn * Kc, -1, -1), v, v, need_weights=False); v = v.reshape(Bn, Kc, SD)
        return torch.einsum("bd,bkd->bk", self.wq(self.drop(q.squeeze(1) + self.cyc(cycle))), self.wv(self.drop(v))) / SD ** 0.5

class SubgoalScorer(CycleScorer):
    def __init__(self, nstage=2):
        super().__init__(); self.tq_stage = nn.Parameter(torch.zeros(nstage, 1, SD))
    def forward(self, txt, tmask, feats, cycle):
        t = self.txt(txt); q, _ = self.tattn(self.tq.expand(len(txt), -1, -1) + self.tq_stage[cycle], t, t, key_padding_mask=~tmask, need_weights=False)
        Bn, Kc = feats.shape[:2]; v = self.vis(feats.reshape(Bn * Kc, 64, 960)) + self.pos; v, _ = self.vattn(self.vq.expand(Bn * Kc, -1, -1), v, v, need_weights=False); v = v.reshape(Bn, Kc, SD)
        return torch.einsum("bd,bkd->bk", self.wq(self.drop(q.squeeze(1) + self.cyc(cycle))), self.wv(self.drop(v))) / SD ** 0.5
