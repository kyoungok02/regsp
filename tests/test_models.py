import numpy as np
import pytest
import torch

import regsp.grounding as R
from regsp.backbone.rotation_encoding import GRIPPER_INDEX_RAW, decode, encode
from regsp.backbone.temporal_decoder import TemporalActionDecoder
from regsp.decoding import (HIST, MODE_NAMES, NMODE, NPROG, NSTAGE, PROG_NAMES, AdaptiveDecoder, BimDecoder, PicProgressHead, ProgressHead, lang_mask, mode_lr)


def test_proposal_head_shape_and_target_decode_roundtrip():
    h = R.Head(64).eval(); assert h(torch.zeros(2, 768, 32, 32)).shape == (2, 3, 64, 64)
    hm, off, msk = R.targets([(321.0, 241.0), (100.0, 400.0)])
    logit = torch.logit(torch.from_numpy(hm).clamp(1e-4, 1 - 1e-4)); out = torch.zeros(3, 64, 64); out[0] = logit; out[1:] = torch.from_numpy(off)
    pk = R.decode(out)
    assert len(pk) == 2 and min(np.hypot(u - 321, v - 241) for u, v, _ in pk) < 1.0 and min(np.hypot(u - 100, v - 400) for u, v, _ in pk) < 1.0


def _scorer_inputs(B=2, K=3, T=5):
    return torch.randn(B, T, 960), torch.ones(B, T, dtype=torch.bool), torch.randn(B, K, 64, 960)


def test_scorer_variants_reduce_to_the_original_at_initialisation():
    torch.manual_seed(0); s = R.Scorer().eval(); txt, tm, f = _scorer_inputs(); ref = s(txt, tm, f); assert ref.shape == (2, 3)
    c = R.CycleScorer().eval(); c.load_state_dict({**s.state_dict(), "cyc.weight": c.cyc.weight.data}); assert torch.allclose(c(txt, tm, f, torch.tensor([0, 1])), ref, atol=1e-6)  # zero-init cycle embedding
    g = R.SubgoalScorer().eval(); g.load_state_dict({**c.state_dict(), "tq_stage": g.tq_stage.data}); assert torch.allclose(g(txt, tm, f, torch.tensor([0, 1])), ref, atol=1e-6)  # zero-init stage query


def test_anchor_head_outputs_cm_offset():
    a = R.AnchorHead(center=(0.0, 0.3, 0.05)).eval(); assert a(torch.randn(2, 64, 960), torch.randn(2, 64, 960), torch.zeros(2, 3)).shape == (2, 3)


def _base(hidden=32, latent=32, horizon=8):
    return TemporalActionDecoder(action_dim=10, state_dim=10, latent_dim=latent, hidden_dim=hidden, horizon=horizon, num_layers=2, num_heads=4, ffn_dim=64, dropout=0.0, use_target_xyz=True, use_grounded_grasp_v2=True).eval()


def test_bimanual_extension_is_zero_initialised_and_adaptive_decoder_emits_two_10d_chunks():
    torch.manual_seed(0); base = _base(); bim = BimDecoder(base).eval(); B, N, H = 2, 6, 8
    tok, pad = torch.randn(B, N, 32), torch.ones(B, N, dtype=torch.bool); s, s2, tgt, gf, ph = torch.randn(B, 10), torch.randn(B, 10), torch.randn(B, 3), torch.randn(B, 32), torch.ones(B, dtype=torch.long)
    mod = torch.zeros(B, N, dtype=torch.long); role = torch.tensor([0, 3]); place = torch.randn(B, 3)
    ref = base(tok, pad, s, target_xyz=tgt, latent_modality_ids=mod, grounded_target_feature=gf, grasp_phase=ph)
    out = bim(tok, pad, s, tgt, mod, gf, ph, s2, role, place)
    assert out.shape == (B, H, 10) and torch.allclose(out, ref, atol=1e-5)
    ad = AdaptiveDecoder(bim).eval(); P, Q = ad(tok, pad, s, s2, tgt, mod, gf, ph, role, place, torch.tensor([0, 2]), torch.tensor([0, 5]), torch.zeros(B, HIST))
    assert P.shape == Q.shape == (B, H, 10)
    # 10-D encoded chunk -> 7-D physical (xyz, rpy, gripper) per arm -> synchronised 14-D [L, R] chunk
    mean, std = torch.zeros(7), torch.ones(7); d7 = [decode(x, mean[3:6], std[3:6]) for x in (P, Q)]
    assert d7[0].shape == (B, H, 7) and torch.cat(d7, -1).shape == (B, H, 14) and GRIPPER_INDEX_RAW == 6
    x = torch.randn(B, H, 7); x[..., 3:6] = torch.rand(B, H, 3) * 5 - 2.5; assert torch.allclose(decode(encode(x, mean[3:6], std[3:6]), mean[3:6], std[3:6]), x, atol=1e-4)


def test_progress_heads_and_label_spaces():
    assert (NPROG, NMODE, NSTAGE, HIST) == (5, 3, 12, 16) and PROG_NAMES[3] == "FAILED_OR_RECOVERY" and MODE_NAMES == ("PRIMARY", "PARTNER", "BIMANUAL")
    B, N = 2, 200; tok, pad = torch.randn(B, N, 960), torch.ones(B, N, dtype=torch.bool); s = torch.randn(B, 10); mem = torch.randn(B, 6); hist = torch.randn(B, HIST); m = torch.tensor([0, 2])
    ph = ProgressHead().eval(); pl, ml = ph(tok, pad, s, s, mem, hist, m); assert pl.shape == (B, NPROG) and ml.shape == (B, NMODE)
    pp = PicProgressHead().eval(); pp.load_state_dict(ph.state_dict(), strict=False)
    pl2, ml2 = pp(tok, pad, s, s, mem, hist, m); assert torch.allclose(pl, pl2, atol=1e-5) and torch.allclose(ml, ml2, atol=1e-5)  # zero-init instruction pathway
    mod = torch.zeros(B, N, dtype=torch.long); mod[:, 192:200] = 1; assert lang_mask(pad, mod).sum() == B * 8 and lang_mask(pad).shape == pad.shape


@pytest.mark.parametrize("mode,primary,expect", [(2, "L", "BIMANUAL"), (2, "R", "BIMANUAL"), (0, "L", "LEFT_ARM"), (0, "R", "RIGHT_ARM"), (1, "L", "RIGHT_ARM"), (1, "R", "LEFT_ARM")])
def test_arm_mode_mapping(mode, primary, expect):
    assert mode_lr(mode, primary) == expect
