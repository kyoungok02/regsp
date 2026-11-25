"""[4-task study, Phase C: decoder RETRAINED (init = Machine-D VLABench-retrained weights, as the baseline adapted decoder) on union(Clean demos, randomized_500 demos)] Per-task post-grasp learned decoder = the ORIGINAL SafeDiff-VLA `temporal_decoder_grounded_grasp_v2` trainable stack (TemporalActionDecoder use_grounded_grasp_v2 + TargetQueryExtractor + VisualGroundingCrossAttention + TargetPointHead) re-trained on ONE RoboTwin task.
Frozen SmolVLA backbone tokens are cached (tokens_full/{task}.pt). Loss = original: (3*mse_pos + 6*mse_rot(sin/cos) + 1*mse_grip)/10 (masked by valid) + lambda_target*masked_mse(target_xyz, grasp_xyz of the first close inside the chunk, pre-grasp frames only).
Numerical adaptations (documented): euler of R@R_REF.T (gimbal), per-task mean/std. INIT=pretrained -> start from the Machine-D VLABench-retrained decoder weights ("adapted decoder"); INIT=random -> from scratch. STEPS=0 evaluation of the pretrained init = "frozen decoder" ablation.
usage: train_decoder_10.py TASK [INIT=pretrained|random] [STEPS]"""
import sys, os, json, copy, hashlib, numpy as np, torch, torch.nn as nn
from regsp import paths as P

from regsp.backbone.loader import load_policy
from regsp.backbone.rotation_encoding import encode, decode
from regsp.backbone.utils import masked_mse
OUT = str(P.clean_root()); t = sys.argv[1]; INIT = sys.argv[2] if len(sys.argv) > 2 else "pretrained"; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 6000; BATCH = 32; SEED = 1000
torch.manual_seed(SEED); rng = np.random.default_rng(SEED); dev = "cuda"
E = torch.load(f"{OUT}/tokens_full/{t}.pt", weights_only=False); [x.__setitem__("src", "clean") for x in E]; E2 = torch.load(f"{P.work_root()}/rand/tokens_full/{t}.pt", weights_only=False); [x.__setitem__("src", "rand") for x in E2]; E = E + E2; del E2; tr = [e for e in E if e["split"] == "train"]; va = [e for e in E if e["split"] == "val"]  # [4-task study] union of original Clean demo tokens and official randomized_500 demo tokens (first NEP episodes); tr = [e for e in E if e["split"] == "train"]; va = [e for e in E if e["split"] == "val"]
allc = np.concatenate([e["chunk"][e["valid"]] for e in tr]); allp = np.stack([e["state_ref"] for e in tr] + list(allc))
mean = torch.tensor(allp.mean(0), dtype=torch.float32, device=dev); std = torch.tensor(allp.std(0) + 1e-3, dtype=torch.float32, device=dev)
pol, pre = load_policy(); D = pol.decoder; Q = pol.target_query_extractor; V = pol.visual_grounding_cross_attn; TPH = pol.target_point_head
mods = nn.ModuleDict({"dec": D, "tqe": Q, "vgc": V, "tph": TPH}).to(dev)
if INIT == "random":
    for m in mods.modules():
        if hasattr(m, "reset_parameters") and m is not mods: m.reset_parameters()
    D.action_queries.data = torch.randn_like(D.action_queries) * 0.02; Q.query.data = torch.randn_like(Q.query) * 0.02
for p in mods.parameters(): p.requires_grad_(STEPS > 0)
def batch(es):
    tok = torch.stack([e["tok"] for e in es]).float().to(dev); pad = torch.stack([e["pad"] for e in es]).to(dev); mod = torch.stack([e["mod"] for e in es]).to(dev)
    s = (torch.tensor(np.stack([e["state_ref"] for e in es]), device=dev) - mean) / std; ch = (torch.tensor(np.stack([e["chunk"] for e in es]), device=dev) - mean) / std; vm = torch.tensor(np.stack([e["valid"] for e in es]), device=dev)
    ph = torch.tensor([e["phase"] for e in es], device=dev); gt = torch.tensor(np.stack([e["grasp_xyz"] if e["grasp_xyz"] is not None else np.zeros(3, np.float32) for e in es]), device=dev)[:, :3]; gt = (gt - mean[:3]) / std[:3]
    tv = torch.tensor([(e["next_close_in"] is not None and e["next_close_in"] <= 50 and e["phase"] == 0) for e in es], device=dev)
    return tok, pad, mod, s, ch, vm, ph, gt, tv
def forward(tok, pad, mod, s, ph):
    q = Q(tok, pad, mod); gf = V(q, tok, pad, mod); txyz = TPH(gf); return D(tok, pad, encode(s, mean[3:6], std[3:6]), target_xyz=txyz, latent_modality_ids=mod, grounded_target_feature=gf, grasp_phase=ph), txyz
def loss_fn(es):
    tok, pad, mod, s, ch, vm, ph, gt, tv = batch(es); pred, txyz = forward(tok, pad, mod, s, ph); clean = encode(ch, mean[3:6], std[3:6])
    lp, lr_, lg = masked_mse(pred[..., :3], clean[..., :3], vm), masked_mse(pred[..., 3:9], clean[..., 3:9], vm), masked_mse(pred[..., 9:10], clean[..., 9:10], vm)
    la = (3 * lp + 6 * lr_ + 1 * lg) / 10; lt = masked_mse(txyz[:, None], gt[:, None], tv[:, None]); return la + lt, (float(lp), float(lr_), float(lg), float(lt))
@torch.no_grad()
def evaluate(es, phase=None):
    mods.eval(); P, R, G, T, n = [], [], [], [], 0; sel = [e for e in es if phase is None or e["phase"] == phase]
    for i in range(0, len(sel), 64):
        b = sel[i:i + 64]; tok, pad, mod, s, ch, vm, ph, gt, tv = batch(b); pred, txyz = forward(tok, pad, mod, s, ph); dec = decode(pred, mean[3:6], std[3:6]); m = vm.float()
        d_pos = ((dec[..., :3] - ch[..., :3]) * std[:3]).norm(dim=-1); d_rot = ((dec[..., 3:6] - ch[..., 3:6]) * std[3:6]); d_rot = torch.atan2(torch.sin(d_rot), torch.cos(d_rot)).abs().mean(-1); d_g = (dec[..., 6] - ch[..., 6]).abs() * std[6]
        for k, v_ in ((P, d_pos), (R, d_rot), (G, d_g)): k.append(((v_ * m).sum(1) / m.sum(1).clamp_min(1)).cpu())
        if tv.any(): T.append(((txyz - gt) * std[:3]).norm(dim=-1)[tv].cpu())
    f = lambda L: float(torch.cat(L).mean()) if L else None
    return {"pos_cm": None if not P else f(P) * 100, "rot_deg_mean_axis": None if not R else f(R) * 57.2958, "grip_mae": f(G), "target_xyz_cm": None if not T else f(T) * 100, "n": len(sel)}
log = {"task": t, "init": INIT, "steps": STEPS, "n_train": len(tr), "n_val": len(va), "step0_val": {"all": evaluate(va), "post": evaluate(va, 1), "pre": evaluate(va, 0)}}; print("step0", json.dumps(log["step0_val"]), flush=True)
if STEPS > 0:
    opt = torch.optim.AdamW([p for p in mods.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-6); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    for step in range(1, STEPS + 1):
        mods.train(); es = [tr[i] for i in rng.choice(len(tr), BATCH, replace=False)]; l, parts = loss_fn(es); opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(mods.parameters(), 1.0); opt.step(); sch.step()
        if step % 1000 == 0 or step == STEPS: v = evaluate(va); log.setdefault("curve", []).append({"step": step, "loss": float(l), "parts(pos,rot,grip,target)": parts, "val": v}); print(step, float(l), json.dumps(v), flush=True)
    log["final_val"] = {"all": evaluate(va), "post": evaluate(va, 1), "pre": evaluate(va, 0)}; log["final_val_clean"] = {"all": evaluate([e for e in va if e["src"] == "clean"]), "post": evaluate([e for e in va if e["src"] == "clean"], 1)}; log["final_val_rand"] = {"all": evaluate([e for e in va if e["src"] == "rand"]), "post": evaluate([e for e in va if e["src"] == "rand"], 1)}; log["n_train_clean"] = sum(e["src"] == "clean" for e in tr); log["n_train_rand"] = sum(e["src"] == "rand" for e in tr)
    os.makedirs((str(P.work_root()) + "/ckpt_sp/") + t, exist_ok=True); fn = (str(P.work_root()) + "/ckpt_sp/") + t + f"/decoder_{INIT}.pt"; torch.save({"mods": {k: v.state_dict() for k, v in mods.items()}, "mean": mean.cpu(), "std": std.cpu(), "init": INIT, "R_REF": "Ry(90deg)"}, fn); log["sha256"] = hashlib.sha256(open(fn, "rb").read()).hexdigest()
json.dump(log, open((str(P.work_root()) + "/training/sp_decoder_") + t + f"_{INIT}.json", "w"), indent=1)
