"""[4-task study, Phase C: StagePlan head RETRAINED on union(Clean demos, randomized_500 demos); recipe identical except steps (env SPSTEPS, default 6000 for ~3x data)] Per-task RETAINED StagePlan-V2 head (original `StagePlanV2Head`, 3 cams, state_dim 7) trained on cached frozen-backbone tokens of THIS task's Clean demos (arm-conditioned 7-D TCP state). Loss / scales / optimiser = original `train_stageplan_v2.py`
(smooth-L1 xyz + MSE sin/cos Euler + (1-cos) approach + smooth-L1 pre_dist/0.10 + smooth-L1 wp/0.25; AdamW 1e-4 wd 1e-6, warm-up then cosine, grad-clip 10, FINAL step, seed 1000).
RoboTwin adaptation (E): xyz normalised by THIS task's train mean/std (VLABench buffers do not fit the RoboTwin frame); steps/batch scaled to the demo count (3000 / 16, warm-up 300).
In V21 this head supplies rot/approach/pre_dist/wp_rel (+ grasp_xyz fallback). usage: train_v2head_10.py TASK -> ckpt/{task}/v2head.pt, training/v2head_{task}.json"""
import sys, os, json, hashlib, numpy as np, torch, torch.nn.functional as F
from regsp import paths as P

from regsp.backbone.stageplan_v2 import PRE_DIST_SCALE_M, WP_SCALE_M, StagePlanV2Head
from scipy.spatial.transform import Rotation as Rot
R_REF = Rot.from_euler('y', 90, degrees=True).as_matrix()  # ONE global embodiment constant (E): RoboTwin grasp labels have pitch==90deg exactly (gimbal lock); Euler is taken of R @ R_REF.T so the target is well-conditioned
OUT = str(P.clean_root()); t = sys.argv[1]; STEPS = int(os.environ.get("SPSTEPS", 6000)); BATCH = 16; WARM = 300; PEAK, FLOOR = 1e-4, 1e-5; dev = "cuda"
E = torch.load(f"{OUT}/tokens/{t}.pt", weights_only=False); [x.__setitem__("src", "clean") for x in E]; E2 = torch.load(f"{P.work_root()}/rand/tokens/{t}.pt", weights_only=False); [x.__setitem__("src", "rand") for x in E2]; E = E + E2; del E2  # [4-task study] union of original Clean demo tokens and official randomized_500 demo tokens (first NEP episodes)
for x in E: x['grasp_R'] = Rot.from_euler('xyz', x['grasp_euler']).as_matrix().astype(np.float32); x['grasp_euler_rel'] = Rot.from_matrix(x['grasp_R'] @ R_REF.T).as_euler('xyz').astype(np.float32)
tr = [x for x in E if x["split"] == "train"]; va = [x for x in E if x["split"] == "val"]
G = np.stack([x["grasp_xyz"] for x in tr]); mean = torch.tensor(G.mean(0), dtype=torch.float32, device=dev); std = torch.tensor(G.std(0).clip(1e-3), dtype=torch.float32, device=dev)
def batch(S):
    f = lambda k: torch.as_tensor(np.stack([x[k] for x in S]), dtype=torch.float32, device=dev)
    return torch.stack([x["tok"] for x in S]).float().to(dev), torch.stack([x["pad"] for x in S]).to(dev), torch.stack([x["mod"] for x in S]).to(dev), torch.stack([x["state"] for x in S]).float().to(dev), {"xyz": f("grasp_xyz"), "euler": f("grasp_euler_rel"), "R": f("grasp_R"), "dir": f("approach"), "pre": f("pre_dist"), "wp": f("wp_rel")}
def loss_fn(o, lab):
    l_xyz = F.smooth_l1_loss(o["grasp_xyz"], (lab["xyz"] - mean) / std); l_rot = F.mse_loss(o["grasp_rot_sc"], torch.stack([torch.sin(lab["euler"]), torch.cos(lab["euler"])], -1)); l_dir = (1 - (o["approach_dir"] * lab["dir"]).sum(-1)).mean()
    l_pre = F.smooth_l1_loss(o["pre_dist"], lab["pre"] / PRE_DIST_SCALE_M); l_wp = F.smooth_l1_loss(o["wp_rel"], lab["wp"] / WP_SCALE_M); return l_xyz + l_rot + l_dir + l_pre + l_wp, {"xyz": l_xyz.item(), "rot": l_rot.item(), "dir": l_dir.item(), "pre": l_pre.item(), "wp": l_wp.item()}
@torch.no_grad()
def evaluate(head, S):
    head.eval(); R = {"xyz_cm": [], "rot_deg": [], "dir_deg": [], "pre_cm": [], "wp_cm": []}
    for i in range(0, len(S), 64):
        tok, pad, mod, st, lab = batch(S[i:i + 64]); o = head(tok, pad, mod, st); g = o["grasp_xyz"] * std + mean; R["xyz_cm"] += (torch.norm(g - lab["xyz"], dim=-1) * 100).tolist()
        eul = torch.atan2(o["grasp_rot_sc"][..., 0], o["grasp_rot_sc"][..., 1]).cpu().numpy(); Rp = Rot.from_euler('xyz', eul).as_matrix() @ R_REF; Rl = lab["R"].cpu().numpy(); R["rot_deg"] += [float(np.degrees(np.arccos(np.clip((np.trace(Rp[i].T @ Rl[i]) - 1) / 2, -1, 1)))) for i in range(len(Rl))]
        R["dir_deg"] += torch.rad2deg(torch.acos((o["approach_dir"] * lab["dir"]).sum(-1).clamp(-1, 1))).tolist(); R["pre_cm"] += (torch.abs(o["pre_dist"] * PRE_DIST_SCALE_M - lab["pre"]) * 100).tolist(); R["wp_cm"] += (torch.norm(o["wp_rel"] * WP_SCALE_M - lab["wp"], dim=-1) * 100).tolist()
    head.train(); return {k: {"median": float(np.median(v)), "mean": float(np.mean(v))} for k, v in R.items()}
torch.manual_seed(1000); rng = np.random.default_rng(1000); head = StagePlanV2Head(960, state_dim=7, d_model=256, n_heads=4, n_layers=2, n_cams=3, dropout=0.1).to(dev); opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-6)
lr_at = lambda s: PEAK * (s + 1) / WARM if s < WARM else FLOOR + 0.5 * (PEAK - FLOOR) * (1 + np.cos(np.pi * min(1.0, (s - WARM) / (STEPS - WARM)))); log = []; head.train()
for step in range(1, STEPS + 1):
    for g_ in opt.param_groups: g_["lr"] = lr_at(step - 1)
    S = [tr[i] for i in rng.integers(0, len(tr), BATCH)]; tok, pad, mod, st, lab = batch(S); o = head(tok, pad, mod, st); loss, parts = loss_fn(o, lab); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0); opt.step()
    if step % 500 == 0: v = evaluate(head, va); log.append({"step": step, "loss": loss.item(), **parts, "val": v}); print(t, step, round(loss.item(), 4), {k: round(x["median"], 2) for k, x in v.items()}, flush=True)
os.makedirs((str(P.work_root()) + "/ckpt_sp/") + t, exist_ok=True); p = (str(P.work_root()) + "/ckpt_sp/") + t + "/v2head.pt"; torch.save({"head": head.state_dict(), "pos_mean": mean.cpu(), "pos_std": std.cpu(), "cfg": {"latent_dim": 960, "state_dim": 7, "d_model": 256, "n_heads": 4, "n_layers": 2, "n_cams": 3}, "task": t}, p)
json.dump({"task": t, "n_train": len(tr), "n_val": len(va), "steps": STEPS, "batch": BATCH, "log": log, "final_val": evaluate(head, va), "final_val_clean": evaluate(head, [x for x in va if x["src"] == "clean"]), "final_val_rand": evaluate(head, [x for x in va if x["src"] == "rand"]), "n_train_clean": sum(x["src"] == "clean" for x in tr), "n_train_rand": sum(x["src"] == "rand" for x in tr), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}, open((str(P.work_root()) + "/training/sp_v2head_") + t + ".json", "w"), indent=1)
