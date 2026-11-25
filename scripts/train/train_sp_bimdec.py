"""[4-task study, Phase C/5: bimanual decoder RETRAINED on union(Clean, randomized_500) tokens, initialised from the RETRAINED task decoder ckpt_sp/{task}/decoder_pretrained.pt] [train_bim_decoder_10] Bimanual-extended learned decoder (bim_common.BimDecoder) for ONE task; INIT from the task-adapted original decoder (ckpt/{task}/decoder_pretrained.pt, which stays immutable) with zero-initialised extension terms. Same objective, optimiser, steps, batch, seed as train_decoder_10.py.
Extension inputs per sample (ep,t,arm) derived from validated EXPERT labels: partner effector state at the same frame (xyz, euler_rel, gripper action), role = arm x min(global cycle index,1), place target = TCP-at-release xyz of the arm's current/next cycle (+ N(0,PLACE_NOISE) training noise matching grounded-placement error).
usage: train_bim_decoder_10.py TASK TAGV [STEPS]    -> ckpt_bim/{task}/bim_decoder_{TAGV}.pt , training/bim_decoder_{task}_{TAGV}.json"""
import sys, os, json, hashlib, pickle, numpy as np, torch, torch.nn as nn
from regsp import paths as P

from regsp.backbone.loader import load_policy
from regsp.backbone.rotation_encoding import encode, decode
from regsp.backbone.utils import masked_mse
from scipy.spatial.transform import Rotation as Rot
from regsp.decoding.bimanual import BimDecoder
OUT = str(P.clean_root()); t = sys.argv[1]; TAGV = sys.argv[2]; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 6000; BATCH = 32; SEED = 1000; PLACE_NOISE = float(os.environ.get("PLACE_NOISE", 0.01)); dev = "cuda"
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
E = torch.load(f"{OUT}/tokens_full/{t}.pt", weights_only=False); [x.__setitem__("src", "clean") for x in E]; E2 = torch.load(f"{P.work_root()}/rand/tokens_full/{t}.pt", weights_only=False); [x.__setitem__("src", "rand") for x in E2]; E = E + E2; del E2; tr = [e for e in E if e["split"] == "train"]; va = [e for e in E if e["split"] == "val"]
L = pickle.load(open(f"{OUT}/labels/{t}.pkl", "rb")); RREF = Rot.from_euler("y", 90, degrees=True).as_matrix(); LAB = {k[1]: v for k, v in L.items() if not v["rejected"]}; L2 = pickle.load(open((str(P.work_root()) + "/rand/labels/") + t + ".pkl", "rb")); LABS = {"clean": LAB, "rand": {k[1]: v for k, v in L2.items() if not v["rejected"]}}
base = torch.load(f"{P.work_root()}/ckpt_sp/{t}/decoder_pretrained.pt", weights_only=False); mean, std = base["mean"].to(dev), base["std"].to(dev)  # SAME normalisation as the baseline decoder
def eul_ref(Rm): return Rot.from_matrix(Rm @ RREF.T).as_euler("xyz")
def extras(e):  # partner state / role / place target from validated expert labels (training-time only; at inference the policy supplies the observed partner state and the grounded/const place target)
    r = LABS[e["src"]][e["ep"]]; a = e["arm"]; p = "R" if a == "L" else "L"; tt = e["t"]; T = r[f"tcp_{p}"][tt]; g = r["action"][tt, (7 if p == "R" else 0) + 6]; pst = np.r_[T[:3, 3], eul_ref(T[:3, :3]), g].astype(np.float32)
    ev = sorted([(c["tc"], s, c) for s in "LR" for c in r["cycles"][s]], key=lambda x: x[0]); mine = [(k, c) for k, (tc, s, c) in enumerate(ev) if s == a]
    cur = next(((k, c) for k, c in mine if c["tp"] > tt), mine[-1] if mine else (0, None)); k, c = cur
    return pst, min(k, 1) * 2 + (a == "R"), (c["place_xyz"] if c is not None else np.zeros(3, np.float32)).astype(np.float32)
for e in E: e["pst"], e["role"], e["place"] = extras(e)
pol, pre = load_policy(); D = pol.decoder; Q = pol.target_query_extractor; V = pol.visual_grounding_cross_attn; TPH = pol.target_point_head; BD = BimDecoder(D)
mods = nn.ModuleDict({"bdec": BD, "tqe": Q, "vgc": V, "tph": TPH}).to(dev)
for p_ in mods.parameters(): p_.requires_grad_(True)  # FIX (Exp2): load_policy() returns frozen params; without this ONLY the new zero-init partner/role/place layers trained and base decoder/Q/V/TPH stayed bit-identical to the init
sd = base["mods"]; BD.base.load_state_dict(sd["dec"]); Q.load_state_dict(sd["tqe"]); V.load_state_dict(sd["vgc"]); TPH.load_state_dict(sd["tph"])
def batch(es, noise):
    tok = torch.stack([e["tok"] for e in es]).float().to(dev); pad = torch.stack([e["pad"] for e in es]).to(dev); mod = torch.stack([e["mod"] for e in es]).to(dev)
    s = (torch.tensor(np.stack([e["state_ref"] for e in es]), device=dev) - mean) / std; ch = (torch.tensor(np.stack([e["chunk"] for e in es]), device=dev) - mean) / std; vm = torch.tensor(np.stack([e["valid"] for e in es]), device=dev)
    ph = torch.tensor([e["phase"] for e in es], device=dev); gt = torch.tensor(np.stack([e["grasp_xyz"] if e["grasp_xyz"] is not None else np.zeros(3, np.float32) for e in es]), device=dev)[:, :3]; gt = (gt - mean[:3]) / std[:3]
    tv = torch.tensor([(e["next_close_in"] is not None and e["next_close_in"] <= 50 and e["phase"] == 0) for e in es], device=dev)
    ps = (torch.tensor(np.stack([e["pst"] for e in es]), device=dev) - mean) / std; role = torch.tensor([e["role"] for e in es], device=dev)
    pl = torch.tensor(np.stack([e["place"] for e in es]), device=dev)
    if noise: pl = pl + torch.randn_like(pl) * PLACE_NOISE
    pl = (pl - mean[:3]) / std[:3]; return tok, pad, mod, s, ch, vm, ph, gt, tv, ps, role, pl
def forward(tok, pad, mod, s, ph, ps, role, pl):
    q = Q(tok, pad, mod); gf = V(q, tok, pad, mod); txyz = TPH(gf); return BD(tok, pad, encode(s, mean[3:6], std[3:6]), txyz, mod, gf, ph, encode(ps, mean[3:6], std[3:6]), role, pl), txyz
def loss_fn(es):
    tok, pad, mod, s, ch, vm, ph, gt, tv, ps, role, pl = batch(es, True); pred, txyz = forward(tok, pad, mod, s, ph, ps, role, pl); clean = encode(ch, mean[3:6], std[3:6])
    lp, lr_, lg = masked_mse(pred[..., :3], clean[..., :3], vm), masked_mse(pred[..., 3:9], clean[..., 3:9], vm), masked_mse(pred[..., 9:10], clean[..., 9:10], vm)
    return (3 * lp + 6 * lr_ + 1 * lg) / 10 + masked_mse(txyz[:, None], gt[:, None], tv[:, None]), (float(lp), float(lr_), float(lg))
@torch.no_grad()
def evaluate(es, phase=None, zero_ext=False):
    mods.eval(); P, G, n = [], [], 0; sel = [e for e in es if phase is None or e["phase"] == phase]
    for i in range(0, len(sel), 64):
        b = sel[i:i + 64]; tok, pad, mod, s, ch, vm, ph, gt, tv, ps, role, pl = batch(b, False)
        if zero_ext: ps = ps * 0 + 0; pl = pl * 0
        pred, _ = forward(tok, pad, mod, s, ph, ps, role, pl); dec = decode(pred, mean[3:6], std[3:6]); m = vm.float()
        d_pos = ((dec[..., :3] - ch[..., :3]) * std[:3]).norm(dim=-1); d_g = (dec[..., 6] - ch[..., 6]).abs() * std[6]
        for k, v_ in ((P, d_pos), (G, d_g)): k.append(((v_ * m).sum(1) / m.sum(1).clamp_min(1)).cpu())
    return {"pos_cm": float(torch.cat(P).mean()) * 100, "grip_mae": float(torch.cat(G).mean()), "n": len(sel)}
log = {"task": t, "tag": TAGV, "steps": STEPS, "place_noise_m": PLACE_NOISE, "n_train": len(tr), "n_val": len(va), "step0_val(=adapted original decoder)": {"all": evaluate(va), "post": evaluate(va, 1), "pre": evaluate(va, 0)}}; print("step0", json.dumps(log["step0_val(=adapted original decoder)"]), flush=True)
opt = torch.optim.AdamW(mods.parameters(), lr=1e-4, weight_decay=1e-6); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
for step in range(1, STEPS + 1):
    mods.train(); es = [tr[i] for i in rng.choice(len(tr), BATCH, replace=False)]; l, parts = loss_fn(es); opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(mods.parameters(), 1.0); opt.step(); sch.step()
    if step % 1000 == 0 or step == STEPS: v = evaluate(va); log.setdefault("curve", []).append({"step": step, "loss": float(l), "val": v}); print(step, float(l), json.dumps(v), flush=True)
log["final_val"] = {"all": evaluate(va), "post": evaluate(va, 1), "pre": evaluate(va, 0)}; log["final_val_clean"] = {"post": evaluate([e for e in va if e["src"] == "clean"], 1)}; log["final_val_rand"] = {"post": evaluate([e for e in va if e["src"] == "rand"], 1)}; log["final_val_extension_inputs_zeroed(ablation)"] = {"post": evaluate(va, 1, True)}
os.makedirs((str(P.work_root()) + "/ckpt_sp/") + t, exist_ok=True); fn = (str(P.work_root()) + "/ckpt_sp/") + t + f"/bim_decoder_{TAGV}.pt"; torch.save({"mods": {k: v.state_dict() for k, v in mods.items()}, "mean": mean.cpu(), "std": std.cpu(), "init": "ckpt_sp/%s/decoder_pretrained.pt" % t, "base_sha256": hashlib.sha256(open(f"{P.work_root()}/ckpt_sp/{t}/decoder_pretrained.pt", "rb").read()).hexdigest(), "R_REF": "Ry(90deg)"}, fn); log["sha256"] = hashlib.sha256(open(fn, "rb").read()).hexdigest()
json.dump(log, open((str(P.work_root()) + "/training/sp_bim_decoder_") + t + f"_{TAGV}.json", "w"), indent=1); print("saved", fn, log["sha256"])
