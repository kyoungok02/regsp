"""[V21-PIC] versioned copy of train_acl.py: ProgressHead -> pic_common.PicProgressHead (explicit instruction pathway, zero-init residual; loaded non-strictly from the frozen acl_a2 head) trained with INSTRUCTION-COUNTERFACTUAL on-policy
samples (pic_onpolicy.py: same observation under both valid instructions, outcome judged against each instruction). Used with ACL_PROG_ONLY=1 ACL_RESUME=ckpt_acl/<task>/acl_a2.pt: decoder / tqe / vgc / tph stay bit-identical.
Output: ckpt_pic/<task>/acl_<TAG>.pt (flag pic_prog), training/pic_acl_<task>_<TAG>.json.   ORIGINAL DOCSTRING: [V21-ACL] Train the Adaptive Single/Bimanual TemporalActionDecoder (acl_common.AdaptiveDecoder) + ProgressHead for ONE task on SYNCHRONISED two-arm demonstrations (separate, explicitly named extension experiment; nothing frozen is modified).
Data  : union(Clean 10-task demos, randomized_500 demos) token caches (frozen backbone tokens every 6th frame, all phases, whole episodes = all cycles and all stage transitions) + validated expert labels.
        Each sample (episode, frame t, primary arm) supervises BOTH effectors: the 50-step TCP+gripper chunk of the primary arm AND of the partner arm at the same frames (synchronised 14-D target), both gripper channels explicitly.
Inputs derived from labels (training only; at inference the executor supplies the observed / grounded / memory values):
  both-arm states, grasp target of the primary arm's current cycle (+1 cm noise; the executor supplies the V21 grounded grasp_xyz. NOTE: the earlier decoders were TRAINED with the TargetPointHead prediction as `target_xyz` but EXECUTED with the grounded
  grasp_xyz - a conditioning mismatch that this trainer removes), place target (+1 cm noise), role, arm mode (which effectors act within the chunk; teacher-forced), stage (= completed releases x primary-holds x partner-holds), 2-step history (dropout 0.3).
Progress labels (per queried stage m, from demonstration event boundaries): t < close_m -> NOT_STARTED ; close_m <= t < release_m + 5 -> IN_PROGRESS ; later -> COMPLETED (TASK_COMPLETED for the last stage). FAILED is never fabricated from demos.
  Optional on-policy samples (ACL_ONPOLICY=dir of training-domain rollouts with training-only privileged outcome labels, see acl_onpolicy.py) add real FAILED / COMPLETED examples for the progress head only.
Init  : ckpt_sp/<task>/bim_decoder_b2.pt (corrected bimanual decoder) ; new terms zero-init. ALL parameters are explicitly unfrozen (load_policy() returns frozen parameters: the parameter-freezing bug of the first bimanual trainer);
        the script logs trainable-parameter counts, per-group gradient norms and per-group relative weight change and aborts if any trainable tensor is bit-identical to its initial value after training.
Objective: (3 pos + 6 rot + 1 grip)/10 for the primary chunk + the same for the partner chunk + original TargetPointHead loss + 0.2 (progress CE + mode CE). AdamW 1e-4 / wd 1e-6, cosine, clip 1.0, batch 32, seed 1000.
usage: train_acl.py TASK TAG [STEPS]   -> ckpt_acl/<task>/acl_<TAG>.pt , ckpt_acl/<task>/acl_stats.json , training/acl_<task>_<TAG>.json"""
import sys, os, json, hashlib, pickle, glob, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from regsp import paths as P
W = str(P.clean_root()); N = str(P.work_root())

from regsp.backbone.loader import load_policy
from regsp.backbone.rotation_encoding import encode, decode
from regsp.backbone.utils import masked_mse
from scipy.spatial.transform import Rotation as Rot
from regsp.decoding.bimanual import BimDecoder
from regsp.decoding.adaptive import AdaptiveDecoder, HIST
from regsp.decoding.progress import ProgressHead
from regsp.decoding.progress import PicProgressHead, lang_mask
t = sys.argv[1]; TAG = sys.argv[2]; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 8000; BATCH = 32; SEED = 1000; TNOISE = 0.01; dev = "cuda"; HZ = 50; INIT = os.environ.get("ACL_INIT", f"{N}/ckpt_sp/{t}/bim_decoder_b2.pt")
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
E = torch.load(f"{W}/tokens_full/{t}.pt", weights_only=False); [x.__setitem__("src", "clean") for x in E]; E2 = torch.load(f"{N}/rand/tokens_full/{t}.pt", weights_only=False); [x.__setitem__("src", "rand") for x in E2]; E = E + E2; del E2
LABS = {s: {k[1]: v for k, v in pickle.load(open(fn, "rb")).items() if not v["rejected"]} for s, fn in (("clean", f"{W}/labels/{t}.pkl"), ("rand", f"{N}/rand/labels/{t}.pkl"))}
RREF = Rot.from_euler("y", 90, degrees=True).as_matrix(); OFF = {"L": 0, "R": 7}; other = lambda a: "R" if a == "L" else "L"
def eul_ref(Rm): return Rot.from_matrix(Rm @ RREF.T).as_euler("xyz")
def state(r, a, tt): T = r[f"tcp_{a}"][tt]; return np.r_[T[:3, 3], eul_ref(T[:3, :3]), r["action"][tt, OFF[a] + 6]].astype(np.float32)
def chunk(r, a, tt):
    Tn = r[f"tcp_{a}"]; g = r["action"][:, OFF[a] + 6]; ix = np.minimum(np.arange(tt + 1, tt + 1 + HZ), len(Tn) - 1); return np.concatenate([Tn[ix][:, :3, 3], np.stack([eul_ref(m) for m in Tn[ix][:, :3, :3]]), g[ix][:, None]], 1).astype(np.float32)
def events(r): return sorted([(c["tc"], c["tp"] if c.get("tp") is not None else 10 ** 9, s, c) for s in "LR" for c in r["cycles"][s]], key=lambda x: x[0])
_cache = {}
def frame(src, ep, tt):  # per (episode, frame): both-arm states/chunks, motion flags, cycle bookkeeping (shared by the two primary-arm samples of that frame)
    k = (src, ep, tt)
    if k not in _cache:
        r = LABS[src][ep]; ev = events(r); d = {"ev": ev, "ndone": min(sum(tp <= tt for _, tp, _, _ in ev), 2), "prog": [0 if tt < tc else (1 if tt < tp + 5 else (4 if m == len(ev) - 1 else 2)) for m, (tc, tp, _, _) in enumerate(ev)]}
        for a in "LR":
            s = state(r, a, tt); ch = chunk(r, a, tt); d["s" + a] = s; d["c" + a] = ch; d["mv" + a] = bool(np.abs(ch[:, :3] - s[:3]).max() > 0.01 or np.abs(ch[:, 6] - s[6]).max() > 0.3); d["hold" + a] = any(tc <= tt < tp for tc, tp, s_, _ in ev if s_ == a)
            d["h" + a] = np.concatenate([np.r_[(r[f"tcp_{a}"][max(tt - lag, 0)][:3, 3] - s[:3]) / 0.05, r["action"][max(tt - lag, 0), OFF[a] + 6]] for lag in (10, 20)]).astype(np.float32)
        nx = [s_ for tc, tp, s_, _ in ev for x in (tc, tp) if x > tt]; d["next_arm"] = None
        nxt = sorted([(x, s_) for tc, tp, s_, _ in ev for x in (tc, tp) if x > tt]); d["next_arm"] = nxt[0][1] if nxt else None; _cache[k] = d
    return _cache[k]
def meta(e):
    d = frame(e["src"], e["ep"], e["t"]); a = e["arm"]; p = other(a); tt = e["t"]; mine = [(k, c, tp) for k, (tc, tp, s_, c) in enumerate(d["ev"]) if s_ == a]; cur = next(((k, c) for k, c, tp in mine if tp > tt), (mine[-1][:2] if mine else (0, None))); k, c = cur
    mp, mq = d["mv" + a], d["mv" + p]; mode = 2 if (mp and mq) else (0 if mp else (1 if mq else (0 if d["next_arm"] in (a, None) else 1)))
    z = np.zeros(3, np.float32); e.update({"sQ": d["s" + p], "cQ": d["c" + p], "role": min(k, 1) * 2 + (a == "R"), "tgt": (c["grasp_xyz"][:3].astype(np.float32) if c is not None else z), "place": (c["place_xyz"].astype(np.float32) if c is not None and c.get("place_xyz") is not None else z),
              "mode": mode, "stage": d["ndone"] * 4 + d["hold" + a] * 2 + d["hold" + p], "hist": np.concatenate([d["h" + a], d["h" + p]]), "prog": d["prog"], "holdP": d["hold" + a], "holdQ": d["hold" + p]})
for e in E: meta(e)
e0 = E[len(E) // 2]; assert np.allclose(chunk(LABS[e0["src"]][e0["ep"]], e0["arm"], e0["t"]), e0["chunk"], atol=1e-5) and np.allclose(state(LABS[e0["src"]][e0["ep"]], e0["arm"], e0["t"]), e0["state_ref"], atol=1e-5), "label/token chunk mismatch"
tr = [e for e in E if e["split"] == "train"]; va = [e for e in E if e["split"] == "val"]; cnt = lambda es, k: {int(v): int(c) for v, c in zip(*np.unique([e[k] for e in es], return_counts=True))}
print(t, "samples train/val", len(tr), len(va), "mode", cnt(tr, "mode"), "stage", cnt(tr, "stage"), "prog(stage0)", {int(v): int(c) for v, c in zip(*np.unique([e["prog"][0] for e in tr], return_counts=True))}, "prog(last)", {int(v): int(c) for v, c in zip(*np.unique([e["prog"][-1] for e in tr], return_counts=True))}, flush=True)
# ---- data-derived executor statistics (TRAIN split only): receiving offset in the donor TCP frame, memory-relative place offsets
def exec_stats():
    recv, rel, relabs = {}, {}, {}
    for src in LABS:
        for ep, r in LABS[src].items():
            if r["split"] != "train": continue
            ev = events(r)
            for k, (tc, tp, s_, c) in enumerate(ev):
                don = [(tc2, tp2, s2) for tc2, tp2, s2, _ in ev if s2 != s_ and tc2 <= tc < tp2]
                if don: Td = r[f"tcp_{don[0][2]}"][tc]; recv.setdefault(str(k), []).append(Td[:3, :3].T @ (r[f"tcp_{s_}"][tc][:3, 3] - Td[:3, 3]))
                if k >= 1 and c.get("place_xyz") is not None and ev[k - 1][3].get("place_xyz") is not None: rel.setdefault(str(k), []).append(c["place_xyz"] - ev[k - 1][3]["place_xyz"]); relabs.setdefault(str(k), []).append(c["place_xyz"])
    st = {"receive_offset_donor_frame": {k: {"median": np.median(v, 0).tolist(), "std": np.std(v, 0).tolist(), "n": len(v)} for k, v in recv.items()},
          "place_relative_to_previous_release": {k: {"median": np.median(v, 0).tolist(), "std": np.std(v, 0).tolist(), "abs_std": np.std(relabs[k], 0).tolist(), "n": len(v), "use": bool(np.max(np.std(v, 0)) <= 0.015)} for k, v in rel.items()}, "source": "expert labels, train split of Clean + randomized_500 demos"}
    return st
os.makedirs(f"{N}/training", exist_ok=True); ST = exec_stats(); print("exec stats", json.dumps(ST)[:600], flush=True)
# ---- optional on-policy progress samples (training-domain rollouts; privileged outcome labels are training-only)
OP = []
if os.environ.get("ACL_ONPOLICY"):
    for dd in os.environ["ACL_ONPOLICY"].split(","):
        for f in sorted(glob.glob(dd + "/*/feat*.pt") + glob.glob(dd + "/feat*.pt")): OP += torch.load(f, weights_only=False)
    for o in OP: o.setdefault("variant", "exec"); o.setdefault("lmode", "orig")
    OP = [o for o in OP if o.get("label") is not None]; rs = np.random.default_rng(7); eps = sorted({(o["cfg"], o["seed"]) for o in OP}); vs = {eps[i] for i in rs.permutation(len(eps))[:max(1, len(eps) // 5)]}
    for o in OP: o["split"] = "val" if (o["cfg"], o["seed"]) in vs else "train"
    print("on-policy progress samples", len(OP), "labels", {int(v): int(c) for v, c in zip(*np.unique([o["label"] for o in OP], return_counts=True))}, "val episodes", len(vs), flush=True)
OPtr = [o for o in OP if o["split"] == "train"]; OPva = [o for o in OP if o["split"] == "val"]
OPW = None
gk = lambda o: (o["label"], o["variant"], o["lmode"])
if OPtr: _c = collections.Counter(gk(o) for o in OPtr); _nl = collections.Counter(k_[0] for k_ in _c); OPW = np.array([1.0 / (_c[gk(o)] * _nl[o["label"]]) for o in OPtr]); OPW = OPW / OPW.sum()  # balanced over the labels, and inside a label over (instruction variant, rollout instruction)
ini = torch.load(INIT, weights_only=False); mean, std = ini["mean"].to(dev), ini["std"].to(dev)
pol, pre = load_policy(); Q = pol.target_query_extractor; V = pol.visual_grounding_cross_attn; TPH = pol.target_point_head; BD = BimDecoder(pol.decoder); BD.load_state_dict(ini["mods"]["bdec"]); Q.load_state_dict(ini["mods"]["tqe"]); V.load_state_dict(ini["mods"]["vgc"]); TPH.load_state_dict(ini["mods"]["tph"])
AD = AdaptiveDecoder(BD); PH = PicProgressHead(); mods = nn.ModuleDict({"adec": AD, "tqe": Q, "vgc": V, "tph": TPH, "prog": PH}).to(dev)
if os.environ.get("ACL_RESUME"):
    rs_ = torch.load(os.environ["ACL_RESUME"], weights_only=False)
    for k_, m_ in mods.items(): mk = m_.load_state_dict(rs_["mods"][k_], strict=(k_ != "prog")); print("resume", k_, "missing", list(mk.missing_keys)[:12], flush=True)
PROG_ONLY = bool(os.environ.get("ACL_PROG_ONLY"))  # second-stage training of the progress head only (decoder weights untouched and verified identical afterwards)
for p_ in mods.parameters(): p_.requires_grad_(True)  # load_policy() returns frozen parameters -> explicit unfreeze of EVERYTHING that is optimised
if PROG_ONLY:
    for k_ in ("adec", "tqe", "vgc", "tph"):
        for p_ in mods[k_].parameters(): p_.requires_grad_(False)
if os.environ.get("PIC_INSTR_ONLY"):  # only the new instruction pathway is optimised (the loaded progress head stays bit-identical; less capacity to memorise episodes)
    for n_, p_ in mods["prog"].named_parameters(): p_.requires_grad_(n_.split(".")[0] in ("itxt", "iq", "iattn", "iprog", "imode"))
P0 = {n_: p_.detach().clone() for n_, p_ in mods.named_parameters()}; ntr = sum(p_.numel() for p_ in mods.parameters() if p_.requires_grad); nfr = sum(p_.numel() for p_ in mods.parameters() if not p_.requires_grad)
def group(n_):
    for g_, pat in (("base.transformer", "adec.bim.base.transformer"), ("base.action_head(P)", "adec.bim.base.action_head"), ("head_partner(Q)", "adec.head_partner"), ("base.queries+encoders", "adec.bim.base."), ("partner/role/place", "adec.bim."), ("mode/stage/hist/slot", "adec."), ("tqe", "tqe."), ("vgc", "vgc."), ("tph", "tph."), ("progress_head", "prog.")):
        if n_.startswith(pat): return g_
    return "other"
print("trainable params", ntr, "frozen params", nfr, "tensors", len(P0), flush=True)
def T_(x, dt=torch.float32): return torch.tensor(np.stack(x), dtype=dt, device=dev)
def batch(es, train):
    tok = torch.stack([e["tok"] for e in es]).float().to(dev); pad = torch.stack([e["pad"] for e in es]).to(dev); mod = torch.stack([e["mod"] for e in es]).to(dev); n = lambda x: (x - mean) / std; n3 = lambda x: (x - mean[:3]) / std[:3]
    sP = n(T_([e["state_ref"] for e in es])); sQ = n(T_([e["sQ"] for e in es])); cP = n(T_([e["chunk"] for e in es])); cQ = n(T_([e["cQ"] for e in es])); vm = T_([e["valid"] for e in es], torch.bool); ph = T_([e["phase"] for e in es], torch.long)
    gt = n3(T_([e["grasp_xyz"][:3] if e["grasp_xyz"] is not None else np.zeros(3, np.float32) for e in es])); tv = T_([(e["next_close_in"] is not None and e["next_close_in"] <= 50 and e["phase"] == 0) for e in es], torch.bool)
    tg = T_([e["tgt"] for e in es]); pl = T_([e["place"] for e in es]); hi = T_([e["hist"] for e in es])
    if train: tg = tg + torch.randn_like(tg) * TNOISE; pl = pl + torch.randn_like(pl) * TNOISE; hi = (hi + torch.randn_like(hi) * 0.05) * (torch.rand(len(es), 1, device=dev) > 0.3)
    m = [int(rng.integers(len(e["prog"]))) if train else None for e in es]
    return {"tok": tok, "pad": pad, "mod": mod, "sP": sP, "sQ": sQ, "cP": cP, "cQ": cQ, "vm": vm, "ph": ph, "gt": gt, "tv": tv, "tg": n3(tg), "pl": n3(pl), "hi": hi, "role": T_([e["role"] for e in es], torch.long), "mode": T_([e["mode"] for e in es], torch.long), "stage": T_([e["stage"] for e in es], torch.long), "m": m}
enc = lambda s: encode(s, mean[3:6], std[3:6])
def forward(b, mode=None, tgt="label"):
    q = Q(b["tok"], b["pad"], b["mod"]); gf = V(q, b["tok"], b["pad"], b["mod"]); txyz = TPH(gf); sP, sQ = enc(b["sP"]), enc(b["sQ"])
    pP, pQ = AD(b["tok"], b["pad"], sP, sQ, b["tg"] if tgt == "label" else txyz, b["mod"], gf, b["ph"], b["role"], b["pl"], b["mode"] if mode is None else mode, b["stage"], b["hi"]); return pP, pQ, txyz, sP, sQ
def prog_fwd(b, sP, sQ, m): return PH(b["tok"], b["pad"], sP, sQ, torch.cat([b["tg"], b["pl"]], -1), b["hi"], torch.tensor(m, device=dev).clamp(max=2), lang=b["tok"], lmask=lang_mask(b["pad"], b["mod"]))
def chunk_loss(pred, ch, vm):
    c = enc(ch); return (3 * masked_mse(pred[..., :3], c[..., :3], vm) + 6 * masked_mse(pred[..., 3:9], c[..., 3:9], vm) + masked_mse(pred[..., 9:10], c[..., 9:10], vm)) / 10
def op_batch(os_):
    L_ = max(o["tok"].shape[0] for o in os_); tk = torch.zeros(len(os_), L_, os_[0]["tok"].shape[1]); pd = torch.zeros(len(os_), L_, dtype=torch.bool)  # variable instruction length -> right-pad (pad mask False = ignored by the attention)
    for i_, o in enumerate(os_): n_ = o["tok"].shape[0]; tk[i_, :n_] = o["tok"].float(); pd[i_, :n_] = o["pad"].bool()
    return {"tok": tk.to(dev), "pad": pd.to(dev), "sP": T_([o["sP"] for o in os_]), "sQ": T_([o["sQ"] for o in os_]), "mem": T_([o["mem"] for o in os_]), "hi": T_([o["hist"] for o in os_]), "m": [o["m"] for o in os_], "y": T_([o["label"] for o in os_], torch.long)}
def op_logits(b, lang=None, lmask=None, blank_instr=False):
    if blank_instr: lang = torch.zeros(len(b["m"]), 1, b["tok"].shape[-1], device=dev); lmask = torch.ones(len(b["m"]), 1, dtype=torch.bool, device=dev)
    return PH(b["tok"], b["pad"], b["sP"], b["sQ"], b["mem"], b["hi"], torch.tensor(b["m"], device=dev).clamp(max=2), lang=lang, lmask=lmask)[0]
def loss_fn(es):
    b = batch(es, True); pP, pQ, txyz, sP, sQ = forward(b); lP = chunk_loss(pP, b["cP"], b["vm"]); lQ = chunk_loss(pQ, b["cQ"], b["vm"]); la = masked_mse(txyz[:, None], b["gt"][:, None], b["tv"][:, None])
    lg, lm = prog_fwd(b, sP.detach(), sQ.detach(), b["m"]); y = torch.tensor([e["prog"][m] for e, m in zip(es, b["m"])], device=dev); lpr = F.cross_entropy(lg, y); lmo = F.cross_entropy(lm, b["mode"]); lop = torch.zeros((), device=dev)
    if OPtr: ob = op_batch([OPtr[i] for i in rng.choice(len(OPtr), min(24, len(OPtr)), replace=False, p=OPW)]); lop = F.cross_entropy(op_logits(ob), ob["y"])  # class-balanced draw of REAL on-policy outcomes
    tot = (0 if PROG_ONLY else lP + lQ + la) + 0.2 * (lpr + lmo) + float(os.environ.get("ACL_OP_W", 0.2)) * lop; return tot, {"P": float(lP), "Q": float(lQ), "tph": float(la), "prog": float(lpr), "mode": float(lmo), "onpolicy_prog": float(lop)}
@torch.no_grad()
def evaluate(es, phase=None, mode_in="gt", tgt="label"):
    mods.eval(); sel = [e for e in es if phase is None or e["phase"] == phase]; acc = {k: [] for k in ("posP", "posQ", "gP", "gQ", "mode", "prog", "relP", "relQ")}
    for i in range(0, len(sel), 64):
        es_ = sel[i:i + 64]; b = batch(es_, False); q = Q(b["tok"], b["pad"], b["mod"]); sP, sQ = enc(b["sP"]), enc(b["sQ"]); lg0, lm = prog_fwd(b, sP, sQ, [0] * len(es_)); mp = lm.argmax(-1)
        pP, pQ, _, _, _ = forward(b, mode=(mp if mode_in == "pred" else None), tgt=tgt); m = b["vm"].float(); acc["mode"].append((mp == b["mode"]).float().cpu())
        for nm, pr, ch in (("P", pP, b["cP"]), ("Q", pQ, b["cQ"])):
            d = decode(pr, mean[3:6], std[3:6]); acc["pos" + nm].append(((((d[..., :3] - ch[..., :3]) * std[:3]).norm(dim=-1) * m).sum(1) / m.sum(1).clamp_min(1)).cpu()); gp = d[..., 6] * std[6] + mean[6]; gg = ch[..., 6] * std[6] + mean[6]
            acc["g" + nm].append((((gp - gg).abs() * m).sum(1) / m.sum(1).clamp_min(1)).cpu()); acc["rel" + nm].append((((gp > 0.5) == (gg > 0.5)).float() * m).sum(1).div(m.sum(1).clamp_min(1)).cpu())
        for mq in range(max(len(e["prog"]) for e in es_)):
            ix = [j for j, e in enumerate(es_) if len(e["prog"]) > mq]
            if ix: lg, _ = prog_fwd(b, sP, sQ, [mq] * len(es_)); acc["prog"].append((lg.argmax(-1)[ix].cpu() == torch.tensor([es_[j]["prog"][mq] for j in ix])).float())
    c = lambda k: float(torch.cat(acc[k]).mean()) if acc[k] else None
    return {"n": len(sel), "primary_pos_cm": c("posP") * 100, "partner_pos_cm": c("posQ") * 100, "primary_grip_mae": c("gP"), "partner_grip_mae": c("gQ"), "primary_grip_state_acc": c("relP"), "partner_grip_state_acc": c("relQ"), "mode_acc": c("mode"), "progress_acc": c("prog")}
@torch.no_grad()
def eval_op(os_):
    if not os_: return None
    mods.eval(); lg = torch.cat([op_logits(op_batch(os_[i:i + 64])).cpu() for i in range(0, len(os_), 64)]); pr = lg.argmax(-1); y = torch.tensor([o["label"] for o in os_]); pf = lg.softmax(-1)[:, 3]
    R_ = {"n": len(os_), "acc": float((pr == y).float().mean()), "confusion(label->pred)": {int(a): {int(b_): int(((y == a) & (pr == b_)).sum()) for b_ in pr.unique()} for a in y.unique()}, "by_variant": {}}
    for v_ in sorted({o["variant"] for o in os_}):
        ix = torch.tensor([o["variant"] == v_ for o in os_]); R_["by_variant"][v_] = {"n": int(ix.sum()), "acc": float((pr[ix] == y[ix]).float().mean()), "labels": {int(a): int((y[ix] == a).sum()) for a in y[ix].unique()}}
    wr = torch.tensor([o.get("why") == "wrong_referent" for o in os_]); ok_ = torch.tensor([o["label"] in (1, 2, 4) for o in os_])
    R_["decision@tau0.8"] = {"wrong_referent_n": int(wr.sum()), "wrong_referent_rejected": int((pf[wr] > 0.8).sum()), "valid_progress_n": int(ok_.sum()), "valid_progress_falsely_rejected": int((pf[ok_] > 0.8).sum())}
    pairs = collections.defaultdict(dict)
    for j_, o in enumerate(os_):
        if "pair" in o: pairs[o["pair"]][o["variant"]] = bool(pr[j_] == y[j_])
    pp = [v_ for v_ in pairs.values() if len(v_) == 2]; R_["same_observation_both_instructions_correct"] = {"n_pairs": len(pp), "rate": float(np.mean([all(v_.values()) for v_ in pp])) if pp else None}
    lgb = torch.cat([op_logits(op_batch(os_[i:i + 64]), blank_instr=True).cpu() for i in range(0, len(os_), 64)]); R_["acc_blank_instruction_pathway"] = float((lgb.argmax(-1) == y).float().mean())
    ep0 = {}
    for o in os_:
        if o["variant"] == "exec": ep0.setdefault((o["lmode"], o["seed"]), o)
    ex = [o for o in os_ if o["variant"] == "exec"]; hit = []
    for o in ex:  # PERSISTENT instruction state: language rows of the FIRST sample of the episode instead of the live rows
        f0 = ep0[(o["lmode"], o["seed"])]; lm = lang_mask(f0["pad"][None].to(dev)); hit.append(int(op_logits(op_batch([o]), lang=f0["tok"].float()[None].to(dev), lmask=lm).argmax(-1)) == o["label"])
    R_["acc_exec_with_cached_episode_start_instruction_state"] = float(np.mean(hit)) if hit else None; R_["acc_exec_live"] = float(np.mean([bool(pr[j_] == y[j_]) for j_, o in enumerate(os_) if o["variant"] == "exec"])) if ex else None
    return R_
def subsets(es): return {"all": evaluate(es), "post": evaluate(es, 1), "pre": evaluate(es, 0)}
log = {"task": t, "tag": TAG, "steps": STEPS, "init": INIT, "n_train": len(tr), "n_val": len(va), "trainable_params": ntr, "frozen_params": nfr, "prog_only": PROG_ONLY, "mode_counts_train": cnt(tr, "mode"), "stage_counts_train": cnt(tr, "stage"), "exec_stats": ST, "n_onpolicy_train": len(OPtr), "n_onpolicy_val": len(OPva)}
if not os.environ.get("ACL_RESUME"):
    log["step0_val(label target = runtime conditioning)"] = subsets(va); log["step0_val(TargetPointHead target = conditioning the b2 decoder was trained with)"] = {"post": evaluate(va, 1, tgt="tph"), "pre": evaluate(va, 0, tgt="tph")}
    print("step0 label-target", json.dumps(log["step0_val(label target = runtime conditioning)"]["post"]), "\nstep0 TPH-target ", json.dumps(log["step0_val(TargetPointHead target = conditioning the b2 decoder was trained with)"]["post"]), flush=True)
opt = torch.optim.AdamW([p_ for p_ in mods.parameters() if p_.requires_grad], lr=float(os.environ.get("ACL_LR", 1e-4)), weight_decay=float(os.environ.get("ACL_WD", 1e-6))); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS); log["grad_norms"] = []
for step in range(1, STEPS + 1):
    mods.train(); es = [tr[i] for i in rng.choice(len(tr), BATCH, replace=False)]; l, parts = loss_fn(es); opt.zero_grad(); l.backward()
    if step in (1, 10, 100) or step % 1000 == 0:
        gn = {}
        for n_, p_ in mods.named_parameters():
            if p_.grad is not None: gn[group(n_)] = gn.get(group(n_), 0.0) + float(p_.grad.pow(2).sum())
        gn = {k: round(v ** 0.5, 6) for k, v in gn.items()}; log["grad_norms"].append({"step": step, **gn}); print("grad", step, json.dumps(gn), flush=True)
    torch.nn.utils.clip_grad_norm_([p_ for p_ in mods.parameters() if p_.requires_grad], 1.0); opt.step(); sch.step()
    if step % 1000 == 0 or step == STEPS: v = evaluate(va); log.setdefault("curve", []).append({"step": step, "loss": float(l), "parts": parts, "val": v, "onpolicy_val": eval_op(OPva)}); print(step, round(float(l), 5), json.dumps(parts), json.dumps(v), flush=True)
# ---- parameter-update verification (prevents the parameter-freezing bug)
upd = {}; same = []
for n_, p_ in mods.named_parameters():
    g_ = group(n_); d_ = float((p_.detach() - P0[n_]).norm()); u = upd.setdefault(g_, [0.0, 0.0, 0, 0]); u[0] += d_ ** 2; u[1] += float(P0[n_].norm()) ** 2; u[2] += 1; ch_ = not torch.equal(p_.detach(), P0[n_]); u[3] += int(ch_)
    if p_.requires_grad and not ch_: same.append(n_)
log["update_check"] = {g_: {"rel_change": (u[0] ** 0.5) / max(u[1] ** 0.5, 1e-12), "tensors": u[2], "tensors_changed": u[3]} for g_, u in upd.items()}; log["trainable_tensors_unchanged"] = same; print("update check", json.dumps(log["update_check"]), "unchanged trainable tensors:", same, flush=True)
assert not same, "parameter-freezing bug: trainable tensors were not updated: %s" % same[:5]
log["final_val"] = subsets(va); log["final_val_mode_predicted(inference setting)"] = subsets(va); log["final_val_mode_predicted(inference setting)"] = {"all": evaluate(va, None, "pred"), "post": evaluate(va, 1, "pred"), "pre": evaluate(va, 0, "pred")}
log["final_val_clean"] = {"post": evaluate([e for e in va if e["src"] == "clean"], 1)}; log["final_val_rand"] = {"post": evaluate([e for e in va if e["src"] == "rand"], 1)}; log["final_onpolicy_progress_val"] = eval_op(OPva)
os.makedirs(f"{N}/ckpt_pic/{t}", exist_ok=True); fn = f"{N}/ckpt_pic/{t}/acl_{TAG}.pt"; torch.save({"pic_prog": True, "resume": os.environ.get("ACL_RESUME"), "mods": {k: v.state_dict() for k, v in mods.items()}, "mean": mean.cpu(), "std": std.cpu(), "init": INIT, "init_sha256": hashlib.sha256(open(INIT, "rb").read()).hexdigest(), "stats": ST, "R_REF": "Ry(90deg)", "steps": STEPS}, fn); log["sha256"] = hashlib.sha256(open(fn, "rb").read()).hexdigest()
json.dump(log, open(f"{N}/training/pic_acl_{t}_{TAG}.json", "w"), indent=1); print("final", json.dumps(log["final_val"]["post"]), "\nsaved", fn, log["sha256"], flush=True)
