"""[4-task study] HR referent scorer training with the ORIGINAL counterfactual sample construction (v21hr_candidate.py build_samples: one sample per valid referent instruction of the SAME scene, CE over the scene's candidates).
Architecture, loss, optimiser, steps, batch, seed identical to the 10-task port (train_v21_task.py scorer stage): Scorer('B') [stack: CycleScorer = Scorer + zero-init cycle embedding, the 10-task extension], CE, AdamW 3e-4/1e-2, 3000 steps, batch 32, seed 1000, from scratch.
MODE (PRE-DECLARED before any CF result was seen):
  base : original frames only (reproduction of the 10-task recipe; reference)
  cf   : each batch = 16 original-frame samples + 16 counterfactual samples = 8 CF scenes x BOTH referent instructions of the same scene (paired)
  ctl  : each batch = 16 original + 16 CF-scene samples with ONE randomly chosen referent per scene (same scenes/images/instructions as cf, unpaired) -> isolates same-scene pairing from extra data
usage: train_cf_scorer.py TASK MODE TAG"""
import sys, os, json, hashlib, numpy as np, torch, torch.nn.functional as F
from regsp import paths as P
W = str(P.clean_root()); OUT = str(P.work_root())

import regsp.grounding as R
from regsp.decoding import bimanual as B
t, MODE, TAG = sys.argv[1:4]; stack = t == "stack_blocks_two"; dev = "cuda"; SEED = 1000; STEPS = int(os.environ.get("HSTEPS", 3000))
d = torch.load(f"{W}/v21cache/{t}.pt", weights_only=False, map_location="cpu"); FR = d["frames"]
if os.environ.get("UNION"): FR = FR + torch.load(f"{OUT}/da_cache/{t}.pt", weights_only=False, map_location="cpu")["frames"]  # [Phase B on Phase A data] original Clean frames + randomized-domain adaptation frames (DA cache), same sample construction
cfp = f"{OUT}/cf_cache/{t}.pt"; cf = torch.load(cfp, weights_only=False) if os.path.exists(cfp) else {"scenes": {}, "samples": []}  # no verified counterfactual referents for place_container_plate / handover_block -> original supervision only (MODE base)
def orig_samples(split):
    S = []
    for f in FR:
        r = f["ref"]
        if f["split"] != split or r is None or r["idx"] is None or r["idx"] not in f["crops_head"] or len(f["crops_head"]) < 2: continue
        ids = sorted(f["crops_head"]); S.append({"txt": f["txt"].float(), "feat": torch.stack([f["crops_head"][k].float() for k in ids]), "t": ids.index(r["idx"]), "cyc": min(f["k"], 1), "src": "orig"})
    return S
def cf_samples(split):
    S = []
    for s in cf["samples"]:
        sc = cf["scenes"][s["scene"]]
        if sc["split"] != split: continue
        S.append({"txt": s["txt"].float(), "feat": sc["feat"].float(), "t": s["t"], "cyc": s["cyc"], "src": "cf", "scene": s["scene"], "variant": s["variant"], "referent": s["referent"], "instruction": s["instruction"]})
    return S
Otr, Ova, Ctr, Cva = orig_samples("train"), orig_samples("val"), cf_samples("train"), cf_samples("val"); print(t, MODE, "orig", len(Otr), len(Ova), "cf", len(Ctr), len(Cva), flush=True)
def coll(S, txts=None):
    txts = txts or [s["txt"] for s in S]; T = max(len(x) for x in txts); Kc = max(len(s["feat"]) for s in S); n = len(S); txt = torch.zeros(n, T, 960); tm = torch.zeros(n, T, dtype=torch.bool); cm = torch.zeros(n, Kc, dtype=torch.bool); f = torch.zeros(n, Kc, 64, 960)
    for b, (s, x) in enumerate(zip(S, txts)): txt[b, :len(x)] = x; tm[b, :len(x)] = True; cm[b, :len(s["feat"])] = True; f[b, :len(s["feat"])] = s["feat"]
    return txt.to(dev), tm.to(dev), f.to(dev), cm.to(dev), torch.tensor([s["t"] for s in S]).to(dev), torch.tensor([s["cyc"] for s in S]).to(dev)
def logits(sc, S, txts=None):
    tx, tm, f, cm, tg, cy = coll(S, txts); return (sc(tx, tm, f, cy) if stack else sc(tx, tm, f)).masked_fill(~cm, -1e9), tg
def batch(rng):
    nb = 32 if MODE == "base" else 16; b = [Otr[i] for i in rng.choice(len(Otr), nb, replace=False)]
    if MODE == "base": return b
    scenes = sorted({s["scene"] for s in Ctr}); pick = rng.choice(len(scenes), min(8, len(scenes)), replace=False)
    for j in pick:
        c = [s for s in Ctr if s["scene"] == scenes[j]]
        if MODE == "cf":  # both referents of the scene (one random instruction variant per referent)
            for ref in sorted({s["referent"] for s in c}): cc = [s for s in c if s["referent"] == ref]; b.append(cc[rng.integers(len(cc))])
        else: b.append(c[rng.integers(len(c))])
    return b
torch.manual_seed(SEED); rng = np.random.default_rng(SEED); sc = (B.CycleScorer() if stack else R.Scorer()).to(dev); opt = torch.optim.AdamW(sc.parameters(), lr=3e-4, weight_decay=1e-2)
for step in range(1, STEPS + 1):
    sc.train(); lg, tg = logits(sc, batch(rng)); loss = F.cross_entropy(lg, tg); opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0: print(t, MODE, step, round(loss.item(), 4), flush=True)
sc.eval(); os.makedirs(f"{OUT}/ckpt_cf/{t}", exist_ok=True); fn = f"{OUT}/ckpt_cf/{t}/scorer_{MODE}_{TAG}.pt"; torch.save(sc.state_dict(), fn)
rng2 = np.random.default_rng(0)
def evaluate(scorer, S, txt_mode="orig"):
    out = []
    with torch.no_grad():
        for i in range(0, len(S), 32):
            ch = S[i:i + 32]; txts = None
            if txt_mode == "shuffled": txts = [S[(i + j + 1 + rng2.integers(len(S) - 1)) % len(S)]["txt"] for j in range(len(ch))] if S[0]["src"] == "orig" else [next(q for q in rng2.permutation(S).tolist() if q["scene"] != s["scene"])["txt"] for s in ch]
            if txt_mode == "blank": txts = [torch.zeros(1, 960) for _ in ch]
            lg, tg = logits(scorer, ch, txts); out += [(l[:len(s["feat"])].cpu(), s) for l, s in zip(lg, ch)]
    return out
def cf_metrics(scorer, S):
    res = {}
    for mode in ("orig", "shuffled", "blank"):
        O = evaluate(scorer, S, mode); acc = float(np.mean([int(l.argmax()) == s["t"] for l, s in O])); by = {}
        for l, s in O: by.setdefault(s["scene"], []).append((int(l.argmax()), s["t"], s["referent"], s["variant"]))
        sw = [all(p == tt for p, tt, _, _ in v if True) for v in by.values() if len({r for _, _, r, _ in v}) >= 2]   # both referents of the scene correct (same scene, different instruction)
        chg = [len({(p) for p, _, _, _ in v}) > 1 for v in by.values() if len({r for _, _, r, _ in v}) >= 2]  # prediction changed when the referent in the instruction changed
        res[mode] = {"acc": acc, "swap_both_correct": float(np.mean(sw)) if sw else None, "pred_changes_on_swap": float(np.mean(chg)) if chg else None, "n_samples": len(O), "n_scenes_paired": len(sw), "chance": float(np.mean([1 / len(s["feat"]) for _, s in O]))}
        if len({v for s in S for v in [s["variant"]]}) > 1:
            for var in sorted({s["variant"] for s in S}): sub = [(l, s) for l, s in O if s["variant"] == var]; res[mode][f"acc_{var}"] = float(np.mean([int(l.argmax()) == s["t"] for l, s in sub]))
    return res
def orig_acc(scorer, S):
    c = []
    with torch.no_grad():
        for i in range(0, len(S), 32): ch = S[i:i + 32]; lg, tg = logits(scorer, ch); c += (lg.argmax(-1) == tg).float().cpu().tolist()
    return float(np.mean(c))
rep = {"task": t, "mode": MODE, "tag": TAG, "steps": STEPS, "sha256": hashlib.sha256(open(fn, "rb").read()).hexdigest(), "n_orig_train": len(Otr), "n_orig_val": len(Ova), "n_cf_train": len(Ctr), "n_cf_val": len(Cva), "orig_val_acc": orig_acc(sc, Ova), "orig_train_acc_subset": orig_acc(sc, Otr[:400]),
       "cf_val": cf_metrics(sc, Cva) if Cva else None, "cf_train": cf_metrics(sc, Ctr[:200]) if Ctr else None, "union_with_da_frames": bool(os.environ.get("UNION"))}
# baseline (frozen 10-task scorer) on the same CF val set
b = (B.CycleScorer() if stack else R.Scorer()).to(dev).eval(); b.load_state_dict(torch.load(f"{W}/ckpt_bim/{t}/v21b_scorer_cycle_b2s.pt" if stack else f"{W}/ckpt/{t}/v21_scorer.pt")); rep["baseline_on_cf_val"] = cf_metrics(b, Cva) if Cva else None; rep["baseline_orig_val_acc"] = orig_acc(b, Ova)
os.makedirs(f"{OUT}/training", exist_ok=True); json.dump(rep, open(f"{OUT}/training/scorer_{t}_{MODE}_{TAG}.json", "w"), indent=1); print(json.dumps(rep)[:3000], flush=True)
