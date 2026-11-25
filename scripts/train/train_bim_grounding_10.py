"""[train_bim_grounding_10] Extra V21 referent scorers (HR Scorer 'B' architecture, original recipe: candidate-level CE, seed 1000, 3000 steps, batch 32, AdamW 3e-4/1e-2) on the task's own v21cache, as NEW versioned checkpoints (the immutable v21_scorer.pt is untouched):
 MODE cycle : CycleScorer (cycle-conditioned referent; target = referent of the CURRENT cycle k = closes so far, incl. distractors) -> ckpt_bim/{task}/v21b_scorer_cycle_{TAGV}.pt
 MODE place : Scorer selecting the PLACE referent (object attribute PLACE_ATTR, e.g. handover `target_box`) among the proposals-equivalent candidates -> ckpt_bim/{task}/v21b_scorer_place_{TAGV}.pt
usage: train_bim_grounding_10.py TASK MODE TAGV [PLACE_ATTR]"""
import sys, os, json, hashlib, numpy as np, torch, torch.nn.functional as F
from regsp import paths as P

import regsp.grounding as R; from regsp.grounding.scorer import CycleScorer
OUT = str(P.clean_root()); t, MODE, TAGV = sys.argv[1:4]; PA = sys.argv[4] if len(sys.argv) > 4 else None; SEED = 1000; dev = "cuda"
d = torch.load(f"{OUT}/v21cache/{t}.pt", weights_only=False); FR = d["frames"]; tr = [i for i, f in enumerate(FR) if f["split"] == "train"]; va = [i for i, f in enumerate(FR) if f["split"] == "val"]
def samples(ix):
    S = []
    for i in ix:
        f = FR[i]; ids = sorted(f["crops_head"])
        if len(ids) < 2: continue
        if MODE == "cycle":
            r = f["ref"]
            if r is None or r["idx"] is None or r["idx"] not in ids: continue
            tgt = r["idx"]; cyc = min(int(f["k"]), 1)
        else:
            c = [j for j, o in enumerate(f["objs"]) if o["attr"] == PA]
            if not c or c[0] not in ids: continue
            tgt = c[0]; cyc = 0
        S.append({"i": i, "txt": f["txt"].float(), "ids": ids, "t": ids.index(tgt), "cyc": cyc, "feat": torch.stack([f["crops_head"][k].float() for k in ids])})
    return S
Str, Sva = samples(tr), samples(va); print(t, MODE, "samples", len(Str), len(Sva), flush=True)
def coll(S):
    T = max(s["txt"].shape[0] for s in S); Kc = max(len(s["ids"]) for s in S); B = len(S); txt = torch.zeros(B, T, 960); tm = torch.zeros(B, T, dtype=torch.bool); cm = torch.zeros(B, Kc, dtype=torch.bool); f = torch.zeros(B, Kc, 64, 960)
    for b, s in enumerate(S): txt[b, :len(s["txt"])] = s["txt"]; tm[b, :len(s["txt"])] = True; cm[b, :len(s["ids"])] = True; f[b, :len(s["ids"])] = s["feat"]
    return txt.cuda(), tm.cuda(), f.cuda(), cm.cuda(), torch.tensor([s["t"] for s in S]).cuda(), torch.tensor([s["cyc"] for s in S]).cuda()
def fwd(sc, txt, tm, f, cyc): return sc(txt, tm, f, cyc) if MODE == "cycle" else sc(txt, tm, f)
torch.manual_seed(SEED); rng = np.random.default_rng(SEED); sc = (CycleScorer() if MODE == "cycle" else R.Scorer()).cuda(); opt = torch.optim.AdamW(sc.parameters(), lr=float(os.environ.get('LR', 3e-4)), weight_decay=1e-2)
if MODE == 'cycle' and os.environ.get('INIT', 'base') == 'base' and False: pass
if MODE == "cycle" and os.environ.get("INIT", "base") == "base": sc.load_state_dict(torch.load(f"{OUT}/ckpt/{t}/v21_scorer.pt"), strict=False)  # init = the task's trained original scorer (cycle embedding zero) -> extension starts exactly at the baseline
for step in range(1, int(os.environ.get("HSTEPS", 3000)) + 1):
    sc.train(); b = [Str[i] for i in rng.choice(len(Str), min(32, len(Str)), replace=False)]; txt, tm, f, cm, tg, cy = coll(b); loss = F.cross_entropy(fwd(sc, txt, tm, f, cy).masked_fill(~cm, -1e9), tg); opt.zero_grad(); loss.backward(); opt.step()
sc.eval(); fn = f"{OUT}/ckpt_bim/{t}/v21b_scorer_{MODE}_{TAGV}.pt"; os.makedirs(os.path.dirname(fn), exist_ok=True); torch.save(sc.state_dict(), fn)
def acc(S, scorer, cyc_override=None):
    c, per = [], {}
    with torch.no_grad():
        for i in range(0, len(S), 32):
            b = S[i:i + 32]; txt, tm, f, cm, tg, cy = coll(b); p = fwd(scorer, txt, tm, f, cy).masked_fill(~cm, -1e9).argmax(-1).cpu()
            for s, pp in zip(b, p): ok = float(pp == s["t"]); c.append(ok); per.setdefault(f"k{FR[s['i']]['k']}", []).append(ok)
    return float(np.mean(c)), {k: (float(np.mean(v)), len(v)) for k, v in per.items()}
rep = {"task": t, "mode": MODE, "place_attr": PA, "sha256": hashlib.sha256(open(fn, "rb").read()).hexdigest(), "n_train": len(Str), "n_val": len(Sva), "val_acc(acc,by_cycle)": acc(Sva, sc), "chance": float(np.mean([1 / len(s["ids"]) for s in Sva]))}
if MODE == "cycle":  # baseline = unchanged scorer on the same val samples
    b = R.Scorer().cuda(); b.load_state_dict(torch.load(f"{OUT}/ckpt/{t}/v21_scorer.pt")); b.eval(); MODE = "baseline"; fwd = lambda sc_, txt, tm, f, cyc: sc_(txt, tm, f); rep["baseline_unchanged_scorer_val_acc"] = acc(Sva, b)
json.dump(rep, open(f"{OUT}/training/bim_grounding_{t}_{sys.argv[2]}_{TAGV}.json", "w"), indent=1); print(json.dumps(rep))
