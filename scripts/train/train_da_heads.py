"""[4-task study, Phase A] DOMAIN-AUGMENTED class-agnostic proposal heads (separately named experiment "DA-heads").
Architecture, loss, optimiser, seed, steps and batch are IDENTICAL to the original V21 heads stage of train_v21_task.py (v21rt.Head(64), focal + masked L1 offset, seed 1000, 4000 steps, batch 16, AdamW 1e-3/1e-4, from scratch, one independent head per camera, frozen SigLIP features).
The ONLY change: the training frames = the task's original Clean cache frames (train split) + randomized-domain frames (demo_randomized, seeds 5000000+, expert view sequences + active observation views, 0-3 distractors) from da_cache/{task}.pt (train split).
Targets = projected centres of ALL task objects (privileged simulator state = LABELS ONLY) with the frame's own K/E (class-agnostic, as the original).
Evaluation (val splits only; baseline heads are re-evaluated on the same frames): recall@10px of visible GT centres, top-1 hit, false-positive rate, referent recall, median nearest-proposal error, per camera and per frame kind, on (a) original Clean val frames, (b) randomized-domain val frames.
Baseline checkpoints are never modified: outputs ckpt_da/{task}/v21_prop_{cam}.pt , training/da_heads_{task}.json.   usage: train_da_heads.py TASK"""
import sys, os, json, hashlib, numpy as np, torch, torch.nn.functional as F
from regsp import paths as P
W = str(P.clean_root()); N = str(P.work_root())

import regsp.grounding as R
t = sys.argv[1]; SEED = 1000; STEPS = int(os.environ.get("HSTEPS", 4000)); TAG = os.environ.get("DATAG", "")
C = torch.load(f"{W}/v21cache/{t}.pt", weights_only=False); D = torch.load(f"{N}/da_cache/{t}.pt", weights_only=False)
if os.environ.get("DA_EXTRA"):  # [Exp2 grounding] additional randomized-domain training units (da_cache/{t}_{DA_EXTRA}.pt, all train split); val sets unchanged -> directly comparable with the earlier heads
    D2 = torch.load(f"{N}/da_cache/{t}_{os.environ['DA_EXTRA']}.pt", weights_only=False); D = {"frames": D["frames"] + D2["frames"], "feats": torch.cat([D["feats"], D2["feats"]])}; del D2
FR = C["frames"] + D["frames"]; src = ["clean"] * len(C["frames"]) + ["rand"] * len(D["frames"]); FE = torch.cat([C["feats"], D["feats"]]); del C, D
tr_c = [i for i, f in enumerate(FR) if f["split"] == "train" and src[i] == "clean"]; va_c = [i for i, f in enumerate(FR) if f["split"] == "val" and src[i] == "clean"]
tr_r = [i for i, f in enumerate(FR) if f["split"] == "train" and src[i] == "rand"]; va_r = [i for i, f in enumerate(FR) if f["split"] == "val" and src[i] == "rand"]
tr = tr_c + tr_r; print(t, "frames clean tr/va", len(tr_c), len(va_c), "rand tr/va", len(tr_r), len(va_r), flush=True)
def sha(fn): return hashlib.sha256(open(fn, "rb").read()).hexdigest()
def centres(f, cam): return [(u, v) for (u, v, dep, ok) in f["uv"][cam] if ok]
def run_head(m, idx, ci):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), 64): out += list(m(FE[idx[i:i + 64], ci].float().cuda()).cpu())
    return out
def evaluate(m, idx, ci, cam):
    outs = run_head(m, idx, ci); hit, err, fp, npr, top1, nf, ref, per = [], [], 0, 0, [], 0, [], {}
    for i, o in zip(idx, outs):
        G = centres(FR[i], cam); P = R.decode(o)
        if not G: continue
        nf += 1; used = set()
        for g in G:
            e = min([np.hypot(p[0] - g[0], p[1] - g[1]) for p in P], default=1e9); h = float(e <= R.MATCH_PX); hit.append(h); err.append(e); per.setdefault(FR[i]["kind"], []).append(h)
        npr += len(P); fp += sum(1 for p in P if min(np.hypot(p[0] - g[0], p[1] - g[1]) for g in G) > R.MATCH_PX)
        if P: top1.append(float(min(np.hypot(max(P, key=lambda q: q[2])[0] - g[0], max(P, key=lambda q: q[2])[1] - g[1]) for g in G) <= R.MATCH_PX))
        else: top1.append(0.0)
        r = FR[i]["ref"]
        if r and r["idx"] is not None and FR[i]["uv"][cam][r["idx"]][3]: u, v = FR[i]["uv"][cam][r["idx"]][:2]; ref.append(float(any(np.hypot(p[0] - u, p[1] - v) <= R.MATCH_PX for p in P)))
    ea = np.array(err); return {"n_frames_with_visible_gt": nf, "n_gt": len(hit), "recall@10px": float(np.mean(hit)) if hit else None, "top1_hit": float(np.mean(top1)) if top1 else None, "referent_recall@10px": float(np.mean(ref)) if ref else None, "n_referent": len(ref),
                                "false_pos_rate": fp / max(npr, 1), "proposals_per_frame": npr / max(nf, 1), "median_err_px_nearest": float(np.median(ea[ea < 1e8])) if (ea < 1e8).any() else None, **{f"recall_{k}": float(np.mean(v)) for k, v in per.items()}}
rng = np.random.default_rng(SEED); res = {"task": t, "experiment": "DA-heads (domain-augmented proposal heads; separately named, see Phase A)", "steps": STEPS, "seed": SEED, "n_train_clean": len(tr_c), "n_train_rand": len(tr_r), "n_val_clean": len(va_c), "n_val_rand": len(va_r), "cams": {}}
os.makedirs(f"{N}/ckpt_da/{t}", exist_ok=True); os.makedirs(f"{N}/training", exist_ok=True)
for ci, cam in enumerate(R.CAMS):
    T = {i: R.targets(centres(FR[i], cam)) for i in tr}; torch.manual_seed(SEED); m = R.Head(64).cuda(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(1, STEPS + 1):
        m.train(); b = rng.choice(tr, 16, replace=False); f = FE[b, ci].float().cuda(); hm, off, mk = [torch.from_numpy(np.stack([T[i][k] for i in b])).cuda() for k in range(3)]
        o = m(f); loss = R.focal(o[:, 0], hm) + F.l1_loss(o[:, 1:] * mk[:, None], off * mk[:, None], reduction="sum") / mk.sum().clamp_min(1); opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0: print(t, cam, step, round(loss.item(), 4), flush=True)
    fn = f"{N}/ckpt_da/{t}/v21_prop_{cam}{TAG}.pt"; torch.save(m.state_dict(), fn); m.eval()
    b0 = R.Head(64).cuda(); b0.load_state_dict(torch.load(f"{W}/ckpt/{t}/v21_prop_{cam}.pt")); b0.eval()
    res["cams"][cam] = {"sha256": sha(fn), "baseline_sha256": sha(f"{W}/ckpt/{t}/v21_prop_{cam}.pt"), **{f"{nm}_{vs}": evaluate(mm, ix, ci, cam) for nm, mm in (("DA", m), ("baseline", b0)) for vs, ix in (("val_clean", va_c), ("val_rand", va_r))}}
    print(t, cam, json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("recall@10px", "top1_hit", "referent_recall@10px", "false_pos_rate")}) for k, v in res["cams"][cam].items() if k != "sha256"}), flush=True)
json.dump(res, open(f"{N}/training/da_heads_{t}{TAG}.json", "w"), indent=1)
