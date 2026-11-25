"""[4-task study, Phase C: ANCHOR stage on the union of original Clean frames + randomized-domain adaptation frames, with the DA obs-heads; otherwise the original train_v21_task.py anchor stage verbatim] Per-task V21 grounding training on the task's own cache (v21cache/{task}.pt; NEVER mixes tasks). STAGES (env STAGE): heads -> scorer -> anchor.
 heads : three CLASS-AGNOSTIC Head(64) (head cam, left wrist, right wrist), ORIGINAL v21q recipe (focal + masked L1 offset, seed 1000, 4000 steps, batch 16, AdamW 1e-3/1e-4) on ALL objects' projected centres. [original-architecture-retrained]
 scorer: HR Scorer('B') on the VLM text-token states of the policy's own backbone pass + 72px crop tokens at the head-camera candidate centres; candidate-level CE over the valid candidates, no null slot (seed 1000, 3000 steps, batch 32, AdamW 3e-4/1e-2). [original-architecture-retrained]
 anchor: AnchorHead trained on triangulated (head x wrist) references from the TRAINED heads (original train_dataset logic), 3000 steps batch 64 AdamW 3e-4/1e-2 smooth-L1 on offset(cm); keep rule (original): learned kept iff DEV median error <= 0.9 x per-task-constant-offset baseline. [original-architecture-retrained; recentre constant = embodiment adapter]
usage: train_v21_task.py TASK"""
import sys, os, json, hashlib, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from regsp import paths as P

import regsp.grounding as R
from regsp.execution import kinematics as rt_kin
OUT = str(P.clean_root()); NN = str(P.work_root()); t = sys.argv[1]; STAGES = ["anchor"]; dev = "cuda"; SEED = 1000; TAG = os.environ.get("V21TAG", "")
d = torch.load(f"{OUT}/v21cache{TAG}/{t}.pt", weights_only=False); D2 = torch.load(f"{NN}/da_cache/{t}.pt", weights_only=False); FR, FE = d["frames"] + D2["frames"], torch.cat([d["feats"], D2["feats"]]); del d, D2; tr = [i for i, f in enumerate(FR) if f["split"] == "train"]; va = [i for i, f in enumerate(FR) if f["split"] == "val"]
os.makedirs(f"{NN}/ckpt_full/{t}", exist_ok=True); rep = {"task": t, "stage": "anchor on union(Clean frames, DA frames) with DA obs-heads"}
fkL = rt_kin.ArmFK("left"); To = fkL.T_origin; Ti = np.linalg.inv(To)
def sha(fn): return hashlib.sha256(open(fn, "rb").read()).hexdigest()
def centres(f, cam): return [(u, v) for (u, v, dep, ok) in f["uv"][cam] if ok]
def match_gt(Pp, G):
    pairs = sorted([(np.hypot(p[0] - g[0], p[1] - g[1]), i, j) for i, p in enumerate(Pp) for j, g in enumerate(G)]); up, ug, m = set(), set(), {}
    for dd, i, j in pairs:
        if dd <= R.MATCH_PX and i not in up and j not in ug: m[i] = j; up.add(i); ug.add(j)
    return m
def run_head(m, idx, ci):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), 64): out += list(m(FE[idx[i:i + 64], ci].float().cuda()).cpu())
    return out
if "heads" in STAGES:
    rng = np.random.default_rng(SEED); res = {}
    for ci, cam in enumerate(R.CAMS):
        T = {i: R.targets(centres(FR[i], cam)) for i in tr}; torch.manual_seed(SEED); m = R.Head(64).cuda(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        for step in range(1, int(os.environ.get("HSTEPS", 4000)) + 1):
            m.train(); b = rng.choice(tr, 16, replace=False); f = FE[b, ci].float().cuda(); hm, off, mk = [torch.from_numpy(np.stack([T[i][k] for i in b])).cuda() for k in range(3)]
            o = m(f); loss = R.focal(o[:, 0], hm) + F.l1_loss(o[:, 1:] * mk[:, None], off * mk[:, None], reduction="sum") / mk.sum().clamp_min(1); opt.zero_grad(); loss.backward(); opt.step()
        fn = f"{OUT}/ckpt{TAG}/{t}/v21_prop_{cam}.pt"; torch.save(m.state_dict(), fn); outs = run_head(m, va, ci); cov, fp, npr, ngt, dup, tot, nz = [], 0, 0, 0, 0, 0, 0; per_kind = {}
        for i, o in zip(va, outs):
            G = centres(FR[i], cam); Pp = R.decode(o); mt = match_gt(Pp, G); used = set(mt.values()); fp += len(Pp) - len(mt); npr += len(Pp); ngt += len(G)
            c = [float(j in used) for j in range(len(G))]; cov += c; per_kind.setdefault(FR[i]["kind"], []).extend(c)
            ref = FR[i]["ref"]
            if ref and ref["idx"] is not None and FR[i]["uv"][cam][ref["idx"]][3]: nz += 1; per_kind.setdefault("referent_" + FR[i]["kind"], []).append(float(any(np.hypot(p[0] - FR[i]["uv"][cam][ref["idx"]][0], p[1] - FR[i]["uv"][cam][ref["idx"]][1]) <= R.MATCH_PX for p in Pp)))
        res[cam] = {"sha256": sha(fn), "n_train_frames": len(tr), "n_val_frames": len(va), "val_gt_centres": ngt, "val_recall@10px": float(np.mean(cov)) if cov else None, "val_proposals": npr, "val_false_pos_rate": fp / max(npr, 1), **{f"recall_{k}": float(np.mean(v)) for k, v in per_kind.items() if v}}
        print(t, cam, json.dumps(res[cam]), flush=True)
    rep["heads"] = res; json.dump(rep, open(f"{NN}/training/full_anchor_{t}.json", "w"), indent=1)
if "scorer" in STAGES:
    def samples(ix):
        S = []
        for i in ix:
            f = FR[i]; r = f["ref"]
            if r is None or r["idx"] is None or r["idx"] not in f["crops_head"] or len(f["crops_head"]) < 2: continue
            ids = sorted(f["crops_head"]); S.append({"i": i, "txt": f["txt"].float(), "ids": ids, "t": ids.index(r["idx"]), "feat": torch.stack([f["crops_head"][k].float() for k in ids])})
        return S
    Str, Sva = samples(tr), samples(va); print(t, "scorer samples", len(Str), len(Sva), flush=True)
    def coll(S):
        T = max(s["txt"].shape[0] for s in S); Kc = max(len(s["ids"]) for s in S); B = len(S); txt = torch.zeros(B, T, 960); tm = torch.zeros(B, T, dtype=torch.bool); cm = torch.zeros(B, Kc, dtype=torch.bool); f = torch.zeros(B, Kc, 64, 960)
        for b, s in enumerate(S): txt[b, :len(s["txt"])] = s["txt"]; tm[b, :len(s["txt"])] = True; cm[b, :len(s["ids"])] = True; f[b, :len(s["ids"])] = s["feat"]
        return txt.cuda(), tm.cuda(), f.cuda(), cm.cuda(), torch.tensor([s["t"] for s in S]).cuda()
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED); sc = R.Scorer().cuda(); opt = torch.optim.AdamW(sc.parameters(), lr=3e-4, weight_decay=1e-2)
    for step in range(1, int(os.environ.get("HSTEPS", 3000)) + 1):
        sc.train(); b = [Str[i] for i in rng.choice(len(Str), min(32, len(Str)), replace=False)]; txt, tm, f, cm, tg = coll(b); loss = F.cross_entropy(sc(txt, tm, f).masked_fill(~cm, -1e9), tg); opt.zero_grad(); loss.backward(); opt.step()
    sc.eval(); fn = f"{OUT}/ckpt{TAG}/{t}/v21_scorer.pt"; torch.save(sc.state_dict(), fn)
    def acc(S):
        c, ch, per = [], [], {}
        with torch.no_grad():
            for i in range(0, len(S), 32):
                b = S[i:i + 32]; txt, tm, f, cm, tg = coll(b); p = sc(txt, tm, f).masked_fill(~cm, -1e9).argmax(-1).cpu()
                for s, pp in zip(b, p): ok = float(pp == s["t"]); c.append(ok); ch.append(1 / len(s["ids"])); per.setdefault("k%d" % FR[s["i"]]["k"], []).append(ok)
        return float(np.mean(c)), float(np.mean(ch)), {k: (float(np.mean(v)), len(v)) for k, v in per.items()}
    a_tr, c_tr, _ = acc(Str[:400]); a_va, c_va, pk = acc(Sva)
    rep["scorer"] = {"sha256": sha(fn), "n_train": len(Str), "n_val": len(Sva), "train_acc_subset": a_tr, "val_acc": a_va, "val_chance": c_va, "val_acc_by_cycle(acc,n)": pk}; print(t, "scorer", json.dumps(rep["scorer"]), flush=True); json.dump(rep, open(f"{NN}/training/full_anchor_{t}.json", "w"), indent=1)
if "anchor" in STAGES:
    heads = {}
    for cam in R.CAMS: m = R.Head(64).cuda(); m.load_state_dict(torch.load(f"{NN}/ckpt_da/{t}/v21_prop_{cam}_obs.pt")); heads[cam] = m.eval()
    from regsp.backbone.loader import load_policy
    pol, _ = load_policy(); ve = pol.backbone.model.vlm_with_expert; import cv2
    def crop_tok(img_half, u, v):
        h = R.CROP_PX // 4; uu, vv = int(round(min(max(u, 0), 639) / 2)), int(round(min(max(v, 0), 479) / 2)); pad = np.pad(img_half, ((h, h), (h, h), (0, 0)), mode="edge"); crop = pad[vv: vv + 2 * h, uu: uu + 2 * h]
        x = torch.from_numpy(cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).float().div(255)[None].cuda() * 2 - 1
        with torch.no_grad(): return ve.embed_image(x)[0].float().cpu().half()
    Dn = {"train": [], "val": []}
    for i, f in enumerate(FR):
        r = f["ref"]
        if r is None or r["idx"] is None or f["grasp_world"] is None: continue
        wc = "left_camera" if r["arm"] == "L" else "right_camera"
        if not (f["uv"]["head_camera"][r["idx"]][3] and f["uv"][wc][r["idx"]][3]): continue
        with torch.no_grad(): Pf = R.decode(heads["head_camera"](FE[i, 0].float().cuda()[None])[0].cpu()); Pw = R.decode(heads[wc](FE[i, R.CAMS.index(wc)].float().cuda()[None])[0].cpu())
        gf = (f["uv"]["head_camera"][r["idx"]][0], f["uv"]["head_camera"][r["idx"]][1]); gw = (f["uv"][wc][r["idx"]][0], f["uv"][wc][r["idx"]][1]); mf = match_gt(Pf, [gf]); mw = match_gt(Pw, [gw])
        if not mf or not mw: continue
        pi, pj = list(mf)[0], list(mw)[0]; A = R.associate(Pf, Pw, f["E"]["head_camera"], f["E"][wc])
        if pi not in A or A[pi][1] != pj: continue
        refw, _, gap = A[pi]; ref_r = Ti[:3, :3] @ refw + Ti[:3, 3]; gt_r = Ti[:3, :3] @ f["grasp_world"] + Ti[:3, 3]
        Dn[f["split"]].append({"tf": crop_tok(f["imgs_half"][0], Pf[pi][0], Pf[pi][1]), "tr": crop_tok(f["imgs_half"][R.CAMS.index(wc)], Pw[pj][0], Pw[pj][1]), "ref": ref_r.astype(np.float32), "res": ((gt_r - ref_r) * 100).astype(np.float32), "gap": gap, "obj_err_cm": float(np.linalg.norm(refw - np.array(f["objs"][r["idx"]]["xyz"])) * 100)})
    print(t, "anchor samples", len(Dn["train"]), len(Dn["val"]), flush=True)
    if len(Dn["train"]) >= 20 and len(Dn["val"]) >= 5:
        REF = torch.tensor(np.stack([x["ref"] for x in Dn["train"]])); Y = torch.tensor(np.stack([x["res"] for x in Dn["train"]])); TF = torch.stack([x["tf"] for x in Dn["train"]]); TRr = torch.stack([x["tr"] for x in Dn["train"]])
        mean_off = Y.mean(0).numpy(); torch.manual_seed(SEED); rng = np.random.default_rng(SEED); ah = R.AnchorHead(center=tuple(REF.mean(0).tolist())).cuda(); opt = torch.optim.AdamW(ah.parameters(), lr=3e-4, weight_decay=1e-2)
        for step in range(1, int(os.environ.get("HSTEPS", 3000)) + 1):
            ah.train(); idx = torch.from_numpy(rng.choice(len(Dn["train"]), min(64, len(Dn["train"])), replace=False)); loss = F.smooth_l1_loss(ah(TF[idx].float().cuda(), TRr[idx].float().cuda(), REF[idx].cuda()), Y[idx].cuda()); opt.zero_grad(); loss.backward(); opt.step()
        ah.eval(); Vr = torch.tensor(np.stack([x["ref"] for x in Dn["val"]])); Vy = np.stack([x["res"] for x in Dn["val"]])
        with torch.no_grad(): pc = ah(torch.stack([x["tf"] for x in Dn["val"]]).float().cuda(), torch.stack([x["tr"] for x in Dn["val"]]).float().cuda(), Vr.cuda()).cpu().numpy()
        eA = np.linalg.norm(Vy, axis=1); eB = np.linalg.norm(Vy - mean_off, axis=1); eC = np.linalg.norm(Vy - pc, axis=1); keep = bool(np.median(eC) <= 0.9 * np.median(eB))
        fn = f"{NN}/ckpt_full/{t}/v21_anchor.pt"; torch.save({"state": ah.state_dict(), "center": REF.mean(0).tolist(), "mean_offset_cm": mean_off.tolist(), "keep_learned": keep}, fn)
        rep["anchor"] = {"sha256": sha(fn), "n_train": len(Dn["train"]), "n_val": len(Dn["val"]), "val_median_err_cm": {"A_triangulated_ref_only": float(np.median(eA)), "B_ref+const_offset": float(np.median(eB)), "C_learned_anchor": float(np.median(eC))}, "keep_learned_C(rule: C<=0.9*B)": keep, "mean_offset_cm": mean_off.tolist(), "triangulation_error_to_object_cm_median": float(np.median([x["obj_err_cm"] for x in Dn["val"]])), "ray_gap_m_median": float(np.median([x["gap"] for x in Dn["val"]]))}
    else: rep["anchor"] = {"error": "too few triangulable training samples", "n_train": len(Dn["train"]), "n_val": len(Dn["val"])}
    print(t, "anchor", json.dumps(rep["anchor"]), flush=True); json.dump(rep, open(f"{NN}/training/full_anchor_{t}.json", "w"), indent=1)
