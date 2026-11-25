"""Per-task V21 training cache from the view sequences (privileged simulator state = LABELS ONLY): SigLIP patch features of the 3 cameras (half-res), candidate projections (all objects, per camera, from the frame's own K/E), referent of the upcoming grasp cycle,
VLM text-token states of the policy's own backbone pass on the frame (3 cameras letterboxed 224 + instruction + zero state = arm-agnostic home state), and 72px-crop local tokens of every valid head-camera candidate at its GT centre (scorer training, as original).
usage: build_v21_cache_10.py TASK -> v21cache/{task}.pt   (frames: every 2nd of each scene's sequence, plus frame 0)"""
import sys, os, glob, pickle, numpy as np, torch, cv2
from regsp import paths as P

from regsp.backbone.loader import load_policy, make_batch
from regsp.envs import cameras as C, v21rt as R
OUT = str(P.clean_root()); t = sys.argv[1]; MIN_SEP = 0.03; TAG = os.environ.get("V21TAG", ""); NMAX = int(os.environ.get("NUNITS", 75)); from validate_units import valid_unit
allf = sorted(glob.glob(f"{OUT}/views/{t}/view*.pkl")); vrep = {f: valid_unit(f, "seq") for f in allf}; files = [f for f in allf if vrep[f][0]][:NMAX]; nval = max(1, len(files) // 5); pol, pre = load_policy(); vm = pol.backbone.model.vlm_with_expert.get_vlm_model().vision_model; ve = pol.backbone.model.vlm_with_expert
def crop_tok(img_half, u, v):  # (u,v) in full-res px; crop CROP_PX/2 window on the half-res image (== CROP_PX on full-res), edge padded, resized to 512, frozen vision tower + connector -> 64 tokens
    h = R.CROP_PX // 4; uu, vv = int(round(min(max(u, 0), 639) / 2)), int(round(min(max(v, 0), 479) / 2)); pad = np.pad(img_half, ((h, h), (h, h), (0, 0)), mode="edge"); crop = pad[vv: vv + 2 * h, uu: uu + 2 * h]
    x = torch.from_numpy(cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).float().div(255)[None].cuda() * 2 - 1
    with torch.no_grad(): return ve.embed_image(x)[0].float().cpu().half()
FR = []; FE = []
valseeds = {pickle.load(open(f, "rb"))["seed"] for f in files[len(files) - nval:]}
allo = sorted(glob.glob(f"{OUT}/obsviews/{t}/obs*.pkl")); orep = {f: valid_unit(f, "obs") for f in allo}; ofiles = [f for f in allo if orep[f][0]][:NMAX]  # split by seed: seeds in the val set of the view units are val, all others train
jobs = [(f, "seq") for f in files] + [(f, "obs") for f in ofiles]
for si, (f, kind) in enumerate(jobs):
    p = pickle.load(open(f, "rb")); z = np.load(f.replace(".pkl", ".npz")); split = "val" if p["seed"] in valseeds else "train"
    for fi, fr in enumerate(p["frames"]):
        if kind == "seq" and fi % 2 and fi != len(p["frames"]) - 1: continue
        ob = []
        for o in fr["objects"]:
            if o["attr"] == "last_actor": continue
            if any(np.linalg.norm(np.array(o["xyz"]) - np.array(q["xyz"])) < 0.005 for q in ob): continue
            ob.append(o)
        imgs = [z[f"f{fi}_{c}"] for c in R.CAMS]; uv = {}
        for c in R.CAMS:
            L = []
            for o in ob:
                (u, v), d = R.project(fr["E"][c], o["xyz"]); L.append((float(u), float(v), d, bool(d > 0.05 and 8 <= u < 632 and 8 <= v < 472)))
            uv[c] = L
        xyz = np.array([o["xyz"] for o in ob]); D_ = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1) + np.eye(len(ob)) * 9; amb = D_.min(1) < MIN_SEP if len(ob) > 1 else np.zeros(len(ob), bool)
        if kind == "seq": k = fr["n_closed_so_far"]
        else: k = [i for i, rf in enumerate(p["refs"]) if tuple(rf) == (fr["arm"], fr["attr"])][0]
        ref = None; gw = None
        if k < len(p["refs"]):
            arm, attr = p["refs"][k]; idx = [i for i, o in enumerate(ob) if o["attr"] == attr]; gw = np.array(p["tcp_at_close"][k][1]) if k < len(p["tcp_at_close"]) else None
            ref = {"arm": arm, "attr": attr, "idx": idx[0] if idx else None}
        FR.append({"kind": kind, "scene": p["seed"], "fi": fi, "split": split, "instruction": p["instruction"], "objs": ob, "uv": uv, "E": {c: R.w2c4(fr["E"][c]) for c in R.CAMS}, "tcp": fr["tcp"], "grip": fr.get("grip"), "k": k, "ref": ref, "grasp_world": gw, "amb": amb.tolist(), "d_obs": fr.get("d_obs")})
        f3 = C.patch_feats(vm, [i for i in imgs]); FE.append(f3)  # [3,768,32,32] fp16
        b = pre(make_batch({"imgs224": [R.sq(i) for i in imgs], "state": np.zeros(7, np.float32)}, p["instruction"]))
        with torch.no_grad(): tok, pad, mod = pol._encode_multimodal_latent(b)
        m = (mod[0] == 1) & pad[0].bool(); FR[-1]["txt"] = tok[0][m].half().cpu()
        FR[-1]["crops_head"] = {i: crop_tok(imgs[0], uv["head_camera"][i][0], uv["head_camera"][i][1]) for i, o in enumerate(ob) if uv["head_camera"][i][3] and not amb[i]}
        FR[-1]["imgs_half"] = [np.ascontiguousarray(i) for i in imgs]  # kept for anchor/crop re-extraction at PROPOSAL locations
    print(t, si + 1, len(jobs), len(FR), flush=True)
import json, collections; os.makedirs(f"{OUT}/v21cache{TAG}", exist_ok=True); json.dump({"view_units_valid": len(files), "obs_units_valid": len(ofiles), "view_rejected": dict(collections.Counter(v[1] for v in vrep.values() if not v[0])), "obs_rejected": dict(collections.Counter(v[1] for v in orep.values() if not v[0])), "tag": TAG or "final"}, open(f"{OUT}/v21cache{TAG}/{t}.units.json", "w")); torch.save({"frames": FR, "feats": torch.stack(FE), "task": t}, f"{OUT}/v21cache{TAG}/{t}.pt"); print("saved", t, len(FR))
