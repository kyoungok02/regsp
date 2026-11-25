"""[4-task study] Counterfactual referent cache = text-token states (policy's own frozen backbone pass, same procedure as build_v21_cache_10.py) for ALL valid instructions of a scene + candidate crop tokens.
click_bell : from gen_cf_bell.py scenes (two bells). Per scene: 2 referents x {with-arm, no-arm} instructions. Split: seeds 3000000-3000079 train, 3000080+ val (last 20 val).
stack_blocks_two : from the 10-task cache's HOME-POSE frames (seq, fi==0, k==0; arms unmoved so the scene does not reveal the target). Per scene: original instruction (referent = block1/red first) + role-swapped instruction (referent = block2/green first); split = the 10-task split.
Output cf_cache/{task}.pt : {"scenes": {key: {"ids","feat"[K,64,960 fp16],"split","names"}}, "samples": [{"scene","variant","instruction","txt","t","cyc","pair"}]}
usage: build_cf_cache.py TASK"""
import sys, os, glob, pickle, numpy as np, torch, cv2
from regsp import paths as P
W = str(P.clean_root()); OUT = str(P.work_root())

from regsp.backbone.loader import load_policy, make_batch
from regsp.envs import cameras as C, v21rt as R
t = sys.argv[1]; MIN_SEP = 0.03; pol, pre = load_policy(); ve = pol.backbone.model.vlm_with_expert
def crop_tok(img_half, u, v):
    h = R.CROP_PX // 4; uu, vv = int(round(min(max(u, 0), 639) / 2)), int(round(min(max(v, 0), 479) / 2)); pad = np.pad(img_half, ((h, h), (h, h), (0, 0)), mode="edge"); crop = pad[vv: vv + 2 * h, uu: uu + 2 * h]
    x = torch.from_numpy(cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).float().div(255)[None].cuda() * 2 - 1
    with torch.no_grad(): return ve.embed_image(x)[0].float().cpu().half()
def text_states(imgs_half, instruction):
    b = pre(make_batch({"imgs224": [R.sq(i) for i in imgs_half], "state": np.zeros(7, np.float32)}, instruction))
    with torch.no_grad(): tok, pad, mod = pol._encode_multimodal_latent(b)
    m = (mod[0] == 1) & pad[0].bool(); return tok[0][m].half().cpu()
def head_candidates(objs, E):  # same validity rule as build_v21_cache_10.py (head camera in frame, depth>0.05, not within 3 cm of another object)
    ob = []
    for o in objs:
        if o["attr"] == "last_actor": continue
        if any(np.linalg.norm(np.array(o["xyz"]) - np.array(q["xyz"])) < 0.005 for q in ob): continue
        ob.append(o)
    xyz = np.array([o["xyz"] for o in ob]); D_ = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1) + np.eye(len(ob)) * 9; amb = D_.min(1) < MIN_SEP
    uv = []
    for o in ob: (u, v), d = R.project(E["head_camera"], o["xyz"]); uv.append((float(u), float(v), d, bool(d > 0.05 and 8 <= u < 632 and 8 <= v < 472)))
    return ob, uv, amb
scenes = {}; samples = []
if t == "click_bell":
    from generate_episode_instructions import generate_episode_descriptions  # noqa: F401
    for f in sorted(glob.glob(f"{OUT}/cf_scenes/click_bell*/cf*.pkl")):
        p = pickle.load(open(f, "rb")); z = np.load(f.replace(".pkl", ".npz")); imgs = [z[c] for c in R.CAMS]; E = {c: p["E"][c] for c in R.CAMS}
        ob, uv, amb = head_candidates(p["objects"], E); ids = [i for i, o in enumerate(ob) if uv[i][3] and not amb[i]]; attrs = [ob[i]["attr"] for i in ids]
        if "bell" not in [ob[i]["attr"] for i in ids] or "bell2" not in [ob[i]["attr"] for i in ids]: continue  # both referents must be valid candidates
        key = f"cf{p['seed']}"; split = "val" if p["seed"] >= 3000080 else "train"
        scenes[key] = {"ids": ids, "feat": torch.stack([crop_tok(imgs[0], uv[i][0], uv[i][1]) for i in ids]), "split": split, "names": attrs, "models": {b["attr"]: b["model"] for b in p["bells"]}}
        for b in p["bells"]:
            for var in ("arm", "noarm"):
                ins = b["instr_" + var]; samples.append({"scene": key, "variant": var, "instruction": ins, "txt": text_states(imgs, ins), "t": attrs.index(b["attr"]), "cyc": 0, "pair": key, "referent": b["attr"], "model": b["model"]})
elif t == "stack_blocks_two":
    from generate_episode_instructions import generate_episode_descriptions
    d = torch.load(f"{W}/v21cache/{t}.pt", weights_only=False, map_location="cpu"); FR = d["frames"]; n = 0
    for f in FR:
        if not (f["kind"] == "seq" and f["fi"] == 0 and f["k"] == 0 and f["ref"] and f["ref"]["idx"] is not None): continue
        ids = sorted(f["crops_head"]); attrs = [f["objs"][i]["attr"] for i in ids]
        if "block1" not in attrs or "block2" not in attrs: continue
        key = f"s{f['scene']}"; rng = np.random.default_rng(f["scene"] + 5); arm = lambda a: "left" if f["objs"][[o["attr"] for o in f["objs"]].index(a)]["xyz"][0] < 0 else "right"
        sw = generate_episode_descriptions(t, [{"{A}": "green block", "{B}": "red block", "{a}": arm("block2"), "{b}": arm("block1")}], 30)[0]["seen"]; ins_sw = str(sw[rng.integers(len(sw))])
        scenes[key] = {"ids": ids, "feat": torch.stack([f["crops_head"][i] for i in ids]), "split": f["split"], "names": attrs, "orig_row": None}
        imgs = [np.ascontiguousarray(x) for x in f["imgs_half"]]
        samples.append({"scene": key, "variant": "orig", "instruction": f["instruction"], "txt": f["txt"].half(), "t": attrs.index("block1"), "cyc": 0, "pair": key, "referent": "block1"})
        samples.append({"scene": key, "variant": "swap", "instruction": ins_sw, "txt": text_states(imgs, ins_sw), "t": attrs.index("block2"), "cyc": 0, "pair": key, "referent": "block2"}); n += 1
    del d, FR
os.makedirs(OUT + "/cf_cache", exist_ok=True); torch.save({"scenes": scenes, "samples": samples, "task": t}, f"{OUT}/cf_cache/{t}.pt")
import collections; print(t, "scenes", len(scenes), "samples", len(samples), collections.Counter(s["split"] for s in scenes.values()), collections.Counter(s["variant"] for s in samples), flush=True)
