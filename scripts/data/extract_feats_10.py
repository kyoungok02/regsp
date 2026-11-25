"""Frozen-SigLIP patch features (the reused SafeDiff-VLA visual module) for ONE task: every 3rd frame before that episode's LAST close event, 3 cameras, accepted Clean demos only. usage: extract_feats_10.py TASK
Output (per task, never shared): feats/{task}_{cam}.npy fp16 [N,768,32,32] + feats/{task}_{cam}.index.json [(ep,t)...]"""
import sys, os, json, pickle, numpy as np, pandas as pd, av, torch
from regsp import paths as P
from regsp.envs import cameras as C
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); H = str(P.clean_data_root()); STRIDE = 3; t = sys.argv[1]
L = pickle.load(open(f"{OUT}/labels/{t}.pkl", "rb")); acc = {k[1]: v for k, v in L.items() if not v["rejected"]}; os.makedirs(OUT + "/feats", exist_ok=True)
want = {ep: list(range(0, max(c["tc"] for s in "LR" for c in r["cycles"][s]), STRIDE)) for ep, r in acc.items()}
b = f"{H}/{t}/aloha-agilex_clean_50"; e = pd.read_parquet(b + "/meta/episodes/chunk-000/file-000.parquet"); pol = C.load_backbone(); vm = pol.backbone.model.vlm_with_expert.get_vlm_model().vision_model
for cam in C.CAMS:
    if os.path.exists(f"{OUT}/feats/{t}_{cam}.index.json"): continue
    ct = av.open(f"{b}/videos/observation.images.{cam}/chunk-000/file-000.mp4"); t0 = {int(x.episode_index): float(x[f"videos/observation.images.{cam}/from_timestamp"]) for _, x in e.iterrows()}; need = {}
    for ep, ts in want.items():
        for i in ts: need[int(round((t0[ep] + i / 30) * 30))] = (ep, i)
    got = {}
    for fr in ct.decode(video=0):
        gi = int(round(fr.time * 30))
        if gi in need: got[need[gi]] = fr.to_ndarray(format="rgb24")
        if len(got) == len(need): break
    assert len(got) == len(need), (t, cam, len(got), len(need)); keys = sorted(got)
    Fe = torch.cat([C.patch_feats(vm, [got[k] for k in keys[i:i + 32]]).cpu() for i in range(0, len(keys), 32)]).numpy().astype(np.float16)
    np.save(f"{OUT}/feats/{t}_{cam}.npy", Fe); json.dump(keys, open(f"{OUT}/feats/{t}_{cam}.index.json", "w")); print(t, cam, Fe.shape, flush=True)
