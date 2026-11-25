"""Backbone multimodal tokens (frozen SafeDiff backbone `_encode_multimodal_latent`: 3x64 image tokens + text tokens + state token; [N,201,960] fp16) for demo frames, per task, per ARM (arm-conditioned 7-D TCP state: xyz + euler + gripper, robot-root frame).
Images: head, left wrist, right wrist (= camera1/2/3 of the policy), letterboxed to square (resize-with-pad convention). Frames: every 6th frame before that arm's LAST close; target = that arm's next cycle (tc > t).
usage: tokens_10.py TASK -> tokens/{task}.pt"""
import sys, os, pickle, numpy as np, torch, cv2, av, pandas as pd
from regsp import paths as P

from regsp.backbone.loader import load_policy, make_batch
from regsp.execution import kinematics as rt_kin
from scipy.spatial.transform import Rotation as Rot
OUT = str(P.clean_root()); H = str(P.clean_data_root()); t = sys.argv[1]; STRIDE = 6
L = pickle.load(open(f"{OUT}/labels/{t}.pkl", "rb")); acc = {k[1]: v for k, v in L.items() if not v["rejected"]}
def sq(img, s=224):  # letterbox to square then resize
    h, w = img.shape[:2]; m = max(h, w); pad = np.zeros((m, m, 3), np.uint8); pad[(m - h) // 2:(m - h) // 2 + h, (m - w) // 2:(m - w) // 2 + w] = img; return cv2.resize(pad, (s, s), interpolation=cv2.INTER_AREA)
want = {ep: sorted({tt for a in "LR" for c in r["cycles"][a] for tt in range(0, c["tc"], STRIDE)}) for ep, r in acc.items()}
b = f"{H}/{t}/aloha-agilex_clean_50"; e = pd.read_parquet(b + "/meta/episodes/chunk-000/file-000.parquet"); tk = pd.read_parquet(b + "/meta/tasks.parquet"); fr = {}
for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
    ct = av.open(f"{b}/videos/observation.images.{cam}/chunk-000/file-000.mp4"); t0 = {int(x.episode_index): float(x[f"videos/observation.images.{cam}/from_timestamp"]) for _, x in e.iterrows()}; need = {}
    for ep, ts in want.items():
        for i in ts: need[int(round((t0[ep] + i / 30) * 30))] = (ep, i)
    for f in ct.decode(video=0):
        gi = int(round(f.time * 30))
        if gi in need: fr[(cam,) + need[gi]] = sq(f.to_ndarray(format="rgb24"))
        if sum(1 for k in fr if k[0] == cam) == len(need): break
pol, pre = load_policy(); ent = []
for ep, r in acc.items():
    for tt in want[ep]:
        for a in "LR":
            nxt = [c for c in r["cycles"][a] if c["tc"] > tt]
            if not nxt: continue
            c = nxt[0]; T = r[f"tcp_{a}"][tt]; eul = Rot.from_matrix(T[:3, :3]).as_euler("xyz"); st = np.r_[T[:3, 3], eul, r["action"][tt][(0 if a == "L" else 7) + 6]].astype(np.float32)
            bb = pre(make_batch({"imgs224": [fr[("cam_high", ep, tt)], fr[("cam_left_wrist", ep, tt)], fr[("cam_right_wrist", ep, tt)]], "state": st}, r["instruction"]))
            with torch.no_grad(): tok, pad, mod = pol._encode_multimodal_latent(bb)
            n = tok.shape[1]; Tk = torch.zeros(256, tok.shape[2], dtype=torch.float16); Tk[:n] = tok[0].half().cpu(); P = torch.zeros(256, dtype=torch.bool); P[:n] = pad[0].cpu().bool(); M = torch.full((256,), 2, dtype=torch.long); M[:n] = mod[0].cpu()
            ent.append({"ep": ep, "t": tt, "arm": a, "split": r["split"], "tok": Tk, "pad": P, "mod": M, "state": pol._current_state(bb)[0].cpu(), "cycle_idx": r["cycles"][a].index(c), "grasp_xyz": c["grasp_xyz"].astype(np.float32), "grasp_euler": c["grasp_euler"].astype(np.float32), "approach": c["approach_dir"].astype(np.float32), "pre_dist": np.float32(c["pre_dist"]), "wp_rel": c["wp_rel"].astype(np.float32), "instruction": r["instruction"]})
os.makedirs(OUT + "/tokens", exist_ok=True); torch.save(ent, f"{OUT}/tokens/{t}.pt"); print(t, "samples", len(ent), "train", sum(x["split"] == "train" for x in ent), "val", sum(x["split"] == "val" for x in ent))
