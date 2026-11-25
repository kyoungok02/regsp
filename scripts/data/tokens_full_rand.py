"""[tokens_full_10] DECODER-TRAINING token cache: same interface as tokens_10 but ALL phases (pre and post grasp), every STRIDE=6th frame of the episode, plus the 50-step EE action-chunk targets (arm TCP xyz, euler of R@R_REF.T [numerical adaptation], gripper), valid mask, phase label (gripper channel), and the next-close grasp target. Original docstring: Backbone multimodal tokens (frozen SafeDiff backbone `_encode_multimodal_latent`: 3x64 image tokens + text tokens + state token; [N,201,960] fp16) for demo frames, per task, per ARM (arm-conditioned 7-D TCP state: xyz + euler + gripper, robot-root frame).
Images: head, left wrist, right wrist (= camera1/2/3 of the policy), letterboxed to square (resize-with-pad convention). Frames: every 6th frame before that arm's LAST close; target = that arm's next cycle (tc > t).
usage: tokens_10.py TASK -> tokens_full/{task}.pt"""
import sys, os, pickle, numpy as np, torch, cv2, av, pandas as pd
from regsp import paths as P

from regsp.backbone.loader import load_policy, make_batch
from regsp.execution import kinematics as rt_kin
from scipy.spatial.transform import Rotation as Rot
OUT = (str(P.work_root()) + "/rand"); H = (str(P.work_root()) + "/data_rand/hf"); t = sys.argv[1]; STRIDE = 6
L = pickle.load(open(f"{OUT}/labels/{t}.pkl", "rb")); acc = {k[1]: v for k, v in L.items() if not v["rejected"]}
def sq(img, s=224):  # letterbox to square then resize
    h, w = img.shape[:2]; m = max(h, w); pad = np.zeros((m, m, 3), np.uint8); pad[(m - h) // 2:(m - h) // 2 + h, (m - w) // 2:(m - w) // 2 + w] = img; return cv2.resize(pad, (s, s), interpolation=cv2.INTER_AREA)
want = {ep: list(range(0, r["len"] - 1, STRIDE)) for ep, r in acc.items()}
b = f"{H}/{t}/aloha-agilex_randomized_500"; e = pd.read_parquet(b + "/meta/episodes/chunk-000/file-000.parquet"); tk = pd.read_parquet(b + "/meta/tasks.parquet"); fr = {}
for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):  # [4-task study] randomized_500 stores several video files; each episode names its own file_index + from_timestamp
    need = {}
    for _, x in e.iterrows():
        ep = int(x.episode_index)
        if ep not in want: continue
        fi = int(x[f"videos/observation.images.{cam}/file_index"]); t0 = float(x[f"videos/observation.images.{cam}/from_timestamp"])
        for i in want[ep]: need.setdefault(fi, {})[int(round((t0 + i / 30) * 30))] = (ep, i)
    for fi, nd in sorted(need.items()):
        ct = av.open(f"{b}/videos/observation.images.{cam}/chunk-000/file-{fi:03d}.mp4"); cnt = 0
        for f in ct.decode(video=0):
            gi = int(round(f.time * 30))
            if gi in nd: fr[(cam,) + nd[gi]] = sq(f.to_ndarray(format="rgb24")); cnt += 1
            if cnt == len(nd): break
        assert cnt == len(nd), (t, cam, fi, cnt, len(nd))
pol, pre = load_policy(); ent = []; RREF = Rot.from_euler("y", 90, degrees=True).as_matrix(); HZ = 50
def eul_ref(Rm): return Rot.from_matrix(Rm @ RREF.T).as_euler("xyz")
for ep, r in acc.items():
    for tt in want[ep]:
        for a in r["active"]:
            Tn = r[f"tcp_{a}"]; g = r["action"][:, (0 if a == "L" else 7) + 6]; T = Tn[tt]; eul = Rot.from_matrix(T[:3, :3]).as_euler("xyz"); st = np.r_[T[:3, 3], eul, g[tt]].astype(np.float32)
            ix = np.minimum(np.arange(tt + 1, tt + 1 + HZ), len(Tn) - 1); valid = (np.arange(tt + 1, tt + 1 + HZ) < len(Tn)); chunk = np.concatenate([Tn[ix][:, :3, 3], np.stack([eul_ref(m) for m in Tn[ix][:, :3, :3]]), g[ix][:, None]], 1).astype(np.float32)
            sref = np.r_[T[:3, 3], eul_ref(T[:3, :3]), g[tt]].astype(np.float32); nxt = [c for c in r["cycles"][a] if c["tc"] > tt]; c = nxt[0] if nxt else None
            bb = pre(make_batch({"imgs224": [fr[("cam_high", ep, tt)], fr[("cam_left_wrist", ep, tt)], fr[("cam_right_wrist", ep, tt)]], "state": st}, r["instruction"]))
            with torch.no_grad(): tok, pad, mod = pol._encode_multimodal_latent(bb)
            n = tok.shape[1]; Tk = torch.zeros(256, tok.shape[2], dtype=torch.float16); Tk[:n] = tok[0].half().cpu(); P = torch.zeros(256, dtype=torch.bool); P[:n] = pad[0].cpu().bool(); M = torch.full((256,), 2, dtype=torch.long); M[:n] = mod[0].cpu()
            ent.append({"ep": ep, "t": tt, "arm": a, "split": r["split"], "tok": Tk, "pad": P, "mod": M, "state_ref": sref, "chunk": chunk, "valid": valid, "phase": int(g[tt] < 0.5), "next_close_in": (None if c is None else int(c["tc"] - tt)), "grasp_xyz": None if c is None else c["grasp_xyz"].astype(np.float32), "instruction": r["instruction"]})
os.makedirs(OUT + "/tokens_full", exist_ok=True); torch.save(ent, f"{OUT}/tokens_full/{t}.pt"); print(t, "samples", len(ent), "train", sum(x["split"] == "train" for x in ent), "val", sum(x["split"] == "val" for x in ent))
