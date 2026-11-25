"""[bim_stats_10] Data-derived runtime statistics for the bimanual extension (from validated EXPERT demo labels + place-pair captures; no hand-set task constants):
 place table per (global cycle k, arm): median expert TCP-at-release xyz (robot frame) and per-axis std; mode 'const' if max std <= CONST_STD (1.5 cm) else 'grounded' (needs the V21 place referent);
 stable-grasp threshold: p95 of the gripper STATE value during expert closed-hold (>=10 steps after a close) + 0.02;
 place-referent calibration (grounded mode, from place_pairs): referent height (robot frame z) and median offset (TCP-at-release - referent xyz), with std.
usage: bim_stats_10.py TASK [PLACE_ATTR] [PAIRS_DIR]"""
import sys, os, json, pickle, glob, numpy as np
from regsp import paths as P
from regsp.execution import kinematics as rt_kin
OUT = str(P.clean_root()); t = sys.argv[1]; PA = sys.argv[2] if len(sys.argv) > 2 else None; PD = sys.argv[3] if len(sys.argv) > 3 else None; CONST_STD = 0.015
To = rt_kin.ArmFK("left").T_origin; Ti = np.linalg.inv(To)
L = pickle.load(open(f"{OUT}/labels/{t}.pkl", "rb")); acc = [v for v in L.values() if not v["rejected"] and v["split"] == "train"]; rows = {}; hold = []; grows = {}
for v in acc:
    ev = sorted([(c["tc"], s, c) for s in "LR" for c in v["cycles"][s]], key=lambda x: x[0])
    for k, (tc, s, c) in enumerate(ev):
        rows.setdefault((min(k, 1), s), []).append(c["place_xyz"]); grows.setdefault((min(k, 1), s), []).append(c["grasp_xyz"]); g = v["state"][:, 6 if s == "L" else 13]; end = c["tp"]
        if end - tc > 12: hold += list(g[tc + 10:end])
tab = {f"{k}_{s}": {"median": np.median(np.array(p), 0).tolist(), "std": np.array(p).std(0).tolist(), "min": np.array(p).min(0).tolist(), "max": np.array(p).max(0).tolist(), "n": len(p), "mode": "const" if np.array(p).std(0).max() <= CONST_STD else "grounded"} for (k, s), p in sorted(rows.items())}
gtab = {f"{k}_{s}": {"median": np.median(np.array(p), 0).tolist(), "std": np.array(p).std(0).tolist(), "n": len(p), "grasp_mode": "const" if np.array(p).std(0).max() <= CONST_STD else "varies"} for (k, s), p in sorted(grows.items())}
S = {"task": t, "place": tab, "grasp": gtab, "stable_g_max": float(np.percentile(hold, 95) + 0.02), "hold_state_percentiles(50,95,99)": np.percentile(hold, [50, 95, 99]).tolist(), "const_std_threshold_m": CONST_STD}
if PA and PD:
    offs, zs = [], []
    for f in sorted(glob.glob(f"{PD}/pair*.json")):
        d = json.load(open(f))
        if PA not in d["init_objs"] or not d["releases"]: continue
        pad = np.array(d["init_objs"][PA]); rel = d["releases"][-1]; r = Ti[:3, :3] @ np.array(rel["tcp_world"]) + Ti[:3, 3]; p = Ti[:3, :3] @ pad + Ti[:3, 3]; offs.append(r - p); zs.append(p)
    offs = np.array(offs); zs = np.array(zs); S["place_referent"] = {"attr": PA, "n_pairs": len(offs), "offset_median": np.median(offs, 0).tolist(), "offset_std": offs.std(0).tolist(), "referent_robot_xyz_median": np.median(zs, 0).tolist(), "referent_robot_xyz_std": zs.std(0).tolist(), "referent_height": float(np.median(zs[:, 2]))}
os.makedirs(f"{OUT}/ckpt_bim/{t}", exist_ok=True); json.dump(S, open(f"{OUT}/ckpt_bim/{t}/bim_stats.json", "w"), indent=1); print(json.dumps(S, indent=1))
