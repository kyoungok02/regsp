"""Per-task StagePlan-v2 labels for the 10 selected RoboTwin tasks from the pinned dataset (hxma/RoboTwin-LeRobot-v3.0 @5d857cc7), clean_50 ONLY. No simulator state, no randomized data.
Semantics = D's build_labels.py = VLABench stageplan_v2_labels (P1 = commanded EE pose immediately before CLOSE; P0 = last RDP corner >= 5 cm back; approach = unit(P1-P0); pre_dist = |P1-P0|; wp_rel = first pose within 25 cm of P1),
computed from validated FK of the recorded joint ACTIONS (rt_kin, audit/fk_ik_validation.json). Frame = robot root; rotation 3x3 + euler.
EXTENSION (documented, not in original/D): MULTI-CYCLE. D labels only the first close of each arm. Here every close event k of each arm (gripper open->closed transition, closed >= MIN_CLOSE steps) is a StagePlan cycle with its own P1/P0/approach/pre_dist/wp_rel,
a release time (next reopen) and a place xyz (TCP at release). P0 for cycle k>0 is searched only on the path since the previous release (a new approach).
Split: episode_index % 5 == 4 -> val (same rule as D). Per-episode validation rejects episodes (counts reported): missing close, workspace violation, non-orthonormal rotation, non-finite values, pre_dist outside [0.02, 0.40], cycles with release before close.
Output: labels/{task}.pkl (one file per task: tasks are never mixed) and audit/label_validation_10.json."""
import sys, json, pickle, os, numpy as np, pandas as pd
from regsp import paths as P

from regsp.execution import kinematics as rt_kin
from scipy.spatial.transform import Rotation as Rot
H = (str(P.work_root()) + "/data_rand/hf"); OUT = (str(P.work_root()) + "/rand"); NEP = int(os.environ.get("NEP", 100))
TASKS = [sys.argv[1]]; WP_R = 0.25; CORNER_EPS = 0.01; MIN_ARC = 0.05; MIN_CLOSE = 3
fk = {"L": rt_kin.ArmFK("left"), "R": rt_kin.ArmFK("right")}; To = fk["L"].T_origin; Ti = np.linalg.inv(To)
def rdp(P, eps):
    if len(P) < 3: return list(range(len(P)))
    a, b = P[0], P[-1]; d = b - a; n = np.linalg.norm(d); dist = np.linalg.norm(np.cross(P - a, d / n), axis=1) if n > 1e-9 else np.linalg.norm(P - a, axis=1); i = int(np.argmax(dist))
    if dist[i] <= eps: return [0, len(P) - 1]
    l, r = rdp(P[: i + 1], eps), rdp(P[i:], eps); return l[:-1] + [x + i for x in r]
def close_events(g):  # open(>=.5) -> closed(<.5) transitions that stay closed >= MIN_CLOSE steps; returns [(tc, release_t or None)]
    c = g < 0.5; ev = []; t = 0; n = len(g)
    while t < n:
        if c[t] and (t == 0 or not c[t - 1]):
            e = t
            while e < n and c[e]: e += 1
            if e - t >= MIN_CLOSE or e == n: ev.append((t, e if e < n else None))
            t = e
        else: t += 1
    return ev
def cycle(P, g, A_len, tc, rel, t_prev):
    t1 = max(tc - 1, t_prev + 1, 1); p1 = P[t1]; xyz = P[t_prev:t1 + 1, :3, 3]; s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]; v = rdp(xyz, CORNER_EPS)
    back = s[-1] - s[v]; ok = [j for j, b in zip(v, back) if b >= MIN_ARC]; j = ok[-1] if ok else 0; p0 = xyz[j]; dv = p1[:3, 3] - p0; r = float(np.linalg.norm(dv))
    d = np.linalg.norm(xyz - p1[:3, 3], axis=1); i = int(np.argmax(d <= WP_R)) if d[0] > WP_R else 0; tp = rel if rel is not None else A_len - 1
    return {"tc": int(tc), "t1": int(t1), "tp": int(tp), "grasp_xyz": p1[:3, 3].copy(), "grasp_R": p1[:3, :3].copy(), "grasp_euler": Rot.from_matrix(p1[:3, :3]).as_euler("xyz"), "approach_dir": dv / max(r, 1e-9), "pre_dist": r, "wp_rel": xyz[i] - p1[:3, 3], "place_xyz": P[tp][:3, 3].copy(), "tool_x": p1[:3, 0].copy(), "released": rel is not None}
def validate(c):
    R = c["grasp_R"]; why = []
    if not np.all(np.isfinite(c["grasp_xyz"])) or not np.all(np.isfinite(R)): why.append("non_finite")
    if np.abs(R @ R.T - np.eye(3)).max() > 1e-3 or abs(np.linalg.det(R) - 1) > 1e-3: why.append("rotation_not_orthonormal")
    x, y, z = c["grasp_xyz"]
    if not (-0.9 < x < 0.9 and -0.9 < y < 0.9 and -0.2 < z < 1.2): why.append("workspace")
    if not (0.02 <= c["pre_dist"] <= 0.40): why.append("pre_dist_range")
    if c["released"] and c["tp"] <= c["tc"]: why.append("release_before_close")
    return why
if __name__ == "__main__":
    os.makedirs(OUT + "/labels", exist_ok=True); os.makedirs(OUT + "/audit", exist_ok=True); rep = {}
    for t in TASKS:
        b = f"{H}/{t}/aloha-agilex_randomized_500"; d = pd.read_parquet(b + "/data/chunk-000/file-000.parquet"); tk = pd.read_parquet(b + "/meta/tasks.parquet"); out = {}; rej = {}; ncyc = {}
        for ep, gdf in d.groupby("episode_index"):
            if ep >= NEP: continue
            A = np.stack(gdf["action"]).astype(np.float64); S = np.stack(gdf["observation.state"]).astype(np.float64); ti = int(gdf["task_index"].iloc[0]); instr = tk.index[tk.task_index == ti][0]
            rec = {"task": t, "episode": int(ep), "len": len(A), "instruction": instr, "split": "val" if ep % 5 == 4 else "train", "cycles": {"L": [], "R": []}}; why = []
            for side, sl in (("L", slice(0, 7)), ("R", slice(7, 14))):
                Aa = A[:, sl]; P = np.stack([Ti @ fk[side].tcp(a[:6], "world") for a in Aa]); rec[f"tcp_{side}"] = P.astype(np.float32); prev = 0
                for tc, rel in close_events(Aa[:, 6]):
                    c = cycle(P, Aa[:, 6], len(Aa), tc, rel, prev); w = validate(c); why += [f"{side}:{x}" for x in w]; rec["cycles"][side].append(c); prev = rel if rel is not None else len(Aa) - 1
            rec["active"] = [s for s in "LR" if rec["cycles"][s]]
            if not rec["active"]: why.append("no_close_event")
            rec["state"] = S.astype(np.float32); rec["action"] = A.astype(np.float32); rec["rejected"] = why
            if why:
                for w_ in set(x.split(":")[-1] for x in why): rej[w_] = rej.get(w_, 0) + 1
            out[(t, int(ep))] = rec
            if not why: ncyc["".join(f"{s}{len(rec['cycles'][s])}" for s in rec["active"])] = ncyc.get("".join(f"{s}{len(rec['cycles'][s])}" for s in rec["active"]), 0) + 1
        acc = [v for v in out.values() if not v["rejected"]]; pickle.dump(out, open(f"{OUT}/labels/{t}.pkl", "wb"))
        pd_ = [c["pre_dist"] for v in acc for s in "LR" for c in v["cycles"][s]]
        rep[t] = {"episodes": len(out), "accepted": len(acc), "rejected": len(out) - len(acc), "reject_reasons": rej, "train_accepted": sum(v["split"] == "train" for v in acc), "val_accepted": sum(v["split"] == "val" for v in acc), "cycle_signature_counts(arm+n_cycles)": ncyc,
                  "pre_dist_median_p10_p90_m": np.percentile(pd_, [50, 10, 90]).round(3).tolist() if pd_ else None, "n_cycles_total": len(pd_)}
        print(t, json.dumps(rep[t]), flush=True)
    json.dump(rep, open(OUT + f"/audit/label_validation_{sys.argv[1]}.json", "w"), indent=1)
