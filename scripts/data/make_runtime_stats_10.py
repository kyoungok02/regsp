"""Per-task runtime statistics from TRAIN labels/scenes (learned, not hard-coded): arm selector (logistic on referent robot-frame xy + previous arm), referent height prior per cycle index, max cycles, max post-grasp chunks, release-deferral rule (fraction of cycles whose release is after the next cycle's close). usage: make_runtime_stats_10.py TASK"""
import sys, os, glob, json, pickle, numpy as np
from regsp import paths as P
from regsp.execution import kinematics as rt_kin
from sklearn.linear_model import LogisticRegression
OUT = str(P.clean_root()); t = sys.argv[1]; To = rt_kin.ArmFK("left").T_origin; Ti = np.linalg.inv(To)
L = pickle.load(open(f"{OUT}/labels/{t}.pkl", "rb")); acc = [v for v in L.values() if not v["rejected"] and v["split"] == "train"]
ncyc = [sum(len(v["cycles"][a]) for a in "LR") for v in acc]; maxcyc = int(np.ceil(np.percentile(ncyc, 95))); chunks = []; over = []
for v in acc:
    cs = sorted([(c["tc"], a, c) for a in "LR" for c in v["cycles"][a]], key=lambda x: x[0])
    for i, (tc, a, c) in enumerate(cs):
        chunks.append((c["tp"] - c["tc"]) / 50.0)
        if i + 1 < len(cs): over.append(float(c["tp"] > cs[i + 1][0]))
maxpost = int(np.ceil(np.percentile(chunks, 95))) + 2; defer = bool(over and np.mean(over) >= 0.5)
X, Y, Z, X2 = [], [], {}, []
for f in sorted(glob.glob(f"{OUT}/scenes/{t}/scene*.pkl")):
    p = pickle.load(open(f, "rb")); ob = {o["attr"]: o["xyz"] for o in p["objects_natural"]}; prev = None
    for k, (arm, attr) in enumerate(p["refs"]):
        if attr not in ob: continue
        xr = Ti[:3, :3] @ np.array(ob[attr]) + Ti[:3, 3]; oth = [Ti[:3, :3] @ np.array(v) + Ti[:3, 3] for kk, v in ob.items() if kk != attr and kk != "last_actor"]; on = min(oth, key=lambda q: np.linalg.norm(q - xr)) if oth else xr; X.append([xr[0], xr[1], float(prev == "L"), float(prev == "R")]); X2.append([xr[0], xr[1], float(prev == "L"), float(prev == "R"), on[0], on[1]]); Y.append(int(arm == "R")); Z.setdefault(k, []).append(ob[attr][2]); prev = arm
X, Y, X2 = np.array(X), np.array(Y), np.array(X2); nv = max(1, len(X) // 5); va1 = float(LogisticRegression(C=100.0, max_iter=2000).fit(X[:-nv], Y[:-nv]).score(X[-nv:], Y[-nv:])); va2 = float(LogisticRegression(C=100.0, max_iter=2000).fit(X2[:-nv], Y[:-nv]).score(X2[-nv:], Y[-nv:]))
use_other = bool(va2 >= va1 + 0.1); XX = X2 if use_other else X; va = va2 if use_other else va1; lr = LogisticRegression(C=100.0, max_iter=2000).fit(XX, Y)
S = {"task": t, "arm_selector": {"w": lr.coef_[0].tolist(), "b": float(lr.intercept_[0]), "n": len(X), "use_other_object_feature": use_other, "val_acc_holdout": va, "val_acc_referent_only": va1, "val_acc_with_other": va2, "majority_baseline": float(max(Y.mean(), 1 - Y.mean()))}, "referent_height_world": {str(k): float(np.median(v)) for k, v in Z.items()}, "max_cycles": maxcyc, "max_post_chunks": maxpost, "defer_release": defer, "defer_fraction": float(np.mean(over)) if over else None, "cycles_per_episode_p50_p95": [float(np.median(ncyc)), float(np.percentile(ncyc, 95))]}
os.makedirs(f"{OUT}/ckpt/{t}", exist_ok=True); json.dump(S, open(f"{OUT}/ckpt/{t}/runtime_stats.json", "w"), indent=1); print(t, json.dumps({k: v for k, v in S.items() if k != "referent_height_world"}))
