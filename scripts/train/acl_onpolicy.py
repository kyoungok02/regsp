"""[V21-ACL] TRAINING-ONLY label builder for on-policy progress samples (ProgressHead; classes NOT_STARTED 0 / IN_PROGRESS 1 / COMPLETED 2 / FAILED_OR_RECOVERY 3 / TASK_COMPLETED 4).
Input : feature files written by acl_rt_policy.py with ACL_SAVE_FEATS on TRAINING-DOMAIN seeds (optionally with ACL_PERTURB so that real failures and the executor's real recoveries occur).
Labels: derived from the simulator state STORED AT THE TIME OF EACH SAMPLE (`priv_training_only`: object poses, env.check_success) = privileged supervision used for training only. Nothing is fabricated: a sample is FAILED only if the
        physical outcome at that moment was a failure, and IN_PROGRESS / COMPLETED only if it physically was. The ProgressHead never receives these values as inputs (inputs = backbone tokens, both-arm proprioception, target/place, history).
Rules (generic; no task-specific action sequence):
  held(arm)      = measured finger opening inside the calibrated hold band AND a movable object's body axis within 4.5 cm of that arm's TCP (privileged object pose)
  chk_grasp / post* (carry, receive) / chk_transfer, queried stage m = current stage : held(primary arm) -> IN_PROGRESS else FAILED_OR_RECOVERY
  chk_release, last stage      : env.check_success at that moment (or at episode end when the episode ended with this stage) -> TASK_COMPLETED else FAILED_OR_RECOVERY
  chk_release, earlier stage   : the released object rests within 4 cm (horizontal) / 5 cm (vertical) of the TCP position at which it was released (first-round files without that field: of the place target) and is not held -> COMPLETED else FAILED_OR_RECOVERY
  retreat* samples are not labelled.
usage: acl_onpolicy.py TASK RAW_DIR [RAW_DIR ...]  -> onpolicy/labelled/<TASK>/<cfg>/feat<seed>.pt + onpolicy/labelled/<TASK>/label_report.json"""
import sys, os, json, glob, collections, numpy as np, torch
from regsp import paths as P
N = str(P.work_root()); t = sys.argv[1]; cal = json.load(open(N + "/acl_calib.json"))[t]; MOV = {"box": 0.10, "block1": 0.0, "block2": 0.0}; NST = 2
def axis_dist(p, c, Rm, L):
    if not L or Rm is None: return float(np.linalg.norm(p - c))
    a = np.array(Rm)[:, 2]; s = float(np.clip((p - c) @ a, -L, L)); return float(np.linalg.norm(p - (c + s * a)))
def held(o, a):
    pv = o["priv_training_only"]; f = o["finger"][a]; tcp = np.array(o["tcp"][a])
    return bool(cal["thr"] < f < cal["open_min"]) and any(axis_dist(tcp, np.array(pv[k]), pv.get(k + "_R"), L) < 0.045 for k, L in MOV.items() if k in pv)
rep = collections.Counter(); per_ep = []; nfiles = 0
for raw in sys.argv[2:]:
    for f in sorted(glob.glob(raw + "/feat*.pt")):
        F = torch.load(f, weights_only=False); out = []; nfiles += 1
        for i, o in enumerate(F):
            tag = o["tag"]; a = o["arm"]; cyc = o["cyc"]; pv = o["priv_training_only"]; lab = None
            if tag.startswith("retreat"): rep["unlabelled_retreat"] += 1; continue
            if tag in ("chk_grasp", "chk_transfer") or tag.startswith("post"): lab = 1 if held(o, a) else 3
            elif tag == "chk_release":
                if cyc >= NST - 1:
                    ok = bool(pv.get("check_success")) or (i == len(F) - 1 and bool(o["final_priv_training_only"].get("check_success"))); lab = 4 if ok else 3
                else:
                    rp = np.array(o["release_tcp"] if o.get("release_tcp") is not None else o["place"]); near = any(np.linalg.norm((np.array(pv[k]) - rp)[:2]) < 0.04 and abs(float(np.array(pv[k])[2] - rp[2])) < 0.05 for k in MOV if k in pv); lab = 2 if (near and not held(o, a)) else 3
            if lab is None: continue
            kind = "carry" if tag.startswith("post") else tag[4:]; rep[f"{o['cfg']}|{kind}|{lab}"] += 1
            out.append({"tok": o["tok"], "pad": o["pad"], "sP": o["sP"], "sQ": o["sQ"], "mem": o["mem"], "hist": o["hist"], "m": min(cyc, 2), "label": lab, "cfg": o["cfg"], "seed": o["seed"], "kind": kind, "cyc": cyc, "call": o["call"]})
        if out:
            d = f"{N}/onpolicy/labelled/{t}/{out[0]['cfg']}"; os.makedirs(d, exist_ok=True); torch.save(out, f"{d}/{os.path.basename(f)}")
            per_ep.append({"seed": out[0]["seed"], "cfg": out[0]["cfg"], "n": len(out), "n_failed": sum(x["label"] == 3 for x in out), "recovered": bool(any(x["label"] == 3 for x in out) and out[-1]["label"] in (2, 4)), "final_success": bool(F[-1]["final_priv_training_only"].get("check_success"))})
R = {"task": t, "raw_dirs": sys.argv[2:], "episodes": len(per_ep), "samples_by_cfg|kind|label": dict(sorted(rep.items())), "label_totals": {str(l): sum(v for k, v in rep.items() if k.endswith(f"|{l}")) for l in (1, 2, 3, 4)},
     "episodes_with_real_failure": sum(e["n_failed"] > 0 for e in per_ep), "episodes_failure_then_success(real recovery)": sum(e["recovered"] and e["final_success"] for e in per_ep), "episodes_success": sum(e["final_success"] for e in per_ep)}
json.dump(R, open(f"{N}/onpolicy/labelled/{t}/label_report.json", "w"), indent=1); print(json.dumps(R, indent=1))
