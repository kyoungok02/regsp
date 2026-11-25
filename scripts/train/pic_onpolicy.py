"""[V21-PIC] TRAINING-ONLY label builder for INSTRUCTION-CONDITIONED progress samples (PicProgressHead; classes NOT_STARTED 0 / IN_PROGRESS 1 / COMPLETED 2 / FAILED_OR_RECOVERY 3 / TASK_COMPLETED 4).
Input : feature files written by pic_rt_policy.py (ACL_SAVE_FEATS) on TRAINING-DOMAIN seeds; stack_blocks_two rollouts under the original (PIC_LANG=orig) and the role-exchanged (PIC_LANG=cf) instruction; every progress-check sample also
        carries the tokens of the SAME observation encoded with the other valid instruction (`tok_alt`).
Labels: privileged simulator state stored at the time of the sample (training only; nothing fabricated), judged AGAINST THE INSTRUCTION that the tokens were encoded with:
  required block of stage m under an instruction = the block that the instruction places first (m=0) / second (m=1)
  grasp / carry / transfer : the REQUIRED block is physically held by the primary arm -> IN_PROGRESS ; anything else (nothing held, or the OTHER block held = wrong referent) -> FAILED_OR_RECOVERY
  release, stage 0         : the REQUIRED block rests within 4 cm / 5 cm of the release point and is not held -> COMPLETED else FAILED_OR_RECOVERY
  release, last stage      : the goal relation OF THAT INSTRUCTION holds (original: env.check_success ; exchanged: the same test with the two blocks exchanged, same tolerances) -> TASK_COMPLETED else FAILED_OR_RECOVERY
  => the same observation gets different labels under the two instructions (e.g. green block grasped first: IN_PROGRESS under the exchanged instruction, FAILED_OR_RECOVERY under the original one).
handover_block has no semantically valid alternative instruction: the unchanged acl_onpolicy.py rules are applied (no `alt` samples).
usage: pic_onpolicy.py TASK OUT_TAG RAW_DIR [RAW_DIR ...]  -> onpolicy/pic_labelled_<OUT_TAG>/<TASK>/feat_<mode>_<seed>.pt + label_report.json"""
import sys, os, json, glob, collections, numpy as np, torch
from regsp import paths as P
N = str(P.work_root()); t = sys.argv[1]; OT = sys.argv[2]; cal = json.load(open(N + "/acl_calib.json"))[t]; MOV = {"box": 0.10, "block1": 0.0, "block2": 0.0}; NST = 2; STACK = t == "stack_blocks_two"; EPS = np.array([0.025, 0.025, 0.012])
def axis_dist(p, c, Rm, L):
    if not L or Rm is None: return float(np.linalg.norm(p - c))
    a = np.array(Rm)[:, 2]; s = float(np.clip((p - c) @ a, -L, L)); return float(np.linalg.norm(p - (c + s * a)))
def held_obj(o, a):
    pv = o["priv_training_only"]; f = o["finger"][a]; tcp = np.array(o["tcp"][a])
    if not (cal["thr"] < f < cal["open_min"]): return None
    c = [(axis_dist(tcp, np.array(pv[k]), pv.get(k + "_R"), L), k) for k, L in MOV.items() if k in pv]; c = [x for x in c if x[0] < 0.045]; return min(c)[1] if c else None
def req(mode, cyc): return None if not STACK else ("block1" if (mode == "orig") == (cyc == 0) else "block2")
def goal(pv, mode, o, last):
    if not STACK or mode == "orig": return bool(pv.get("check_success")) or (last and bool(o["final_priv_training_only"].get("check_success")))
    b1, b2 = np.array(pv["block1"]), np.array(pv["block2"]); return bool(np.all(np.abs(b1 - (b2 + [0, 0, 0.05])) < EPS))  # exchanged goal: block1 (red) on block2 (green); grippers are open at a release check
def label(o, mode, i, n):
    tag = o["tag"]; a = o["arm"]; cyc = o["cyc"]; pv = o["priv_training_only"]; r = req(mode, cyc)
    if tag in ("chk_grasp", "chk_transfer") or tag.startswith("post"): h = held_obj(o, a); return (1 if (h is not None and (r is None or h == r)) else 3), ("wrong_referent" if (h is not None and r is not None and h != r) else None)
    if tag == "chk_release":
        if cyc >= NST - 1: return (4 if goal(pv, mode, o, i == n - 1) else 3), None
        rp = np.array(o["release_tcp"] if o.get("release_tcp") is not None else o["place"]); ks = [r] if r else [k for k in MOV if k in pv]; near = any(np.linalg.norm((np.array(pv[k]) - rp)[:2]) < 0.04 and abs(float(np.array(pv[k])[2] - rp[2])) < 0.05 for k in ks if k in pv)
        wrong = bool(r and not near and any(np.linalg.norm((np.array(pv[k]) - rp)[:2]) < 0.04 for k in MOV if k in pv and k != r)); return (2 if (near and held_obj(o, a) is None) else 3), ("wrong_referent" if wrong else None)
    return None, None
rep = collections.Counter(); neps = 0; D = f"{N}/onpolicy/pic_labelled_{OT}/{t}"; os.makedirs(D, exist_ok=True)
for raw in sys.argv[3:]:
    for f in sorted(glob.glob(raw + "/feat*.pt")):
        F = torch.load(f, weights_only=False); out = []; mode = F[0].get("lang_mode", "orig")
        if mode not in ("orig", "cf"): continue
        for i, o in enumerate(F):
            if o["tag"].startswith("retreat"): continue
            lab, why = label(o, mode, i, len(F))
            if lab is None: continue
            kind = "carry" if o["tag"].startswith("post") else o["tag"][4:]; base = {"sP": o["sP"], "sQ": o["sQ"], "mem": o["mem"], "hist": o["hist"], "m": min(o["cyc"], 2), "cfg": o["cfg"], "seed": o["seed"], "kind": kind, "cyc": o["cyc"], "call": o["call"], "lmode": mode, "pair": f"{mode}{o['seed']}_{i}"}
            out.append({**base, "tok": o["tok"], "pad": o["pad"], "label": lab, "variant": "exec", "why": why}); rep[f"{o['cfg']}|{mode}|exec|{kind}|{lab}" + (f"|{why}" if why else "")] += 1
            if STACK and o.get("tok_alt") is not None and (mode == "orig" or o.get("alt_ok")):  # round-1 cf-mode files: the stored alternative tokens equal the executed ones (recorder bug) -> no alt sample
                la, wa = label(o, "cf" if mode == "orig" else "orig", i, len(F)); out.append({**base, "tok": o["tok_alt"], "pad": o["pad_alt"], "label": la, "variant": "alt", "why": wa}); rep[f"{o['cfg']}|{mode}|alt|{kind}|{la}" + (f"|{wa}" if wa else "")] += 1
        if out: torch.save(out, f"{D}/feat_{mode}_{out[0]['seed']}.pt"); neps += 1
R = {"task": t, "raw_dirs": sys.argv[3:], "episodes": neps, "samples_by_cfg|mode|variant|kind|label": dict(sorted(rep.items())), "label_totals": {str(l): sum(v for k, v in rep.items() if k.split("|")[4] == str(l)) for l in (1, 2, 3, 4)}}
json.dump(R, open(f"{D}/label_report.json", "w"), indent=1); print(json.dumps(R, indent=1))
