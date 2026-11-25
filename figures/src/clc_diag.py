"""[V21-CLC] ANALYSIS ONLY (privileged object poses from the ACL_TRACE log; never read by the policy): decomposition of the grounding-to-action discrepancy at every grasp closure.
  ground  = g_final - object centre at cycle start (horizontal)            perception / grounding
  exec    = measured TCP just before the close - g_final                    IK / controller / arrival tolerance
  track   = measured TCP - commanded TCP per call (position, rotation)      execution error per phase
  disp    = object centre just before the close - object centre at cycle start   target displaced during the approach
  align   = measured TCP at the close - object centre just before the close  what the fingers actually see
usage: clc_diag.py DIR [DIR ...]"""
import json, glob, sys, numpy as np
OBJ = {"handover_block": ["box"], "place_container_plate": ["container"], "stack_blocks_two": ["block1", "block2"], "click_bell": ["bell"]}
def rotdeg(A, B): return float(np.degrees(np.arccos(np.clip((np.trace(np.array(A).T @ np.array(B)) - 1) / 2, -1, 1))))
def load(D):
    rows = []; track = {}; eps = []
    for f in sorted(glob.glob(D + "/seed*.json")):
        d = json.load(open(f)); tr = d.get("trace_analysis_only", []); task = d.get("task") or next(t for t in OBJ if t in D); eps.append((d.get("seed"), bool(d.get("success"))))
        for i, t in enumerate(tr):
            ph = "obs" if "obs_view" in t["tag"] else "pre" if t["tag"].startswith("pre") else "close" if t["tag"].startswith(("close", "settle")) else "post" if t["tag"].startswith("post") else "other"
            for a, c in t.get("cmd", {}).items():
                c = np.array(c); m = np.array(t["tcp"][a]); track.setdefault(ph, []).append((float(np.linalg.norm(c[:, 3] - m[:, 3]) * 100), rotdeg(c[:, :3], m[:, :3])))
        groups = [i for i, t in enumerate(tr) if t["tag"].startswith("close") and (i == 0 or not tr[i - 1]["tag"].startswith("close"))]
        cyc = [c for c in d.get("cycles", []) if "grasp_verify" in c]
        for gi, c in zip(groups, cyc):
            arm = c["arm"]; g = np.array(c["g_final"]); st = c["priv_analysis_only"]["start"]; names = [o for o in OBJ[task] if o in st]
            if not names or gi == 0: continue
            o = min(names, key=lambda n: np.linalg.norm((np.array(st[n]) - g)[:2])); o0 = np.array(st[o]); pre = tr[gi - 1]; o1 = np.array(pre["obj"][o]); tc0 = np.array(pre["tcp"][arm])[:, 3]; j = gi
            while j + 1 < len(tr) and tr[j + 1]["tag"].startswith(("close", "settle")): j += 1
            tc1 = np.array(tr[j]["tcp"][arm])[:, 3]; o2 = np.array(tr[j]["obj"][o]); k = min(j + 12, len(tr) - 1); o3 = np.array(tr[k]["obj"][o])
            rows.append({"seed": d.get("seed"), "succ": bool(d.get("success")), "cyc": c["cycle"], "att": c["attempt"], "arm": arm, "src": c.get("source"), "hold": c["grasp_verify"].get("holding"), "obj": o,
                         "ground": (g - o0)[:2] * 100, "ground_z": (g - o0)[2] * 100, "exec": (tc0 - g)[:2] * 100, "exec_z": (tc0 - g)[2] * 100, "exec_close": (tc1 - g)[:2] * 100, "disp": (o1 - o0)[:2] * 100, "disp_close": (o2 - o1)[:2] * 100, "align": (tc1 - o1)[:2] * 100,
                         "lift_cm": (o3 - o0)[2] * 100, "arrived": c.get("arrived"), "arr": c.get("arrival_dist_cm", [None])[-1], "nview_ok": sum(bool(v.get("ok")) for v in c.get("views", [])), "nview": len(c.get("views", []))})
    return rows, track, eps
if __name__ == "__main__":
    n = lambda v: float(np.linalg.norm(v))
    for D in sys.argv[1:]:
        rows, track, eps = load(D); print(f"\n=== {D}  episodes {len(eps)} success {sum(s for _, s in eps)}")
        for ph, v in track.items(): v = np.array(v); print(f"  track[{ph:5s}] n={len(v):4d} pos cm median {np.median(v[:,0]):.2f} p90 {np.percentile(v[:,0],90):.2f} max {v[:,0].max():.1f} | rot deg median {np.median(v[:,1]):.1f} p90 {np.percentile(v[:,1],90):.1f}")
        for cy in sorted({r["cyc"] for r in rows}):
            R_ = [r for r in rows if r["cyc"] == cy]; print(f"  -- stage {cy}: closures {len(R_)}  sources { {s: sum(r['src'] == s for r in R_) for s in {r['src'] for r in R_}} }")
            for nm in ("ground", "exec", "exec_close", "disp", "disp_close", "align"):
                v = np.array([n(r[nm]) for r in R_]); print(f"     {nm:10s} |xy| cm median {np.median(v):.2f} mean {v.mean():.2f} p75 {np.percentile(v,75):.2f} p90 {np.percentile(v,90):.2f}   >1.2cm: {int((v > 1.2).sum())}/{len(v)}")
            print("     exec_z median %.2f  ground_z median %.2f" % (np.median([r["exec_z"] for r in R_]), np.median([r["ground_z"] for r in R_])))
            for r in R_: print("     ", r["seed"], "S" if r["succ"] else "F", f"att{r['att']}{r['arm']}", f"{str(r['src'])[:14]:14s}", "hold", str(r["hold"])[:1], "ground %4.1f (%5.1f,%5.1f) z%5.1f" % (n(r["ground"]), *r["ground"], r["ground_z"]), "exec %4.1f z%5.1f" % (n(r["exec"]), r["exec_z"]), "disp %4.1f" % n(r["disp"]), "dispC %4.1f" % n(r["disp_close"]), "align %4.1f (%5.1f,%5.1f)" % (n(r["align"]), *r["align"]), "lift %5.1f" % r["lift_cm"], "views %d/%d" % (r["nview_ok"], r["nview"]))
