"""[V21-CLC] data-derived fallback offset (no privileged state, no success flag): median over TRAINING-DOMAIN rollouts of this policy (seeds < 7000000; disjoint from every development / final seed) of
(two-view grounded grasp point - head-ray estimate of the same referent), per arm. Used by CLC_FB when two-view grounding fails but the head referent is confident.  usage: clc_build_stats.py"""
import json, glob, numpy as np, collections
from regsp import paths as P
N = str(P.work_root()); out = {}
for task in ("place_container_plate",):
    R = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in glob.glob(f"{N}/eval/*/*/{task}/seed*.json"):
        d = json.load(open(f)); sd = d.get("seed")
        if sd is None or sd >= 7000000 or not d.get("cycles"): continue
        c = d["cycles"][0]
        if c.get("source") == "grounded" and c.get("coarse_xyz_r"): R[c["arm"]][sd].append(np.array(c["g_final"]) - np.array(c["coarse_xyz_r"]))
    out[task] = {"grasp_minus_coarse": {a: np.median(np.array([np.mean(v, 0) for v in R[a].values()]), 0).round(4).tolist() for a in R}, "n_seeds": {a: len(R[a]) for a in R}, "seed_range": "training-domain rollouts < 7000000 (5300000+/5350000+ blocks)",
                 "definition": "median of (two-view grounded grasp point - head-ray x height-prior estimate of the referent), robot frame, metres"}
json.dump(out, open(N + "/clc_stats.json", "w"), indent=1); print(json.dumps(out, indent=1))
