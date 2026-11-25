#!/usr/bin/env python
"""Load the frozen ReGSP policy of each task from the weight bundle and compare every learned module / statistic with configs/reference_module_digests.json
(digests recorded from the frozen research-tree V21-CLC policy). Needs REGSP_BACKBONE_CKPT, REGSP_CKPT_ROOT, REGSP_ROBOTWIN_ROOT and one CUDA GPU; no simulator episode is run.
usage: verify_bundle_load.py [task ...]"""
import json, sys
from pathlib import Path

from regsp import paths
from regsp.execution.factory import build_policy
from regsp.verify import module_digests

ref = json.loads((Path(__file__).resolve().parents[2] / "configs" / "reference_module_digests.json").read_text())
bad = 0
for t in sys.argv[1:] or paths.TASKS:
    got = module_digests(build_policy(t))
    diff = [k for k in ref[t] if k in got and ref[t][k][1] != got[k][1]]; missing = [k for k in ref[t] if k not in got and k != "sc"]
    print(f"{t}: {len(ref[t])} reference entries, mismatching={diff}, missing={missing}"); bad += bool(diff or missing)
sys.exit(1 if bad else 0)
