"""Checkpoint-compatibility tests against the local frozen weights (not shipped in git). Require the weight bundle, backbone checkpoint, RoboTwin and CUDA."""
import json
import os
from pathlib import Path

import pytest

from conftest import REQ_WEIGHTS
from regsp import paths

ROOT = Path(__file__).resolve().parents[1]


@REQ_WEIGHTS
def test_bundle_hashes():
    import subprocess, sys
    r = subprocess.run([sys.executable, str(ROOT / "scripts/utils/assemble_bundle.py"), "--verify", os.environ["REGSP_CKPT_ROOT"]], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


@REQ_WEIGHTS
@pytest.mark.parametrize("task", paths.TASKS)
def test_policy_weights_match_reference_digests(task):
    """Every learned module of the ReGSP policy (backbone, heads, scorers, anchor, decoders, progress head, statistics, executor flags) equals the
    digest recorded from the frozen research-tree V21-CLC policy (configs/reference_module_digests.json)."""
    from regsp.execution.factory import build_policy
    from regsp.verify import module_digests
    ref = json.loads((ROOT / "configs" / "reference_module_digests.json").read_text())[task]
    got = module_digests(build_policy(task))
    assert not [k for k in ref if k in got and ref[k][1] != got[k][1]], [k for k in ref if k in got and ref[k][1] != got[k][1]]
    assert set(got) >= set(ref) - {"sc"}  # 'sc' (plain scorer) is superseded by the subgoal scorer for stack_blocks_two and not loaded
