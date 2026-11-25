import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import REQ_ROBOTWIN
from regsp.planning import StageVars, build_pregrasp_chunk

PKG = Path(__import__("regsp").__file__).parent


def _src(rel):  # read the module text without importing it (importing the executor needs a RoboTwin checkout for the URDF)
    return (PKG / rel).read_text()


G = np.array([0.10, -0.20, 0.05]); D = np.array([0.0, 0.0, -1.0]); WP = np.array([0.0, 0.0, 0.30]); EE = np.array([-0.2, -0.2, 0.25]); R_PRE = 0.10


def test_pregrasp_chunk_waypoint_standoff_grasp_samples():
    pos, eul = build_pregrasp_chunk(EE, np.zeros(3), StageVars(G, np.array([0.0, 0.0, 0.5]), D, R_PRE, True, WP))
    assert pos.shape == (50, 3) and eul.shape == (50, 3)
    assert np.allclose(pos[19], WP) and np.allclose(pos[34], G - R_PRE * D) and np.allclose(pos[49], G)  # waypoint / standoff / grasp
    assert np.allclose(eul[-1], [0.0, 0.0, 0.5], atol=1e-6)


def test_stride_sampling_misses_the_standoff_and_dense_sampling_visits_it():
    stride = list(range(9, 50, 10)); dense = [int(x) for x in "9,19,29,34,39,44,49".split(",")]
    assert 34 not in stride and 34 in dense and stride == [9, 19, 29, 39, 49]
    src = _src("execution/closed_loop.py")
    assert '"CLC_PRE_IDX", "9,19,29,34,39,44,49"' in src   # frozen default of the closed-loop correction


def test_cycle_and_role_indexing_contract():
    src = _src("execution/bimanual_policy.py")
    assert 'min(cyc, 1) * 2 + (arm == "R")' in src and "min(self.cur_cyc, 1)" in src  # role in {0..3}, cycle clipped to {0,1}
    from regsp.decoding import BimDecoder, AdaptiveDecoder  # noqa
    from regsp.grounding import CycleScorer
    assert CycleScorer().cyc.num_embeddings == 2


class _Env:
    def __init__(self, q): self.q = q
    def get_obs(self): return {"joint_action": {"vector": self.q}}


@REQ_ROBOTWIN
def test_ee_action_layout_and_gripper_polarity():
    """16-D end-effector action = per arm [xyz(3), quat wxyz(4), gripper(1)]; gripper 1.0 = open, 0.0 = closed (RoboTwin / ReGSP convention)."""
    from regsp.execution.base_policy import V21Policy, FK
    q = np.zeros(14); q[6] = 0.7; q[13] = 0.3; T = FK["L"].tcp(q[0:6], "world")
    pose = (T[:3, 3], T[:3, :3])
    v_close = V21Policy.act_vec(None, _Env(q), {"L": pose}, {"L": 0.0}); v_open = V21Policy.act_vec(None, _Env(q), {"L": pose}, {"L": 1.0})
    assert v_close.shape == (16,) and v_close[7] == 0.0 and v_open[7] == 1.0
    assert v_close[15] == pytest.approx(0.3)                  # the arm that is not commanded keeps its measured gripper reading
    assert np.linalg.norm(v_close[3:7]) == pytest.approx(1.0)  # unit quaternion


def test_finger_opening_polarity_documented():
    from regsp.execution import proprio
    assert "1 = open" in inspect.getsource(proprio.finger_opening)
    src = _src("execution/base_policy.py")
    assert re.search(r'\{arm: gp\}, \{arm: 0\.0\}, "close"', src.replace("self.step(env, ", "")) or '{arm: 0.0}, "close"' in src   # close command = 0.0
    assert "dd[-1, 6] > 0.5" in src   # decoder gripper > 0.5 = open/release proposal
