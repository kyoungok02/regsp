"""[E] Robot-proprioception helpers (no object state). Verbatim from ``acl_common.py``."""
import numpy as np

def finger_opening(env, arm):
    """[E] measured normalised finger opening of one gripper (1 = open, ~0.18 = fully closed on nothing, in between = fingers blocked by an object). Robot proprioception only (gripper joint encoder); no object state."""
    side = "left" if arm == "L" else "right"; r = env.robot; ent = getattr(r, f"{side}_entity"); js = getattr(r, f"{side}_gripper"); sc = getattr(r, f"{side}_gripper_scale"); aj = ent.get_active_joints(); q = ent.get_qpos()
    return float(np.mean([((q[aj.index(j[0])] - j[2]) / j[1] - sc[0]) / (sc[1] - sc[0]) for j in js]))
def joint_vec(env):
    """[E] 14-D joint/gripper proprioception without rendering the cameras (same vector as obs['joint_action']['vector'])"""
    return np.array(list(env.robot.get_left_arm_jointState()) + list(env.robot.get_right_arm_jointState()), float)
