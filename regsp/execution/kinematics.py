"""Independent numpy FK for the RoboTwin aloha-agilex (ARX5-based) arms, parsed from the embodiment URDF. Used to (a) validate against the simulator, (b) label demo end-effector poses from recorded joint states."""
import os
import numpy as np, xml.etree.ElementTree as ET, transforms3d as t3d, yaml

from regsp.paths import robotwin_root
def _emb(): return str(robotwin_root()) + "/assets/embodiments/aloha-agilex"

def _rpy(r): return t3d.euler.euler2mat(*r, axes="sxyz")  # URDF rpy: fixed-axis xyz

class ArmFK:
    def __init__(self, side="left"):
        EMB = _emb(); cfg = yaml.safe_load(open(EMB + "/config.yml")); self.cfg = cfg; i = 0 if side == "left" else 1; p = "fl" if side == "left" else "fr"
        root = ET.parse(EMB + "/urdf/arx5_description_isaac.urdf").getroot(); J = {}
        for j in root.findall("joint"):
            o = j.find("origin"); xyz = np.array(o.get("xyz", "0 0 0").split(), float); rpy = np.array(o.get("rpy", "0 0 0").split(), float); ax = j.find("axis")
            J[j.get("name")] = dict(type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"), T=self._mat(xyz, rpy), axis=np.array(ax.get("xyz").split(), float) if ax is not None else None)
        # full kinematic path URDF-root -> link6 (fixed joints included): the dual-arm entity root is NOT the arm base link
        byc = {v["child"]: (k, v) for k, v in J.items()}; path = []; l = f"{p}_link6"
        while l in byc: k, v = byc[l]; path.append(v); l = v["parent"]
        self.root_link = l; self.chain = path[::-1]; self.names = cfg["arm_joints_name"][i]
        self.bias = cfg["gripper_bias"]; self.gt = np.array(cfg["global_trans_matrix"]); self.dm = np.array(cfg["delta_matrix"]); self.origin = np.array(cfg["robot_pose"][0])  # xyz + wxyz quaternion
        self.T_origin = np.eye(4); self.T_origin[:3, :3] = t3d.quaternions.quat2mat(self.origin[3:]); self.T_origin[:3, 3] = self.origin[:3]
    @staticmethod
    def _mat(xyz, rpy): T = np.eye(4); T[:3, :3] = _rpy(rpy); T[:3, 3] = xyz; return T
    def link6(self, q):  # pose of link6 in the arm-base link frame
        T = np.eye(4); qi = iter(q)
        for j in self.chain:
            R = np.eye(4)
            if j["type"] in ("revolute", "continuous"): R[:3, :3] = t3d.axangles.axangle2mat(j["axis"], next(qi))
            T = T @ j["T"] @ R
        return T
    def tcp(self, q, frame="base"):  # RoboTwin TCP pose: sim ee link already carries global_trans (verified numerically: R_tcp = R_link6 @ delta); + gripper_bias along local x
        T = self.link6(q); R = T[:3, :3] @ self.dm; out = np.eye(4); out[:3, :3] = R; out[:3, 3] = T[:3, 3] + R @ np.array([self.bias, 0, 0])
        return out if frame == "base" else self.T_origin @ out

