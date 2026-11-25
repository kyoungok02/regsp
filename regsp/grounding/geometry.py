"""Camera geometry and two-view association (RoboTwin Large_D435, 640x480 pinhole).

Verbatim extraction of ``v21rt.py`` (research tree). ORIGINAL SafeDiff-VLA V21 definitions: ``triangulate`` (closest-approach midpoint
and gap), ``associate`` (Hungarian assignment on the two-ray gap, ``GAP_MAX`` = 3 cm gate). EMBODIMENT ADAPTERS: the camera model ``K``,
the half-resolution decimation ``half`` used consistently for features/crops, the letterboxed 224 input ``sq``.
"""
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

K = np.array([[717.3, 0, 320.0], [0, 717.3, 240.0], [0, 0, 1.0]]); KINV = np.linalg.inv(K); W, Hh = 640, 480
THR, GAP_MAX, KCAP, CROP_PX, MATCH_PX = 0.10, 0.03, 10, 72, 10.0
CAMS = ("head_camera", "left_camera", "right_camera")
def half(img): return np.ascontiguousarray(img[::2, ::2])
def sq(img, s=224):
    h, w = img.shape[:2]; m = max(h, w); pad = np.zeros((m, m, 3), np.uint8); pad[(m - h) // 2:(m - h) // 2 + h, (m - w) // 2:(m - w) // 2 + w] = img; return cv2.resize(pad, (s, s), interpolation=cv2.INTER_AREA)
def w2c4(E):
    E = np.asarray(E, float); return np.vstack([E, [0, 0, 0, 1]]) if E.shape[0] == 3 else E
def project(E, p):
    pc = w2c4(E)[:3, :3] @ np.asarray(p) + w2c4(E)[:3, 3]; return (K @ pc)[:2] / pc[2], float(pc[2])
def ray(E, uv):  # world-frame origin and unit direction for pixel uv (full-res coords)
    M = np.linalg.inv(w2c4(E)); d = M[:3, :3] @ (KINV @ np.array([uv[0], uv[1], 1.0])); return M[:3, 3], d / np.linalg.norm(d)
def triangulate(E1, uv1, E2, uv2):  # closest-approach midpoint + gap (original v21g_geometry.triangulate formula)
    c1, d1 = ray(E1, uv1); c2, d2 = ray(E2, uv2); A = np.array([[d1 @ d1, -d1 @ d2], [d1 @ d2, -d2 @ d2]]); b = np.array([(c2 - c1) @ d1, (c2 - c1) @ d2])
    try: s, t = np.linalg.solve(A, b)
    except np.linalg.LinAlgError: return None, 1e9
    p1, p2 = c1 + s * d1, c2 + t * d2
    if s < 0 or t < 0: return None, 1e9
    return (p1 + p2) / 2, float(np.linalg.norm(p1 - p2))
def associate(Pf, Pr, Ef, Er):  # original Hungarian on two-ray gap, GAP_MAX gate; returns {front_idx: (ref_world, right_idx, gap)}
    if not Pf or not Pr: return {}
    C = np.full((len(Pf), len(Pr)), 1e3); pts = {}
    for i, f in enumerate(Pf):
        for j, g in enumerate(Pr):
            p, gap = triangulate(Ef, f[:2], Er, g[:2])
            if p is not None: C[i, j] = gap; pts[(i, j)] = p
    a, b = linear_sum_assignment(C); out = {}
    for i, j in zip(a, b):
        if C[i, j] <= GAP_MAX: out[i] = (pts[(i, j)], j, C[i, j])
    return out
