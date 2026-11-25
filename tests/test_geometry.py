import numpy as np
import regsp.grounding as R


def look_at(c, tgt, up=(0, 0, 1.0)):
    """OpenCV-convention world->camera 4x4 for a camera at c looking at tgt."""
    z = (tgt - c) / np.linalg.norm(tgt - c); x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
    Rw = np.stack([x, y, z], 1)  # camera axes in world (columns)
    E = np.eye(4); E[:3, :3] = Rw.T; E[:3, 3] = -Rw.T @ c
    return E


def test_project_ray_roundtrip():
    E = look_at(np.array([0.5, -0.4, 0.8]), np.array([0.0, 0.1, 0.0]))
    p = np.array([0.05, 0.12, 0.02]); uv, d = R.project(E, p)
    c, dirn = R.ray(E, uv)
    assert d > 0 and abs(np.linalg.norm(dirn) - 1) < 1e-9
    assert np.linalg.norm(np.cross(dirn, p - c)) < 1e-6          # ray passes through p
    assert np.allclose(c, -E[:3, :3].T @ E[:3, 3])               # origin is the camera centre


def test_triangulate_recovers_point_and_gap_is_small():
    p = np.array([0.02, 0.15, 0.03]); E1 = look_at(np.array([0.0, -0.5, 0.7]), p); E2 = look_at(np.array([0.3, 0.1, 0.5]), p)
    (u1, _), (u2, _) = R.project(E1, p)[0], R.project(E2, p)[0]
    est, gap = R.triangulate(E1, R.project(E1, p)[0], E2, R.project(E2, p)[0])
    assert np.linalg.norm(est - p) < 1e-6 and gap < 1e-6


def test_associate_hungarian_pairs_matching_views_and_gates_on_3cm():
    pts = [np.array([0.0, 0.1, 0.02]), np.array([0.15, 0.2, 0.02])]
    E1 = look_at(np.array([0.0, -0.5, 0.7]), pts[0]); E2 = look_at(np.array([0.3, 0.1, 0.5]), pts[0])
    Pf = [(*R.project(E1, p)[0], 1.0) for p in pts]; Pw = [(*R.project(E2, p)[0], 1.0) for p in pts[::-1]]  # wrist list in the opposite order
    A = R.associate(Pf, Pw, E1, E2)
    assert {i: A[i][1] for i in A} == {0: 1, 1: 0} and all(A[i][2] <= R.GAP_MAX for i in A)
    assert R.associate([], Pw, E1, E2) == {} and R.GAP_MAX == 0.03


def test_decimation_and_letterbox():
    img = np.arange(480 * 640 * 3, dtype=np.uint32).reshape(480, 640, 3).astype(np.uint8)
    assert R.half(img).shape == (240, 320, 3) and R.sq(R.half(img)).shape == (224, 224, 3)
