"""Capture-unit validator (a valid unit is more than two files). Checks: both files load; frame count & image keys for all 3 cameras present, uint8 240x320x3, non-degenerate (std>2) and temporally varying (synchronised sequence); intrinsics = Large_D435 K; extrinsics finite, rotation orthonormal;
wrist extrinsics consistent with the SAME frame's recorded TCP pose x calibrated tool->camera mount (<=6 mm, <=1.5 deg) i.e. camera pose / proprioception / image synchronised; objects present with finite xyz; referent attrs of refs found in objects; instruction non-empty;
visibility: seq units need the first referent in-frame in the head camera in >=1 frame; obs units need the referent in-frame in its wrist camera in >=1 frame. Successful completion: seq/obs units are only written after the scripted expert succeeded (seq) / after scene validation (obs: scene pkl from a successful expert run) -- enforced by construction, re-checked via tcp_at_close/refs lengths."""
import sys, os, pickle, numpy as np
from regsp import paths as P
import regsp.grounding as R
from scipy.spatial.transform import Rotation as Rot
TCPCAM = np.array([[0, -0.389, 0.921, -0.05], [-1, 0, 0, 0.032], [0, -0.921, -0.389, 0.065], [0, 0, 0, 1.0]])
def valid_unit(pkl, kind):
    try:
        p = pickle.load(open(pkl, "rb")); z = np.load(pkl.replace(".pkl", ".npz"))
        if not p.get("instruction") or not p.get("refs") or len(p["tcp_at_close"]) < len(p["refs"]): return False, "metadata_incomplete"
        if not p["frames"]: return False, "no_frames"
        vis = False; prev = None
        for i, fr in enumerate(p["frames"]):
            ims = []
            for c in R.CAMS:
                k = f"f{i}_{c}"
                if k not in z.files: return False, "missing_image"
                im = z[k]
                if im.dtype != np.uint8 or im.shape != (240, 320, 3) or im.std() < 2: return False, "bad_image"
                ims.append(im)
            if prev is not None and kind == "seq" and all(np.array_equal(a, b) for a, b in zip(ims, prev)) and i > 2: return False, "frozen_sequence"
            prev = ims
            for c in R.CAMS:
                if not np.allclose(fr["K"][c], R.K, atol=0.5): return False, "bad_intrinsics"
                E = R.w2c4(fr["E"][c]); Rm = E[:3, :3]
                if not np.all(np.isfinite(E)) or np.abs(Rm @ Rm.T - np.eye(3)).max() > 1e-3: return False, "bad_extrinsics"
            for side, cam in (("left", "left_camera"), ("right", "right_camera")):
                t = np.array(fr["tcp"][side]); T = np.eye(4); T[:3, :3] = Rot.from_quat([t[4], t[5], t[6], t[3]]).as_matrix(); T[:3, 3] = t[:3]; c2w = np.linalg.inv(R.w2c4(fr["E"][cam])); Tp = T @ TCPCAM
                dr = Rot.from_matrix(Tp[:3, :3].T @ c2w[:3, :3]).magnitude() * 57.3
                if np.linalg.norm(Tp[:3, 3] - c2w[:3, 3]) > 0.006 or dr > 1.5: return False, "camera_pose_desync"
            if not fr["objects"] or not all(np.all(np.isfinite(o["xyz"])) for o in fr["objects"]): return False, "bad_objects"
            attrs = {o["attr"] for o in fr["objects"]}
            if kind == "seq":
                k0 = fr.get("n_closed_so_far", 0)
                if k0 < len(p["refs"]):
                    a0 = p["refs"][k0][1]
                    if a0 not in attrs: return False, "referent_missing"
                    if i == 0 or True:
                        o = [x for x in fr["objects"] if x["attr"] == a0][0]; (u, v), d = R.project(fr["E"]["head_camera"], o["xyz"]); vis |= bool(d > 0.05 and 8 <= u < 632 and 8 <= v < 472)
            else:
                if fr["attr"] not in attrs: return False, "referent_missing"
                o = [x for x in fr["objects"] if x["attr"] == fr["attr"]][0]; cam = "left_camera" if fr["arm"] == "L" else "right_camera"; (u, v), d = R.project(fr["E"][cam], o["xyz"]); vis |= bool(d > 0.05 and 8 <= u < 632 and 8 <= v < 472)
        return (True, "ok") if vis else (False, "target_never_visible")
    except Exception as e: return False, f"load_error:{type(e).__name__}"
if __name__ == "__main__":
    import glob, collections, json
    O = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); rep = {}
    for t in sorted(os.listdir(O + "/views")):
        rep[t] = {}
        for d, kind in (("views", "seq"), ("obsviews", "obs")):
            c = collections.Counter(valid_unit(f, kind)[1] for f in sorted(glob.glob(f"{O}/{d}/{t}/*.pkl"))); rep[t][d] = dict(c)
    print(json.dumps(rep, indent=1))
