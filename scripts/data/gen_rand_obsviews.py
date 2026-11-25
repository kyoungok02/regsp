"""[gen_obsviews_10] ACTIVE-VIEW training frames: for each valid scene (same seeds/distractors as scenes/), command the referent arm (cuRobo ee) to OBSERVATION poses (target at 0.22-0.40 m along the wrist-camera axis, xy jitter sigma 3 cm = coarse-localisation error, tool rotation = train-median grasp rotation +-10 deg yaw) and record (head, wrist L, wrist R) frames, K/E, objects. Original: [gen_views_10] V21 per-task TRAINING-DOMAIN VIEW SEQUENCES: same as gen_scenes_10 but the scripted expert is run WITH 0..3 distractors spawned and (head, left-wrist, right-wrist) frames + per-camera intrinsics/extrinsics + all object xyz + both TCP poses are recorded every STEP simulator steps along the expert path (half resolution). Used to train class-agnostic proposal heads for WRIST views and the two-stage grounding. PRIVILEGED state = labels only. Original docstring:  (Clean config only; seeds from 7000000, disjoint from RGL-50 final 4000000+, dev 3000000+/5000000+/6000000+). PRIVILEGED simulator state is used for TRAINING LABELS ONLY (candidate centres, referent identity, expert grasp points), never at inference.
For each valid seed (scripted expert succeeds): variant N (natural scene) = 3 camera frames at t0 + all object poses + expert referent actor name and first-close TCP per (arm, cycle) + instruction; variant D (same seed + 0..3 distractor objects spawned) = 3 camera frames + all object poses.
Class-agnostic proposal targets = projected centres of ALL objects; scorer candidates = proposals; anchor labels = expert grasp - referent keypoint. usage: gen_scenes_10.py TASK N START_SEED OUTDIR"""
import sys; 
from regsp import paths as P
import os, json, re, pickle, numpy as np, sapien
import sapien.core as sp
_od = sapien.render.set_ray_tracing_denoiser; sapien.render.set_ray_tracing_denoiser = lambda n: _od(os.environ.get("RGL_DENOISER", "optix"))
from regsp.envs import robotwin as S
task, N, st, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), os.path.abspath(sys.argv[4]); os.makedirs(out, exist_ok=True)
from envs.utils import create_actor
from generate_episode_instructions import generate_episode_descriptions
env, args = S.make(task, os.environ.get("GEN_CONFIG", "demo_clean"))
SPEC = {"bottle": dict(modelname="001_bottle", ids=[13, 16], qpos=[[0.707, 0, 0, -0.707], [0.707, 0, 0, 0.707]], z=0.752), "roller": dict(modelname="102_roller", ids=[0, 2], qpos=[[0.5, 0.5, 0.5, 0.5], [0, 0, 0.707, 0.707]], z=0.741), "hammer": dict(modelname="020_hammer", ids=[0], qpos=[[0, 0, 0.995, 0.105]], z=0.783),
        "cup": dict(modelname="021_cup", ids=[0], qpos=[[0.5, 0.5, 0.5, 0.5]], z=0.741), "coaster": dict(modelname="019_coaster", ids=[0], qpos=[[0.5, 0.5, 0.5, 0.5]], z=0.741)}
SCEN = ("table", "wall", "ground")
from envs.utils.actor_utils import Actor, ArticulationActor
EXTRA = []  # spawned distractor actors of the current scene
def handles(env):  # task objects = the env's own Actor/ArticulationActor attributes (how the expert references them), scenery excluded; lists indexed
    H = {}
    for k, v in vars(env).items():
        vs = [(k, v)] if isinstance(v, Actor) else ([(f"{k}[{i}]", x) for i, x in enumerate(v)] if isinstance(v, (list, tuple)) and v and all(isinstance(x, Actor) for x in v) else [])
        for n, a in vs:
            try: nm = a.get_name()
            except Exception: nm = ""
            if nm in SCEN or n in SCEN: continue
            H[n] = a
    return H
def objs(env): return [{"attr": n, "name": (a.get_name() or n), "xyz": np.array(a.get_pose().p).tolist()} for n, a in handles(env).items()] + [{"attr": f"extra{i}", "name": a.get_name(), "xyz": np.array(a.get_pose().p).tolist()} for i, a in enumerate(EXTRA)]
def side2a(s): return "L" if "left" in str(s) else "R"
def recorders(env):
    X = {"refs": [], "events": [], "tcp_at_close": []}
    og = env.grasp_actor
    def ga(actor, arm_tag=None, *aa, **kk): X["refs"].append((side2a(arm_tag), next((n for n, a in handles(env).items() if a is actor), actor.get_name()))); return og(actor, arm_tag, *aa, **kk)
    env.grasp_actor = ga; os_ = env.set_gripper
    def sg(left_pos=None, right_pos=None, set_tag="all", **kk):
        for side, pos in (("left", left_pos), ("right", right_pos)):
            if pos is not None and pos < 0.5 and set_tag in (side, "all"): X["tcp_at_close"].append((side2a(side), list(map(float, getattr(env.robot, f"get_{side}_tcp_pose")()[:3]))))
        return os_(left_pos=left_pos, right_pos=right_pos, set_tag=set_tag, **kk)
    env.set_gripper = sg; return X
def spawn(rng, cls, taken):
    s = SPEC[cls]
    for _ in range(40):
        xy = np.array([rng.uniform(-0.32, 0.32), rng.uniform(-0.22, 0.08)])
        if all(np.hypot(*(xy - t)) > 0.14 for t in taken):
            k = rng.integers(len(s["ids"]))
            try: a = create_actor(scene=env, pose=sp.Pose([xy[0], xy[1], s["z"]], s["qpos"][k]), modelname=s["modelname"], convex=True, model_id=s["ids"][k], is_static=False); return a, xy
            except Exception: return None, None
    return None, None
def frames(env): o = env.get_obs()["observation"]; return {c: o[c]["rgb"] for c in ("head_camera", "left_camera", "right_camera")}


import glob, transforms3d as t3d
from regsp.execution import kinematics as rt_kin
from scipy.spatial.transform import Rotation as Rot
LAB = pickle.load(open(os.path.dirname(out.rstrip("/")) .replace("/obsviews", "") + f"/labels/{task}.pkl", "rb")) if False else None
OUTR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); LAB = pickle.load(open(f"{OUTR}/labels/{task}.pkl", "rb"))
fk = {"L": rt_kin.ArmFK("left"), "R": rt_kin.ArmFK("right")}; To = fk["L"].T_origin; Ti = np.linalg.inv(To)
RMED = {}
for a in "LR":
    Rs = [c["grasp_R"] for v in LAB.values() if not v["rejected"] for c in v["cycles"][a]]
    if Rs: RMED[a] = Rot.from_matrix(np.stack(Rs)).mean().as_matrix()
TCPCAM = np.array([[0, -0.389, 0.921, -0.05], [-1, 0, 0, 0.032], [0, -0.921, -0.389, 0.065], [0, 0, 0, 1.0]])  # tool->camera (verified rigid: std 0.0 over frames; = calibration mount composed with link6->TCP)
cams = ("head_camera", "left_camera", "right_camera")
def snap(env):
    o = env.get_obs()["observation"]; fr = {}; K = {}; E = {}
    for c in cams: fr[c] = np.ascontiguousarray(o[c]["rgb"][::2, ::2]); K[c] = np.array(o[c]["intrinsic_cv"]); E[c] = np.array(o[c]["extrinsic_cv"])
    tcp = {s: np.array(getattr(env.robot, f"get_{s}_tcp_pose")()) for s in ("left", "right")}; return fr, K, E, tcp, objs(env)
def ee_vec(env, poses):  # poses {arm:(p_w,R_w)}; other arm holds
    q = np.array(env.get_obs()["joint_action"]["vector"]); v = []
    for a in "LR":
        if a in poses: p, Rw = poses[a]
        else: T = fk[a].tcp(q[:6] if a == "L" else q[7:13], "world"); p, Rw = T[:3, 3], T[:3, :3]
        v += list(p - Rw[:, 0] * 0.12) + list(t3d.quaternions.mat2quat(Rw)) + [1.0]
    return np.array(v)
files = sorted(glob.glob(f"{os.path.dirname(out.rstrip('/'))}/../scenes/{task}/scene*.pkl")) if False else sorted(glob.glob(f"{OUTR}/scenes/{task}/scene*.pkl"))
done = 0
for sf in files[:N]:
    sc = pickle.load(open(sf, "rb")); seed = sc["seed"]; f = f"{out}/obs{seed}.pkl"
    if os.path.exists(f) and os.path.exists(f.replace(".pkl", ".npz")): done += 1; continue
    af = f"{out}/att{seed}.txt"; na = int(open(af).read()) if os.path.exists(af) else 0
    if na >= 2: open(out + "/failed_seeds.txt", "a").write(f"{seed} crashed {na}x -> permanently skipped\n"); continue
    open(af, "w").write(str(na + 1))
    try:
        with S.Quiet():
            env.setup_demo(now_ep_num=done, seed=seed, is_test=True, **args); r = np.random.default_rng(seed); EXTRA.clear(); taken = [np.array(o["xyz"][:2]) for o in objs(env)]; own = {o["name"] for o in objs(env)}
            cl = [c for c in SPEC if SPEC[c]["modelname"] not in own]; extra = list(r.choice(cl, size=int(r.integers(0, 4)), replace=False)) if cl else []
            for c in extra:
                a, xy = spawn(r, c, taken)
                if a is not None: taken.append(xy); a.set_name(SPEC[c]["modelname"]); EXTRA.append(a)
            for _ in range(10): env.scene.step()
            H = handles(env); rr = np.random.default_rng(seed + 991); SEQ = []; nfail = 0
            for (arm, attr) in list(dict.fromkeys(map(tuple, sc["refs"]))):
                if attr not in H or arm not in RMED: print("skip ref", seed, arm, attr, attr in H, sorted(H)[:6], sorted(RMED), file=sys.stderr, flush=True); continue
                try:
                    from envs.utils import ArmTag as _AT
                    if not hasattr(env, "arm_tag") or True: env.arm_tag = _AT("left" if arm == "L" else "right")  # some tasks (open_laptop) read self.arm_tag in check_success during take_action; set it from the referent arm
                except Exception: pass
                for k in range(3):
                    tw = np.array(H[attr].get_pose().p); tr_ = Ti[:3, :3] @ tw + Ti[:3, 3] + np.r_[rr.normal(0, 0.03, 2), 0.0]; d = rr.uniform(0.22, 0.40)
                    Rr = Rot.from_euler("z", rr.normal(0, np.deg2rad(10))).as_matrix() @ RMED[arm]; Rw = To[:3, :3] @ Rr; zc = (Rw @ TCPCAM[:3, :3])[:, 2]; pcam = (To[:3, :3] @ tr_ + To[:3, 3]) - d * zc; ptcp = pcam - Rw @ TCPCAM[:3, 3]
                    try: env.take_action(ee_vec(env, {arm: (ptcp, Rw)}), action_type="ee")
                    except Exception as e_: nfail += 1; print("move fail", seed, arm, attr, type(e_).__name__, str(e_)[:80], file=sys.stderr, flush=True); continue
                    SEQ.append({"arm": arm, "attr": attr, "d_obs": float(d), "snap": snap(env)})
            env.close_env()
    except Exception as e:
        try: env.close_env()
        except Exception: pass
        print("obs fail", seed, type(e).__name__, str(e)[:100], flush=True)
        if any(x in str(e) for x in ("DeviceLost", "vk::", "Vulkan", "out of memory")): os._exit(3)  # device is gone: exit now; bounded wrapper restarts; continue
    if not SEQ: print("no views", seed, flush=True); continue
    arr = {}
    for i, s_ in enumerate(SEQ):
        for cm in cams: arr[f"f{i}_{cm}"] = s_["snap"][0][cm]
    pickle.dump({"task": task, "seed": seed, "instruction": sc["instruction"], "n_extra": len(extra), "refs": sc["refs"], "tcp_at_close": sc["tcp_at_close"], "frames": [{"arm": s_["arm"], "attr": s_["attr"], "d_obs": s_["d_obs"], "K": s_["snap"][1], "E": s_["snap"][2], "tcp": s_["snap"][3], "objects": s_["snap"][4]} for s_ in SEQ]}, open(f, "wb"))
    np.savez_compressed(f.replace(".pkl", ".npz"), **arr); done += 1; print(task, done, N, len(SEQ), "move_fail", nfail, flush=True)
open(out + "/DONE", "w").write("ok")
