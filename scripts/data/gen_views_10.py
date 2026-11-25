"""[gen_views_10] V21 per-task TRAINING-DOMAIN VIEW SEQUENCES: same as gen_scenes_10 but the scripted expert is run WITH 0..3 distractors spawned and (head, left-wrist, right-wrist) frames + per-camera intrinsics/extrinsics + all object xyz + both TCP poses are recorded every STEP simulator steps along the expert path (half resolution). Used to train class-agnostic proposal heads for WRIST views and the two-stage grounding. PRIVILEGED state = labels only. Original docstring:  (Clean config only; seeds from 7000000, disjoint from RGL-50 final 4000000+, dev 3000000+/5000000+/6000000+). PRIVILEGED simulator state is used for TRAINING LABELS ONLY (candidate centres, referent identity, expert grasp points), never at inference.
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
env, args = S.make(task, "demo_clean")
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

STEP = int(os.environ.get("VIEW_STEP", 100)); CAP = 36; cams = ("head_camera", "left_camera", "right_camera")
def snap(env):
    o = env.get_obs()["observation"]; fr = {}; K = {}; E = {}
    for c in cams: fr[c] = np.ascontiguousarray(o[c]["rgb"][::2, ::2]); K[c] = np.array(o[c]["intrinsic_cv"]); E[c] = np.array(o[c]["extrinsic_cv"])
    tcp = {s: np.array(getattr(env.robot, f"get_{s}_tcp_pose")()) for s in ("left", "right")}; g = {s: float(getattr(env.robot, f"get_{s}_gripper_val")()) for s in ("left", "right")}
    return fr, K, E, tcp, g, objs(env)
done = 0; seed = st
while done < N and seed < st + 900:
    f = f"{out}/view{seed}.pkl"
    if os.path.exists(f) and os.path.exists(f.replace(".pkl", ".npz")): done += 1; seed += 1; continue
    af = f"{out}/att{seed}.txt"; na = int(open(af).read()) if os.path.exists(af) else 0
    if na >= 2: open(out + "/failed_seeds.txt", "a").write(f"{seed} crashed {na}x -> permanently skipped\n"); seed += 1; continue
    open(af, "w").write(str(na + 1))
    try:
        with S.Quiet():
            env.setup_demo(now_ep_num=done, seed=seed, is_test=True, **args); r = np.random.default_rng(seed); EXTRA.clear(); taken = [np.array(o["xyz"][:2]) for o in objs(env)]; own = {o["name"] for o in objs(env)}
            cl = [c for c in SPEC if SPEC[c]["modelname"] not in own]; extra = list(r.choice(cl, size=int(r.integers(0, 4)), replace=False)) if cl else []
            for c in extra:
                a, xy = spawn(r, c, taken)
                if a is not None: taken.append(xy); a.set_name(SPEC[c]["modelname"]); EXTRA.append(a)
            for _ in range(10): env.scene.step()
            X = recorders(env); SEQ = []; cnt = [0]; ostep = env.scene.step; busy = [False]
            def st_(*a, **k):
                r_ = ostep(*a, **k); cnt[0] += 1
                if cnt[0] % STEP == 0 and not busy[0] and len(SEQ) < CAP:
                    busy[0] = True
                    try: SEQ.append((cnt[0],) + snap(env) + (len(X["tcp_at_close"]),))
                    finally: busy[0] = False
                return r_
            env.scene.step = st_; SEQ.append((0,) + snap(env) + (0,)); info = env.play_once(); env.scene.step = ostep; ok = bool(env.plan_success) and bool(env.check_success())
            if not ok: env.close_env(); seed += 1; print("expert fail", seed - 1, flush=True); continue
            instr = str(np.random.default_rng(seed).choice(generate_episode_descriptions(task, [info["info"]], 10)[0]["seen"])); env.close_env()
    except Exception as e:
        try: env.close_env()
        except Exception: pass
        print("view fail", seed, type(e).__name__, str(e)[:100], flush=True)
        if any(x in str(e) for x in ("DeviceLost", "vk::", "Vulkan", "out of memory")): os._exit(3)  # device is gone: exit now; bounded wrapper restarts; seed += 1; continue
    arr = {}
    for i, (c, fr, K, E, tcp, g, ob, nclose) in enumerate(SEQ):
        for cm in cams: arr[f"f{i}_{cm}"] = fr[cm]
    pickle.dump({"task": task, "seed": seed, "instruction": instr, "n_extra": len(extra), "refs": X["refs"], "tcp_at_close": X["tcp_at_close"], "close_events": X.get("events", []), "frames": [{"step": c, "K": K, "E": E, "tcp": tcp, "grip": g, "objects": ob, "n_closed_so_far": nc} for (c, fr, K, E, tcp, g, ob, nc) in SEQ]}, open(f, "wb"))
    np.savez_compressed(f.replace(".pkl", ".npz"), **arr); done += 1; seed += 1; print(task, done, N, len(SEQ), flush=True)
open(out + "/DONE", "w").write("ok")
