"""V21 per-task TRAINING-DOMAIN scenes (Clean config only; seeds from 7000000, disjoint from RGL-50 final 4000000+, dev 3000000+/5000000+/6000000+). PRIVILEGED simulator state is used for TRAINING LABELS ONLY (candidate centres, referent identity, expert grasp points), never at inference.
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
done = 0; seed = st
while done < N and seed < st + 600:
    f = f"{out}/scene{seed}.pkl"
    if os.path.exists(f): done += 1; seed += 1; continue
    try:
        with S.Quiet():
            env.setup_demo(now_ep_num=done, seed=seed, is_test=True, **args); rgb_n = frames(env); objs_n = objs(env); X = recorders(env); info = env.play_once(); ok = bool(env.plan_success) and bool(env.check_success())
            if not ok: env.close_env(); seed += 1; print("expert fail", seed - 1, flush=True); continue
            instr = str(np.random.default_rng(seed).choice(generate_episode_descriptions(task, [info["info"]], 10)[0]["seen"])); env.close_env()
            env.setup_demo(now_ep_num=done, seed=seed, is_test=True, **args); r = np.random.default_rng(seed); EXTRA.clear(); taken = [np.array(o["xyz"][:2]) for o in objs(env)]; own = {o["name"] for o in objs(env)}
            cl = [c for c in SPEC if SPEC[c]["modelname"] not in own]; extra = list(r.choice(cl, size=int(r.integers(0, 4)), replace=False)) if cl else []
            for c in extra:
                a, xy = spawn(r, c, taken)
                if a is not None: taken.append(xy); a.set_name(SPEC[c]["modelname"]); EXTRA.append(a)
            for _ in range(10): env.scene.step()
            rgb_d = frames(env); objs_d = objs(env); env.close_env()
    except Exception as e:
        try: env.close_env()
        except Exception: pass
        print("scene fail", seed, type(e).__name__, str(e)[:100], flush=True); seed += 1; continue
    np.savez_compressed(f.replace(".pkl", ".npz"), **{f"n_{k}": v for k, v in rgb_n.items()}, **{f"d_{k}": v for k, v in rgb_d.items()})
    pickle.dump({"task": task, "seed": seed, "instruction": instr, "objects_natural": objs_n, "objects_distractor": objs_d, "n_extra": len(extra), "refs": X["refs"], "tcp_at_close": X["tcp_at_close"]}, open(f, "wb")); done += 1; seed += 1; print(task, done, N, "extra", extra, "refs", X["refs"], flush=True)
open(out + "/DONE", "w").write("ok")
