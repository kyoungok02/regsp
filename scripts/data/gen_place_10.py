"""[gen_place_10] PLACE-TARGET pairing capture (training labels only; privileged simulator state never used at inference). For each scene seed (7000000+, Clean, no distractors) run the scripted expert once and record, per gripper-RELEASE event (set_gripper to open after a close): arm, TCP xyz (world), step; plus the initial world xyz of every task object (incl. the place target, e.g. handover `target_box`). Output: place_pairs/{task}/pair{seed}.json. Used to (a) fit the per-task constant pad->TCP-release offset (analogous to the AnchorHead constant-offset fallback) and (b) the height prior of the place referent. usage: gen_place_10.py TASK N START_SEED OUTDIR"""
import sys, os, json, numpy as np
from regsp import paths as P

import sapien
_od = sapien.render.set_ray_tracing_denoiser; sapien.render.set_ray_tracing_denoiser = lambda n: _od(os.environ.get("RGL_DENOISER", "optix"))
from regsp.envs import robotwin as S
task, N, st, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), os.path.abspath(sys.argv[4]); os.makedirs(out, exist_ok=True)
from envs.utils.actor_utils import Actor
env, args = S.make(task, "demo_clean"); ORIG_SG = env.set_gripper
def side2a(s): return "L" if "left" in str(s) else "R"
def objs(env):
    H = {}
    for k, v in vars(env).items():
        if isinstance(v, Actor):
            try: nm = v.get_name()
            except Exception: nm = ""
            if nm in ("table", "wall", "ground") or k in ("table", "wall", "ground"): continue
            H[k] = np.array(v.get_pose().p).tolist()
    return H
done = 0; seed = st
while done < N and seed < st + 400:
    f = f"{out}/pair{seed}.json"
    if os.path.exists(f): done += 1; seed += 1; continue
    af = f"{out}/att{seed}.txt"; na = int(open(af).read()) if os.path.exists(af) else 0
    if na >= 2: open(out + "/failed_seeds.txt", "a").write(f"{seed} crashed {na}x\n"); seed += 1; continue
    open(af, "w").write(str(na + 1))
    try:
        with S.Quiet():
            env.setup_demo(now_ep_num=done, seed=seed, is_test=True, **args); o0 = objs(env); ev = []; cl = {"left": False, "right": False}; os_ = ORIG_SG
            def sg(left_pos=None, right_pos=None, set_tag="all", **kk):
                for side, pos in (("left", left_pos), ("right", right_pos)):
                    if pos is None or set_tag not in (side, "all"): continue
                    if pos < 0.5: cl[side] = True
                    elif cl[side]: ev.append({"arm": side2a(side), "tcp_world": list(map(float, getattr(env.robot, f"get_{side}_tcp_pose")()[:3])), "closed_before": True, "objs": objs(env)}); cl[side] = False
                return os_(left_pos=left_pos, right_pos=right_pos, set_tag=set_tag, **kk)
            env.set_gripper = sg; env.play_once(); ok = bool(env.plan_success) and bool(env.check_success()); o1 = objs(env); env.close_env()
        if not ok: print("expert fail", seed, flush=True); seed += 1; continue
        json.dump({"task": task, "seed": seed, "init_objs": o0, "final_objs": o1, "releases": ev}, open(f, "w")); done += 1; seed += 1; print(task, done, N, len(ev), flush=True)
    except Exception as e:
        try: env.close_env()
        except Exception: pass
        print("fail", seed, type(e).__name__, str(e)[:100], flush=True)
        if any(x in str(e) for x in ("DeviceLost", "vk::", "Vulkan", "out of memory")): os._exit(3)
        seed += 1
open(out + "/DONE", "w").write("ok")
