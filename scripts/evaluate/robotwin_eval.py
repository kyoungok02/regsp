"""[ReGSP] NATIVE RoboTwin evaluation of the frozen V21-CLC policy (``regsp.execution.factory.build_policy``).
Evaluation infrastructure (seed slots, expert-validity screening with the SCRIPTED EXPERT used ONLY to decide whether a seed is a valid episode and for post-hoc
analysis, 3-attempt infrastructure cap, per-seed timeout, native ``eval_success``) is taken unchanged from the research tree's ``v21_eval.py``; the policy never
sees expert or simulator object state. The diagnostic ``--oracle`` option of the research script (privileged expert grasp points) is removed.
usage: robotwin_eval.py --task T --config demo_clean|demo_randomized --n 20 --st_seed S --out DIR
Required environment: REGSP_BACKBONE_CKPT, REGSP_CKPT_ROOT, REGSP_ROBOTWIN_ROOT (docs/INSTALL.md)."""
import sys, os, json, time, argparse, traceback, threading, re, numpy as np, cv2
import sapien  # INFRASTRUCTURE FIX (documented in INFRA_RENDER_FINDING.md): the default OIDN denoiser fails on this GPU ("unsupported device type: CUDA") and leaves ray-traced frames grainy; use OptiX denoising instead (RGL_DENOISER=oidn restores the old behaviour)
_od = sapien.render.set_ray_tracing_denoiser; sapien.render.set_ray_tracing_denoiser = lambda n: _od(os.environ.get("RGL_DENOISER", "optix"))
from regsp.envs import robotwin as S
ap = argparse.ArgumentParser(description="ReGSP native RoboTwin evaluation (frozen V21-CLC configuration, configs/final_v21_clc.json)")
ap.add_argument("--task", required=True, choices=["click_bell", "place_container_plate", "stack_blocks_two", "handover_block"]); ap.add_argument("--config", default="demo_clean", choices=["demo_clean", "demo_randomized"]); ap.add_argument("--n", type=int, default=10); ap.add_argument("--st_seed", type=int, default=1000000)
ap.add_argument("--out", required=True); ap.add_argument("--max_seed_tries", type=int, default=100); ap.add_argument("--instruction_type", default="seen"); ap.add_argument("--seed_timeout", type=int, default=1500); ap.add_argument("--variant", default="ReGSP"); ap.add_argument("--ground", default="v21"); ap.add_argument("--geom", default="live")
a = ap.parse_args(); S.setup(); from generate_episode_instructions import generate_episode_descriptions
from regsp.execution.factory import build_policy
a.out = os.path.abspath(a.out); os.makedirs(a.out, exist_ok=True); env, args = S.make(a.task, a.config); pol = build_policy(a.task, ground=a.ground, geom=a.geom)
done = 0; seed = a.st_seed; results = []; skipped = []; t_start = time.time(); MAXTRY = 3
INFRA_SIG = ("DeviceLost", "cannot create buffer", "vk::", "Vulkan", "out of memory", "OutOfMemory"); infra = lambda s: any(x in s for x in INFRA_SIG)
def attempts(seed):
    f = f"{a.out}/attempts{seed}.txt"; n = int(open(f).read()) if os.path.exists(f) else 0; return n, f
def alog(seed, msg): open(f"{a.out}/attempts{seed}.log", "a").write(f"{time.strftime('%FT%TZ', time.gmtime())} {msg}\n")
def scene_all(env):
    try: return {x.get_name(): np.array(x.get_pose().p).tolist() for x in env.scene.get_all_actors() if x.get_name()}
    except Exception: return {}
def arm_timer(seed):
    def fire(): alog(seed, f"TIMEOUT>{a.seed_timeout}s -> process exit(3) (infrastructure attempt)"); os._exit(3)
    t = threading.Timer(a.seed_timeout, fire); t.daemon = True; t.start(); return t
while done < a.n and seed < a.st_seed + a.max_seed_tries:
    args["render_freq"] = 0; info = None; gt = {}
    if os.path.exists(f"{a.out}/seed{seed}.json"): results.append(json.load(open(f"{a.out}/seed{seed}.json"))); done += 1; seed += 1; continue
    if os.path.exists(f"{a.out}/skip{seed}.json"): seed += 1; continue
    n, af = attempts(seed)
    if n >= MAXTRY:  # three attempts ended in abort / device loss / OOM / timeout: explicit UNRESOLVED infrastructure failure (consumes one of the n slots; validity of the seed unknown)
        log = {"task": a.task, "seed": seed, "config": a.config, "variant": a.variant, "success": False, "infra_failure": True, "attempts": n, "error": "unresolved infrastructure failure: abort/DeviceLost/Vulkan OOM/timeout in all attempts (see attempts%d.log)" % seed}
        json.dump(log, open(f"{a.out}/seed{seed}.json", "w"), indent=1); results.append(log); done += 1; seed += 1; continue
    open(af, "w").write(str(n + 1)); alog(seed, f"attempt {n + 1} start"); tm = arm_timer(seed); ok = False; ex_flags = {}
    try:
        with S.Quiet():
            env.setup_demo(now_ep_num=done, seed=seed, is_test=True, **args); rec = S.install_grasp_hook(env); info = env.play_once(); ex_flags = {"plan_success": bool(env.plan_success), "check_success": bool(env.check_success())}; ok = ex_flags["plan_success"] and ex_flags["check_success"]
        gt = {("L" if s == "left" else "R"): p for s, p in rec.items()}
    except Exception as e:
        try: env.close_env()
        except Exception: pass
        r = f"{type(e).__name__}: {e}"[:300]
        if infra(r): alog(seed, "infra exception in expert screening: " + r[:120]); tm.cancel(); raise SystemExit(3)
        tm.cancel(); json.dump({"reason": type(e).__name__, "detail": r, "stage": "expert_screening"}, open(f"{a.out}/skip{seed}.json", "w")); skipped.append({"seed": seed, "reason": type(e).__name__}); alog(seed, "expert-invalid skip: " + r[:120]); seed += 1; continue
    if not ok:
        tm.cancel(); json.dump({"reason": "expert_fail", "flags": ex_flags}, open(f"{a.out}/skip{seed}.json", "w")); skipped.append({"seed": seed, "reason": "expert_fail"}); alog(seed, f"expert-invalid skip: expert_fail {ex_flags}"); seed += 1; continue
    log = {"task": a.task, "seed": seed, "config": a.config, "variant": a.variant, "calls": [], "error": None, "attempts": n + 1, "expert_flags": ex_flags}; video = []
    try:
        with S.Quiet():
            env.setup_demo(now_ep_num=done, seed=seed, is_test=True, **args)
            instr = np.random.default_rng(seed).choice(generate_episode_descriptions(a.task, [info["info"]], a.n)[0][a.instruction_type]) if info else "pick"; env.set_instruction(instruction=str(instr)); log["instruction"] = str(instr)
            planner_log = []; r = env.robot
            for side in ("left", "right"):
                fn = getattr(r, f"{side}_plan_path")
                def wrap(*aa, _fn=fn, _s=side, **kk): res = _fn(*aa, **kk); planner_log.append((_s, res.get("status"))); return res
                setattr(r, f"{side}_plan_path", wrap)
            log["priv_analysis_only"] = {"scene_start": scene_all(env), "expert_grasp_world": {k: list(map(float, v)) for k, v in gt.items()}}  # PRIVILEGED: failure analysis only, never given to the policy
            t0 = time.time(); success = bool(pol.run(env, str(instr), log, video)); log["wall_s"] = time.time() - t0; log["priv_analysis_only"]["scene_end"] = scene_all(env)
        log["success"] = bool(env.eval_success or success); log["n_calls"] = env.take_action_cnt; log["step_limit"] = env.step_lim; log["planner_failures"] = sum(1 for _, s in planner_log if s != "Success"); log["planner_calls"] = len(planner_log); log["planner_status"] = [s for _, s in planner_log]
    except Exception as e:
        tb = traceback.format_exc()[-1500:]
        if infra(tb):
            try: env.close_env()
            except Exception: pass
            alog(seed, "infra exception in policy rollout: " + tb[-160:]); tm.cancel(); raise SystemExit(3)
        log["success"] = False; log["error"] = tb; log["policy_exception"] = True
    finally:
        try: env.close_env()
        except Exception: pass
    tm.cancel()
    if video:
        vw = cv2.VideoWriter(f"{a.out}/seed{seed}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 5, (640, 480))
        for fr in video: vw.write(fr[:, :, ::-1])
        vw.release()
    alog(seed, f"attempt {n + 1} finished success={log['success']} policy_exception={bool(log.get('policy_exception'))}")
    json.dump(log, open(f"{a.out}/seed{seed}.json", "w"), indent=1, default=str); results.append(log); done += 1; seed += 1
    print(f"[{a.task} {a.config}] {done}/{a.n} seed {log['seed']} success={log['success']} cycles={[(c.get('arm'), c.get('source')) for c in log.get('cycles', [])]} total_succ={sum(r['success'] for r in results)} t={time.time() - t_start:.0f}s", flush=True)
json.dump({"task": a.task, "config": a.config, "variant": a.variant, "n_requested": a.n, "n_episodes": len(results), "n_success": int(sum(r["success"] for r in results)), "n_unresolved_infra": int(sum(bool(r.get("infra_failure")) for r in results)), "st_seed": a.st_seed, "max_seed_tries": a.max_seed_tries, "seeds": [r["seed"] for r in results], "skipped": skipped, "insufficient_valid_episodes": len(results) < a.n, "wall_s": time.time() - t_start}, open(f"{a.out}/summary.json", "w"), indent=1)
open(a.out + "/DONE", "w").write("ok")
