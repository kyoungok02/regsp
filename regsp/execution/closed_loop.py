"""[ReGSP / research-tree clc_rt_policy.py, frozen V21-CLC executor] [V21-CLC] closed-loop grounding-to-action error correction: versioned copy of the frozen pic_rt_policy.py (freeze3; not modified). CLC_* options default off = behaviour identical to pic_rt_policy.py.
[V21-PIC] (versioned copy of the frozen acl_rt_policy.py; the frozen file is not modified) Persistent Instruction-Conditioned closed-loop executor. PIC_LANG=orig|blank|shuffle|wrong|cf|init_only = development-only instruction interventions.
[V21-ACL] Multi-Stage Closed-Loop executor for the genuine V21 pipeline + Adaptive Single/Bimanual TemporalActionDecoder ([X] documented extension experiment).
Scientific identity: Original Counterfactual Grounding + Adaptive Single/Bimanual TemporalActionDecoder + Multi-Stage Closed-Loop Execution. NO RGL/KSP, NO scripted expert actions.
Nothing frozen is modified: the frozen policies (v21_rt_policy.V21Policy / bim_rt_policy.BimV21Policy) are SUBCLASSED; proposals, HR/cycle scorer, two-ray association (<=3 cm), triangulation, AnchorHead, StagePlan head and
build_pregrasp_chunk are the unchanged code paths (called through super()).

Loop (per stage; one stage = one grasp-to-release cycle of one arm):
  OBSERVE   3 RGB cameras + joint/gripper proprioception
  GROUND    V21 proposals -> referent scorer -> active view -> two-view triangulation -> AnchorHead   (re-run for EVERY stage and every retry: no stale grasp_xyz / waypoints)
            sequential MEMORY gates on the proposals: (i) locations where this episode already released an object are excluded; (ii) while another arm is VERIFIED to hold the object (hand-over), the object's location is known
            from that arm's proprioception (donor TCP (+) data-derived receive offset, train-split statistic) -> proposals are restricted to that image region and a grounded point is accepted only if it agrees (<=4 cm), else the
            memory point is used ("memory_receive").  No simulator object pose anywhere.
  PROGRESS  ProgressHead on the same backbone tokens (stage-completion probabilities for every stage so far; logged, optionally used) + MEASURED finger opening (robot proprioception) for physical grasp / retention / release checks
  PLAN      unchanged StagePlan head (rotation / approach / pre_dist / waypoint) ; place target = const | grounded place referent (unchanged BimV21Policy.place_target) | memory-relative (previous verified release + data offset)
  ARM MODE  ProgressHead arm-mode logits (PRIMARY / PARTNER / BIMANUAL) -> conditions the joint decoder and decides whether the partner arm's synchronised chunk is executed
  DECODE    AdaptiveDecoder: joint synchronised 2 x 50-step chunk (primary 7-D + partner 7-D = 14-D); receding horizon: only the first K of the 5 strided control steps are executed, then re-observe / re-decode
  VERIFY    after close: measured finger opening must exceed the calibrated empty-gripper reading (acl_calib.json; rule fixed before evaluation) -> else open, re-ground, retry (<= ACL_RETRY per stage)
            every re-plan: retention check of every holding arm ; hand-over: the donor is released ONLY after the receiver's grasp is verified, and the transfer counts only if the receiver still holds after the donor opened
            release: gated on the place target (<= ACL_GATE_CM, at most 2 suppressions) and verified by the measured opening
  ADVANCE / RECOVER   a stage advances on verified events, never because a chunk finished; bounded retries; hard call budget far below the environment step limit.
`env.eval_success` is only the environment's own episode termination (identical to the frozen protocol: v21_rt_policy.step returns it) - it is never an input to any stage decision.
Privileged values (object poses, check_success) are written to `priv_analysis_only` / feature files for failure analysis and for TRAINING-ONLY progress labels (acl_onpolicy.py); they are never read back by the policy.

env: ACL_CKPT (acl_<tag>.pt; absent -> original decoders, used for the single-arm regression controls) ACL_K (5) ACL_RETRY (2) ACL_CALIB ACL_SAVE_FEATS (dir) ACL_PROG (0) ACL_MAXCALLS (300) ACL_RECV_VIEWS (2) ACL_GATE_CM (2.5)
     toggles (default 1): ACL_VERIFY ACL_MEMGATE ACL_RECEIVE ACL_RETREAT ACL_RELGATE ACL_TERM (grounded terminal correction of the release point) ACL_CARRYCHK (per-step measured retention) ACL_PIVOT (receive point re-grounded from the head-camera referent ray about the donor pinch point); ACL_TRACE=1 = analysis-only privileged trace
ACL_PROG=1: the ProgressHead verdict on a FRESH observation is a DECISION input (not only logged): (D1) after the close, P(FAILED_OR_RECOVERY | stage) > ACL_PROG_TAU rejects the grasp -> open, re-ground, retry. D1 overrides a POSITIVE measured-finger
     verdict only with ACL_PROG_D1=1 (default 0: on development seeds the measured finger test made no error while the head's 3 % false-failure rate cost a correct grasp; D1 decides alone when the finger calibration is disabled for the task);
     (D2) after the measured release, P(FAILED_OR_RECOVERY | stage) > ACL_PROG_TAU means the stage outcome was not achieved -> the stage bookkeeping is undone and the stage is re-grounded and repeated (bounded by ACL_RETRY). A stage therefore advances
     only if the release was measured AND the observation-based verdict does not contradict it. ACL_PROG_TAU default 0.8.
ACL_CONS=1 (first-grasp grounding under domain randomisation; the gripper straddles the object only if the horizontal error is below ~1.2 cm): a grounded point is accepted when TWO active views give triangulations that agree within ACL_CONS_CM (1.5)
     -> their mean is used (halves the single-view noise and rejects wrong head/wrist associations); if the views are exhausted with a single / disagreeing candidates, the candidate closest to the head-ray estimate is used.
     In this mode an AnchorHead horizontal offset larger than 1.5 cm is not applied (the learned offset is either ~0 or unreliable on randomized scenes; its validation set had 7 samples).
ACL_AXIS2D=1 (diagnostic; measured to be equivalent to the 3-D triangulation, not used) (hand-over first-grasp grounding under domain randomisation): the horizontal position of an UPRIGHT ELONGATED referent is taken from the intersection of the head ray and the associated wrist ray(s) PROJECTED ON THE HORIZONTAL PLANE
     (every ray through the silhouette centre passes through the object's vertical axis whatever height the two proposal peaks lie at; the 3-D closest-point midpoint is displaced along the rays when the two peaks sit at different heights of a tall object).
     Used only if the projected rays subtend >= ACL_AZ_MIN degrees; the AnchorHead xy offset is then not applied (height still from the StagePlan head / triangulation). Observation only; rays of all associated views are accumulated (least squares).
ACL_PERTURB="p_grasp,r_max_m,p_place" (ONLY honoured together with ACL_SAVE_FEATS = training-domain data generation; ignored in any evaluation): random horizontal offsets on the grasp point / place target to obtain REAL failed grasps, slips,
     off-target releases and the executor's real recoveries; outcome labels come from the simulator state afterwards (training-only, acl_onpolicy.py), nothing is fabricated.
usage: python acl_rt_policy.py <v21_eval.py args>"""
import sys, os, json, math, time, traceback, numpy as np, torch
from regsp.execution import base_policy as KP, bimanual_policy as BP
import regsp.grounding as R
from regsp.execution.base_policy import To, Ti, FK, STRIDE, eul_rel, mat_rel, Rot, StageVars, build_pregrasp_chunk, PRE_DIST_SCALE_M, WP_SCALE_M, encode, decode
from regsp.paths import data_file
from regsp.decoding.bimanual import BimDecoder
from regsp.decoding.adaptive import AdaptiveDecoder, mode_lr, PROG_NAMES, MODE_NAMES
from regsp.decoding.progress import ProgressHead, PicProgressHead
from regsp.execution.proprio import finger_opening, joint_vec
from regsp.grounding.scorer import SubgoalScorer
dev = "cuda"; IDX = list(range(STRIDE - 1, 50, STRIDE)); INFRA = ("DeviceLost", "vk::", "Vulkan", "out of memory"); PRIV_ATTRS = ("block1", "block2", "box", "target_box", "bell", "container", "plate")
VIEWS = [(0.30, 0), (0.36, -20), (0.26, 20), (0.33, -10), (0.28, 10), (0.40, 0)]
BASE_TOKENS = KP.V21Policy.tokens
def other(a): return "R" if a == "L" else "L"
class Done(BaseException):
    """the environment terminated the episode (frozen-protocol termination signal); BaseException so that no policy-level `except Exception` can swallow it"""


WRONG = {"click_bell": "Shift red block and green block to the center, then position green block on red block.", "place_container_plate": "Pass the red block from the left to the right.", "stack_blocks_two": "Tap the bell's top center.", "handover_block": "Tap the bell's top center."}
def cf_instruction(task, ins):
    """valid same-scene counterfactual instruction (only where one exists): stack_blocks_two = the two block roles exchanged"""
    if task == "stack_blocks_two": return ins.replace("red block", "\0").replace("green block", "red block").replace("\0", "green block")
    raise ValueError("no semantically valid same-scene counterfactual instruction for " + task)
class AclMixin:
    def __init__(self, *a, **k):
        super().__init__(*a, **k); E = os.environ; self.lang_mode = E.get("PIC_LANG", "orig"); self._grasped_once = False; self.log = None; self.sg_dir = E.get("PIC_SAVE_SG"); self.sg_stop = E.get("PIC_SG_STOP") == "1"; self.sg = []; self.recv3d = E.get("PIC_RECV3D", "0") == "1"; self.views = [tuple(float(x) for x in v.split(":")) for v in E["PIC_VIEWS"].split(",")] if E.get("PIC_VIEWS") else VIEWS; self.reach = E.get("PIC_REACH", "0") == "1"; self.probe = E.get("PIC_PROBE", "0") == "1"; self.topdown = E.get("PIC_TOPDOWN", "0") == "1"; self.td_deg = float(E.get("PIC_TOPDOWN_DEG", 12)); self.sp_refresh = E.get("PIC_SP_REFRESH", "0")  # 0 | log (StagePlan re-predicted at every observation view, logged only) | use (the rotation of the LAST observation view replaces the home-pose prediction)
        self.clc_approach = E.get("CLC_APPROACH", "0") in ("1", "2"); self.clc_lead = float(E.get("CLC_LEAD_CM", 6)) / 100 if E.get("CLC_APPROACH") == "2" else 0.0; self.pre_idx = [int(x) for x in E.get("CLC_PRE_IDX", "9,19,29,34,39,44,49").split(",")]  # [CLC] the UNCHANGED pre-grasp chunk (waypoint -> standoff -> grasp, 20/15/15 samples) executed with a SHORT horizon near the grasp: the stride-10 sampling (9,19,29,39,49) never visits the standoff (sample 34) and cuts the corner, so the open gripper reaches the object on a diagonal
        self.clc_assoc = E.get("CLC_ASSOC", "0") == "1"; self.clc_recenter = E.get("CLC_RECENTER", "0") == "1"; self.conv_tol = float(E.get("CLC_CONV_CM", 1.0)) / 100; self.clc_recvxy = E.get("CLC_RECVXY", "0") == "1"; self.clc_fb = E.get("CLC_FB", "0") == "1"; self.clc_reverify = E.get("CLC_REVERIFY", "0") == "1"
        self.clc_near = E.get("CLC_NEAR", "0") == "1"; self.near_d = [float(x) for x in E.get("CLC_NEAR_D", "0.20,0.16").split(",")]; self.near_tol = float(E.get("CLC_NEAR_TOL_CM", 3)) / 100; self.near_tol_fb = float(E.get("CLC_NEAR_TOL_FB_CM", 7)) / 100
        self.clc_stats = json.load(open(E.get("CLC_STATS", data_file("clc_stats.json")))).get(self.task, {}) if os.path.exists(E.get("CLC_STATS", data_file("clc_stats.json"))) else {}
        self.K = int(E.get("ACL_K", 5)); self.max_retry = int(E.get("ACL_RETRY", 2)); self.use_prog = E.get("ACL_PROG", "0") == "1"; self.maxcalls = int(E.get("ACL_MAXCALLS", 300))
        self.recv_views = int(E.get("ACL_RECV_VIEWS", 2)); self.gate_tol = float(E.get("ACL_GATE_CM", 2.5)) / 100; self.opt = {o: E.get("ACL_" + o.upper(), "1") == "1" for o in ("verify", "memgate", "receive", "retreat", "relgate", "fuse", "lowthr", "conf", "term", "carrychk", "pivot")}; self.trace = E.get("ACL_TRACE") == "1"; self.anchor_tol = float(E.get("ACL_ANCHOR_CM", 4)) / 100; self.tau = float(E.get("ACL_PROG_TAU", 0.8)); self.prog_d1 = E.get("ACL_PROG_D1", "0") == "1"; self.axis2d = E.get("ACL_AXIS2D", "0") == "1"; self.cons = E.get("ACL_CONS", "0") == "1"; self.anch_max = float(E.get("ACL_ANCHOR_MAX_CM", 0)) / 100; self.ref_tol = float(E.get("ACL_REF_RAY_CM", float(E.get("ACL_ANCHOR_CM", 4)) + 6)) / 100; self.cons_tol = float(E.get("ACL_CONS_CM", 1.5)) / 100; self.az_min = float(E.get("ACL_AZ_MIN", 15)); self.thr_low = float(E.get("ACL_THR_LOW", 0.02))
        cal = json.load(open(E.get("ACL_CALIB", data_file("acl_calib.json")))).get(self.task, {}); self.cal = cal if (cal.get("enabled") and self.opt["verify"]) else None; self.open_min = float(cal.get("open_min", 0.85))
        self.adec = None; self.PH = None; self.ast = {}; ck = str(self.bundle.path("progress")) if self.bundle.has("progress") else None; self.save_dir = E.get("ACL_SAVE_FEATS"); self.pert = [float(x) for x in E["ACL_PERTURB"].split(",")] if (E.get("ACL_PERTURB") and self.save_dir) else None
        if ck:
            d = torch.load(ck, weights_only=False); bim = getattr(self, "bimdec", None)
            if bim is None: bim = BimDecoder(self.D).to(dev)
            self.adec = AdaptiveDecoder(bim).to(dev); self.adec.load_state_dict(d["mods"]["adec"]); self.PH = (PicProgressHead() if d.get("pic_prog") else ProgressHead()).to(dev); self.PH.load_state_dict(d["mods"]["prog"])
            self.Q.load_state_dict(d["mods"]["tqe"]); self.Vg.load_state_dict(d["mods"]["vgc"]); self.TPH.load_state_dict(d["mods"]["tph"]); self.dm, self.dsd = d["mean"].to(dev), d["std"].to(dev); self.ast = d["stats"]
            for m in (self.adec, self.PH, self.Q, self.Vg, self.TPH): m.eval()
            self._acl = ck
        self.instr_src = E.get("PIC_INSTR", "cached"); self.L0 = None; sgp = str(self.bundle.path("subgoal_scorer")) if self.bundle.has("subgoal_scorer") else None
        if sgp:  # [PIC] stage-conditioned subgoal scorer replaces the cycle scorer (same inputs: instruction-token states, candidate crop tokens, stage index)
            self.csc = SubgoalScorer().to(dev); self.csc.load_state_dict(torch.load(sgp)); self.csc.eval(); self._subgoal = sgp
        self.newg = self.adec is not None or E.get("ACL_NEWG") == "1"  # grounding cross-checks (confidence gate, low-threshold re-proposal, estimator fusion): two-arm experiment only; the single-arm regression controls keep the original grounding path
        print(f"[ACL] task={self.task} ckpt={ck} K={self.K} retry={self.max_retry} verify={'on' if self.cal else 'off'} opt={self.opt} prog_used={self.use_prog}", flush=True)

    # ------------------------------------------------------------------ observation helpers (no simulator state)
    def props(self, imgs): return super().props(imgs)
    def score(self, imgs, instruction, Pf, tok, pad, mod): return super().score(imgs, instruction, Pf, tok, pad, mod)
    def lang(self, instruction):
        """[PIC] development-only instruction interventions (PIC_LANG; default orig = the episode instruction at every VLM pass of the rollout)"""
        m = self.lang_mode
        if m == "orig": return instruction
        if m == "blank": return ""
        if m == "shuffle": w = instruction.split(); return " ".join(w[i] for i in np.random.default_rng(len(instruction) + 7).permutation(len(w)))
        if m == "wrong": return WRONG[self.task]
        if m == "cf": return cf_instruction(self.task, instruction)
        if m == "init_only": return instruction if not self._grasped_once else ""  # the instruction is available only until the first verified grasp of the episode
        raise ValueError(m)
    def alt_instruction(self, instruction):
        """the OTHER valid instruction of the same scene w.r.t. the one being executed (stack_blocks_two only)"""
        eff = self.lang(instruction); return cf_instruction(self.task, eff) if self.task == "stack_blocks_two" else None
    def raw_tokens(self, imgs, text, st7): return BASE_TOKENS(self, imgs, text, st7)  # the frozen policy's own tokenisation + VLM pass with EXACTLY this text (no PIC_LANG substitution)
    def sg_record(self, imgs, obs, instruction, Pf, keep, lg, ft, where):
        """[PIC] TRAINING-DOMAIN DATA GENERATION ONLY (PIC_SAVE_SG): candidate crop tokens of the head proposals + language-token states of the executed AND the alternative valid instruction + PRIVILEGED referent projections (training labels only; never read by the policy)"""
        if not self.sg_dir or not len(Pf): return
        env = self._env; eff = self.lang(instruction); alt = self.alt_instruction(instruction); rows = lambda o: o[0][0][(o[2][0] == 1) & o[1][0].bool()].half().cpu()
        rec = {"where": where, "cyc": self.cur_cyc, "done": self.mem["done"], "n_placed": len(self.mem["placed"]), "P": [list(map(float, p)) for p in Pf], "keep": np.asarray(keep).tolist(), "logits": np.asarray(lg, float).tolist(), "feat": torch.stack(ft).half().cpu(),
               "instr_exec": eff, "instr_alt": alt, "txt_exec": rows(self.raw_tokens(imgs, eff, np.zeros(7))), "txt_alt": None if alt is None else rows(self.raw_tokens(imgs, alt, np.zeros(7))), "priv_training_only": {}, "txt_ok": True}
        E3 = self.head_E(obs)
        for at in PRIV_ATTRS:
            try: pw = np.array(getattr(env, at).get_pose().p, float); (u, v), d = R.project(E3, pw); rec["priv_training_only"][at] = {"uv": [float(u), float(v)], "depth": float(d), "xyz_r": (Ti[:3, :3] @ pw + Ti[:3, 3]).tolist()}
            except Exception: pass
        self.sg.append(rec)
    def tokens(self, imgs, instruction, st7):
        t0 = time.time(); eff = self.lang(instruction); o = super().tokens(imgs, eff, st7); self.stat["vlm_forwards"] += 1; self.stat["t_vlm_s"] += time.time() - t0
        if self.L0 is None and eff: self.L0 = o[0][0][(o[2][0] == 1) & o[1][0].bool()].float()[None]  # PERSISTENT INSTRUCTION STATE: the instruction-token states of the first VLM pass of the episode, kept for every later stage / arm mode / retry
        if getattr(self, "log", None) is not None and self.lang_mode != "orig": self.log.setdefault("pic_lang", {"mode": self.lang_mode, "effective": []}); e_ = self.log["pic_lang"]["effective"]; (e_.append(eff) if eff not in e_ else None)
        return o
    def look(self, env):
        self.stat["renders"] += 1; return env.get_obs()
    def st(self, env, arm):
        """TCP (robot frame, 4x4) and commanded gripper value of one arm from joint proprioception (no camera render)"""
        q = joint_vec(env); a0 = 0 if arm == "L" else 7; return Ti @ FK[arm].tcp(q[a0:a0 + 6], "world"), float(q[a0 + 6])
    def s7(self, env, arm):
        Tr, g = self.st(env, arm); return np.r_[Tr[:3, 3], eul_rel(Tr[:3, :3]), g]
    def act_vec(self, env, poses, grips):  # same as the frozen act_vec, proprioception read without a camera render
        import transforms3d as t3d; q = joint_vec(env); v = []
        for a in "LR":
            if a in poses: p, Rw = poses[a]
            else: T = FK[a].tcp(q[(0 if a == "L" else 7):(6 if a == "L" else 13)], "world"); p, Rw = T[:3, 3], T[:3, :3]
            gv = grips[a] if a in grips else float(q[6 if a == "L" else 13]); v += list(p - Rw[:, 0] * 0.12) + list(t3d.quaternions.mat2quat(Rw)) + [gv]
        return np.array(v)
    def snap(self, env): self.H.append({a: (self.st(env, a)[0][:3, 3].copy(), self.st(env, a)[1]) for a in "LR"})
    def hist_vec(self, arm):
        """both arms' TCP displacement and gripper command 1 and 2 control steps ago (primary first) = the trainer's 10/20-frame history"""
        now = self.H[-1]; out = []
        for a in (arm, other(arm)):
            for lag in (1, 2): h = self.H[max(len(self.H) - 1 - lag, 0)]; out += list(np.clip((h[a][0] - now[a][0]) / 0.05, -1, 1)) + [h[a][1]]
        return np.array(out, np.float32)
    def over(self, env): return env.take_action_cnt >= min(env.step_lim, self.maxcalls) - 1
    def step(self, env, poses, grips, tag, log):
        poses = dict(poses); grips = dict(grips)
        if self.pq:  # synchronised partner motion queued by the joint decoder (retreat of an arm that has finished its stage)
            a, p_w, R_w = self.pq[0]
            if self.mem["hold"][a]: self.pq = []
            elif a not in poses: self.pq.pop(0); poses[a] = (p_w, R_w); grips.setdefault(a, 1.0); tag += f"+retreat_{a}"
        t0 = time.time(); r = super().step(env, poses, grips, tag, log); self.stat["t_env_s"] += time.time() - t0; self.snap(env)
        if self.trace:  # ANALYSIS-ONLY debugging trace (privileged object poses; never read by any decision code)
            tr_ = {"tag": tag, "obj": self.priv(env), "open": {a: round(finger_opening(env, a), 3) for a in "LR"}, "tcp": {a: np.round(self.st(env, a)[0][:3, :4], 4).tolist() for a in "LR"}, "cmd": {a: np.round(np.c_[Ti[:3, :3] @ poses[a][1], Ti[:3, :3] @ poses[a][0] + Ti[:3, 3]], 4).tolist() for a in poses}, "grip_cmd": {a: float(v) for a, v in grips.items()}}
            for at in ("box", "block1", "block2"):
                try: tr_[at + "_R"] = np.round(Ti[:3, :3] @ getattr(env, at).get_pose().to_transformation_matrix()[:3, :3], 3).tolist()
                except Exception: pass
            log.setdefault("trace_analysis_only", []).append(tr_)
        if r:
            if self.save_dir and self.PH is not None and getattr(self, "_ctx", None) is not None and not self._in_chk:
                self._in_chk = True
                try: self.prog_check(env, *self._ctx, "release", {})
                except Exception: pass
            raise Done()
        return r
    def holding(self, env, arm):
        """measured physical grasp test (None when the calibration rule disables it for this task)"""
        return None if self.cal is None else bool(self.cal["thr"] < finger_opening(env, arm) < self.open_min)  # fingers blocked by an object: neither fully closed (empty) nor still open (close command not executed)
    def priv(self, env, full=False):
        """PRIVILEGED - analysis / training-only labels; never returned to any decision code"""
        o = {}
        for at in PRIV_ATTRS:
            try: o[at] = [float(x) for x in (Ti[:3, :3] @ np.array(getattr(env, at).get_pose().p, float) + Ti[:3, 3])]
            except Exception: pass
        if full:
            try: o["check_success"] = bool(env.check_success())
            except Exception: pass
            for at in PRIV_ATTRS:
                try: o[at + "_R"] = (Ti[:3, :3] @ getattr(env, at).get_pose().to_transformation_matrix()[:3, :3]).tolist()
                except Exception: pass
        return o

    # ------------------------------------------------------------------ sequential memory gates on V21 proposals
    def gate(self, Pf, obs, cyc, recv, donor, env):
        keep = np.ones(len(Pf), bool); info = {}
        if not self.opt["memgate"] or not len(Pf): return keep, info
        E3 = obs["observation"]["head_camera"]["extrinsic_cv"]
        if recv is not None:  # the object is in the donor's gripper: image region known from donor proprioception
            ctr = 0.5 * (recv + self.st(env, donor)[0][:3, 3]); (u, v), d = R.project(E3, To[:3, :3] @ ctr + To[:3, 3]); px = float(os.environ.get("ACL_RECV_PX", 90))
            keep = np.array([np.hypot(p[0] - u, p[1] - v) <= px for p in Pf]); info = {"kind": "held_object_region", "uv": [float(u), float(v)], "px": px, "kept": int(keep.sum()), "n": len(Pf)}; return keep, info
        if self.mem["placed"] or self.mem["rejected"]:  # objects already placed by this episode (and objects whose grasp the instruction verification rejected in this stage) must not be selected again
            HE = self.head_E(obs); ex = []; avoid = self.mem["placed"] + self.mem["rejected"]
            for i, p in enumerate(Pf):
                c = self.coarse_xyz(p[:2], cyc, HE)
                if c is not None and min(np.linalg.norm(c[:2] - q[:2]) for q in avoid) < 0.04: keep[i] = False; ex.append(i)
            info = {"kind": "placed_exclusion", "excluded": ex, "n": len(Pf)}
            if not keep.any(): keep[:] = True; info["all_excluded_kept_all"] = True
        return keep, info

    def head_refs(self, imgs, obs, instruction, Pf, tok, pad, mod, kz, recv, donor, env):
        """memory-gated referent candidates of the head camera with a CONFIDENCE check: the referent scorer is trained with a per-proposal binary target, so logit <= 0 for every candidate means "the referent is not among the proposals"
        (proposal-recall failure). RECOVER: re-decode the SAME head heat-map with a lower peak threshold and re-score; the result is only used if it contains a confident referent."""
        def ev(Pc):
            keep, info = self.gate(Pc, obs, kz, recv, donor, env); lg, ft = self.score(imgs, instruction, Pc, tok, pad, mod) if len(Pc) else (np.zeros(0), []); lg = np.asarray(lg, float)
            info["confident"] = bool(len(Pc) and keep.any() and lg[keep].max() > 0); return keep, lg, ft, info
        keep, lg, ft, info = ev(Pf)
        if not info["confident"] and self.newg and self.opt["lowthr"]:
            with torch.no_grad(): Pl = R.decode(self.heads["head_camera"](self.feat(imgs[0]))[0].cpu(), thr=self.thr_low, k=2 * R.KCAP)
            if len(Pl) > len(Pf):
                k2, l2, f2, i2 = ev(Pl); i2["low_thr"] = {"n": len(Pl), "n_default": len(Pf), "confident": i2["confident"]}; self.stat["low_thr"] += 1
                if i2["confident"]: return Pl, k2, l2, f2, i2
                info["low_thr"] = i2["low_thr"]
        return Pf, keep, lg, ft, info
    def assoc_prior(self, pf, Pw, Ef, Er, expect_r, vr):
        """[CLC] PRIOR-CONSISTENT head/wrist association of the instruction-selected head referent. The original association (Hungarian assignment on the two-ray gap over ALL class-agnostic proposals) pairs the referent with any wrist peak whose ray
        passes within 3 cm of the head ray; with randomized clutter that is frequently a peak far from the image centre although the view was commanded to centre the referent. Here every wrist proposal within the ORIGINAL gap gate is triangulated
        with the referent's head ray and the one whose 3-D point is closest to the CURRENT estimate of the referent (the point the view was centred on) is used. Same proposals, same triangulation, same gap gate; no privileged state."""
        c = []
        for j, g in enumerate(Pw):
            pw_, gap = R.triangulate(Ef, pf[:2], Er, g[:2])
            if pw_ is None or gap > R.GAP_MAX: continue
            c.append((float(np.linalg.norm(Ti[:3, :3] @ pw_ + Ti[:3, 3] - expect_r)), j, pw_, float(gap)))
        vr["assoc_prior"] = {"n_wrist": len(Pw), "n_gap_ok": len(c), "dist_to_expected_cm": sorted(round(x[0] * 100, 1) for x in c)}
        if not c: return None
        d_, j, pw_, gap = min(c, key=lambda x: x[0]); return pw_, j, gap
    def near_correct(self, env, instruction, arm, g, Rg, src, cl, log):
        """[CLC] CLOSE-RANGE LATERAL CORRECTION immediately before the approach (observation refresh + re-estimation of the grasp position; no new network).
        Measured decomposition (development traces): the head ray fixes the along-approach coordinate of the grasp point (~0.3 cm) while the lateral coordinate - the one the open fingers must straddle - comes from the WRIST ray, whose metric error
        grows with the camera range. The wrist camera is therefore moved to a short range in the GRASP orientation, centred on the current estimate; the SAME wrist proposal head is decoded and the estimate is shifted horizontally, perpendicular to
        the viewing direction, onto the ray of the proposal that is most consistent with the current estimate (3-D point-to-ray distance <= CLC_NEAR_TOL_CM; larger for fallback estimates). The along-ray coordinate and the height are not changed.
        Bounded: one correction per range in CLC_NEAR_D; a view without a consistent proposal changes nothing."""
        wc = "left_camera" if arm == "L" else "right_camera"; g = np.asarray(g, float).copy(); rec = []; cl["clc_near"] = rec; tol = self.near_tol if src == "grounded" else self.near_tol_fb
        for k, dn in enumerate(self.near_d):
            if self.over(env): break
            p_w, R_w = self.obs_pose(g, Rg, dn); e = {"d_obs": dn, "before": g.tolist()}; rec.append(e)
            try: self.step(env, {arm: (p_w, R_w)}, {arm: 1.0}, f"near_view{k}", log)
            except Exception as ex:
                if any(s_ in str(ex) for s_ in INFRA): raise
                e["moved"] = False; continue
            Tn, _ = self.st(env, arm); e["pose_err_cm"] = round(float(np.linalg.norm((To[:3, :3] @ Tn[:3, 3] + To[:3, 3]) - p_w) * 100), 2)
            if e["pose_err_cm"] > 3.0: e["reached"] = False; continue
            ob = self.look(env); im = self.imgs(ob)
            with torch.no_grad(): Pw = self.props(im)[wc]
            WE = self.wrist_E(ob, arm); c = []
            for j, q in enumerate(Pw):
                o_, d_ = self.ray_r(WE, q[:2]); w_ = g - o_; perp = w_ - (w_ @ d_) * d_; c.append((float(np.linalg.norm(perp)), j, o_, d_))
            e["n_wrist"] = len(Pw); e["ray_dist_cm"] = sorted(round(x[0] * 100, 1) for x in c)
            c = [x for x in c if x[0] <= tol]
            if not c: e["corrected"] = False; continue
            _, j, o_, d_ = min(c, key=lambda x: x[0]); u = d_[:2] / max(np.linalg.norm(d_[:2]), 1e-9); w2 = g[:2] - o_[:2]; lat = float(u[0] * w2[1] - u[1] * w2[0]); g[:2] = g[:2] - lat * np.array([-u[1], u[0]])
            e.update({"corrected": True, "lateral_shift_cm": round(lat * 100, 2), "wrist_uv": [float(x) for x in Pw[j][:2]], "after": g.tolist()}); tol = self.near_tol
        if rec and self.mem["last_view"] is not None and self.mem["last_view"][0] == arm and not self.over(env):  # back to the last regular observation pose: the pre-grasp chunk (waypoint -> lead-in -> standoff) must not start inside the object's neighbourhood
            self.step(env, {arm: (self.mem["last_view"][1], self.mem["last_view"][2])}, {arm: 1.0}, "near_return", log)
        return g
    def verify_placed(self, env, instruction, cyc, arm, cl, log):
        """[CLC] closed-loop MANIPULATION VERIFICATION for the stages without a usable finger test (thin-walled container): after the post-grasp chunks the instruction-selected referent is RE-GROUNDED from a fresh observation (same proposals + referent scorer +
        head-ray estimate) and compared with the MEASURED position where the gripper ended (proprioception). Referent confidently found far from the gripper's end point = the object was not carried -> the stage is repeated (bounded by ACL_RETRY).
        No confident referent -> no decision (the stage is not repeated blindly). No simulator state."""
        Tn, _ = self.st(env, arm); end = Tn[:3, 3].copy(); rec = {"gripper_end_xyz_r": end.tolist()}; cl["clc_verify"] = rec
        self.step(env, {arm: self.to_world(Tn[:3, 3] + np.r_[0, 0, 0.06], Tn[:3, :3])}, {arm: 1.0}, "verify_open_lift", log)
        if self.mem["last_view"] is not None and not self.over(env): a_, p_w, R_w = self.mem["last_view"]; self.step(env, {a_: (p_w, R_w)}, {a_: 1.0}, "verify_view", log)
        obs = self.look(env); imgs = self.imgs(obs); tok, pad, mod, _ = self.tokens(imgs, instruction, np.zeros(7)); P = self.props(imgs)
        Pf, keep, lg, ft, gi = self.head_refs(imgs, obs, instruction, P["head_camera"], tok, pad, mod, cyc, None, None, env); rec["confident"] = bool(gi.get("confident")); rec["n_head"] = len(Pf)
        if not (len(Pf) and keep.any() and gi.get("confident")): rec["decision"] = "no_confident_referent"; return None
        c = self.coarse_xyz(Pf[int(np.argmax(np.where(keep, lg, -np.inf)))][:2], cyc, self.head_E(obs))
        if c is None: rec["decision"] = "no_estimate"; return None
        d_ = float(np.linalg.norm((c - end)[:2])); rec.update({"referent_xyz_r": c.tolist(), "referent_to_gripper_end_cm": round(d_ * 100, 1)}); ok = d_ <= float(os.environ.get("CLC_VERIFY_CM", 10)) / 100; rec["decision"] = "carried" if ok else "not_carried"; return ok
    def ray_r(self, E_, uv):
        c_w, d_w = R.ray(E_, uv); return Ti[:3, :3] @ c_w + Ti[:3, 3], Ti[:3, :3] @ d_w
    @staticmethod
    def axis_xy(lines):
        """least-squares horizontal point closest to the horizontally projected rays [(origin, direction)]; returns (xy, largest pairwise angle in degrees, rms distance)"""
        A = np.zeros((2, 2)); b = np.zeros(2); dn = []
        for o, d in lines:
            u = d[:2] / max(np.linalg.norm(d[:2]), 1e-9); Pm = np.eye(2) - np.outer(u, u); A += Pm; b += Pm @ o[:2]; dn.append(u)
        az = max((float(np.degrees(np.arcsin(min(1.0, abs(dn[i][0] * dn[j][1] - dn[i][1] * dn[j][0]))))) for i in range(len(dn)) for j in range(i)), default=0.0)
        if az < 1.0: return None, az, None
        p = np.linalg.solve(A, b); rms = float(np.sqrt(np.mean([np.sum(((np.eye(2) - np.outer(u, u)) @ (p - o[:2])) ** 2) for (o, _), u in zip(lines, dn)]))); return p, az, rms
    def zfix(self, p, g_sp, cyc, arm, info):
        """stages whose grasp HEIGHT is constant in the training demonstrations (std < 2 mm, bim_stats.json): the height is taken from the StagePlan head (learned regression), the horizontal position from grounding"""
        e = (getattr(self, "bst", None) or {}).get("grasp", {}).get(f"{min(cyc, 1)}_{arm}"); p = np.array(p, float)
        if e is not None and e["std"][2] < 0.002: info["z_from_stageplan_cm"] = round(float(g_sp[2] - p[2]) * 100, 1); p[2] = g_sp[2]
        return p
    def fuse(self, cand, ref, coarse, g_sp, cyc, arm):
        """VERIFY the grounded point by cross-checking three independent estimators (no privileged state): AnchorHead output `cand`, the triangulated reference `ref`, the head-ray x height-prior estimate `coarse`, the StagePlan head `g_sp`.
        gross disagreement of the triangulation with the head ray (> tol + 6 cm) -> the view is rejected; AnchorHead xy disagreeing with the head ray (> tol) -> the learned offset is dropped (triangulated xy kept); height: see zfix."""
        info = {}; out = np.array(cand, float)
        if coarse is not None:
            dr = float(np.linalg.norm(ref[:2] - coarse[:2])); dc = float(np.linalg.norm(out[:2] - coarse[:2])); info.update({"ref_vs_ray_cm": round(dr * 100, 1), "anchor_vs_ray_cm": round(dc * 100, 1)})
            if dr > self.ref_tol: return None, info  # ACL_REF_RAY_CM (default = ACL_ANCHOR_CM + 6): triangulation must agree with the head-ray estimate, else the view is rejected and the next view is tried
            if dc > self.anchor_tol: out[:2] = ref[:2]; info["anchor_xy_dropped"] = True
        return self.zfix(out, g_sp, cyc, arm, info), info

    # ------------------------------------------------------------------ joint decoder + progress / arm-mode head
    @torch.no_grad()
    def acl_decode(self, env, obs, instruction, arm, cyc, target, place, phase, tag):
        t0 = time.time(); M = self.mem; p = other(arm); im = self.imgs(obs); TrP, gP = self.st(env, arm); st7 = np.r_[TrP[:3, 3], Rot.from_matrix(TrP[:3, :3]).as_euler("xyz"), gP]; tok, pad, mod, _ = self.tokens(im, instruction, st7)
        tok = tok.float(); padb = pad.bool(); T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev); n3 = lambda x: ((T(x) - self.dm[:3]) / self.dsd[:3])[None]
        sP = encode(((T(self.s7(env, arm)) - self.dm) / self.dsd)[None], self.dm[3:6], self.dsd[3:6]); sQ = encode(((T(self.s7(env, p)) - self.dm) / self.dsd)[None], self.dm[3:6], self.dsd[3:6])
        tg, pl = n3(target), n3(place); hist = T(self.hist_vec(arm))[None]; mem = torch.cat([tg, pl], -1); stage = min(M["done"], 2) * 4 + int(M["hold"][arm]) * 2 + int(M["hold"][p]); role = min(cyc, 1) * 2 + int(arm == "R"); L = lambda v: torch.tensor([v], dtype=torch.long, device=dev)
        nm = min(cyc, 2) + 1; kw = {}
        if isinstance(self.PH, PicProgressHead) and self.instr_src == "cached" and self.L0 is not None and self.lang_mode != "init_only": kw = {"lang": self.L0.expand(nm, -1, -1), "lmask": torch.ones(nm, self.L0.shape[1], dtype=torch.bool, device=dev)}
        pg, ml = self.PH(tok.expand(nm, -1, -1), padb.expand(nm, -1), sP.expand(nm, -1), sQ.expand(nm, -1), mem.expand(nm, -1), hist.expand(nm, -1), torch.arange(nm, device=dev), **kw)
        pp = pg.softmax(-1).cpu().numpy(); mp = ml[0].softmax(-1).cpu().numpy(); mode = int(mp.argmax())
        q = self.Q(tok, pad, mod); gf = self.Vg(q, tok, pad, mod); pP, pQ = self.adec(tok, padb, sP, sQ, tg, mod, gf, L(phase), L(role), pl, L(mode), L(stage), hist)
        dP = (decode(pP, self.dm[3:6], self.dsd[3:6])[0] * self.dsd + self.dm).float().cpu().numpy().astype(float); dQ = (decode(pQ, self.dm[3:6], self.dsd[3:6])[0] * self.dsd + self.dm).float().cpu().numpy().astype(float)
        self.stat["decodes"] += 1; self.stat["t_dec_s"] += time.time() - t0; pr_ = None
        if self.probe and not tag.startswith("chk_"):  # [PIC] DEVELOPMENT-ONLY intervention probe (logged, never used): the SAME observation / state / grounded target re-encoded with other instructions -> change of the decoder action chunk, arm mode and progress verdict
            pr_ = {}; alts = {"blank": "", "wrong_task": WRONG[self.task], "shuffled": " ".join(reversed(self.lang(instruction).split()))}
            if self.task == "stack_blocks_two": alts["counterfactual"] = cf_instruction(self.task, self.lang(instruction))
            for nm_, txt_ in alts.items():
                tk, pd, md, _ = self.raw_tokens(im, txt_, st7); tk = tk.float(); pdb = pd.bool(); kw2 = {}
                if isinstance(self.PH, PicProgressHead): lr_ = tk[0][(md[0] == 1) & pdb[0]][None]; kw2 = {"lang": lr_.expand(nm, -1, -1), "lmask": torch.ones(nm, lr_.shape[1], dtype=torch.bool, device=dev)} if lr_.shape[1] else {}
                pg2, ml2 = self.PH(tk.expand(nm, -1, -1), pdb.expand(nm, -1), sP.expand(nm, -1), sQ.expand(nm, -1), mem.expand(nm, -1), hist.expand(nm, -1), torch.arange(nm, device=dev), **kw2); q2 = self.Q(tk, pd, md); gf2 = self.Vg(q2, tk, pd, md)
                pP2, pQ2 = self.adec(tk, pdb, sP, sQ, tg, md, gf2, L(phase), L(role), pl, L(mode), L(stage), hist); dP2 = (decode(pP2, self.dm[3:6], self.dsd[3:6])[0] * self.dsd + self.dm).float().cpu().numpy().astype(float); dQ2 = (decode(pQ2, self.dm[3:6], self.dsd[3:6])[0] * self.dsd + self.dm).float().cpu().numpy().astype(float)
                pr_[nm_] = {"P_pos_cm_mean": round(float(np.linalg.norm(dP2[:, :3] - dP[:, :3], axis=1).mean() * 100), 2), "P_pos_cm_max": round(float(np.linalg.norm(dP2[:, :3] - dP[:, :3], axis=1).max() * 100), 2), "Q_pos_cm_max": round(float(np.linalg.norm(dQ2[:, :3] - dQ[:, :3], axis=1).max() * 100), 2),
                            "P_grip_flip": bool(((dP2[IDX, 6] > 0.5) != (dP[IDX, 6] > 0.5)).any()), "Q_grip_flip": bool(((dQ2[IDX, 6] > 0.5) != (dQ[IDX, 6] > 0.5)).any()), "mode_flip": bool(int(ml2[0].argmax()) != mode), "progress_flip": bool(int(pg2[min(cyc, 2)].argmax()) != int(pg[min(cyc, 2)].argmax())), "p_failed": round(float(pg2[min(cyc, 2)].softmax(-1)[3]), 3)}
        info = {"tag": tag, "call": len(self.log["calls"]), "stage": stage, "role": role, "phase": phase, "mode": mode, "arm_mode": mode_lr(mode, arm), "mode_probs": [round(float(x), 3) for x in mp],
                "progress": {str(m): {PROG_NAMES[j]: round(float(pp[m, j]), 3) for j in range(len(PROG_NAMES))} for m in range(nm)}, "grip_P": [round(float(dP[i, 6]), 2) for i in IDX], "grip_Q": [round(float(dQ[i, 6]), 2) for i in IDX],
                "lang_probe": pr_, "span_P_cm": round(float(np.abs(dP[:, :3] - TrP[:3, 3]).max() * 100), 1), "end_to_place_cm": round(float(np.linalg.norm(dP[-1, :3] - place) * 100), 1)}
        if self.save_dir:
            self.feats.append({"tok": tok[0].half().cpu(), "pad": padb[0].cpu(), "sP": sP[0].cpu().numpy(), "sQ": sQ[0].cpu().numpy(), "mem": mem[0].cpu().numpy(), "hist": hist[0].cpu().numpy(), "cyc": cyc, "arm": arm, "tag": tag, "call": info["call"],
                               "stage": stage, "mode_pred": mode, "hold": dict(M["hold"]), "done": M["done"], "finger": {a: finger_opening(env, a) for a in "LR"}, "tcp": {a: self.st(env, a)[0][:3, 3].tolist() for a in "LR"},
                               "target": np.asarray(target, float).tolist(), "place": np.asarray(place, float).tolist(), "priv_training_only": self.priv(env, True), "instr_exec": self.lang(instruction), "lang_mode": self.lang_mode})
            alt = self.alt_instruction(instruction) if tag.startswith("chk_") else None
            if alt is not None: ta, pa_, _, _ = self.raw_tokens(im, alt, st7); self.feats[-1].update({"tok_alt": ta[0].half().cpu(), "pad_alt": pa_[0].bool().cpu(), "instr_alt": alt, "alt_ok": True})
        return dP, dQ, info
    def prog_check(self, env, instruction, arm, cyc, target, place, kind, cl):
        """OBSERVATION-BASED verdict of the ProgressHead for the current stage on a FRESH observation (cameras + proprioception; no simulator state). kind: grasp | transfer | release"""
        if self.PH is None or self.over(env): return None
        _, _, info = self.acl_decode(env, self.look(env), instruction, arm, cyc, target, place, 1, "chk_" + kind); pr = info["progress"][str(min(cyc, 2))]
        rec = {"kind": kind, "stage": cyc, "probs": pr, "verdict": max(pr, key=pr.get), "all_stages": info["progress"], "decision_input": self.use_prog}; cl.setdefault("prog_checks", []).append(rec); self.stat["prog_checks"] += 1
        if self.save_dir and self.feats and kind == "release" and self.mem["placed"]: self.feats[-1]["release_tcp"] = self.mem["placed"][-1].tolist()  # training-only label geometry
        return rec
    def place_for(self, env, cyc, arm, cl):
        """place / hand-over target of this stage and whether it is reliable enough to gate the release"""
        M = self.mem; rel = self.ast.get("place_relative_to_previous_release", {}).get(str(cyc))
        if rel and rel.get("use") and M["placed"]:
            est = M["placed"][-1] + np.array(rel["median"], float); cl["place"] = {"mode": "memory_relative", "previous_release_xyz_r": M["placed"][-1].tolist()}; cl["place_xyz_r"] = est.tolist(); return est, True
        if getattr(self, "bst", None) is None: return np.array(self.pm, float), False
        est = self.place_target(env, self.look(env), cyc, arm, cl); info = cl.get("place", {}); reliable = info.get("mode") == "const" or bool(info.get("grounded"))
        if cyc >= 1 and info.get("mode") == "grounded" and M["place0"] is not None:  # the place referent is static: prefer the estimate taken from the unoccluded episode-start view
            info["live_estimate"] = est.tolist(); info["used"] = "episode_start_estimate"; est = M["place0"].copy(); reliable = True
        cl["place_xyz_r"] = est.tolist(); return est, reliable
    def init_place(self, env, log):
        bst = getattr(self, "bst", None)
        if self.adec is None or not bst or getattr(self, "psc", None) is None: return
        ks = [k for k, e in bst["place"].items() if e["mode"] == "grounded"]
        if not ks: return
        tmp = {}; est = self.place_target(env, self.look(env), int(ks[0][0]), ks[0][2], tmp)
        if tmp.get("place", {}).get("grounded"): self.mem["place0"] = np.asarray(est, float)
        log["place_episode_start"] = {"xyz_r": np.asarray(est, float).tolist(), **{k: v for k, v in tmp.get("place", {}).items() if k != "scores"}}

    # ------------------------------------------------------------------ main loop
    def run(self, env, instruction, log, video=None):
        self._env = env; self.cur_instruction = instruction; self._static = {}; log["calls"] = []; log["cycles"] = []; self.log = log; t_run = time.time()
        self._grasped_once = False; self.L0 = None; self.mem = {"rejected": [], "done": 0, "hold": {"L": False, "R": False}, "placed": [], "retired": set(), "last_view": None, "place0": None}; self._ctx = None; self._in_chk = False; self.pq = []; self.H = []; self.feats = []; self.sg = []
        self.stat = {"vlm_forwards": 0, "decodes": 0, "renders": 0, "t_vlm_s": 0.0, "t_dec_s": 0.0, "t_env_s": 0.0, "retries": 0, "low_thr": 0, "prog_checks": 0, "prog_rejects": 0}; self.prng = np.random.default_rng(int(log.get("seed") or 0) + 17); self.snap(env); held = {}
        try:
            self.init_place(env, log); cyc = 0; attempt = 0; prev_arm = None
            while cyc < self.maxcyc and not self.over(env):
                cl = {"cycle": cyc, "attempt": attempt}; log["cycles"].append(cl)
                try: st = self.run_cycle(env, instruction, cyc, prev_arm, held, cl, log, attempt)
                except Exception:
                    cl["error"] = traceback.format_exc()[-1500:]
                    if any(s in cl["error"] for s in INFRA): raise
                    cl["status"] = "exception"; break
                cl["status"] = st; cl["priv_analysis_only"]["end"] = self.priv(env)
                if st == "advance": prev_arm = cl.get("arm", prev_arm); cyc += 1; attempt = 0; self.mem["rejected"] = []
                elif st == "retry" and attempt < self.max_retry: attempt += 1; self.stat["retries"] += 1
                else: break
            for a in "LR":  # release anything still held at the end (as in the frozen executor)
                if self.mem["hold"][a] or a in held:
                    try: self.step(env, {}, {a: 1.0}, f"final_open_{a}", log)
                    except Exception: pass
        except Done: pass
        finally:
            log["priv_final_analysis_only"] = self.priv(env, True); self.stat["wall_s"] = time.time() - t_run; self.stat["calls"] = int(env.take_action_cnt); self.stat["budget_hit"] = bool(self.over(env))
            log["acl"] = {**{k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.stat.items()}, "K": self.K, "ckpt": getattr(self, "_acl", None), "verify": self.cal is not None, "opt": self.opt, "mem": {"done": self.mem["done"], "hold": dict(self.mem["hold"]), "placed": [p.tolist() for p in self.mem["placed"]]}, "pic": {"lang_mode": self.lang_mode, "subgoal_scorer": getattr(self, "_subgoal", None), "instr_progress_head": isinstance(self.PH, PicProgressHead), "instr_state": self.instr_src, "recv3d": self.recv3d}}
            if self.save_dir and self.feats:
                os.makedirs(self.save_dir, exist_ok=True); fin = self.priv(env, True)
                for f in self.feats: f.update({"seed": log.get("seed"), "cfg": log.get("config"), "task": self.task, "final_priv_training_only": fin, "n_calls_total": len(log["calls"])})
                torch.save(self.feats, f"{self.save_dir}/feat{log.get('seed')}.pt")
            if self.sg_dir and self.sg:
                os.makedirs(self.sg_dir, exist_ok=True); fin = self.priv(env, True)
                for r_ in self.sg: r_.update({"seed": log.get("seed"), "cfg": log.get("config"), "task": self.task, "instruction_env": instruction, "lang_mode": self.lang_mode})
                torch.save({"frames": self.sg, "final_priv_training_only": fin}, f"{self.sg_dir}/sg{log.get('seed')}.pt")
        return bool(env.eval_success)

    def run_cycle(self, env, instruction, cyc, prev_arm, held, cl, log, attempt):
        self.cur_cyc = cyc; M = self.mem; cl["priv_analysis_only"] = {"start": self.priv(env, True)}; cl["mem"] = {"done": M["done"], "hold": dict(M["hold"]), "n_placed": len(M["placed"])}
        if attempt and M["last_view"] is not None and not M["hold"][M["last_view"][0]]:  # RECOVER: back to the last observation pose with an open gripper, then re-observe and re-ground
            a_, p_w, R_w = M["last_view"]; self.step(env, {a_: (p_w, R_w)}, {a_: 1.0}, "recover_view", log)
        # ---- OBSERVE + coarse V21 referent selection (memory-gated)
        obs = self.look(env); imgs = self.imgs(obs); tok, pad, mod, _ = self.tokens(imgs, instruction, np.zeros(7)); P = self.props(imgs); cl["n_props_t0"] = {c: len(v) for c, v in P.items()}
        donor = next((a for a in "LR" if M["hold"][a]), None); recv = None; ro = self.ast.get("receive_offset_donor_frame", {}).get(str(cyc)); recv_stage = ro is not None
        if donor is not None and ro is not None and self.opt["receive"]:
            Td, _ = self.st(env, donor); recv = Td[:3, 3] + Td[:3, :3] @ np.array(ro["median"], float); cl["recv_memory_xyz_r"] = recv.tolist(); cl["donor"] = donor
        kz = 0 if (recv_stage and donor is None) else cyc  # a hand-over stage without a holding donor: the object is on the table again -> table height prior
        Pf, keep, lg, ft, ginfo = self.head_refs(imgs, obs, instruction, P["head_camera"], tok, pad, mod, kz, recv, donor, env); cl["gate"] = ginfo; self.sg_record(imgs, obs, instruction, Pf, keep, lg, ft, "t0"); noprop = not (len(Pf) and keep.any()); xo = None; conf = bool(ginfo.get("confident")); cl["referent_confident"] = conf
        if noprop:
            cl["abort"] = "no_head_proposals" if not len(Pf) else "no_head_proposal_in_memory_region"; xyz_c = recv.copy() if recv is not None else np.array(self.pm, float)
        else:
            sel = int(np.argmax(np.where(keep, lg, -np.inf))); cl["coarse_uv"] = [float(x) for x in Pf[sel][:2]]; cl["scores"] = [float(x) for x in lg]; cl["sel"] = sel
            xyz_c = self.coarse_xyz(Pf[sel][:2], kz, self.head_E(obs)); cl["coarse_xyz_r"] = None if xyz_c is None else xyz_c.tolist()
            if xyz_c is None: xyz_c = np.array(self.pm, float)
            if recv is not None and self.opt["pivot"] and conf:  # RE-GROUND the receive point: the held object pivots in the donor's pinch grasp, so its direction is re-observed (confident head-camera referent ray) instead of assumed rigid
                c_w, d_w = R.ray(self.head_E(obs), Pf[sel][:2]); o_ = Ti[:3, :3] @ c_w + Ti[:3, 3]; d_ = Ti[:3, :3] @ d_w; Pv = self.st(env, donor)[0][:3, 3]; Lr = float(np.linalg.norm(recv - Pv)); cm = 0.5 * (recv + Pv)
                c_obs = o_ + float((cm - o_) @ d_) * d_; u_ = (c_obs - Pv) / max(np.linalg.norm(c_obs - Pv), 1e-6); rn = Pv + Lr * u_; sh = float(np.linalg.norm(rn - recv))
                cl["pivot"] = {"centre_memory": cm.tolist(), "centre_observed": c_obs.tolist(), "recv_memory": recv.tolist(), "recv_observed": rn.tolist(), "shift_cm": round(sh * 100, 1), "used": bool(sh <= 0.08)}
                if sh <= 0.08: recv = rn; cl["recv_memory_xyz_r"] = recv.tolist()
            if recv is not None: xyz_c = recv.copy()
            oth = [i for i in range(len(Pf)) if i != sel]; xo = self.coarse_xyz(Pf[oth[int(np.argmax([Pf[i][2] for i in oth]))]][:2], kz, self.head_E(obs)) if oth else None
        # ---- ARM
        arm = other(donor) if donor is not None else self.pick_arm(xyz_c, prev_arm, xo if xo is not None else xyz_c); cl["arm"] = arm; p = other(arm)
        # ---- PLAN: unchanged StagePlan head
        Tr, g0 = self.state_of(obs, arm); st7 = np.r_[Tr[:3, 3], Rot.from_matrix(Tr[:3, :3]).as_euler("xyz"), g0]; tok1, pad1, mod1, stn = self.tokens(imgs, instruction, st7)
        with torch.no_grad(): o = self.v2(tok1, pad1, mod1, stn)
        g2 = o["grasp_xyz"][0].float().cpu().numpy().astype(float) * self.ps + self.pm; eul = torch.atan2(o["grasp_rot_sc"][0, :, 0], o["grasp_rot_sc"][0, :, 1]).float().cpu().numpy().astype(float)
        r = float(np.clip(o["pre_dist"][0].item() * PRE_DIST_SCALE_M, 0.05, 0.15)); wp_rel = o["wp_rel"][0].float().cpu().numpy().astype(float) * WP_SCALE_M; ad = o["approach_dir"][0].float().cpu().numpy().astype(float); Rg = mat_rel(eul)
        cl["v2"] = {"xyz_fallback": g2.tolist(), "pre_dist": r, "eul_rel": eul.tolist(), "approach": ad.tolist()}
        if self.probe:  # [PIC] development-only: StagePlan head under other instructions (same observation and state)
            cl["v2"]["lang_probe"] = {}; alts = {"blank": "", "wrong_task": WRONG[self.task], "shuffled": " ".join(reversed(self.lang(instruction).split()))}
            if self.task == "stack_blocks_two": alts["counterfactual"] = cf_instruction(self.task, self.lang(instruction))
            for nm_, txt_ in alts.items():
                tk, pd, md, sn = self.raw_tokens(imgs, txt_, st7)
                with torch.no_grad(): o2 = self.v2(tk, pd, md, sn)
                e2 = torch.atan2(o2["grasp_rot_sc"][0, :, 0], o2["grasp_rot_sc"][0, :, 1]).float().cpu().numpy().astype(float); x2 = o2["grasp_xyz"][0].float().cpu().numpy().astype(float) * self.ps + self.pm
                cl["v2"]["lang_probe"][nm_] = {"xyz_cm": round(float(np.linalg.norm(x2 - g2)) * 100, 2), "rot_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(mat_rel(e2).T @ Rg) - 1) / 2, -1, 1)))), 1), "pre_dist_cm": round(abs(float(np.clip(o2["pre_dist"][0].item() * PRE_DIST_SCALE_M, 0.05, 0.15)) - r) * 100, 2)}
        gate_conf = bool(self.newg and self.opt["conf"]); _ =  gate_conf = bool(self.newg and self.opt["conf"])
        if gate_conf and not conf and recv is None: xyz_c = g2.copy(); cl["unconfident_referent"] = "no confident referent among the head proposals -> the active views are centred on the StagePlan estimate; a grounded point needs a confident referent"
        # ---- ARM-MODE SWITCH: the arm that finished the previous stage retreats, synchronised with this arm's approach (partner chunk of the joint decoder)
        retreat = bool(self.adec is not None and self.opt["retreat"] and prev_arm is not None and prev_arm != arm and prev_arm in M["retired"] and not M["hold"][prev_arm])
        if retreat: self.queue_retreat(env, obs, instruction, arm, cyc, xyz_c, cl, "retreat0")
        # ---- active view + genuine V21 two-view grounding (re-run for every stage / retry)
        g_final, src = g2, "v2_fallback"; cl["views"] = []; lines2d = []; cands = []; eul_ref_last = None; dseq = [] if (noprop and recv is None) else (self.views if (self.reach and recv is None) else self.views[:self.nview]); used = 0
        if recv is not None: dseq = VIEWS[:self.nview][:self.recv_views]
        if attempt and dseq: k_ = attempt % len(dseq); dseq = dseq[k_:] + dseq[:k_]
        wc = "left_camera" if arm == "L" else "right_camera"; vcentre = np.asarray(xyz_c, float).copy(); ests = []; rrefs = []; expect = vcentre.copy() if recv is None else 0.5 * (recv + self.st(env, donor)[0][:3, 3])
        for vi, (dobs, pitch) in enumerate(dseq):
            if self.over(env) or (self.reach and recv is None and used >= self.nview): break  # [PIC] REACHABILITY-AWARE ACTIVE VIEWS: a commanded observation pose that the arm did not reach (measured, proprioception) does not consume one of the nview grounding views; the next candidate view is tried
            vr = {"attempt": vi, "d_obs": dobs, "tilt_deg": pitch}; p_w, R_w = self.obs_pose(vcentre, Rg @ Rot.from_euler("y", pitch, degrees=True).as_matrix(), dobs)
            if self.clc_recenter and recv is None and ests: vr["centred_on_previous_estimate"] = True
            try: self.step(env, {arm: (p_w, R_w)}, {arm: 1.0}, f"obs_view{vi}", log); vr["moved"] = True
            except Exception as e:
                if any(s in str(e) for s in INFRA): raise
                vr["moved"] = False; cl["views"].append(vr); continue
            M["last_view"] = (arm, p_w, R_w); Tn, _ = self.st(env, arm); vr["pose_err_cm"] = float(np.linalg.norm((To[:3, :3] @ Tn[:3, 3] + To[:3, 3]) - p_w) * 100); vr["reached"] = bool(vr["pose_err_cm"] <= 3.0); used += int(vr["reached"])
            ob2 = self.look(env); im2 = self.imgs(ob2); Pn = self.props(im2); Pf2, Pw = Pn["head_camera"], Pn[wc]; vr.update({"n_head": len(Pf2), "n_wrist": len(Pw)})
            if not Pw or not (Pf2 or (self.newg and self.opt["lowthr"])): cl["views"].append(vr); continue
            if self.sp_refresh != "0":  # [PIC] closed-loop StagePlan: re-plan the grasp rotation from the refreshed (close wrist view) observation instead of keeping the home-pose prediction
                Tv, gv = self.st(env, arm); tkv, pdv, mdv, stv = self.tokens(im2, instruction, np.r_[Tv[:3, 3], Rot.from_matrix(Tv[:3, :3]).as_euler("xyz"), gv])
                with torch.no_grad(): ov = self.v2(tkv, pdv, mdv, stv)
                ev = torch.atan2(ov["grasp_rot_sc"][0, :, 0], ov["grasp_rot_sc"][0, :, 1]).float().cpu().numpy().astype(float); vr["v2_refresh"] = {"eul_rel": ev.tolist(), "xyz": (ov["grasp_xyz"][0].float().cpu().numpy().astype(float) * self.ps + self.pm).tolist()}; eul_ref_last = ev
            tok2, pad2, mod2, _ = self.tokens(im2, instruction, np.zeros(7)); Pf2, keep2, lg2, ft2, g2i = self.head_refs(im2, ob2, instruction, Pf2, tok2, pad2, mod2, kz, recv, donor, env); vr["gate"] = g2i; self.sg_record(im2, ob2, instruction, Pf2, keep2, lg2, ft2, f"view{vi}")
            if not (len(Pf2) and keep2.any()): vr["ok"] = False; cl["views"].append(vr); continue
            if gate_conf and not g2i["confident"]: vr["ok"] = False; vr["rejected"] = "no_confident_referent"; cl["views"].append(vr); continue
            s2 = int(np.argmax(np.where(keep2, lg2, -np.inf)))
            if self.clc_assoc:
                ap_ = self.assoc_prior(Pf2[s2], Pw, self.head_E(ob2), self.wrist_E(ob2, arm), expect, vr); A = {} if ap_ is None else {s2: ap_}
            else: A = R.associate(Pf2, Pw, self.head_E(ob2), self.wrist_E(ob2, arm))
            vr["assoc"] = sorted(int(x) for x in A.keys()); vr["selected"] = s2
            if s2 in A:
                ref_w, j, gap = A[s2]; ref = Ti[:3, :3] @ ref_w + Ti[:3, 3]; off = np.zeros(3)
                if recv is None:  # [PIC] ray geometry of the associated pair (always logged). TOP-DOWN WRIST REFINEMENT (PIC_TOPDOWN=1): when the wrist ray of the referent is within PIC_TOPDOWN_DEG of the vertical, the horizontal position is taken from
                    # the wrist ray alone (intersection with the horizontal plane at the triangulated height): the wrist camera is ~3x closer than the head camera, and a near-vertical ray is insensitive to the height estimate
                    (oh, dh), (ow, dw) = self.ray_r(self.head_E(ob2), Pf2[s2][:2]), self.ray_r(self.wrist_E(ob2, arm), Pw[j][:2]); vert = float(np.degrees(np.arccos(min(1.0, abs(dw[2]))))); sw = (ref[2] - ow[2]) / dw[2] if abs(dw[2]) > 1e-6 else -1.0
                    vr["rays"] = {"head": [oh.tolist(), dh.tolist()], "wrist": [ow.tolist(), dw.tolist()], "wrist_from_vertical_deg": round(vert, 1), "head_uv": [float(x) for x in Pf2[s2][:2]], "wrist_uv": [float(x) for x in Pw[j][:2]]}
                    if sw > 0: pw_ = ow + sw * dw; vr["rays"]["wrist_plane_xy"] = pw_[:2].tolist()
                    if self.topdown and sw > 0 and vert <= self.td_deg and np.linalg.norm(pw_[:2] - ref[:2]) <= 0.04: vr["topdown"] = {"midpoint_xy": ref[:2].tolist(), "shift_cm": round(float(np.linalg.norm(pw_[:2] - ref[:2])) * 100, 2)}; ref = np.r_[pw_[:2], ref[2]]
                if self.anchor is not None:
                    if self.anchor["keep"]:
                        with torch.no_grad(): off = self.ah(ft2[s2][None], self.crop(im2[R.CAMS.index(wc)], Pw[j][0], Pw[j][1])[None], torch.tensor(ref, dtype=torch.float32, device=dev)[None])[0].cpu().numpy() / 100
                    else: off = self.anchor["mean_off"]
                if (self.cons or self.anch_max > 0) and recv is None and float(np.linalg.norm(np.asarray(off)[:2])) > (self.anch_max if self.anch_max > 0 else 0.015): off = np.r_[0.0, 0.0, np.asarray(off)[2]]; vr["anchor_xy_gated"] = True  # ACL_ANCHOR_MAX_CM: a learned AnchorHead horizontal offset larger than this is not applied (triangulated xy kept)
                cand = ref + off; vr.update({"ref": ref.tolist(), "offset_cm": (np.asarray(off) * 100).tolist(), "gap_m": float(gap)})
                if recv is not None and "recv_ref3d" not in cl: cl["recv_ref3d"] = ref.tolist()
                if recv is not None and self.clc_recvxy:  # [CLC] collect the triangulated centre of the held object from every receive view (no early exit)
                    rrefs.append(np.asarray(ref, float)); vr["ok"] = True; vr["recv_centre_vs_memory_cm"] = round(float(np.linalg.norm(ref - expect)) * 100, 1); cl["views"].append(vr); continue
                if recv is None and self.newg:  # horizontal-plane (vertical-axis) triangulation; always logged, used with ACL_AXIS2D=1
                    hl = self.ray_r(self.head_E(ob2), Pf2[s2][:2]); wl = self.ray_r(self.wrist_E(ob2, arm), Pw[j][:2]); lines2d = [hl] + lines2d[1:] + [wl]; pxy, az, rms = self.axis_xy(lines2d)
                    vr["axis2d"] = {"xy": None if pxy is None else pxy.tolist(), "az_deg": round(az, 1), "rms_cm": None if rms is None else round(rms * 100, 2), "n_rays": len(lines2d), "pair_xy": (lambda q: None if q[0] is None else q[0].tolist())(self.axis_xy([hl, wl])), "used": False}
                    if self.axis2d and pxy is not None and az >= self.az_min and np.linalg.norm(pxy - ref[:2]) <= 0.10: ref = np.r_[pxy, ref[2]]; cand = ref.copy(); vr["axis2d"]["used"] = True
                if recv is None and self.newg and self.opt["fuse"]:
                    cand, vr["fuse"] = self.fuse(cand, ref, self.coarse_xyz(Pf2[s2][:2], kz, self.head_E(ob2)), g2, cyc, arm)
                    if cand is None: vr["ok"] = False; vr["rejected"] = "triangulation_inconsistent_with_head_ray"; cl["views"].append(vr); continue
                if recv is None and M["placed"] and self.opt["memgate"] and min(np.linalg.norm(cand[:2] - q[:2]) for q in M["placed"]) < 0.03: vr["ok"] = False; vr["rejected"] = "grounded_point_is_an_already_placed_object"; cl["views"].append(vr); continue
                if self.cons and recv is None:  # two-view CONSENSUS
                    cands.append(np.asarray(cand, float)); vr["ok"] = True; cl["views"].append(vr)
                    pair = next(((a_, b_) for i_, a_ in enumerate(cands) for b_ in cands[:i_] if np.linalg.norm((a_ - b_)[:2]) <= self.cons_tol), None)
                    if pair is not None: g_final = 0.5 * (pair[0] + pair[1]); src = "grounded"; cl["consensus"] = {"n_candidates": len(cands), "agree_cm": round(float(np.linalg.norm((pair[0] - pair[1])[:2])) * 100, 2)}; break
                    continue
                if self.clc_recenter and recv is None:  # [CLC] CLOSED-LOOP ACTIVE GROUNDING: the next view is centred on the CURRENT estimate (not on the stale coarse point); the estimate is accepted when two successive estimates agree (convergence), else the loop continues (bounded by nview)
                    ests.append(np.asarray(cand, float)); vr["ok"] = True; cl["views"].append(vr); vcentre = np.asarray(ref, float).copy(); expect = vcentre.copy()
                    if len(ests) >= 2 and np.linalg.norm((ests[-1] - ests[-2])[:2]) <= self.conv_tol: g_final = 0.5 * (ests[-1] + ests[-2]); src = "grounded"; cl["clc_ground"] = {"converged": True, "n_estimates": len(ests), "last_change_cm": round(float(np.linalg.norm((ests[-1] - ests[-2])[:2])) * 100, 2)}; break
                    continue
                g_final = cand; src = "grounded"; vr["ok"] = True; cl["views"].append(vr); break
            vr["ok"] = False; cl["views"].append(vr)
        if self.clc_recenter and recv is None and src != "grounded" and ests:
            g_final = ests[-1]; src = "grounded"; cl["clc_ground"] = {"converged": False, "n_estimates": len(ests), "changes_cm": [round(float(np.linalg.norm((ests[i] - ests[i - 1])[:2])) * 100, 2) for i in range(1, len(ests))]}
        if self.cons and recv is None and src != "grounded" and cands:
            g_final = min(cands, key=lambda c_: np.linalg.norm((c_ - xyz_c)[:2])) if not noprop else cands[0]; src = "grounded"; cl["consensus"] = {"n_candidates": len(cands), "agree_cm": None, "single_or_disagreeing": True}
        if recv is not None:  # hand-over: vision and memory must agree, else the proprioceptive memory of the held object is used
            dv = float(np.linalg.norm(g_final - recv)) if src == "grounded" else None; cl["recv_gate"] = {"vision_source": src, "vision_minus_memory_cm": None if dv is None else dv * 100}
            if src != "grounded" or dv > 0.04: g_final, src = recv.copy(), "memory_receive"
            if self.clc_recvxy:  # [CLC] RE-GROUNDED RECEIVE POINT: the held object is upright, so the horizontal position of the receive point equals that of the object's triangulated centre (current observation); the height stays the proprioceptive memory value.
                # used when two views agree (<= CLC_CONV_CM) or a single view lies within 3 cm of the memory point; otherwise the memory point is kept
                ok_ = [r_ for r_ in rrefs if np.linalg.norm(r_ - expect) <= 0.06]; pair = next(((a_, b_) for i_, a_ in enumerate(ok_) for b_ in ok_[:i_] if np.linalg.norm((a_ - b_)[:2]) <= self.conv_tol), None); xy = None
                if pair is not None: xy = 0.5 * (pair[0] + pair[1])[:2]; how = "two_views_agree"
                elif len(ok_) == 1 and np.linalg.norm(ok_[0][:2] - recv[:2]) <= 0.03: xy = ok_[0][:2]; how = "single_view"
                cl["clc_recv"] = {"n_centres": len(rrefs), "n_consistent": len(ok_), "centres": [r_.tolist() for r_ in rrefs], "memory": recv.tolist(), "used": xy is not None}
                if xy is not None: cl["clc_recv"].update({"how": how, "shift_cm": round(float(np.linalg.norm(xy - recv[:2])) * 100, 2)}); g_final = np.r_[xy, recv[2]]; src = "memory_receive"
            if "recv_ref3d" in cl:  # [PIC] 3-D re-grounding of the receive point: the held object's centre is TRIANGULATED (head + receiver wrist view, memory-gated proposals); the receive point is the memory point rotated about the donor pinch point towards the observed centre (adds the depth component the head-ray pivot cannot observe)
                Pv = self.st(env, donor)[0][:3, 3]; c3 = np.array(cl["recv_ref3d"], float); cm_ = 0.5 * (recv + Pv); Lr = float(np.linalg.norm(recv - Pv)); dc = float(np.linalg.norm(c3 - cm_)); u_ = (c3 - Pv) / max(np.linalg.norm(c3 - Pv), 1e-6); r3 = Pv + Lr * u_
                cl["recv3d"] = {"centre_triangulated": c3.tolist(), "centre_memory": cm_.tolist(), "centre_diff_cm": round(dc * 100, 1), "recv_3d": r3.tolist(), "shift_cm": round(float(np.linalg.norm(r3 - recv)) * 100, 1), "used": False}
                if self.recv3d and src == "memory_receive" and dc <= 0.06 and np.linalg.norm(r3 - recv) <= 0.05: g_final = r3; cl["recv3d"]["used"] = True
        else:
            fo = self.clc_stats.get("grasp_minus_coarse", {}).get(arm)
            if self.clc_fb and src == "v2_fallback" and not noprop and conf and fo is not None and cl.get("coarse_xyz_r") is not None:  # [CLC] GROUNDED FALLBACK: two-view grounding failed but the head camera holds a CONFIDENT instruction-selected referent -> the grasp point is the
                # head-ray estimate of the referent + the data-derived offset between that estimate and the two-view grounded grasp point (clc_stats.json; training-domain rollouts of this policy, no privileged state), instead of the ungrounded StagePlan regression
                cl["clc_fallback"] = {"stageplan_xyz_r": np.asarray(g_final, float).tolist(), "offset": fo, "stageplan_minus_grounded_fallback_cm": round(float(np.linalg.norm((np.asarray(g_final) - (np.array(cl["coarse_xyz_r"]) + np.array(fo)))[:2])) * 100, 1)}; g_final = np.array(cl["coarse_xyz_r"], float) + np.array(fo, float); src = "coarse_offset_fallback"
            g_final, src = self.fallback_choice(g_final, src, None if noprop else xyz_c, cl, kz, arm)
            if src == "coarse_fallback" and self.newg and self.opt["fuse"]: cl["fuse_fallback"] = {}; g_final = self.zfix(g_final, g2, cyc, arm, cl["fuse_fallback"])
        if self.clc_near and recv is None and not noprop and src in ("grounded", "coarse_fallback", "coarse_offset_fallback"): g_final = self.near_correct(env, instruction, arm, np.asarray(g_final, float), Rg, src, cl, log)
        cl["source"] = src if not (noprop and recv is None) else "v2_fallback_no_proposals"; cl["g_final"] = np.asarray(g_final, float).tolist(); cl["g_v2_fallback"] = g2.tolist()
        if self.sg_stop and self.sg_dir and cyc >= 1: cl["sg_stop"] = True; raise Done()  # data generation only: the stage-1 grounding frames are recorded, the rest of the episode is not needed
        if self.pert and self.prng.random() < self.pert[0]:  # DATA GENERATION ONLY (training-domain rollouts): real failures / recoveries
            a_ = self.prng.uniform(0, 2 * np.pi); r_ = self.prng.uniform(0.012, self.pert[1]); g_final = np.asarray(g_final, float) + np.r_[r_ * np.cos(a_), r_ * np.sin(a_), 0.0]; cl["perturb_grasp_cm"] = round(r_ * 100, 1)
        if self.sp_refresh == "use" and eul_ref_last is not None and recv is None: cl["v2"]["eul_rel_home"] = eul.tolist(); eul = eul_ref_last; cl["v2"]["eul_rel"] = eul.tolist(); cl["v2"]["rotation_source"] = "observation_view"
        # ---- PRE_GRASP: unchanged analytic chunk builder executed as ee calls
        sv = StageVars(np.asarray(g_final, float), eul, ad, r, True, np.asarray(g_final, float) + wp_rel); arrived = False
        for rep in range(4):
            if self.over(env): break
            if retreat and not self.pq and rep == 0: self.queue_retreat(env, self.look(env), instruction, arm, cyc, np.asarray(g_final, float), cl, "retreat1")
            Tr, _ = self.st(env, arm); pos, eu = build_pregrasp_chunk(Tr[:3, 3].copy(), eul_rel(Tr[:3, :3]), sv)
            if self.clc_lead > 0 and len(pos) == 50 and np.linalg.norm(Tr[:3, 3] - g_final) >= 0.05:
                # [CLC] ALIGNED LEAD-IN (CLC_APPROACH=2): the measured traces show the finger tips reach ~5 cm beyond the tool point, so the lateral/descending sweep from the waypoint must end BEHIND the standoff. The waypoint (sample 19) is followed by a lead-in point
                # L = grasp - (pre_dist + CLC_LEAD_CM) * a, the standoff S = grasp - pre_dist * a and a straight three-step approach along a. a = finger (tool x) axis of the StagePlan grasp rotation when it is within 45 deg of the StagePlan approach_dir
                # (the two StagePlan outputs are separate regressions; an approach not parallel to the fingers sweeps them sideways), else approach_dir. Same StagePlan quantities, same waypoint, same grasp pose.
                G_ = np.asarray(g_final, float); a_ = np.asarray(ad, float) / max(np.linalg.norm(ad), 1e-9); xg = mat_rel(eul)[:, 0]; ang = float(np.degrees(np.arccos(np.clip(a_ @ xg, -1, 1)))); a_ = xg if ang <= 45 else a_
                S_ = G_ - r * a_; pos = pos.copy(); eu = eu.copy(); pos[29] = G_ - (r + self.clc_lead) * a_; pos[34] = S_; pos[39] = S_ + (G_ - S_) / 3; pos[44] = S_ + 2 * (G_ - S_) / 3; eu[29] = eu[34]
                if rep == 0: cl["lead_in"] = {"approach_vs_finger_axis_deg": round(ang, 1), "axis": "finger" if ang <= 45 else "approach_dir", "lead_cm": self.clc_lead * 100}
            for i in (self.pre_idx if (self.clc_approach and len(pos) == 50 and np.linalg.norm(Tr[:3, 3] - g_final) >= 0.05) else IDX):
                self.step(env, {arm: self.to_world(pos[i], mat_rel(eu[i]))}, {arm: 1.0}, f"pre{rep}_{i}", log)
                if self.clc_approach and i == 34:  # [CLC] STANDOFF CHECK (commanded vs measured end-effector state, proprioception): the straight final approach starts only from the standoff point pre = grasp - pre_dist * approach_dir
                    Tm, _ = self.st(env, arm); e_ = float(np.linalg.norm(Tm[:3, 3] - pos[i])); cl.setdefault("standoff", []).append({"rep": rep, "cmd_minus_measured_cm": round(e_ * 100, 2), "lateral_cm": round(float(np.linalg.norm((Tm[:3, 3] - g_final) - ((Tm[:3, 3] - g_final) @ ad) * ad / max(float(ad @ ad), 1e-9))) * 100, 2)})
                    if e_ > 0.01 and not self.over(env): self.step(env, {arm: self.to_world(pos[i], mat_rel(eu[i]))}, {arm: 1.0}, f"pre{rep}_{i}r", log); Tm, _ = self.st(env, arm); cl["standoff"][-1]["after_repeat_cm"] = round(float(np.linalg.norm(Tm[:3, 3] - pos[i])) * 100, 2)
            Tr, _ = self.st(env, arm); d = float(np.linalg.norm(Tr[:3, 3] - g_final)); cl.setdefault("arrival_dist_cm", []).append(d * 100)
            if d <= 0.02: arrived = True; break
        cl["arrived"] = arrived; gp = self.to_world(np.asarray(g_final, float), mat_rel(eul)); tr = []
        # ---- CLOSE + VERIFY the physical grasp from the measured finger opening
        for _ in range(2): self.step(env, {arm: gp}, {arm: 0.0}, "close", log); tr.append(finger_opening(env, arm))
        if self.cal is not None:
            for k_ in range(3):
                if abs(tr[-1] - tr[-2]) < 0.01: break
                self.step(env, {arm: gp}, {arm: 0.0}, f"settle{k_}", log); tr.append(finger_opening(env, arm))
        ok = self.holding(env, arm); cl["grasp_verify"] = {"finger_trace": [round(x, 4) for x in tr], "thr": None if self.cal is None else self.cal["thr"], "holding": ok}
        if donor is not None: cl["grasp_verify"]["donor_holding"] = self.holding(env, donor)
        pl_ = None
        if self.adec is not None:  # D1: observation-based grasp verdict of the ProgressHead (fresh observation), for every close
            pl_ = self.place_for(env, cyc, arm, cl); cl["place_reliable"] = pl_[1]; pc = self.prog_check(env, instruction, arm, cyc, np.asarray(g_final, float), pl_[0], "grasp", cl)
            if pc: pc["finger_holding"] = ok
            if ok is not False and self.use_prog and pc and pc["probs"]["FAILED_OR_RECOVERY"] > self.tau and (self.prog_d1 or ok is None):
                pc["rejected"] = True; self.stat["prog_rejects"] += 1; cl["prog_reject"] = "grasp"
                if ok: self.mem["rejected"].append(np.asarray(g_final, float).copy()); cl["rejected_referent_memory"] = True  # the fingers hold an object but the instruction-conditioned verdict says the stage is not in progress -> wrong referent: put it down, do not select it again in this stage
                ok = False
        if ok is False:  # miss -> open, re-ground, retry (bounded)
            Tn, _ = self.st(env, arm); self.step(env, {arm: self.to_world(Tn[:3, 3], Tn[:3, :3])}, {arm: 1.0}, "open_after_miss", log); cl["grasp_miss"] = True; return "retry"
        M["hold"][arm] = True; self._grasped_once = True
        if recv_stage and donor is None: M["done"] = max(M["done"], cyc)  # the transfer stage ended without a transfer (object re-grasped from the table)
        if self.adec is None:  # original decoders (single-arm regression controls): unchanged post-grasp path of the frozen policy
            self.after_close(env, arm, cl, log)
            for a_ in [x for x in list(held) if x != arm]:
                Tn, _ = self.st(env, a_); self.step(env, {a_: self.to_world(Tn[:3, 3], Tn[:3, :3])}, {a_: 1.0}, f"deferred_release_{a_}", log); held.pop(a_, None); M["hold"][a_] = False; M["done"] += 1; cl["released_prev"] = a_
            held[arm] = True; super().post_grasp(env, instruction, cyc, arm, np.asarray(g_final, float), held, cl, log); M["hold"][arm] = arm in held
            if self.clc_reverify and attempt < self.max_retry and not self.over(env):
                vp = self.verify_placed(env, instruction, cyc, arm, cl, log)
                if vp is False: held.pop(arm, None); M["hold"][arm] = False; cl["clc_retry"] = True; return "retry"
            if arm not in held: M["done"] += 1; M["retired"].add(arm)
            return "advance"
        return self.acl_post(env, instruction, cyc, arm, np.asarray(g_final, float), cl, log, pl_)
    def queue_retreat(self, env, obs, instruction, arm, cyc, target, cl, tag):
        place, _ = self.place_for(env, cyc, arm, {}); dP, dQ, info = self.acl_decode(env, obs, instruction, arm, cyc, target, place, 0, tag); p = other(arm); cl.setdefault("retreat", []).append(info)
        if info["mode"] in (1, 2): self.pq = [(p, *self.to_world(dQ[i, :3], mat_rel(dQ[i, 3:6]))) for i in IDX]

    # ------------------------------------------------------------------ POST_GRASP: joint decoder, receding horizon, verified transitions
    def acl_post(self, env, instruction, cyc, arm, g_final, cl, log, pl_=None):
        M = self.mem; p = other(arm); cl["post"] = []; cl["acl"] = True; self.pq = []; place, reliable = pl_ if pl_ is not None else self.place_for(env, cyc, arm, cl); cl["place_reliable"] = reliable
        if self.pert and self.prng.random() < self.pert[2]:
            a_ = self.prng.uniform(0, 2 * np.pi); r_ = self.prng.uniform(0.03, 0.06); place = np.asarray(place, float) + np.r_[r_ * np.cos(a_), r_ * np.sin(a_), 0.0]; cl["perturb_place_cm"] = round(r_ * 100, 1)
        self._ctx = (instruction, arm, cyc, g_final, place)
        donor_cycle = bool(self.defer and cyc + 1 < self.maxcyc); n_it = math.ceil(self.post_chunks * 5 / self.K) + 3; sup = 0; wait = 0; released = False; delta = np.zeros(3)
        for it in range(n_it):
            if self.over(env): cl["budget_stop"] = True; break
            if self.cal is not None:  # RETENTION (measured) of every arm that should be holding
                if M["hold"][arm] and not self.holding(env, arm):
                    cl["lost"] = {"arm": arm, "it": it}; M["hold"][arm] = False; Tn, _ = self.st(env, arm); self.step(env, {arm: self.to_world(Tn[:3, 3], Tn[:3, :3])}, {arm: 1.0}, "open_after_loss", log); return "retry"
                if M["hold"][p] and not self.holding(env, p): cl["partner_lost"] = {"arm": p, "it": it}; M["hold"][p] = False; M["done"] += 1; M["retired"].add(p)
            obs = self.look(env); dP, dQ, info = self.acl_decode(env, obs, instruction, arm, cyc, g_final, place, 1, f"post{it}"); rec = {"it": it, **info}; cl["post"].append(rec)
            if M["hold"][p]:  # ---- A: RECEIVING. The receiver's grasp is verified; the donor releases on the decoder's partner-gripper prediction; nothing moves until the donor's opening is measured
                rec["state"] = "receive"; want = bool((dQ[IDX[:self.K], 6] > 0.5).any()); rec["donor_release_predicted"] = want
                if not want and wait < 2: wait += 1; self.step(env, {}, {arm: 0.0, p: 0.0}, f"hold{it}", log); continue
                if not want: rec["release_fallback"] = True
                self.step(env, {}, {arm: 0.0, p: 1.0}, f"donor_release{it}", log)
                for k_ in range(2):
                    if finger_opening(env, p) > self.open_min: break
                    self.step(env, {}, {arm: 0.0, p: 1.0}, f"donor_release_settle{k_}", log)
                still = self.holding(env, arm); M["hold"][p] = False; M["done"] += 1; M["retired"].add(p)
                cl["transfer"] = {"donor": p, "receiver": arm, "donor_opening": round(finger_opening(env, p), 4), "receiver_opening": round(finger_opening(env, arm), 4), "receiver_holds_after_donor_release": still, "decoder_released": want}
                self.prog_check(env, instruction, arm, cyc, g_final, place, "transfer", cl)
                if still is False:
                    M["hold"][arm] = False; Tn, _ = self.st(env, arm); self.step(env, {arm: self.to_world(Tn[:3, 3], Tn[:3, :3])}, {arm: 1.0}, "open_after_loss", log); cl["lost"] = {"arm": arm, "it": it, "at": "transfer"}; return "retry"
                continue
            rec["state"] = "carry"
            if donor_cycle and it >= 1 and (info["span_P_cm"] < 2.0 or info["mode"] == 1):  # the object is at the exchange pose (no further primary motion predicted / next action belongs to the partner)
                rec["exchange_pose_reached"] = True; cl["deferred_release"] = True; return "advance"
            for i in IDX[:self.K]:
                gcmd = float(np.clip(dP[i, 6], 0, 1)); releasing = False; pos = dP[i, :3] + delta
                if gcmd > 0.5 and not released:
                    if donor_cycle: rec["deferred"] = True; cl["deferred_release"] = True; return "advance"  # the object is handed over in the next stage; never dropped
                    term = bool(self.opt["term"] and reliable); dist = float(np.linalg.norm((pos - place)[:2 if term else 3]))
                    if self.opt["relgate"] and reliable and dist > self.gate_tol and sup < 2:
                        sup += 1; rec.setdefault("release_suppressed_cm", []).append(round(dist * 100, 1)); gcmd = 0.0
                        if term and dist <= 0.12:  # GROUNDED TERMINAL CORRECTION: the remaining decoder trajectory is translated horizontally so that its predicted release point lies on the grounded place target; the object stays grasped during the move
                            d_ = np.r_[(place - pos)[:2], 0.0]; delta = delta + d_; pos = pos + d_; rec.setdefault("terminal_correction_cm", []).append([round(float(x) * 100, 1) for x in d_[:2]])
                    else: releasing = True; rec["release_dist_to_place_cm"] = round(dist * 100, 1)
                poses = {arm: self.to_world(pos, mat_rel(dP[i, 3:6]))}; grips = {arm: 1.0 if released else gcmd}
                if p in M["retired"] and not M["hold"][p] and info["mode"] in (1, 2): poses[p] = self.to_world(dQ[i, :3], mat_rel(dQ[i, 3:6])); grips[p] = 1.0  # synchronised partner motion (BIMANUAL / PARTNER arm mode)
                self.step(env, poses, grips, f"post{it}_{i}", log)
                if self.cal is not None and self.opt["carrychk"] and not released and not releasing and finger_opening(env, arm) < self.cal["thr"]:  # RETENTION during the carry (measured every control step): fingers fully closed = object lost -> stop, open, re-ground, re-grasp
                    cl["lost"] = {"arm": arm, "it": it, "step": i, "at": "carry"}; M["hold"][arm] = False; Tn, _ = self.st(env, arm); self.step(env, {arm: self.to_world(Tn[:3, 3], Tn[:3, :3])}, {arm: 1.0}, "open_after_loss", log); return "retry"
                if releasing:
                    for k_ in range(2):
                        if finger_opening(env, arm) > self.open_min: break
                        self.step(env, {}, {arm: 1.0}, f"release_settle{k_}", log)
                    Tn, _ = self.st(env, arm); released = True; M["hold"][arm] = False; M["done"] += 1; M["placed"].append(Tn[:3, 3].copy()); M["retired"].add(arm)
                    cl["release"] = {"tcp_xyz_r": Tn[:3, 3].tolist(), "opening": round(finger_opening(env, arm), 4), "tcp_to_place_cm": round(float(np.linalg.norm(Tn[:3, 3] - place) * 100), 1), "suppressed": sup}
            if released: break
        cl["released"] = released
        if released:  # D2: observation-based stage-completion verdict after the measured release
            pc = self.prog_check(env, instruction, arm, cyc, g_final, place, "release", cl)
            if self.use_prog and pc and pc["probs"]["FAILED_OR_RECOVERY"] > self.tau and cl.get("attempt", 0) < self.max_retry:
                pc["rejected"] = True; self.stat["prog_rejects"] += 1; cl["prog_reject"] = "release"; M["done"] = max(M["done"] - 1, 0); M["placed"].pop(); M["retired"].discard(arm); return "retry"
        if not released and not donor_cycle: cl["timeout_no_release"] = True
        return "advance"


class ReGSPBimanualPolicy(AclMixin, BP.BimV21Policy):
    """two-arm tasks (stack_blocks_two, handover_block): BimV21Policy + adaptive decoder + progress head + closed-loop executor"""


class ReGSPSingleArmPolicy(AclMixin, KP.V21Policy):
    """single-arm tasks (click_bell, place_container_plate): original decoders + closed-loop executor (unchanged V21 post-grasp path)"""
