"""[ReGSP / research-tree v21_rt_policy.py] SafeDiff-VLA StagePlan-v2.1 (grounded-final V21) deployed on ONE RoboTwin task (task-specific checkpoints). Inference uses ONLY raw RGB (3 cameras), the instruction, joint/gripper proprioception and calibrated camera geometry. NO simulator object state, expert actions or KSP/RGL module.
Component labels: [O]=original code unchanged, [R]=original architecture retrained on this task, [E]=embodiment adapter, [X]=new extension.
 per CYCLE (outer loop [X]; one cycle = one gripper-close event = the original single-grasp StagePlan episode):
  1 OBSERVE  head+wrist frames -> frozen SmolVLA backbone tokens (3x64 image, text, state) [O backbone]
  2 COARSE   [O] class-agnostic Head(64) proposals on the head camera -> [O] HR Scorer('B') on VLM text-token states + 72px crops selects the referent proposal; [X] head ray x learned height prior -> coarse 3D (ONLY used to choose a view pose; never replaces the V21 decision)
  3 ARM      [X] learned arm selector (referent xy + previous arm)
  4 STAGEPLAN [O] StagePlanV2Head -> grasp rotation / approach / pre_dist / wp_rel (+ grasp_xyz fallback) [R]
  5 VIEW     [X] active-view acquisition: arm moves (cuRobo ee) to an observation pose that puts the coarse point on the wrist-camera axis (calibrated tool->camera mount) -> up to NVIEW attempts
  6 GROUND   [O] genuine V21 path: head proposals x wrist proposals -> Hungarian two-ray association (gap<=3 cm) -> selected referent triangulated -> [O] AnchorHead (kept per the original DEV rule else per-task constant offset) -> grasp_xyz; original fallback = StagePlan grasp_xyz
  7 PRE_GRASP [O] build_pregrasp_chunk (waypoint -> pre -> grasp, orientation slerp) executed as cuRobo ee calls [E]; arrival -> CLOSE [O semantics]
  8 POST_GRASP [R] learned TemporalActionDecoder (+TargetQueryExtractor/VisualGroundingCrossAttention) 50-step chunks replanned each chunk, executed as ee calls [E]; RELEASE = decoder gripper passthrough [O], optionally deferred while the next cycle's arm has not yet closed [X]
Variants (ablations): ground in {v21 (active view), orig (no active view: second view = wrist at its current pose), noscorer, noanchor}; dec in {pretrained, random, frozen}; v2: adapted."""
import sys, os, json, pickle, numpy as np, torch, transforms3d as t3d
import regsp.grounding as R
from regsp.execution import kinematics as rt_kin
from regsp.envs import cameras as C
from regsp.backbone.loader import load_policy, make_batch
from regsp.bundle import TaskBundle
from regsp.paths import data_file
from scipy.spatial.transform import Rotation as Rot
from regsp.backbone.stageplan_v2 import StagePlanV2Head, StageVars, build_pregrasp_chunk, PRE_DIST_SCALE_M, WP_SCALE_M
from regsp.backbone.rotation_encoding import encode, decode
dev = "cuda"
FK = {"L": rt_kin.ArmFK("left"), "R": rt_kin.ArmFK("right")}; To = FK["L"].T_origin; Ti = np.linalg.inv(To); RREF = Rot.from_euler("y", 90, degrees=True).as_matrix()
TCPCAM = np.array([[0, -0.389, 0.921, -0.05], [-1, 0, 0, 0.032], [0, -0.921, -0.389, 0.065], [0, 0, 0, 1.0]])  # [E] tool->wrist-camera mount (calibration; verified rigid in sim)
STRIDE = 10
def eul_rel(Rm): return Rot.from_matrix(Rm @ RREF.T).as_euler("xyz")
def mat_rel(e): return Rot.from_euler("xyz", e).as_matrix() @ RREF
class V21Policy:
    def __init__(self, task, ground="v21", dec="pretrained", nview=3, defer_release=None, verbose=False, tag="", geom="live", bundle=None):
        assert dec == "pretrained", "only the task-adapted pretrained decoder is part of the released pipeline"
        self.task, self.ground, self.decn, self.nview = task, ground, dec, nview; self.bundle = bundle if bundle is not None else TaskBundle.for_task(task); bu = self.bundle
        self.pol, self.pre = load_policy(); self.vm = self.pol.backbone.model.vlm_with_expert.get_vlm_model().vision_model; self.ve = self.pol.backbone.model.vlm_with_expert; self.tag = tag
        # stage-variable head (per task); StagePlan fallback statistics pm/ps come from the same checkpoint
        c = torch.load(bu.path("stageplan_head"), weights_only=False); self.v2 = StagePlanV2Head(c["cfg"]["latent_dim"], **{k: v for k, v in c["cfg"].items() if k != "latent_dim"}).to(dev).eval(); self.v2.load_state_dict(c["head"]); self.pm, self.ps = c["pos_mean"].numpy().astype(float), c["pos_std"].numpy().astype(float)
        self.heads = {}
        for cam in R.CAMS: m = R.Head(64).to(dev); m.load_state_dict(torch.load(bu.path(f"proposal_{cam}"))); self.heads[cam] = m.eval()
        self.sc = None; self.anchor = None
        if bu.has("referent_scorer") and bu.referent_scorer_kind() == "plain": self.sc = R.Scorer().to(dev); self.sc.load_state_dict(torch.load(bu.path("referent_scorer"))); self.sc.eval()
        if bu.has("anchor_head"):
            a = torch.load(bu.path("anchor_head"), weights_only=False); self.ah = R.AnchorHead(center=tuple(a["center"])).to(dev); self.ah.load_state_dict(a["state"]); self.ah.eval(); self.anchor = {"keep": a["keep_learned"], "mean_off": np.array(a["mean_offset_cm"]) / 100}
        d = torch.load(bu.path("decoder"), weights_only=False)
        self.D, self.Q, self.Vg, self.TPH = self.pol.decoder.to(dev).eval(), self.pol.target_query_extractor.to(dev).eval(), self.pol.visual_grounding_cross_attn.to(dev).eval(), self.pol.target_point_head.to(dev).eval()
        for k, m in (("dec", self.D), ("tqe", self.Q), ("vgc", self.Vg), ("tph", self.TPH)): m.load_state_dict(d["mods"][k])
        self.dm, self.dsd = d["mean"].to(dev), d["std"].to(dev)
        S = json.load(open(data_file("tasks", task, "runtime_stats.json"))); self.stats = S; self.arm_w = S["arm_selector"]; self.zprior = {int(k): v for k, v in S["referent_height_world"].items()}; self.maxcyc = S["max_cycles"]; self.post_chunks = S["max_post_chunks"]; self.defer = S["defer_release"] if defer_release is None else defer_release
        self.verbose = verbose; self.hw = C.head_w2c(); self.geom = geom; self._static = {}; self.oracle = None  # DIAGNOSTIC ONLY: {"order":[arm..],"xyz":{arm: robot-frame grasp}} set by an evaluation driver for oracle interventions; never set in normal evaluation
    # ------------------------------------------------------------------ observation helpers
    def state_of(self, obs, arm):
        q = np.array(obs["joint_action"]["vector"], float); a = (0, 7) if arm == "L" else (7, 14); T = FK[arm].tcp(q[a[0]:a[0] + 6], "world"); Tr = Ti @ T; g = float(q[a[0] + 6]); return Tr, g
    def head_E(self, obs):  # geom=live: per-frame extrinsic_cv delivered with the observation; fk/static: calibration file
        return R.w2c4(obs["observation"]["head_camera"]["extrinsic_cv"]) if self.geom == "live" else self.hw
    def wrist_E(self, obs, arm):  # [E] per-frame wrist extrinsics. live = observation extrinsic_cv (synchronised with the image by construction); fk = FK(joints) x calibrated mount; static = frozen at the first observation of the episode (PREVIOUS-IMPLEMENTATION FAILURE MODE, A/B only)
        if self.geom == "live": return R.w2c4(obs["observation"]["left_camera" if arm == "L" else "right_camera"]["extrinsic_cv"])
        if self.geom == "static":
            if arm not in self._static: self._static[arm] = self._wrist_fk(obs, arm)
            return self._static[arm]
        return self._wrist_fk(obs, arm)
    def _wrist_fk(self, obs, arm):
        q = np.array(obs["joint_action"]["vector"], float); a = (0, 7) if arm == "L" else (7, 14); T = FK[arm].tcp(q[a[0]:a[0] + 6], "world"); return np.linalg.inv(T @ TCPCAM)
    def imgs(self, obs):
        o = obs["observation"]; return [R.half(o[c]["rgb"]) for c in R.CAMS]
    @torch.no_grad()
    def tokens(self, imgs, instruction, st7):
        b = self.pre(make_batch({"imgs224": [R.sq(i) for i in imgs], "state": np.asarray(st7, np.float32)}, instruction)); tok, pad, mod = self.pol._encode_multimodal_latent(b); return tok, pad, mod, self.pol._current_state(b)
    @torch.no_grad()
    def feat(self, img):
        return C.patch_feats(self.vm, [img])[0].float().to(dev)[None]
    @torch.no_grad()
    def crop(self, img_half, u, v):
        h = R.CROP_PX // 4; uu, vv = int(round(min(max(u, 0), 639) / 2)), int(round(min(max(v, 0), 479) / 2)); import cv2; pad = np.pad(img_half, ((h, h), (h, h), (0, 0)), mode="edge"); cr = pad[vv: vv + 2 * h, uu: uu + 2 * h]
        x = torch.from_numpy(cv2.resize(cr, (512, 512), interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).float().div(255)[None].to(dev) * 2 - 1; return self.ve.embed_image(x)[0].float()
    @torch.no_grad()
    def props(self, imgs):
        out = {}
        for ci, cam in enumerate(R.CAMS): out[cam] = R.decode(self.heads[cam](self.feat(imgs[ci]))[0].cpu())
        return out
    @torch.no_grad()
    def score(self, imgs, instruction, Pf, tok, pad, mod):
        txt_m = (mod[0] == 1) & pad[0].bool(); txt = tok[0][txt_m].float()[None]; tm = torch.ones(1, txt.shape[1], dtype=torch.bool, device=dev)
        ft = [self.crop(imgs[0], p[0], p[1]) for p in Pf]; lg = self.sc(txt, tm, torch.stack(ft)[None])[0]; return lg.cpu().numpy(), ft
    # ------------------------------------------------------------------ grounding pieces
    def coarse_xyz(self, uv, k, E=None):  # [X] head ray x learned referent-height prior (robot frame out)
        c, d = R.ray(self.hw if E is None else E, uv); z = self.zprior.get(k, list(self.zprior.values())[-1]); s = (z - c[2]) / d[2]
        return None if s <= 0 else Ti[:3, :3] @ (c + s * d) + Ti[:3, 3]
    def pick_arm(self, xyz_r, prev_arm, other_r=None):
        w = np.array(self.arm_w["w"]); b = self.arm_w["b"]; f = np.array([xyz_r[0], xyz_r[1], float(prev_arm == "L"), float(prev_arm == "R")] + ([other_r[0], other_r[1]] if self.arm_w.get("use_other_object_feature") else [])); return "R" if float(w @ f + b) > 0 else "L"
    def obs_pose(self, xyz_r, Rr, d):  # [X] TCP pose (world) that places xyz_r at depth d on the wrist-camera axis
        Rw = To[:3, :3] @ Rr; zc = (Rw @ TCPCAM[:3, :3])[:, 2]; pc = (To[:3, :3] @ xyz_r + To[:3, 3]) - d * zc; return pc - Rw @ TCPCAM[:3, 3], Rw
    # ------------------------------------------------------------------ execution primitives [E]
    def act_vec(self, env, poses, grips):
        q = np.array(env.get_obs()["joint_action"]["vector"], float); v = []
        for a in "LR":
            if a in poses: p, Rw = poses[a]
            else: T = FK[a].tcp(q[(0 if a == "L" else 7):(6 if a == "L" else 13)], "world"); p, Rw = T[:3, 3], T[:3, :3]
            gv = grips[a] if a in grips else float(q[6 if a == "L" else 13]); v += list(p - Rw[:, 0] * 0.12) + list(t3d.quaternions.mat2quat(Rw)) + [gv]
        return np.array(v)
    def step(self, env, poses, grips, tag, log):
        env.take_action(self.act_vec(env, poses, grips), action_type="ee"); log["calls"].append(tag); return bool(env.eval_success)
    def to_world(self, p_r, Rr): return To[:3, :3] @ p_r + To[:3, 3], To[:3, :3] @ Rr
    # ------------------------------------------------------------------ main loop
    def run(self, env, instruction, log, video=None):
        self._static = ({a: self._wrist_fk(env.get_obs(), a) for a in "LR"} if self.geom == "static" else {}); log["calls"] = []; log["cycles"] = []; prev_arm = None; held = {}  # held: arms still closed awaiting deferred release; static geom = wrist extrinsics frozen at the episode-initial pose
        for cyc in range(self.maxcyc):
            cl = {"cycle": cyc}; log["cycles"].append(cl)
            try:
                if self.run_cycle(env, instruction, cyc, prev_arm, held, cl, log, video): return True
            except Exception as e:
                import traceback; cl["error"] = traceback.format_exc()[-800:]
                if any(s in cl["error"] for s in ("DeviceLost", "vk::", "Vulkan", "out of memory")): raise
                break
            prev_arm = cl.get("arm", prev_arm)
            if env.eval_success: return True
        for a in list(held):  # release anything still held at the end (decoder passthrough would have opened it)
            try: self.step(env, {}, {a: 1.0}, f"final_open_{a}", log)
            except Exception: pass
        return bool(env.eval_success)
    def run_cycle(self, env, instruction, cyc, prev_arm, held, cl, log, video):
        obs = env.get_obs(); imgs = self.imgs(obs); arm0 = prev_arm or "L"; Tr, g0 = self.state_of(obs, arm0); st0 = np.r_[Tr[:3, 3], Rot.from_matrix(Tr[:3, :3]).as_euler("xyz"), g0]
        tok, pad, mod, _ = self.tokens(imgs, instruction, np.zeros(7)); P = self.props(imgs); cl["n_props_t0"] = {c: len(v) for c, v in P.items()}
        noprop = not P["head_camera"]
        if noprop:  # original V21: no proposals -> StagePlan-v2 fallback (logged, never an oracle)
            cl["abort"] = "no_head_proposals"; xyz_c = np.array(self.pm); sel = None; oth = []; xo = None
        else:
            lg, ft = self.score(imgs, instruction, P["head_camera"], tok, pad, mod) if self.ground != "noscorer" else (np.array([p[2] for p in P["head_camera"]]), None); sel = int(np.argmax(lg)); cl["coarse_uv"] = P["head_camera"][sel][:2]; cl["scores"] = lg.tolist()
            xyz_c = self.coarse_xyz(P["head_camera"][sel][:2], cyc, self.head_E(obs)); cl["coarse_xyz_r"] = None if xyz_c is None else xyz_c.tolist()
            if xyz_c is None: xyz_c = np.array(self.pm)
            oth = [i for i in range(len(P["head_camera"])) if i != sel]; xo = self.coarse_xyz(P["head_camera"][oth[int(np.argmax([P["head_camera"][i][2] for i in oth]))]][:2], cyc, self.head_E(obs)) if oth else None
        arm = self.pick_arm(xyz_c, prev_arm, xo if xo is not None else xyz_c); cl["arm"] = arm
        # StagePlan head (arm-conditioned state, 3 cams, frozen backbone tokens) -> rotation / approach / pre_dist / wp_rel (+ xyz fallback)
        Tr, g0 = self.state_of(obs, arm); st7 = np.r_[Tr[:3, 3], Rot.from_matrix(Tr[:3, :3]).as_euler("xyz"), g0]; tok, pad, mod, stn = self.tokens(imgs, instruction, st7)
        with torch.no_grad(): o = self.v2(tok, pad, mod, stn)
        g2 = o["grasp_xyz"][0].float().cpu().numpy().astype(float) * self.ps + self.pm; eul = torch.atan2(o["grasp_rot_sc"][0, :, 0], o["grasp_rot_sc"][0, :, 1]).float().cpu().numpy().astype(float)
        r = float(np.clip(o["pre_dist"][0].item() * PRE_DIST_SCALE_M, 0.05, 0.15)); wp_rel = o["wp_rel"][0].float().cpu().numpy().astype(float) * WP_SCALE_M; ad = o["approach_dir"][0].float().cpu().numpy().astype(float); Rg = mat_rel(eul)
        cl["v2"] = {"xyz_fallback": g2.tolist(), "pre_dist": r, "eul_rel": eul.tolist(), "approach": ad.tolist()}
        # ---- active-view acquisition [X] + genuine V21 two-view grounding [O]
        g_final, src = g2, "v2_fallback"; cl["views"] = []
        if noprop: dseq = []
        elif self.ground in ("v21", "noscorer", "noanchor"): dseq = [(0.30, 0), (0.36, -20), (0.26, 20), (0.33, -10), (0.28, 10), (0.40, 0)][:self.nview]  # first three = baseline acquisition; 4-6 only when --nview > 3 (acquisition-procedure adaptation [X])
        else: dseq = [(None, 0)]  # 'orig': no active view -> wrist at its current (home) pose
        for vi, (dobs, pitch) in enumerate(dseq):
            vr = {"attempt": vi, "d_obs": dobs, "tilt_deg": pitch}
            if dobs is not None:
                p_w, R_w = self.obs_pose(xyz_c, Rg @ Rot.from_euler("y", pitch, degrees=True).as_matrix(), dobs)
                try: self.step(env, {arm: (p_w, R_w)}, {arm: 1.0}, f"obs_view{vi}", log); vr["moved"] = True; vr["target_tcp_world"] = p_w.tolist(); vr["R_w"] = R_w.tolist()
                except Exception as e:
                    if any(s in str(e) for s in ("DeviceLost", "Vulkan", "vk::")): raise
                    vr["moved"] = False; cl["views"].append(vr); continue
                Tn, _ = self.state_of(env.get_obs(), arm); vr["pose_err_cm"] = float(np.linalg.norm((To[:3, :3] @ Tn[:3, 3] + To[:3, 3]) - p_w) * 100)
            ob2 = env.get_obs(); im2 = self.imgs(ob2); wc = "left_camera" if arm == "L" else "right_camera"; Pn = self.props(im2); Pf, Pw = Pn["head_camera"], Pn[wc]; vr.update({"n_head": len(Pf), "n_wrist": len(Pw)})
            if not Pf or not Pw: cl["views"].append(vr); continue
            vr["wrist_has_proposal"] = True; tok2, pad2, mod2, _ = self.tokens(im2, instruction, np.zeros(7)); lg2, ft2 = self.score(im2, instruction, Pf, tok2, pad2, mod2) if self.ground != "noscorer" else (np.array([p[2] for p in Pf]), [self.crop(im2[0], p[0], p[1]) for p in Pf]); s2 = int(np.argmax(lg2))
            A = R.associate(Pf, Pw, self.head_E(ob2), self.wrist_E(ob2, arm)); vr["geom"] = self.geom; vr["assoc"] = sorted(A.keys()); vr["selected"] = s2
            if s2 in A:
                ref_w, j, gap = A[s2]; ref = Ti[:3, :3] @ ref_w + Ti[:3, 3]; off = np.zeros(3)
                if self.anchor is not None and self.ground != "noanchor":
                    if self.anchor["keep"]:
                        with torch.no_grad(): off = self.ah(ft2[s2][None], self.crop(im2[R.CAMS.index(wc)], Pw[j][0], Pw[j][1])[None], torch.tensor(ref, dtype=torch.float32, device=dev)[None])[0].cpu().numpy() / 100
                    else: off = self.anchor["mean_off"]
                g_final = ref + off; src = "grounded"; vr.update({"ok": True, "ref": ref.tolist(), "offset_cm": (off * 100).tolist(), "gap_m": float(gap), "anchor": "learned" if (self.anchor and self.anchor["keep"] and self.ground != "noanchor") else ("const" if self.anchor and self.ground != "noanchor" else "none")}); cl["views"].append(vr); break
            vr["ok"] = False; cl["views"].append(vr)
        g_final, src = self.fallback_choice(g_final, src, None if noprop else xyz_c, cl, cyc, arm)  # hook (identity in the baseline policy)
        if self.oracle is not None and cyc < len(self.oracle["order"]):  # DIAGNOSTIC oracle intervention (arm and/or grasp target replaced by the expert's); recorded in the log
            oa = self.oracle["order"][cyc]
            if "arm" in self.oracle["mode"] and oa != arm: cl["oracle_arm_replaced"] = [arm, oa]; arm = oa; cl["arm"] = arm
            if "xyz" in self.oracle["mode"] and arm in self.oracle["xyz"]: cl["oracle_xyz_replaced"] = g_final.tolist(); g_final = np.array(self.oracle["xyz"][arm], float); src = src + "+oracle"
        cl["source"] = src if not noprop else "v2_fallback_no_proposals"; cl["g_final"] = g_final.tolist(); cl["g_v2_fallback"] = g2.tolist()
        # ---- PRE_GRASP [O] chunk builder, executed as ee calls
        sv = StageVars(np.asarray(g_final, float), eul, ad, r, True, np.asarray(g_final, float) + wp_rel); grip_open = 1.0; arrived = False
        for rep in range(4):
            Tr, _ = self.state_of(env.get_obs(), arm); ee = Tr[:3, 3].copy(); pos, eu = build_pregrasp_chunk(ee, eul_rel(Tr[:3, :3]), sv)
            for i in range(STRIDE - 1, 50, STRIDE):
                if self.step(env, {arm: self.to_world(pos[i], mat_rel(eu[i]))}, {arm: grip_open}, f"pre{rep}_{i}", log): return True
            Tr, _ = self.state_of(env.get_obs(), arm); d = float(np.linalg.norm(Tr[:3, 3] - g_final)); cl.setdefault("arrival_dist_cm", []).append(d * 100)
            if d <= 0.02: arrived = True; break
        cl["arrived"] = arrived
        gp = self.to_world(g_final, mat_rel(eul))
        for _ in range(2):
            if self.step(env, {arm: gp}, {arm: 0.0}, "close", log): return True
        self.after_close(env, arm, cl, log)  # hook (no-op in the baseline policy)
        for a_ in [x for x in list(held) if x != arm]:  # [X] the previous arm is released only now that this cycle's arm holds the object
            Tn, _ = self.state_of(env.get_obs(), a_); self.step(env, {a_: self.to_world(Tn[:3, 3], Tn[:3, :3])}, {a_: 1.0}, f"deferred_release_{a_}", log); held.pop(a_, None); cl["released_prev"] = a_
        held[arm] = True
        return self.post_grasp(env, instruction, cyc, arm, g_final, held, cl, log)
    def fallback_choice(self, g_final, src, xyz_c, cl, cyc, arm): return g_final, src
    def after_close(self, env, arm, cl, log): pass
    def post_grasp(self, env, instruction, cyc, arm, g_final, held, cl, log):  # unchanged original-adapter post-grasp stage (factored out so the bimanual extension can override it; baseline behaviour is byte-identical)
        # ---- POST_GRASP [R] decoder chunks (phase=1); release passthrough [O] with optional deferral [X]
        tgt_n = torch.tensor((g_final - self.pm) / self.ps, dtype=torch.float32, device=dev)[None]; tgt_dec = ((torch.tensor(g_final, dtype=torch.float32, device=dev) - self.dm[:3]) / self.dsd[:3])[None]; rel = False; cl["post"] = []
        for ch in range(self.post_chunks):
            ob = env.get_obs(); im = self.imgs(ob); Tr, gv = self.state_of(ob, arm); st7 = np.r_[Tr[:3, 3], Rot.from_matrix(Tr[:3, :3]).as_euler("xyz"), gv]; tok, pad, mod, _ = self.tokens(im, instruction, st7)
            s = torch.tensor(np.r_[Tr[:3, 3], eul_rel(Tr[:3, :3]), gv], dtype=torch.float32, device=dev); s = (s - self.dm) / self.dsd
            with torch.no_grad():
                q = self.Q(tok, pad, mod); gf = self.Vg(q, tok, pad, mod); pred = self.D(tok, pad, encode(s[None], self.dm[3:6], self.dsd[3:6]), target_xyz=tgt_dec, latent_modality_ids=mod, grounded_target_feature=gf, grasp_phase=torch.ones(1, dtype=torch.long, device=dev))
                dd = decode(pred, self.dm[3:6], self.dsd[3:6])[0] * self.dsd + self.dm
            dd = dd.float().cpu().numpy().astype(float); will_release = dd[-1, 6] > 0.5 or dd[STRIDE - 1:, 6].max() > 0.5
            deferred = False
            for i in range(STRIDE - 1, 50, STRIDE):
                gcmd = float(np.clip(dd[i, 6], 0, 1))
                if self.defer and cyc + 1 < self.maxcyc and gcmd > 0.5: deferred = True; break  # [X] keep holding until the next cycle's arm has closed
                if self.step(env, {arm: self.to_world(dd[i, :3], mat_rel(dd[i, 3:6]))}, {arm: gcmd}, f"post{ch}_{i}", log): return True
            cl["post"].append({"chunk": ch, "grip_pred_last": float(dd[-1, 6]), "deferred": deferred})
            if deferred: cl["deferred_release"] = True; break
            if will_release and dd[-1, 6] > 0.5: rel = True; held.pop(arm, None); break
        cl["released"] = rel; return bool(env.eval_success)
