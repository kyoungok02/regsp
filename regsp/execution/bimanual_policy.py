"""[ReGSP / research-tree bim_rt_policy.py] BimV21Policy = V21Policy (grounded-final V21, unchanged stages 1-7) + the documented BIMANUAL EMBODIMENT EXTENSION [X] for post-grasp / receiving behaviour:
 (a) learned decoder = bim_common.BimDecoder (original TemporalActionDecoder + partner-arm state, arm/cycle role, place-target conditioning; zero-init extension, objective unchanged);
 (b) PLACE TARGET: 'const' per (cycle, arm) = data-derived median of expert release TCP (when expert variance <= 1.5 cm) or 'grounded' = V21 proposals (class-agnostic head Head(64)) -> place-referent Scorer on the same HR architecture -> head-camera ray x place-referent height -> + data-derived constant referent->TCP-release offset; no privileged state;
 (c) CYCLE-CONDITIONED referent scorer (CycleScorer) for multi-referent tasks (stack_blocks_two);
 (d) STABLE-GRASP gate: after the receiving/second arm closes, extra close steps until its gripper reading has settled (<= expert hold p95 + 0.02 or change < 0.005 per step, max 3) BEFORE the previous arm's deferred release.
Everything else (proposals, HR scorer, triangulation, AnchorHead, StagePlan head, build_pregrasp_chunk, arrival-close) is the unchanged baseline code path."""
import sys, os, json, numpy as np, torch
from regsp.execution import base_policy as B
from regsp.execution.base_policy import *  # noqa
import regsp.grounding as R
from regsp.decoding.bimanual import BimDecoder
from regsp.grounding.scorer import CycleScorer
from regsp.backbone.rotation_encoding import encode, decode
from regsp.paths import data_file
dev = "cuda"
class BimV21Policy(B.V21Policy):
    def __init__(self, task, tagv="b1", cycle_scorer=True, place=True, bim_dec=True, **kw):
        super().__init__(task, **kw); bu = self.bundle; self.tagv = tagv; self.cur_cyc = 0; self.use_place = place; self.bimdec = None
        if bim_dec and bu.has("bim_decoder"):
            d = torch.load(bu.path("bim_decoder"), weights_only=False); self.bimdec = BimDecoder(self.D).to(dev).eval(); self.bimdec.load_state_dict(d["mods"]["bdec"]); self.Q.load_state_dict(d["mods"]["tqe"]); self.Vg.load_state_dict(d["mods"]["vgc"]); self.TPH.load_state_dict(d["mods"]["tph"]); self.dm, self.dsd = d["mean"].to(dev), d["std"].to(dev)
        self.bst = json.load(open(data_file("tasks", task, "bim_stats.json"))) if data_file("tasks", task, "bim_stats.json").exists() else None
        self.csc = None
        if cycle_scorer and bu.has("referent_scorer") and bu.referent_scorer_kind() == "cycle": self.csc = CycleScorer().to(dev); self.csc.load_state_dict(torch.load(bu.path("referent_scorer"))); self.csc.eval()
        self.psc = None
        if place and bu.has("place_scorer"): self.psc = R.Scorer().to(dev); self.psc.load_state_dict(torch.load(bu.path("place_scorer"))); self.psc.eval()
    # ---- (c) cycle-conditioned referent scoring
    @torch.no_grad()
    def score(self, imgs, instruction, Pf, tok, pad, mod):
        if self.csc is None or self.ground == "noscorer": return super().score(imgs, instruction, Pf, tok, pad, mod)
        txt_m = (mod[0] == 1) & pad[0].bool(); txt = tok[0][txt_m].float()[None]; tm = torch.ones(1, txt.shape[1], dtype=torch.bool, device=dev); ft = [self.crop(imgs[0], p[0], p[1]) for p in Pf]
        return self.csc(txt, tm, torch.stack(ft)[None], torch.tensor([min(self.cur_cyc, 1)], device=dev))[0].cpu().numpy(), ft
    def run_cycle(self, env, instruction, cyc, prev_arm, held, cl, log, video):
        self.cur_cyc = cyc; return super().run_cycle(env, instruction, cyc, prev_arm, held, cl, log, video)
    # ---- (d) stable-grasp gate
    def after_close(self, env, arm, cl, log):
        if self.bst is None: return
        thr = self.bst["stable_g_max"]; gs = []; extra = 0; side = "left" if arm == "L" else "right"
        for k in range(4):
            g = float(getattr(env.robot, f"get_{side}_gripper_val")()); gs.append(g)  # MEASURED gripper opening (not the commanded joint action)
            if g <= thr or (len(gs) > 1 and abs(gs[-1] - gs[-2]) < 0.005) or k == 3: break
            Tn, _ = self.state_of(env.get_obs(), arm); Tg = self.to_world(Tn[:3, 3], Tn[:3, :3]); self.step(env, {arm: Tg}, {arm: 0.0}, f"stable_close{k}", log); extra += 1
        cl["stable_grasp"] = {"g_trace": gs, "thr": thr, "extra_close_steps": extra}
    # ---- (b) place target
    @torch.no_grad()
    def place_target(self, env, obs, cyc, arm, cl):
        k = min(cyc, 1); tab = self.bst["place"]; e = tab.get(f"{k}_{arm}") or tab.get(f"{k}_{'L' if arm == 'R' else 'R'}") or list(tab.values())[-1]; med = np.array(e["median"], float); info = {"mode": e["mode"]}
        if e["mode"] == "grounded" and self.psc is not None and "place_referent" in self.bst:
            try:
                imgs = self.imgs(obs); Pf = self.props(imgs)["head_camera"]
                if len(Pf) >= 1:
                    tok, pad, mod, _ = self.tokens(imgs, "", np.zeros(7)) if False else self.tokens(imgs, self.cur_instruction, np.zeros(7)); txt_m = (mod[0] == 1) & pad[0].bool(); txt = tok[0][txt_m].float()[None]; tm = torch.ones(1, txt.shape[1], dtype=torch.bool, device=dev)
                    ft = [self.crop(imgs[0], p[0], p[1]) for p in Pf]; lg = self.psc(txt, tm, torch.stack(ft)[None])[0].cpu().numpy(); pr = self.bst["place_referent"]; lo, hi = np.array(e["min"]) - 0.02, np.array(e["max"]) + 0.02
                    zw = float((B.To[:3, :3] @ np.array(pr["referent_robot_xyz_median"]) + B.To[:3, 3])[2]); cands = []
                    for j in np.argsort(-lg):  # best-scored candidate whose implied TCP-release lies inside the expert release range (+-2 cm): data-derived plausibility prior (no privileged state)
                        c, d = R.ray(self.head_E(obs), Pf[j][:2]); s_ = (zw - c[2]) / d[2]
                        if s_ <= 0: continue
                        ref = B.Ti[:3, :3] @ (c + s_ * d) + B.Ti[:3, 3]; est = ref + np.array(pr["offset_median"], float); ok = bool(np.all(est >= lo) and np.all(est <= hi)); cands.append((int(j), ok))
                        if ok: info.update({"grounded": True, "referent_xyz_r": ref.tolist(), "n_prop": len(Pf), "sel": int(j), "rank": len(cands) - 1, "scores": lg.tolist()}); cl["place"] = info; return est
                    info["no_plausible_candidate"] = True
            except Exception as ex: info["error"] = str(ex)[:120]
            info["fallback_const"] = True
        cl["place"] = info; return med
    def partner_state(self, obs, arm):
        p = "R" if arm == "L" else "L"; Tr, g = self.state_of(obs, p); return np.r_[Tr[:3, 3], B.eul_rel(Tr[:3, :3]), g]
    def fallback_choice(self, g_final, src, xyz_c, cl, cyc, arm):  # [X] when V21 two-view grounding failed: if the expert grasp for this (cycle,arm) VARIES across demos (>1.5 cm), the constant StagePlan xyz is imprecise -> use the V21 coarse head estimate (proposal->HR scorer->ray x height prior; no privileged state); constant-grasp slots keep the StagePlan fallback
        if src != "v2_fallback" or xyz_c is None or self.bst is None or not os.environ.get("BIM_COARSE_FB", "1") == "1": return g_final, src
        e = self.bst.get("grasp", {}).get(f"{min(cyc, 1)}_{arm}")
        if e is not None and e["grasp_mode"] == "varies": cl["fallback_coarse"] = True; return np.asarray(xyz_c, float), "coarse_fallback"
        return g_final, src
    def run(self, env, instruction, log, video=None):
        self.cur_instruction = instruction; return super().run(env, instruction, log, video)
    # ---- post-grasp with the bimanual decoder
    def post_grasp(self, env, instruction, cyc, arm, g_final, held, cl, log):
        if self.bimdec is None: return super().post_grasp(env, instruction, cyc, arm, g_final, held, cl, log)
        STRIDE = B.STRIDE; tgt_dec = ((torch.tensor(g_final, dtype=torch.float32, device=dev) - self.dm[:3]) / self.dsd[:3])[None]; rel = False; cl["post"] = []; cl["bimanual_decoder"] = True
        place_r = self.place_target(env, env.get_obs(), cyc, arm, cl); cl["place_xyz_r"] = place_r.tolist(); role = torch.tensor([min(cyc, 1) * 2 + (arm == "R")], device=dev); pl = ((torch.tensor(place_r, dtype=torch.float32, device=dev) - self.dm[:3]) / self.dsd[:3])[None]
        for ch in range(self.post_chunks):
            ob = env.get_obs(); im = self.imgs(ob); Tr, gv = self.state_of(ob, arm); st7 = np.r_[Tr[:3, 3], Rot.from_matrix(Tr[:3, :3]).as_euler("xyz"), gv]; tok, pad, mod, _ = self.tokens(im, instruction, st7)
            s = (torch.tensor(np.r_[Tr[:3, 3], B.eul_rel(Tr[:3, :3]), gv], dtype=torch.float32, device=dev) - self.dm) / self.dsd; ps = (torch.tensor(self.partner_state(ob, arm), dtype=torch.float32, device=dev) - self.dm) / self.dsd
            with torch.no_grad():
                q = self.Q(tok, pad, mod); gf = self.Vg(q, tok, pad, mod); pred = self.bimdec(tok, pad, encode(s[None], self.dm[3:6], self.dsd[3:6]), tgt_dec, mod, gf, torch.ones(1, dtype=torch.long, device=dev), encode(ps[None], self.dm[3:6], self.dsd[3:6]), role, pl)
                dd = decode(pred, self.dm[3:6], self.dsd[3:6])[0] * self.dsd + self.dm
            dd = dd.float().cpu().numpy().astype(float); will_release = dd[-1, 6] > 0.5 or dd[STRIDE - 1:, 6].max() > 0.5; deferred = False
            for i in range(STRIDE - 1, 50, STRIDE):
                gcmd = float(np.clip(dd[i, 6], 0, 1))
                if self.defer and cyc + 1 < self.maxcyc and gcmd > 0.5: deferred = True; break
                if self.step(env, {arm: self.to_world(dd[i, :3], B.mat_rel(dd[i, 3:6]))}, {arm: gcmd}, f"post{ch}_{i}", log): return True
            cl["post"].append({"chunk": ch, "grip_pred_last": float(dd[-1, 6]), "deferred": deferred})
            if deferred: cl["deferred_release"] = True; break
            if will_release and dd[-1, 6] > 0.5: rel = True; held.pop(arm, None); break
        cl["released"] = rel; return bool(env.eval_success)
