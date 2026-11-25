"""[V21-PIC] TRAINING-ONLY sample builder for the stage-conditioned subgoal scorer (stack_blocks_two) from on-policy training-domain rollouts recorded by pic_rt_policy.py (PIC_SAVE_SG) under the original and the role-exchanged instruction.
Candidates = the head-camera proposals of the policy itself (crop tokens). Target index = the proposal within 12 px of the PRIVILEGED projection of the instructed block (training label only).
 stage 0 (nothing placed): BOTH valid instructions of the same frame -> executed instruction => its first block ; alternative instruction => the other block (same-scene counterfactual pair)
 stage 1 (first block released): executed instruction => the block that the instruction places SECOND (rollouts under the original instruction: green; under the exchanged instruction: red). The alternative instruction is NOT a valid
          instruction for that scene (its first block is not the one already placed) and is not used as supervision (kept as `txt_probe` for a language-dependence probe only).
Split: seed % 5 == 0 -> val. usage: pic_build_sg.py RAW_DIR [RAW_DIR ...] -> sg/stack_blocks_two_sg.pt + sg/sg_report.json"""
import sys, os, glob, json, collections, numpy as np, torch
from regsp import paths as P
N = str(P.work_root()); PX = 12.0; S = []; rep = collections.Counter()
def pos(rec, attr):
    pv = rec["priv_training_only"].get(attr)
    if pv is None or pv["depth"] <= 0.05 or not rec["P"]: return None
    d = [np.hypot(p[0] - pv["uv"][0], p[1] - pv["uv"][1]) for p in rec["P"]]; j = int(np.argmin(d)); return j if d[j] <= PX else None
for raw in sys.argv[1:]:
    for f in sorted(glob.glob(raw + "/sg*.pt")):
        D = torch.load(f, weights_only=False)
        for i, r in enumerate(D["frames"]):
            mode = r["lang_mode"]
            if mode == "cf" and not r.get("txt_ok"): r["txt_exec"], r["txt_alt"] = r["txt_alt"], r["txt_exec"]; rep["round1_cf_text_states_swapped_back"] += 1  # round-1 recorder bug: in PIC_LANG=cf rollouts the two text-state tensors were stored under each other's name (both are genuine VLM passes of the two valid instructions on the same images)
            if mode not in ("orig", "cf") or len(r["P"]) < 2: rep["skip_mode_or_single_candidate"] += 1; continue
            first, second = ("block1", "block2") if mode == "orig" else ("block2", "block1"); base = {"feat": r["feat"], "seed": r["seed"], "cfg": r["cfg"], "mode": mode, "where": r["where"], "split": "val" if r["seed"] % 5 == 0 else "train", "pair": f"{mode}{r['seed']}_{i}", "keep": r["keep"], "P": r["P"]}
            if r["n_placed"] == 0 and r["done"] == 0 and r["cyc"] == 0:
                a, b = pos(r, first), pos(r, second)
                if a is not None and b is not None and a != b:
                    S.append({**base, "cyc": 0, "txt": r["txt_exec"], "t": a, "variant": "exec", "referent": first, "instruction": r["instr_exec"]}); S.append({**base, "cyc": 0, "txt": r["txt_alt"], "t": b, "variant": "alt", "referent": second, "instruction": r["instr_alt"]}); rep[f"stage0_pair|{r['cfg']}|{mode}"] += 1
                else: rep["stage0_skip_referent_not_among_proposals"] += 1
            elif r["cyc"] >= 1 and r["n_placed"] >= 1:
                b = pos(r, second); fz = r["priv_training_only"].get(first)
                if b is None: rep["stage1_skip_referent_not_among_proposals"] += 1; continue
                if not r["keep"][b]: rep["stage1_skip_wrong_block_was_placed_first"] += 1; continue  # the proposal of the "second" block lies on the release location = the rollout placed the wrong block first: not a valid stage-1 sample
                S.append({**base, "cyc": 1, "txt": r["txt_exec"], "t": b, "variant": "exec", "referent": second, "instruction": r["instr_exec"], "txt_probe": r["txt_alt"], "placed_idx": pos(r, first)}); rep[f"stage1|{r['cfg']}|{mode}"] += 1
            else: rep["skip_other_state"] += 1
os.makedirs(N + "/sg", exist_ok=True); torch.save({"samples": S}, N + "/sg/stack_blocks_two_sg.pt")
R = {"raw": sys.argv[1:], "n_samples": len(S), "by": dict(sorted(rep.items())), "split": dict(collections.Counter(s["split"] for s in S)), "stage_referent": dict(collections.Counter(f"stage{s['cyc']}|{s['variant']}|{s['referent']}|{s['split']}" for s in S)), "episodes": len({(s["mode"], s["seed"]) for s in S})}
json.dump(R, open(N + "/sg/sg_report.json", "w"), indent=1); print(json.dumps(R, indent=1))
