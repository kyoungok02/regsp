# Architecture and code map

Tags used in the module docstrings: `[O]` original V21 code unchanged, `[R]` original architecture retrained on this task, `[E]` embodiment adapter, `[X]` extension.

## Per-stage flow of the frozen executor (`regsp/execution/closed_loop.py`)

1. **OBSERVE** three RGB cameras + joint/gripper proprioception; the frozen backbone encodes the instruction with every refreshed observation (`BASE_TOKENS`).
2. **GROUND** (re-run for every stage and every retry; no stale `grasp_xyz`): proposals on the head camera `[O]`; referent scorer `[O]/[X]` selects the instructed candidate; coarse 3-D = head ray x learned referent-height prior `[X]` (used only to choose a view pose); arm selector `[X]`; active-view acquisition `[X]` moves the chosen arm so that the coarse point lies on the wrist-camera axis (up to `nview` attempts, view table `VIEWS`); wrist proposals are associated with the head proposals by Hungarian two-ray gap (<= 3 cm) `[O]`; triangulation `[O]`; `AnchorHead` offset `[O]` (kept only if its horizontal offset is plausible: `ACL_ANCHOR_MAX_CM`, `ACL_REF_RAY_CM`); fallbacks: coarse head estimate, StagePlan regression.
   Sequential **memory gates** `[X]`: previously released locations are excluded; while the other arm verifiably holds the object (hand-over) the object's location is known from that arm's proprioception and proposals are restricted to that image region.
3. **PLAN** unchanged StagePlan head `[R]` -> grasp rotation, approach direction, pre-grasp distance, waypoint; place target = constant | grounded place referent | memory-relative `[X]`.
4. **PRE-GRASP** `build_pregrasp_chunk` `[O]`: 50 samples (20 to the waypoint, 15 to the standoff = sample 34, 15 to the grasp point). *Original* executor commanded samples 9, 19, 29, 39, 49 (never the standoff). *Released* (`CLC_APPROACH=2`, handover_block and place_container_plate): samples 9, 19, 29, 34, 39, 44, 49 through an aligned lead-in point, the standoff and a straight final segment along the gripper finger axis; the commanded pose at the standoff is compared with the measured pose and repeated once if they differ by more than 1 cm.
5. **CLOSE and VERIFY** measured finger opening must exceed the calibrated empty-gripper reading (`acl_calib.json`); the progress head's observation-based verdict (`ACL_PROG=1`, threshold `ACL_PROG_TAU = 0.8`) decides alone only where the task's finger calibration is disabled, and overrides a positive finger measurement only with `ACL_PROG_D1=1` (off in the release); a miss -> open, re-ground, retry (<= `ACL_RETRY = 2` per stage).
6. **POST-GRASP** adaptive decoder `[X]` over the same frozen tokens: joint synchronised 2 x 50 chunk (primary + partner), receding horizon (`ACL_K = 5` of 5 strided steps executed, then re-observe), arm mode from the progress head (PRIMARY / PARTNER / BIMANUAL), retention check every control step, hand-over: the donor releases only after the receiver's grasp is verified and the transfer counts only if the receiver still holds after the donor opened. Single-arm controls use the original task decoder path.
7. **RELEASE and VERIFY** release gated on the place target (<= `ACL_GATE_CM = 2.5` cm, at most two suppressions) with a grounded terminal correction of the release point; after release the progress head must not contradict the outcome. For `place_container_plate` (`CLC_REVERIFY=1`) the container is re-grounded after release and the stage is retried if it is confidently seen more than 10 cm from the release point.
8. **ADVANCE / RECOVER** a stage advances only on verified events; hard call budget `ACL_MAXCALLS = 300` far below the environment's step limit. `env.eval_success` is only the environment's termination signal and is never a decision input.

## Where each requested component lives

| Component | Module |
|---|---|
| Visual-language backbone | `regsp/backbone/modeling_safediff_vla.py`, `stageplan_v2.py` (`SafeDiffVLAStagePlanV2Policy`), loader `regsp/backbone/loader.py` |
| 64x64 class-agnostic proposal heatmaps | `regsp/grounding/proposals.py` |
| Counterfactual referent learning | `scripts/data/gen_cf_bell.py`, `build_cf_cache.py`; `scripts/train/train_cf_scorer.py` (modes `base` / `cf` / `ctl`) |
| HR referent scoring | `regsp/grounding/scorer.py` |
| Two-view triangulation | `regsp/grounding/geometry.py` (`triangulate`, `associate`) |
| AnchorHead | `regsp/grounding/anchor.py`, `scripts/train/train_da_anchor.py` |
| StagePlan and waypoint prediction | `regsp/planning`, `regsp/backbone/stageplan_v2.py`, `scripts/train/train_sp_v2head.py` |
| Adaptive single-arm / bimanual decoder | `regsp/decoding/bimanual.py`, `adaptive.py`, `regsp/backbone/temporal_decoder.py` |
| Progress estimation | `regsp/decoding/progress.py` |
| Sequential manipulation memory | `closed_loop.py` (`self.mem`, memory gates, `place_for`) |
| Multi-stage closed-loop execution | `closed_loop.py`, `regsp/execution/factory.py` |
| Re-grounding and bounded recovery | `closed_loop.py` (retry loop, `ACL_RETRY`) |
| Grounding-to-action alignment | `closed_loop.py` (`CLC_APPROACH`, `CLC_REVERIFY`, `CLC_FB`) |

## Original V21 vs RoboTwin extensions in one line each

* Original methodology: frozen-backbone tokens -> class-agnostic proposals -> HR scorer -> two-ray triangulation -> AnchorHead -> StagePlan variables -> geometric pre-grasp chunk -> arrival close -> learned decoder resumes -> decoder gripper = release.
* RoboTwin adaptations/extensions are listed in the README table and in the docstrings of `base_policy.py`, `bimanual_policy.py`, `closed_loop.py`.
