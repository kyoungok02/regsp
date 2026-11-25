# FINAL REPORT - 4-task SafeDiff-VLA V21 with the ACL extension

Scope: four RoboTwin tasks on Machine D. Two-arm extension tasks stack_blocks_two and handover_block run the V21-ACL configuration frozen in `freeze2/PROTOCOL2.json` (commit e1c92aa2, 2026-11-16T04:19:37Z); click_bell and place_container_plate are single-arm regression controls that run the same closed-loop executor with their immutable frozen checkpoints and original decoders.

This is a Randomized-domain adaptation experiment (training used Clean and randomized demonstrations/views), not Clean-only zero-shot. All earlier frozen records (10-task results, `freeze/`, `eval/final_full`, `eval/final_baseline`) are untouched; their numbers are quoted for reference only and were measured on different seeds (6000000+/6500000+), so differences of a few episodes at n=20 are within sampling noise (run-to-run nondeterminism alone is about +-2 at n=12).

## Final native results (fresh seeds: Clean 8000000+, Randomized 8500000+; n valid episodes per cell; Wilson 95% interval)

| task | Clean | 95% CI | earlier frozen full | earlier frozen baseline | Randomized | 95% CI | earlier frozen full | earlier frozen baseline |
|---|---|---|---|---|---|---|---|---|
| click_bell | 20/20 | [0.839, 1.0] | 20/20 | 17/20 | 17/20 | [0.64, 0.948] | 18/20 | 6/20 |
| place_container_plate | 16/20 | [0.584, 0.919] | 14/20 | 14/20 | 8/20 | [0.219, 0.613] | 11/20 | 5/20 |
| stack_blocks_two | 18/20 | [0.699, 0.972] | 13/20 | 11/20 | 15/20 | [0.531, 0.888] | 4/20 | 0/20 |
| handover_block | 13/20 | [0.433, 0.819] | 4/20 | 0/20 | 6/20 | [0.145, 0.519] | 0/20 | 0/20 |


Two-arm extension tasks pooled: Clean 31/40, Randomized 21/40 (fresh 80-episode final = 52/80).


## Stage-level metrics of the final cells

| task | config | stage-1 first-attempt grasp | stage-1 completed | later-stage first-attempt grasp | later stages completed | release within 2.5 cm / measured releases | transfer ok / transfers | episodes recovered / with retry | object-lost events | progress rejects grasp/release | grounding sources (all attempts) | mean policy wall s | mean env calls | mean VLM forwards | mean decodes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| click_bell | clean | 0/20 | 20/20 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0 | 0/0 | {"grounded": 20} | 2.09 | 10.15 | 4.0 | 0.0 |
| handover_block | clean | 13/20 | 13/20 | 13/13 | 13/13 | 0/0 | 13/13 | 0/7 | 0 | 0/0 | {"coarse_fallback": 15, "grounded": 19, "memory_receive": 13} | 8.83 | 37.25 | 18.15 | 6.25 |
| place_container_plate | clean | 0/20 | 20/20 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0 | 0/0 | {"grounded": 19, "v2_fallback": 1} | 3.34 | 16.0 | 5.1 | 0.0 |
| stack_blocks_two | clean | 20/20 | 20/20 | 18/20 | 19/20 | 21/21 | 0/0 | 1/3 | 0 | 0/1 | {"coarse_fallback": 14, "grounded": 31} | 8.51 | 43.7 | 14.4 | 7.9 |
| click_bell | rand | 0/20 | 20/20 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0 | 0/0 | {"grounded": 18, "v2_fallback": 2} | 3.79 | 13.15 | 4.35 | 0.0 |
| handover_block | rand | 12/20 | 14/20 | 8/14 | 8/14 | 2/2 | 8/8 | 2/15 | 0 | 0/1 | {"coarse_fallback": 16, "memory_receive": 26, "grounded": 19, "v2_fallback": 2} | 13.52 | 50.4 | 22.65 | 6.45 |
| place_container_plate | rand | 0/20 | 20/20 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0 | 0/0 | {"v2_fallback": 16, "grounded": 3, "v2_fallback_no_proposals": 1} | 4.93 | 18.6 | 4.9 | 0.0 |
| stack_blocks_two | rand | 19/20 | 19/20 | 18/19 | 19/19 | 25/25 | 0/0 | 0/4 | 0 | 0/3 | {"grounded": 26, "coarse_fallback": 21} | 11.84 | 49.4 | 17.65 | 9.3 |


## Development selection (development seeds only; see DEVELOPMENT_ABLATIONS.csv)

| label | what | task | config | success | K |
|---|---|---|---|---|---|
| dev_acl1 | first V21-ACL closed-loop executor + adaptive decoder a1 (stack cells ran before the terminal-correction / carry-check patches); single-arm controls | click_bell | clean | 12/12 | 5 |
| dev_acl1 | first V21-ACL closed-loop executor + adaptive decoder a1 (stack cells ran before the terminal-correction / carry-check patches); single-arm controls | place_container_plate | clean | 9/12 | 5 |
| dev_acl1 | first V21-ACL closed-loop executor + adaptive decoder a1 (stack cells ran before the terminal-correction / carry-check patches); single-arm controls | stack_blocks_two | clean | 12/12 | 5 |
| dev_acl1 | first V21-ACL closed-loop executor + adaptive decoder a1 (stack cells ran before the terminal-correction / carry-check patches); single-arm controls | click_bell | rand | 9/12 | 5 |
| dev_acl1 | first V21-ACL closed-loop executor + adaptive decoder a1 (stack cells ran before the terminal-correction / carry-check patches); single-arm controls | place_container_plate | rand | 6/12 | 5 |
| dev_acl1 | first V21-ACL closed-loop executor + adaptive decoder a1 (stack cells ran before the terminal-correction / carry-check patches); single-arm controls | stack_blocks_two | rand | 11/12 | 5 |
| dev_acl2 | V21-ACL executor after the recovered-state patches, a1, original DA obs heads | handover_block | clean | 9/12 | 5 |
| dev_acl2 | V21-ACL executor after the recovered-state patches, a1, original DA obs heads | stack_blocks_two | clean | 12/12 | 5 |
| dev_acl2 | V21-ACL executor after the recovered-state patches, a1, original DA obs heads | handover_block | rand | 4/12 | 5 |
| dev_acl2 | V21-ACL executor after the recovered-state patches, a1, original DA obs heads | stack_blocks_two | rand | 10/12 | 5 |
| dev_acl3 | as dev_acl2 + refreshed proposal heads *_obs2 | handover_block | clean | 8/12 | 5 |
| dev_acl3 | as dev_acl2 + refreshed proposal heads *_obs2 | stack_blocks_two | clean | 12/12 | 5 |
| dev_acl3 | as dev_acl2 + refreshed proposal heads *_obs2 | handover_block | rand | 7/12 | 5 |
| dev_acl3 | as dev_acl2 + refreshed proposal heads *_obs2 | stack_blocks_two | rand | 10/12 | 5 |
| dev_acl3g | *_obs2 heads + AnchorHead xy-offset gate 1.5 cm | handover_block | clean | 8/12 | 5 |
| dev_acl3g | *_obs2 heads + AnchorHead xy-offset gate 1.5 cm | handover_block | rand | 5/12 | 5 |
| dev_acl3h | *_obs2 heads + anchor gate 1.5 cm + triangulation/head-ray tolerance 4 cm (selected handover grounding) | handover_block | clean | 9/12 | 5 |
| dev_acl3h | *_obs2 heads + anchor gate 1.5 cm + triangulation/head-ray tolerance 4 cm (selected handover grounding) | handover_block | rand | 8/12 | 5 |
| dev_c0b | same as dev_c1b with progress decisions off (ACL_PROG=0) | handover_block | clean | 11/20 | 5 |
| dev_c0b | same as dev_c1b with progress decisions off (ACL_PROG=0) | stack_blocks_two | clean | 20/20 | 5 |
| dev_c0b | same as dev_c1b with progress decisions off (ACL_PROG=0) | handover_block | rand | 8/20 | 5 |
| dev_c0b | same as dev_c1b with progress decisions off (ACL_PROG=0) | stack_blocks_two | rand | 13/20 | 5 |
| dev_c1 | SELECTED configuration (a2, K=5, D2 on), development block A (seeds 7200000+/7500000+) | click_bell | clean | 12/12 | 5 |
| dev_c1 | SELECTED configuration (a2, K=5, D2 on), development block A (seeds 7200000+/7500000+) | handover_block | clean | 6/12 | 5 |
| dev_c1 | SELECTED configuration (a2, K=5, D2 on), development block A (seeds 7200000+/7500000+) | place_container_plate | clean | 9/12 | 5 |
| dev_c1 | SELECTED configuration (a2, K=5, D2 on), development block A (seeds 7200000+/7500000+) | stack_blocks_two | clean | 11/12 | 5 |
| dev_c1 | SELECTED configuration (a2, K=5, D2 on), development block A (seeds 7200000+/7500000+) | click_bell | rand | 9/12 | 5 |
| dev_c1 | SELECTED configuration (a2, K=5, D2 on), development block A (seeds 7200000+/7500000+) | handover_block | rand | 6/12 | 5 |
| dev_c1 | SELECTED configuration (a2, K=5, D2 on), development block A (seeds 7200000+/7500000+) | place_container_plate | rand | 3/12 | 5 |
| dev_c1 | SELECTED configuration (a2, K=5, D2 on), development block A (seeds 7200000+/7500000+) | stack_blocks_two | rand | 10/12 | 5 |
| dev_c1b | SELECTED configuration, development block B (seeds 7200100+/7500100+, n=20) | handover_block | clean | 11/20 | 5 |
| dev_c1b | SELECTED configuration, development block B (seeds 7200100+/7500100+, n=20) | stack_blocks_two | clean | 17/20 | 5 |
| dev_c1b | SELECTED configuration, development block B (seeds 7200100+/7500100+, n=20) | handover_block | rand | 11/20 | 5 |
| dev_c1b | SELECTED configuration, development block B (seeds 7200100+/7500100+, n=20) | stack_blocks_two | rand | 15/20 | 5 |
| dev_exp2 | corrected bimanual decoder b2 (parameter-freezing bug fixed), old executor | handover_block | clean | 0/12 | None |
| dev_exp2 | corrected bimanual decoder b2 (parameter-freezing bug fixed), old executor | stack_blocks_two | clean | 9/12 | None |
| dev_exp2 | corrected bimanual decoder b2 (parameter-freezing bug fixed), old executor | handover_block | randomized | 0/12 | None |
| dev_exp2 | corrected bimanual decoder b2 (parameter-freezing bug fixed), old executor | stack_blocks_two | randomized | 4/12 | None |
| dev_full | earlier frozen full model (retrained StagePlan head / decoder / bimanual decoder b1), old executor | click_bell | clean | 12/12 | None |
| dev_full | earlier frozen full model (retrained StagePlan head / decoder / bimanual decoder b1), old executor | handover_block | clean | 0/12 | None |
| dev_full | earlier frozen full model (retrained StagePlan head / decoder / bimanual decoder b1), old executor | place_container_plate | clean | 11/12 | None |
| dev_full | earlier frozen full model (retrained StagePlan head / decoder / bimanual decoder b1), old executor | stack_blocks_two | clean | 6/12 | None |
| dev_full | earlier frozen full model (retrained StagePlan head / decoder / bimanual decoder b1), old executor | click_bell | randomized | 10/12 | None |
| dev_full | earlier frozen full model (retrained StagePlan head / decoder / bimanual decoder b1), old executor | handover_block | randomized | 0/12 | None |
| dev_full | earlier frozen full model (retrained StagePlan head / decoder / bimanual decoder b1), old executor | place_container_plate | randomized | 5/12 | None |
| dev_full | earlier frozen full model (retrained StagePlan head / decoder / bimanual decoder b1), old executor | stack_blocks_two | randomized | 3/12 | None |
| dev_k2 | replanning interval K=2 (INCOMPLETE: stopped for time) | handover_block | rand | 4/11 | 2 |
| dev_k3 | replanning interval K=3 (a1, progress decisions off) | handover_block | clean | 10/12 | 3 |
| dev_k3 | replanning interval K=3 (a1, progress decisions off) | stack_blocks_two | clean | 11/12 | 3 |
| dev_k3 | replanning interval K=3 (a1, progress decisions off) | handover_block | rand | 4/12 | 3 |
| dev_k3 | replanning interval K=3 (a1, progress decisions off) | stack_blocks_two | rand | 7/12 | 3 |
| dev_prog1 | a2 progress head, ACL_PROG=1 with the grasp veto D1 still enabled (D1 later disabled by default) | handover_block | clean | 8/12 | 5 |
| dev_prog1 | a2 progress head, ACL_PROG=1 with the grasp veto D1 still enabled (D1 later disabled by default) | handover_block | rand | 6/12 | 5 |


## Independence of the final seeds (`audit/independence_final2.json`)

verdict_final2_independent = True; set sizes {'TRAIN_10TASK': 315, 'DA': 249, 'DA2': 240, 'ONPOLICY': 322, 'DEV': 230, 'FINAL_FROZEN_OLD': 162, 'FINAL2': 160}; identifier overlaps of FINAL2 with other sets: {"DA": 0, "DA2": 0, "DEV": 0, "FINAL_FROZEN_OLD": 0, "ONPOLICY": 0, "TRAIN_10TASK": 0}; identical layouts: {"DA": 0, "DA2": 0, "DEV": 1, "FINAL_FROZEN_OLD": 0, "ONPOLICY": 0, "TRAIN_10TASK": 0}


## What was implemented in this session (2026-11-16)

1. Preservation: commits 092ccbce (recovered snapshot), 37587db5 (progress decisions, on-policy labels, head refresh), e1c92aa2 (freeze); durable copy of all git-ignored artifacts under outputs/backup_v21_4task_acl_20261116.
2. Hand-over first-grasp grounding: the dominant Randomized failure was a lateral grasp error above ~1.2 cm (the gripper pushes the 6 cm box during the final approach); cause = low wrist-proposal recall on randomized frames (few associated views, coarse fallback). Fix = proposal heads retrained with 3x more randomized view units (`*_obs2`), AnchorHead offset gate, tighter triangulation/head-ray consistency. Two alternatives were measured and rejected (two-view consensus, horizontal-axis triangulation).
3. Progress head: on-policy failure/recovery samples generated with real perturbations, FAILED_OR_RECOVERY class trained (held-out on-policy accuracy 0.945 for both tasks), verdicts integrated as decisions (D2 after each measured release; D1 optional).
4. K validated on development seeds (K=5).

## Honest limitations

* Independence audit: no final2 identifier or object layout coincides with any training, calibration or on-policy set. One single-object control scene (click_bell Clean seed 8000010) has its bell within 0.7 mm of a development episode (seed 7000006, an evaluation-only record); it is a different random draw, reported for completeness. Two place_container_plate scenes share only the plate position with earlier final scenes (different container object).
* Episodes are the first 20 expert-valid seeds per cell (frozen protocol of `v21_eval.py`); expert-invalid/unstable seeds are skipped and listed in each cell's `summary.json` (e.g. 13 skipped for handover_block Randomized).
* Single-arm controls: place_container_plate Randomized is 8/20 against 11/20 in the earlier frozen final (different seeds, identical checkpoints; development repeats of this cell ranged 3-6/12). The difference is inside the Wilson intervals but a small regression from the new executor cannot be excluded.
* Sample sizes are small (n=20 per cell); see the Wilson intervals. No claim is made from differences of 1-3 episodes.
* hand-over remains limited by the first grasp (a miss usually topples the box and the bounded retries rarely recover) and by the absence of a table re-grasp in the receive stage.
* stack_blocks_two Randomized failures are mostly second-block placements beside the first block; the progress head detects them, recovery succeeds rarely.
* The freeze decision (ACL_PROG=1) was taken when the ACL_PROG=0 stack cells of development block B were still running; those cells are reported in DEVELOPMENT_ABLATIONS.csv and did not enter the decision.
* `freeze/MANIFEST_FREEZE.sha256` (earlier freeze) lists two training/generation scripts (`gen_rand_views.py`, `train_sp_bimdec.py`) that were edited after that freeze (bug fix and da_views2 generation); no frozen model or result file changed.
