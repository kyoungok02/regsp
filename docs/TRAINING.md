# Training

**Status: ported, compiled, import-checked; not re-run.** The scripts below are mechanical ports (paths -> environment variables, imports -> `regsp.*`) of the research-tree scripts that produced the released weights. No training job was launched while extracting the repository, so none of these commands has been executed in this repository. Evidence that the ported *model code* is identical to the trained code: `tests/` and the weight-equivalence digests (docs/VERIFICATION.md).

## Layout

```
REGSP_CLEAN_DATA   RoboTwin-LeRobot-v3.0 Clean-50 demos     <root>/<task>/aloha-agilex_clean_50/{meta,data,videos}
REGSP_CLEAN_ROOT   Clean-domain working dir                  labels/ tokens/ tokens_full/ v21cache/ scenes/ views/ obsviews/ feats/ ckpt*/
REGSP_WORK_ROOT    Randomized / counterfactual / adaptive    data_rand/ rand/{labels,tokens,tokens_full} da_* cf_* onpolicy/ sg/ ckpt_*/ training/
REGSP_BACKBONE_CKPT frozen backbone                           (needed by every script that builds tokens)
REGSP_ROBOTWIN_ROOT RoboTwin checkout                         (scene/view generation, FK)
```

All generators and trainers take `TASK` in `{click_bell, place_container_plate, stack_blocks_two, handover_block}` and use seed 1000 where they train. Run one SAPIEN worker per GPU and never SAPIEN and training on the same GPU (rendering and the optimiser compete for memory).

## Order (commands of the research runs; arguments are those recorded in the research queue)

0. **Backbone** (once): `python scripts/train/train_backbone.py` (20k steps, batch 4; needs the SafeDiff research fork of `lerobot-train`, see below).
1. **Clean-domain data** (original ten-task stage): `scripts/data/build_labels_10.py` (expert-replay grasp labels), `gen_scenes_10.py TASK 75 7000000 DIR`, `gen_views_10.py`, `gen_obsviews_10.py`, `extract_feats_10.py TASK`, `build_v21_cache_10.py TASK`, `tokens_10.py TASK`, `tokens_full_10.py TASK`, `make_runtime_stats_10.py TASK`, bimanual tasks: `gen_place_10.py`, `bim_stats_10.py`. Wrap simulator jobs with a restart loop (see `scripts/evaluate/run_eval.sh`); `validate_units.py` checks capture units.
2. **Randomized-domain data**: `scripts/data/dl_rand.py`, `gen_rand_scenes.py`, `gen_rand_views.py`, `gen_rand_obsviews.py`, `build_labels_rand.py TASK`, `tokens_rand.py TASK`, `tokens_full_rand.py TASK`, `build_da_cache.py TASK`.
3. **Counterfactual data**: `gen_cf_bell.py 100 3000000 DIR` (two-bell scenes), `build_cf_cache.py TASK` (role-swapped `stack_blocks_two` instructions). `place_container_plate` and `handover_block` have no valid counterfactual referent.
4. **Proposal heads**: `train_da_heads.py TASK` (4000 steps, batch 16, one head per camera, frozen SigLIP features).
5. **Referent scorer**: `UNION=1 train_cf_scorer.py click_bell cf u1`, `... stack_blocks_two cf u1`, `... place_container_plate base u1`, `... handover_block base u1` (3000 steps, CE over the scene's candidates; `cf` mode = original frames + paired same-scene counterfactual instructions). Place-referent scorer (`handover_block`): `train_bim_grounding_10.py handover_block ...`.
6. **AnchorHead**: `train_da_anchor.py TASK`.
7. **StagePlan head**: `train_sp_v2head.py TASK` (6000 steps).
8. **Task decoder**: `train_sp_decoder.py TASK pretrained 10000` (initialised from the backbone's decoder, task-normalised).
9. **Bimanual decoder** (two-arm tasks): `train_sp_bimdec.py TASK b1 8000` (zero-initialised extension terms: step 0 equals the task decoder).
10. **Adaptive decoder + progress head** (two-arm tasks): `train_acl.py TASK a1 8000`; on-policy progress data from the executor (`REGSP_DATAGEN=1` with `ACL_PERTURB=0.3,0.035,0.3 ACL_SAVE_FEATS=DIR` on training-domain seeds 5300000+/5350000+, needs a bundle whose `progress.pt` is `acl_a1`), labels with `acl_onpolicy.py TASK RAW_DIR...`, then progress-head retraining with the decoder frozen: `ACL_PROG_ONLY=1 ACL_RESUME=<acl_a1.pt> ACL_ONPOLICY=<labelled> ACL_OP_W=1.0 train_acl.py TASK a2 3000` -> `progress.pt` of `handover_block`.
11. **Persistent instruction conditioning** (`stack_blocks_two`): `pic_build_sg.py RAW_DIR...`, `train_pic_scorer.py stack_blocks_two cf s2` -> `subgoal_scorer.pt`; `pic_onpolicy.py ...`, `train_pic.py stack_blocks_two p2` -> `progress.pt` of `stack_blocks_two`.
12. **Closed-loop correction**: no weights; `clc_build_stats.py` recomputes `regsp/data/clc_stats.json` (mean grasp offsets from training-domain rollouts); the executor options are in `configs/final_v21_clc.json`.

The exact historical commands and the queue are preserved in `docs/history/RESEARCH_REPRODUCE_*.md`; per-stage training metrics are summarised in `docs/history/TASK_SPECIFIC_V21_TRAINING.md`, `HEATMAP_TRAINING_REPORT.md`, `COUNTERFACTUAL_REFERENT_REPORT.md`, `BIMANUAL_DECODER_REPORT.md`.

## Known gaps

* Backbone training (`train_backbone.py`) calls `lerobot.scripts.lerobot_train.train`, which builds the policy through LeRobot's factory; upstream `lerobot` has no `lerobot.policies.safediff_vla`, so this single stage needs the SafeDiff research fork. Everything downstream of the backbone checkpoint works with upstream LeRobot.
* Some data-preparation scripts still read intermediate directory names of the research tree (`ckpt/`, `ckpt_bim/`, `v21cache/` under `REGSP_CLEAN_ROOT`); create those directories with the research layout when re-running a stage.
* Seeds used for training-domain data (<= 7000000, e.g. 5300000+/5350000+, 3000000+) are disjoint from every development (7200000+/7500000+) and final (6000000+...8700000+) range; `docs/history/*` documents the ledger and the one seed-leakage correction that was made during development.
