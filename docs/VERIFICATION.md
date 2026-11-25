# Verification performed during extraction

Environment: the `robotwin` conda environment on the development machine (Python 3.12.16, torch 2.7.1+cu128, transformers 5.22.2, numpy 1.26.4), with **upstream `lerobot==0.7.0` shadowing the research fork** via `PYTHONPATH` (see docs/INSTALL.md), RoboTwin commit 0aeea2d6. One RTX PRO 4500 per job.

| # | Check | Result |
|---|---|---|
| 1 | Import of every module of `regsp` and every script (static AST resolution of all imports against the environment) | all resolved (the two `validate_units` references were fixed by porting that module) |
| 2 | `pytest tests` | **73 passed** with the weight bundle, backbone checkpoint, RoboTwin and CUDA available; **67 passed, 6 skipped** without those (checkpoint-free subset; the weight/RoboTwin-dependent tests skip themselves). 4 of the 73 are the per-task weight-equivalence tests |
| 3 | Line-by-line comparison of the ported executor/model modules with the research sources | every non-import, non-path line of `v21rt`, `bim_common`, `acl_common`, `pic_common`, `v21_rt_policy`, `bim_rt_policy`, `clc_rt_policy` is present verbatim in `regsp/`; the only differences are the import headers, path constants, the bundle-based constructors, removal of the diagnostic pass-through wrapper and removal of `runpy`/monkey-patching. (This check found and fixed one slicing error: the last line of `CycleScorer.forward` had been dropped; the unit test `test_scorer_variants_reduce_to_the_original_at_initialisation` also covers it.) |
| 4 | Backbone loading: `regsp` on upstream `lerobot==0.7.0` vs research fork | 472,169,389 parameters, identical state-dict digest; `lerobot==0.6.1` fails on the tokenizer artifact (so 0.7.0 is pinned) |
| 5 | Checkpoint-compatible loading of all four tasks: frozen research policy vs `regsp` policy built from the canonical bundle | module digests equal for backbone, StagePlan head, 3 proposal heads, scorers, AnchorHead, decoders, bimanual/adaptive decoders, progress head, statistics and 40+ executor flags (36 / 36 / 40 / 41 compared entries for click_bell / place_container_plate / stack_blocks_two / handover_block; 0 differences). The only reference entry not reproduced is the plain scorer `sc` of `stack_blocks_two`, which the frozen executor loads but never uses (superseded by the subgoal scorer) |
| 6 | Bundle hashes | all 33 bundle files match `configs/bundle_manifest.json`; the 32 that originate from the freeze4 manifest also match its hashes (the 33rd, the handover place-referent scorer, is not in that manifest) |
| 7 | 7-D / 14-D interfaces; gripper polarity; stage/cycle indexing | `tests/test_models.py`, `tests/test_stageplan_and_actions.py` (10-D encoded -> 7-D per arm -> 14-D [L,R]; end-effector action 16-D; gripper 1.0 open / 0.0 closed; role index `min(cyc,1)*2+(arm=="R")`; pre-grasp samples 19/34/49) |
| 8 | Entry points | `scripts/evaluate/robotwin_eval.py --help` runs; training/data scripts compile (`tests/test_scripts.py`); training scripts were **not executed** |
| 9 | Aggregation | `scripts/evaluate/aggregate_results.py` applied to the research tree's frozen final episode files reproduces 71/80 Clean, 55/80 Randomized and every Wilson interval |
| 10 | End-to-end smoke episodes with the released ReGSP code path (not a benchmark) | Clean seed 8200000, n = 1 per task, bundle assembled from the research checkpoints: 4 episodes (3 success, 1 failure that the original executor reproduces); see "Smoke episodes" |
| 11 | Figures | both plotting scripts regenerated from the research tree: PNG byte-identical, SVG identical except date/ids |
| 12 | Scans | no machine paths (`tests/test_configs.py`), no files > 2 MB, no checkpoint/dataset extensions tracked, no credentials/tokens |

## Smoke episodes

The bundle was assembled from the frozen research checkpoints (symlinks, hashes verified) and `scripts/evaluate/robotwin_eval.py` was run with `--n 1` per task, Clean, first valid seed from 8200000 (not a benchmark; 4 episodes):

| Task | Seed | ReGSP | Frozen run, same seed | Notes |
|---|---|---|---|---|
| click_bell | 8200000 | success, 10 calls | success, 10 calls | same arm (R), same call-tag sequence, first-cycle grasp target 0.06 cm from the frozen run's |
| stack_blocks_two | 8200000 | success, 37 calls | success, 35 calls | stages L/grounded then R/grounded in both |
| handover_block | 8200000 | success, 36 calls | success, 36 calls | L/grounded then R/memory_receive in both (adaptive decoder, progress head, memory path exercised) |
| place_container_plate | 8200001 | **failure** (twice) | success | see below |

`place_container_plate` seed 8200001 is a hard scene (first attempt falls back to the StagePlan regression in every run). To separate port errors from run-to-run variation it was run through (a) the ReGSP code path and (b) the original research executor (`clc_rt_policy.py` under the frozen `freeze4` environment, written to scratch), 6 episodes each (seeds 8200001-8200006, `--n 6`): **per-seed outcomes identical, 5/6 vs 5/6 (only 8200001 fails in both)**; the frozen run had 8200003 instead of 8200001 as its failure among these six. The instruction text is re-sampled in every run (`generate_episode_descriptions` is not seeded by the evaluation seed), and SAPIEN is not bit-reproducible, so single-seed outcomes can differ between runs; these checks show that the extraction does not change behaviour, not that every seed reproduces.

## Not verified

* No training stage was re-run (the task forbade launching training).
* The frozen 160-episode final evaluation was not re-run; the reported numbers are those of the frozen run, re-aggregated from its per-episode records.
* A clean-room environment built from `requirements.txt` alone was not created (see docs/INSTALL.md).
* Backbone training through `lerobot-train` outside the research fork.
