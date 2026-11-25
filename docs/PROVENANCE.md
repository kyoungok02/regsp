# Provenance and extraction manifest

This file records where the code, configurations and numbers of this repository come from, what was left out and why. Paths are relative names of research outputs (no machine paths).

## Identification of the released model

* **Released configuration: V21-CLC**, frozen as `freeze4` (research branch `robotwin/safediff-v21-4task-counterfactual`, freeze commit `ddc8b50d`, results commit `d69d2e08`, 2026-11-17). Chosen because it is the last frozen version of the RoboTwin V21 line whose final evaluation (160 episodes, fresh seeds) exists *and* whose protocol, executor source and 73 files are hash-frozen. It is **not** simply the newest script: the research tree also contains later, unrelated work that was deliberately not used (D2 50-task zero-additional-training screening, `final_stageplan_v21e_eval.py` / `v21e_*` expansion scripts, RGL/KSP ablations, VLABench Grounded-Stage-Execution studies; see "Excluded").
* **Executor entry point actually invoked by the frozen run**: `scripts/run_final4.sh clc` -> `freeze4/selected_env.sh` -> `scripts/run_clc.sh` -> `scripts/clc_rt_policy.py` (a versioned copy of the frozen `pic_rt_policy.py`, itself a copy of `acl_rt_policy.py`), which subclasses `bim_rt_policy.BimV21Policy` / `v21_rt_policy.V21Policy` through `v21_diag` (whose diagnostic logging is bypassed by the executor). The import closure of that entry point was computed from the sources and by tracing every file opened while constructing each task's policy under the frozen environment.
* **Per-task checkpoints actually used** (frozen manifest `results/final_v21_clc/MANIFEST_FREEZE4.sha256`): see `configs/bundle_manifest.json`. Files that the frozen runner loaded and then overwrote (original ten-task heads, scorer, anchor, decoders; `scorer_cf_u1` of `stack_blocks_two`, replaced by the subgoal scorer) are not part of the bundle.
* **Source commits**: SafeDiff-VLA policy package (vendored): research fork LeRobot, tag `safediff-stageplan-v2.1-grounded-final` (`1f4b7906`) lineage, files taken from fork commit `e9179d50` (last change to the package: `98f910aa`); RoboTwin V21 line: `bc8ccd23` (ten-task, bimanual extension), `e1c92aa2` (freeze2, ACL), `65c3de48` (freeze3, PIC), `ddc8b50d` (freeze4, CLC); RoboTwin simulator commit `0aeea2d6`.

## Ported files (original source -> ReGSP)

| ReGSP file | Research source | SHA-256 prefix of the source | Relation |
|---|---|---|---|
| `regsp/grounding/{geometry,proposals,scorer,anchor}.py` | `scripts/v21rt.py` | `19f0bc4b526f4c3a` | verbatim (split) |
| `regsp/grounding/scorer.py (CycleScorer)` | `scripts/bim_common.py` | `95fa0ebd79aaf732` | verbatim (split) |
| `regsp/decoding/bimanual.py` | `scripts/bim_common.py` | `95fa0ebd79aaf732` | verbatim (split) |
| `regsp/execution/base_policy.py` | `scripts/v21_rt_policy.py` | `de39d0c46e28f1ba` | adapted: sys.path/path constants removed; constructor loads the per-task bundle directly |
| `regsp/execution/bimanual_policy.py` | `scripts/bim_rt_policy.py` | `85ca77db1a9f5f50` | adapted: constructor loads the bundle directly |
| `regsp/execution/closed_loop.py` | `scripts/clc_rt_policy.py` | `ac371e11326c70a4` | adapted: imports, bundle paths, no monkey-patching/runpy; DiagMixin pass-through removed |
| `regsp/decoding/adaptive.py` | `scripts/acl_common.py` | `1c46d22d406abaa8` | verbatim (split) |
| `regsp/decoding/progress.py` | `scripts/acl_common.py + pic_common.py` | `1c46d22d406abaa8` | verbatim (split) |
| `regsp/execution/proprio.py` | `scripts/acl_common.py` | `1c46d22d406abaa8` | verbatim (split) |
| `regsp/execution/kinematics.py` | `robotwin_stageplan/scripts/rt_kin.py` | `fb60cf9bc8be00b5` | adapted: RoboTwin root from REGSP_ROBOTWIN_ROOT |
| `regsp/envs/cameras.py` | `robotwin_stageplan/scripts/rt_common.py` | `0f01312467aba44f` | subset; calibration JSON shipped in regsp/data |
| `scripts/evaluate/robotwin_eval.py` | `scripts/v21_eval.py` | `bbac0df3aa74bea4` | adapted: policy via factory; diagnostic --oracle removed |
| `regsp/envs/robotwin.py` | `robotwin_stageplan/scripts/rt_env.py + robotwin_stageplan_d2/scripts/d2_scene.py` | `5eae3e4378ea81e3` | merged, paths from REGSP_ROBOTWIN_ROOT |
| `regsp/backbone/*.py` | `src/lerobot/policies/safediff_vla/{configuration,modeling}_safediff_vla.py, domain_adapter, execution, grasp_correction_head, grounded_grasp_v2/v3, language_grounded_target_pooling, local_grasp_refiner, rotation_encoding, state_predictor, target_point_head, temporal_decoder, utils, stageplan_v2` | (fork commit `e9179d50`) | vendored; changes: `from lerobot.action_semantics` -> relative import, two config fields (`action_representation`, `action_domain`) declared on `SafeDiffVLAConfig` because the fork defined them on LeRobot's base config class |
| `regsp/backbone/action_semantics/` | `src/lerobot/action_semantics/*.py` | (fork commit `e9179d50`) | vendored unchanged |
| `regsp/backbone/loader.py` | `examples/safediff_vla/v21_common.py` (`load_policy`, `make_batch`) | | adapted: no `chdir`, checkpoint from `REGSP_BACKBONE_CKPT` |
| `scripts/data/*`, `scripts/train/*` | `scripts/*` of `robotwin_v21_4task_counterfactual` and `robotwin_safediff_original_10task` | | mechanical path/import port (`tx()` rules: literals of the two research output roots -> `regsp.paths`, `sys.path` hacks removed, local imports -> `regsp.*`); logic untouched; compiled only |
| `scripts/train/train_backbone.py` | `examples/safediff_vla/train_grounded_grasp_v2_20k.py` | | import path only |
| `regsp/data/*.json`, `configs/final_v21_clc.json` | `acl_calib.json`, `clc_stats.json`, `camera_calibration.json`, `ckpt/<task>/runtime_stats.json`, `ckpt_bim/<task>/bim_stats.json`, `freeze4/PROTOCOL4.json`, `freeze4/selected_env.sh`, `scripts/run_clc.sh` | | copied / transcribed |
| `docs/history/*`, `results/*`, `figures/*` | research reports, CSVs, `figures_supp/` | | copied with absolute paths replaced by relative names |

## Included (and why)

Model definitions, losses/trainers, data preparation and valid-label generation, counterfactual training, proposal/anchor/StagePlan/decoder/adaptive/PIC trainers, the closed-loop executor, RoboTwin adapter, evaluation driver, minimal configs, tests, figures with plotting scripts and provenance, frozen result tables.

## Excluded from the publication copy (reasons)

| Component | Reason |
|---|---|
| `acl_rt_policy.py`, `pic_rt_policy.py`, `run_acl*.sh`, `run_pic.sh`, `run_final*.sh`, `run_old_exec.sh` | superseded earlier copies of the executor; `clc_rt_policy.py` reproduces them with every `CLC_*` option off (frozen behaviour) |
| `v21_diag.py`, `diag`-style analysis (`analyze_components.py`, `audit_independence*.py`, `analysis/`) | diagnostic wrappers with privileged logging, bypassed by the final executor; independence audits are summarised in `docs/history` |
| `freeze*.py`, `make_reports.py`, `clc_make_reports.py`, `pic_make_reports.py` | research bookkeeping |
| `sched_*.py`, `gpuq.sh`, `queue.jsonl*`, `chain_*.sh`, `run_*` job scripts, watchdogs, logs | temporary schedulers/queues/logs |
| `ckpt*/`, `da_cache/`, `cf_cache/`, `scenes/`, `views/`, `feats/`, `tokens*/`, `onpolicy/`, `sg/`, `data_rand/` (128 GB tree) | training intermediates and checkpoints; checkpoints are provided only as hashes |
| `da_views_*_discarded`, `backup_v21_4task_*`, `eval/*smoke*`, development evaluation directories | failed/discarded experiments and development artifacts (summarised in the history reports) |
| `eval/final*` episode logs and videos | large; per-episode success is kept in `results/final_v21_clc/episodes_*.csv` |
| D2 50-task screening (`robotwin_stageplan_d2`, `robotwin_stageplan_50task_screening`), RGL 50-task, `v21e_*` expansion scripts, `final_stageplan_v21*_eval.py` (uncommitted modifications in the research repository at extraction time) | different studies/protocols (zero-additional-training screening, RGL/KSP); not part of the V21-CLC pipeline |
| `regsp_component_ablations/`, `regsp_stage_intervention/`, `rgl_training_scope_ablation/` | separate ablation studies (VLABench Grounded-Stage-Execution and RoboTwin component ablations on restored physics states, different seeds); they are not the frozen final evaluation, so their numbers are not used here |
| VLABench pipeline (`final_stageplan_v21_eval.py --grounded`, `v21_provider.py`, GSE) | a different embodiment of the same method family; this repository releases the RoboTwin implementation only |
| `libero_*`, `legacy/` policy files, other SafeDiff policy variants | abandoned/unused variants; the vendored package keeps only the modules that the checkpoint-loading code imports |
| Original `v21_eval.py --oracle` option | privileged diagnostic |

## Preserved, untouched

The original repositories, branches, worktrees (`/lerobot`, the four-task, original ten-task and other worktrees), checkpoints, experimental results and backups were only read. The `origin` remote of the research repository was not modified. The only filesystem writes outside `outputs/ReGSP` were scratch files in the session scratchpad.
