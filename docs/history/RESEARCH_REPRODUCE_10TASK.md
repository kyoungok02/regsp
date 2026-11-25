# Reproduction (Machine D, GPUs 0-2, env `<home>`)

Worktree `<worktree:original-10task>`, branch `robotwin/safediff-original-10task-task-specific`, outputs `outputs/robotwin_safediff_original_10task/` (git-ignored; committed with `git add -f`). Dataset `hxma/RoboTwin-LeRobot-v3.0` rev `5d857cc7` (Clean demos only). `PY=python`, `export RGL_DENOISER=optix OMP_NUM_THREADS=3`.

1. Labels: `$PY scripts/build_labels_10.py`; scenes `scripts/gen_scenes_10.py T 75 7000000 scenes/T`; view sequences `scripts/gen_views_10.py T 75 7000000 views/T`; active-view frames `scripts/gen_obsviews_10.py T 75 7000000 obsviews/T` (wrap with `scripts/run_cell_bounded.sh GPU DIR cmd...`; validate with `scripts/validate_units.py`).
2. Grounding: `NUNITS=75 $PY scripts/build_v21_cache_10.py T`; `$PY scripts/train_v21_task.py T` (heads, scorer, anchor).
3. Decoder / StagePlan / stats: `scripts/tokens_full_10.py T`, `scripts/train_decoder_10.py T pretrained 6000`, `scripts/train_v2head_10.py T`, `scripts/make_runtime_stats_10.py T`.
4. Bimanual extension (handover, stack): `scripts/gen_place_10.py handover_block 40 7000000 place_pairs_v3/handover_block`; `scripts/bim_stats_10.py handover_block target_box place_pairs_v3/handover_block`; `scripts/bim_stats_10.py T`; `scripts/train_bim_decoder_10.py T b1 6000`; `INIT=scratch $PY scripts/train_bim_grounding_10.py stack_blocks_two cycle b2s`; `$PY scripts/train_bim_grounding_10.py handover_block place b1 target_box`.
5. Development evaluation: `scripts/v21_eval.py --task T --config demo_clean --n 10 --st_seed 8000000 --out DIR --ground v21 --tag "" [--bim b1 --nview 6 ...]`; ablations `scripts/run_ablate_all.sh GPU T...`; bimanual ablations `scripts/run_bim_ablations.sh`.
6. Frozen final evaluation: `scripts/run_final.sh GPU T...` (Clean seeds 9000000+, Randomized 9500000+; 10 valid episodes each).
7. Tables: `scripts/summarize_eval.py OUT.csv label=dir ...`. Hashes: `CHECKPOINTS_SHA256.txt`, `scripts_sha256_frozen.txt`, `MANIFEST_FINAL.sha256`.
