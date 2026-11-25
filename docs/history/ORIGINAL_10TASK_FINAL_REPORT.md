# FINAL REPORT — SafeDiff-VLA StagePlan-v2.1 (V21), RoboTwin 10-task task-specific, Machine D

**Label: mixed-completeness V21-family results.** Ten independent task-specific V21 policies (shared frozen SmolVLA backbone only). Component labels: [O] original unchanged, [R] original architecture retrained per task, [E] embodiment adapter, [X] new extension. Not RGL/KSP/D2; no scripted executor; no privileged inference inputs. These results are frozen and must not be relabelled or overwritten.

## Headline (frozen protocol, fresh seeds Clean 9000000+, Randomized 9500000+, 10 valid episodes per task per setting = 200 native episodes, 0 infrastructure failures)
| task | Clean | Randomized | model completeness |
|---|---|---|---|
| handover_block | 0/10 | 0/10 | 75 view / 72 obs units; bimanual extension (b1 decoder, const place) |
| stack_blocks_two | 6/10 | 0/10 | 75 / 63; cycle scorer + coarse fallback + 6 views |
| place_dual_shoes | 0/10 | 0/10 | 75 / 74; **no AnchorHead** (too few triangulable samples) |
| place_object_basket | 2/10 | 0/10 | **68 / 64** |
| place_bread_basket | 2/10 | 1/10 | 75 / 66 |
| beat_block_hammer | 3/10 | 1/10 | 75 / **41** |
| click_bell | 9/10 | 1/10 | 74 / 62 |
| place_container_plate | 7/10 | 0/10 | **67 / 55** |
| open_laptop | 3/10 | 1/10 | **60** / 75 |
| shake_bottle | 4/10 | 2/10 | 75 / 58; no AnchorHead |
| **Total** | **36/100 (36 %)** | **6/100 (6 %)** | counts above are the actual verified units used |

Multi-cycle/bimanual-capable tasks (5): Clean 10/50, Randomized 1/50. Single-cycle tasks (5): Clean 26/50, Randomized 5/50. Wilson intervals: `eval/FINAL_CLEAN.csv`, `eval/FINAL_RANDOMIZED.csv`; per task: `eval/FINAL_PER_TASK.csv`. Grounded/fallback cycles: `audit/FINAL_FAILURE_STATS.md` (Clean: 74/150 cycles grounded, 5 coarse fallback, 58 StagePlan fallback, 13 no proposals; Randomized: 60/9/73/8).

Caveats: n=10 per cell (very wide intervals); capture caps left several tasks under the 75-unit target (counts above, never silently reduced); the per-task configuration of handover and stack was selected on development seeds (block A is optimistic, block B and the final are unbiased); the final handover result (0/10) shows the development gain did not replicate; Randomized performance collapses because training used Clean demonstrations only (by protocol).

## What was corrected beyond the first V21 port
* Camera geometry audit: live per-frame observation extrinsics (CAMERA_MOTION_GEOMETRY_AUDIT.md); a frozen wrist pose was shown to give ~14 cm triangulation error.
* Bimanual failure audit and extension (BIMANUAL_V21_EXTENSION_REPORT.md): stack improved on development (5/20 -> 14/20 pooled) and final Clean 6/10; handover not solved (pooled dev 2/20 -> 6/20, final 0/10).

## Ablations (development seeds 8000000+, Clean, live geometry, 10 episodes each; `eval/ABLATIONS_DEV.csv`)
| task | full (dev ref.) | no HR scorer | no AnchorHead | no active view (orig) | decoder frozen | decoder random-init |
|---|---|---|---|---|---|---|
| handover_block | 6/10 | 0 | 0 | 2 | 0 | 1 |
| stack_blocks_two | 9/10 | 6 | 8 | 4 | 0 | 6 |
| place_dual_shoes | 0 | 0 | 0 | 0 | 0 | 0 |
| place_object_basket | 1 | 0 | 2 | 1 | 0 | 1 |
| place_bread_basket | 2 | 1 | 2 | 0 | 0 | 1 |
| beat_block_hammer | 3 | 1 | 4 | 4 | 0 | 4 |
| click_bell | 9 | 9 | 8 | 5 | 0 | 8 |
| place_container_plate | 7 | 5 | 4 | 6 | 0 | 5 |
| open_laptop | 3 | 3 | 3 | 3 | 3 | 4 |
| shake_bottle | 8 | 6 | 7 | 9 | 2 | 7 |

Reading: the adapted learned decoder is the most consistent contributor (frozen decoder ≈ 0 on 9/10 tasks); the active-view extension helps click_bell/stack/handover and not shake_bottle/beat_hammer; scorer and anchor effects are mostly within n=10 noise except stack/handover. The "full" reference for each task is the nearest matched dev run (documented in the script that built the CSV): handover/stack = adopted bimanual configs (block A, optimistic); dual_shoes = extension config; click_bell/container = live A/B arm; others = dev_clean_live. **Not run:** frozen-vs-adapted StagePlan head ablation (the original-weights StagePlan head is not loaded by this runtime) — explicitly blocked, not claimed.

## Software defects found and fixed (versioned in the repository history)
`--dec frozen` branch bug (raise None) fixed post-freeze without touching the final path (PROTOCOL_FROZEN.md addendum); earlier: obs-view filter bug, arm_tag bug, NameError in live-geometry patch, no-proposal fallback, CKPT path/cwd — all documented in the commit history and earlier audits.

## Provenance
Checkpoint hashes `CHECKPOINTS_SHA256.txt`; scripts `scripts_sha256_frozen.txt`; protocol `audit/PROTOCOL_FROZEN.md`; training `training/TRAINING_REPORT.md`; runtime verification `audit/METHOD_RUNTIME_VERIFICATION.md`; failure analysis `audit/FAILURE_ANALYSIS.md`; commands `REPRODUCE.md`; stopped RGL/KSP experiment `STOPPED_RGL_KSP_EXPERIMENT.md`. Historical RGL-50 and task-specific RGL artifacts are untouched.
