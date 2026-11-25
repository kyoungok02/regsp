# Task-specific V21 training report (StagePlan head, TemporalActionDecoder, AnchorHead)

Task-specific training of the original V21 modules on the union of Clean and randomized_500 demonstrations (Randomized-domain adaptation, not Clean-only zero-shot). These checkpoints were trained on 2026-11-14 and are reused unchanged by the ACL extension (StagePlan head from `exp2_sp/<task>` = the same file as `ckpt_sp/<task>/v2head.pt` for the two-arm tasks; the single-arm controls use the immutable `ckpt_sp/<task>` checkpoints with the original decoders).

## StagePlan head (`training/sp_v2head_<task>.json`)

| task | steps | train clean+rand | val | grasp xyz median cm | Clean | Rand | rot median deg | approach-dir median deg | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| click_bell | 6000 | 286+572 | 219 | 3.77 | 2.44 | 4.98 | 0.8 | 9.43 | e009efa36bc8 |
| place_container_plate | 6000 | 411+810 | 303 | 3.37 | 2.58 | 4.26 | 0.6 | 0.66 | 2c3156f57e2e |
| stack_blocks_two | 6000 | 1756+3613 | 1314 | 4.64 | 4.41 | 4.73 | 21.2 | 0.51 | 6bdb3fed2f14 |
| handover_block | 6000 | 1486+3125 | 1166 | 0.31 | 0.34 | 0.28 | 4.1 | 3.92 | 895ae0575e54 |


## TemporalActionDecoder (`training/sp_decoder_<task>_pretrained.json`)

| task | steps | train clean+rand | val | post-grasp pos cm (step 0) | final | Clean | Rand | grip MAE | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| click_bell | 10000 | 526+1042 | 392 | 5.5 | 1.44 | 0.89 | 1.72 | 0.005 | 2431fc90c5ca |
| place_container_plate | 10000 | 1070+2115 | 791 | 11.08 | 2.99 | 2.18 | 3.4 | 0.052 | ce99d9217782 |
| stack_blocks_two | 10000 | 3788+8034 | 2813 | 8.81 | 1.41 | 0.98 | 1.63 | 0.035 | af614f7605d2 |
| handover_block | 10000 | 3668+7640 | 2854 | 13.33 | 1.68 | 1.15 | 1.95 | 0.031 | 971220835838 |


## AnchorHead (`training/full_anchor_<task>.json`)

| task | train | val | A triangulated ref only (cm) | B ref + const offset | C learned anchor | sha256 |
|---|---|---|---|---|---|---|
| click_bell | 540 | 77 | 13.11 | 0.64 | 0.28 | 7092ad33a765 |
| place_container_plate | 610 | 19 | 8.53 | 5.49 | 0.68 | 7e808d9df3f4 |
| stack_blocks_two | 595 | 30 | 1.13 | 0.45 | 0.41 | 410c67e9996d |
| handover_block | 528 | 7 | 21.39 | 12.65 | 0.68 | adfeaa3609f0 |


Note: the AnchorHead validation set of handover_block has only 7 samples and its learned horizontal offset is unreliable on randomized scenes (median |offset| 9-15 cm for views whose triangulation was correct on training-domain rollouts); the ACL executor therefore keeps the triangulated xy when the offset exceeds 1.5 cm (handover_block) or disagrees with the head ray by more than 4 cm (both tasks).