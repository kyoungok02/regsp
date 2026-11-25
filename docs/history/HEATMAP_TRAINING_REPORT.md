# Heat-map (class-agnostic 64x64 proposal head) training report

Randomized-domain adaptation of the ORIGINAL V21 class-agnostic proposal heads (v21rt.Head(64), focal + masked L1 offset, 4000 steps, batch 16, seed 1000, one independent head per camera, frozen SigLIP features). Architecture, loss and optimiser are unchanged; only the training frames differ. `*_obs2` (2026-11-16) adds the additional randomized view units of `da_views2` (seeds 5100000+, train split only) for stack_blocks_two and handover_block; the validation frames are the same as for the earlier heads, so the rows are directly comparable.

All numbers are read from `training/da_heads_<task><tag>.json` (validation splits only; the baseline heads are re-evaluated on the same frames). recall = recall@10px of visible ground-truth centres; ref = referent recall@10px; fp = false-positive rate.


## click_bell

| task | heads | camera | train frames clean+rand | Clean recall | Clean ref | Rand recall | Rand ref | Rand fp | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| click_bell | original Clean-only head | head_camera | - | 0.471 | 0.888 | 0.277 | 0.542 | 0.814 | d10af00d2ef6 |
| click_bell | DA v1 (views) | head_camera | 999+672 | 0.523 | 0.802 | 0.396 | 0.861 | 0.426 | 2af69cb64469 |
| click_bell | original Clean-only head | left_camera | - | 0.705 | 0.725 | 0.035 | 0.0 | 0.961 | e92311e01fda |
| click_bell | DA v1 (views) | left_camera | 999+672 | 0.712 | 0.75 | 0.765 | 0.5 | 0.235 | 1868b6788146 |
| click_bell | original Clean-only head | right_camera | - | 0.648 | 0.655 | 0.092 | 0.067 | 0.881 | aa0a42a9dbe9 |
| click_bell | DA v1 (views) | right_camera | 999+672 | 0.659 | 0.759 | 0.553 | 0.533 | 0.125 | c9f81c71496e |
| click_bell | DA obs (views + observation views) | head_camera | 999+768 | 0.538 | 0.966 | 0.413 | 0.912 | 0.456 | 677d96a8f305 |
| click_bell | DA obs (views + observation views) | left_camera | 999+768 | 0.712 | 0.75 | 0.696 | 0.432 | 0.413 | 01b99efc6697 |
| click_bell | DA obs (views + observation views) | right_camera | 999+768 | 0.693 | 0.828 | 0.527 | 0.63 | 0.461 | 6b085a4dd87c |


## place_container_plate

| task | heads | camera | train frames clean+rand | Clean recall | Clean ref | Rand recall | Rand ref | Rand fp | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| place_container_plate | original Clean-only head | head_camera | - | 0.394 | 0.581 | 0.1 | 0.228 | 0.961 | ff0564a57bb1 |
| place_container_plate | DA v1 (views) | head_camera | 1161+779 | 0.401 | 0.588 | 0.223 | 0.424 | 0.76 | 2abd875d3b50 |
| place_container_plate | original Clean-only head | left_camera | - | 0.317 | 0.283 | 0.017 | 0.062 | 0.994 | 9448cf3ac630 |
| place_container_plate | DA v1 (views) | left_camera | 1161+779 | 0.5 | 0.483 | 0.0 | 0.0 | 1.0 | c55355ebf717 |
| place_container_plate | original Clean-only head | right_camera | - | 0.348 | 0.32 | 0.117 | 0.149 | 0.787 | f816072cc412 |
| place_container_plate | DA v1 (views) | right_camera | 1161+779 | 0.404 | 0.32 | 0.482 | 0.447 | 0.764 | bd6d2b7548e4 |
| place_container_plate | DA obs (views + observation views) | head_camera | 1161+857 | 0.407 | 0.628 | 0.307 | 0.421 | 0.668 | 9f82e3ea3f7b |
| place_container_plate | DA obs (views + observation views) | left_camera | 1161+857 | 0.341 | 0.3 | 0.015 | 0.056 | 0.974 | 96a2d9ab20fa |
| place_container_plate | DA obs (views + observation views) | right_camera | 1161+857 | 0.416 | 0.44 | 0.456 | 0.436 | 0.736 | 56484994ba43 |


## stack_blocks_two

| task | heads | camera | train frames clean+rand | Clean recall | Clean ref | Rand recall | Rand ref | Rand fp | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| stack_blocks_two | original Clean-only head | head_camera | - | 0.597 | 0.916 | 0.333 | 0.485 | 0.733 | 4492c60ac3ad |
| stack_blocks_two | DA v1 (views) | head_camera | 1458+722 | 0.614 | 0.959 | 0.541 | 0.737 | 0.5 | 612eb9a112bd |
| stack_blocks_two | original Clean-only head | left_camera | - | 0.253 | 0.432 | 0.009 | 0.022 | 0.994 | 5b586f87fc9d |
| stack_blocks_two | DA v1 (views) | left_camera | 1458+722 | 0.48 | 0.75 | 0.101 | 0.244 | 0.934 | f571733827b1 |
| stack_blocks_two | original Clean-only head | right_camera | - | 0.237 | 0.298 | 0.172 | 0.2 | 0.947 | cc804fee1703 |
| stack_blocks_two | DA v1 (views) | right_camera | 1458+722 | 0.35 | 0.404 | 0.31 | 0.32 | 0.894 | f0c316760e0c |
| stack_blocks_two | DA obs (views + observation views) | head_camera | 1458+932 | 0.716 | 0.983 | 0.57 | 0.826 | 0.718 | c3b0f17b8753 |
| stack_blocks_two | DA obs (views + observation views) | left_camera | 1458+932 | 0.36 | 0.545 | 0.092 | 0.226 | 0.921 | f3d0054e4fb6 |
| stack_blocks_two | DA obs (views + observation views) | right_camera | 1458+932 | 0.356 | 0.423 | 0.395 | 0.424 | 0.916 | 1468ddd96393 |
| stack_blocks_two | DA obs2 (+ da_views2 units) | head_camera | 1458+3212 | 0.714 | 0.983 | 0.581 | 0.925 | 0.61 | c9363dde7c97 |
| stack_blocks_two | DA obs2 (+ da_views2 units) | left_camera | 1458+3212 | 0.44 | 0.682 | 0.115 | 0.245 | 0.935 | d1012690264d |
| stack_blocks_two | DA obs2 (+ da_views2 units) | right_camera | 1458+3212 | 0.275 | 0.337 | 0.209 | 0.273 | 0.939 | 414743ba712e |


## handover_block

| task | heads | camera | train frames clean+rand | Clean recall | Clean ref | Rand recall | Rand ref | Rand fp | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| handover_block | original Clean-only head | head_camera | - | 0.556 | 0.691 | 0.363 | 0.251 | 0.774 | 1b277676e507 |
| handover_block | DA v1 (views) | head_camera | 1494+912 | 0.592 | 0.691 | 0.447 | 0.583 | 0.626 | 9862f3e8232b |
| handover_block | original Clean-only head | left_camera | - | 0.227 | 0.194 | 0.029 | 0.0 | 0.995 | 27291e7248fd |
| handover_block | DA v1 (views) | left_camera | 1494+912 | 0.283 | 0.278 | 0.118 | 0.0 | 0.974 | 2220363e3217 |
| handover_block | original Clean-only head | right_camera | - | 0.395 | 0.243 | 0.048 |  | 0.967 | 0ed16d9e33f1 |
| handover_block | DA v1 (views) | right_camera | 1494+912 | 0.456 | 0.27 | 0.424 |  | 0.587 | aa312679c3fc |
| handover_block | DA obs (views + observation views) | head_camera | 1494+1158 | 0.588 | 0.665 | 0.393 | 0.486 | 0.556 | 22ab0ab1678f |
| handover_block | DA obs (views + observation views) | left_camera | 1494+1158 | 0.243 | 0.278 | 0.047 | 0.044 | 0.984 | 2affa5597136 |
| handover_block | DA obs (views + observation views) | right_camera | 1494+1158 | 0.451 | 0.351 | 0.201 | 0.238 | 0.851 | 821cf182b8f0 |
| handover_block | DA obs2 (+ da_views2 units) | head_camera | 1494+3343 | 0.637 | 0.712 | 0.506 | 0.577 | 0.556 | e85aeff2fb3b |
| handover_block | DA obs2 (+ da_views2 units) | left_camera | 1494+3343 | 0.255 | 0.278 | 0.133 | 0.111 | 0.959 | e3393097145b |
| handover_block | DA obs2 (+ da_views2 units) | right_camera | 1494+3343 | 0.642 | 0.514 | 0.323 | 0.333 | 0.821 | e7fc43c10e34 |


Selection (development seeds only): handover_block uses `*_obs2` (first-grasp success on training-domain seeds 5350100+ rose from 4/16 to 13/16 with otherwise identical code, and development Randomized success from 4/12 to 7-8/12); stack_blocks_two keeps `*_obs` (development result unchanged with `*_obs2`: Randomized 10/12 vs 10-11/12, and its right-wrist recall is lower). The wrist-camera recall on randomized frames remains the weakest component (left wrist 0.13 for handover_block).