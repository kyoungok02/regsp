# Counterfactual referent learning report (HR referent scorer)

HR referent scorer on the VLM text-token states of the policy's own backbone pass (original V21 architecture). `cf` = trained with counterfactual referent pairs (same scene, instruction naming the other object), `base` = without, `ctl` = control; `u1` = trained on the union with the randomized-domain frames. Counterfactual scenes exist for click_bell and stack_blocks_two only (see COUNTERFACTUAL_DATA_MANIFEST.md); place_container_plate and handover_block have a single referent class and use the `base` scorer. This component was not retrained in the ACL extension.

| task | mode | tag | steps | orig train/val | CF train/val | orig val acc | CF val acc | CF swap both correct | pred changes on swap | chance | baseline CF acc | baseline swap both | sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| click_bell | base | v1 | 3000 | 398/104 | 320/80 | 1.0 | 0.5 | 0.0 | 0.0 | 0.287 | 0.5 | 0.0 | 5091f4ca7f69 |
| click_bell | cf | u1 | 3000 | 722/170 | 320/80 | 1.0 | 1.0 | 1.0 | 1.0 | 0.287 | 0.5 | 0.0 | a8cbda6eaa8c |
| click_bell | cf | v1 | 3000 | 398/104 | 320/80 | 1.0 | 1.0 | 1.0 | 1.0 | 0.287 | 0.5 | 0.0 | b0f4ee0984af |
| click_bell | ctl | v1 | 3000 | 398/104 | 320/80 | 1.0 | 1.0 | 1.0 | 1.0 | 0.287 | 0.5 | 0.0 | c3558f6edffe |
| handover_block | base | u1 | 3000 | 2498/590 | 0/0 | 1.0 |  |  |  |  |  |  | 91642493be10 |
| place_container_plate | base | u1 | 3000 | 1089/255 | 0/0 | 1.0 |  |  |  |  |  |  | 79bcf14f118c |
| stack_blocks_two | base | v1 | 3000 | 1431/345 | 116/30 | 1.0 | 0.5 | 0.0 | 0.0 | 0.266 | 0.5 | 0.0 | fd9b2d4537f1 |
| stack_blocks_two | cf | u1 | 3000 | 2361/546 | 116/30 | 1.0 | 1.0 | 1.0 | 1.0 | 0.266 | 0.5 | 0.0 | cec2c3ff7e98 |
| stack_blocks_two | cf | v1 | 3000 | 1431/345 | 116/30 | 0.991 | 1.0 | 1.0 | 1.0 | 0.266 | 0.5 | 0.0 | d41df9f22dc3 |
| stack_blocks_two | ctl | v1 | 3000 | 1431/345 | 116/30 | 1.0 | 1.0 | 1.0 | 1.0 | 0.266 | 0.5 | 0.0 | fb591abe33a5 |


Frozen selection: `scorer_cf_u1` for click_bell and stack_blocks_two, `scorer_base_u1` for place_container_plate and handover_block (hashes in freeze2/MANIFEST_FREEZE2.sha256).