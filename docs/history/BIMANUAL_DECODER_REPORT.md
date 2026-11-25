# Bimanual and Adaptive Single/Bimanual TemporalActionDecoder report

Identity: Original Counterfactual Grounding + Adaptive Single/Bimanual TemporalActionDecoder + Multi-Stage Closed-Loop Execution (documented extension of grounded-final SafeDiff-VLA V21; no RGL/KSP, no scripted expert actions).

* `b1`: first bimanual adaptation; its trainer left the base decoder frozen (parameter-freezing bug). `b2`: corrected retraining (all parameters explicitly unfrozen), 8000 steps.
* `acl_a1`: AdaptiveDecoder = the original TemporalActionDecoder (initialised from `b2`) run over 2 x 50 action queries (primary slot P, partner slot Q) -> two 7-D heads = synchronised 14-D bimanual chunk; conditioning on both-arm states and grippers, grounded target, place/receiving target, role, ARM MODE (LEFT_ARM / RIGHT_ARM / BIMANUAL), stage, 2-step observation/action history. 8000 steps, all 24,057,823 parameters trainable, every tensor verified changed.
* `acl_a2`: the same decoder weights (bit-identical, `audit/acl_a1_vs_a2_weights.json`) with the ProgressHead retrained for 3000 steps on demonstration labels plus REAL on-policy outcomes (training-domain rollouts with random grasp/place perturbations; labels from the simulator state stored at each sample = training-only privileged supervision, never a policy input). This adds the FAILED_OR_RECOVERY class, which has no demonstration examples.

## Bimanual decoder adaptation (`training/sp_bim_decoder_*.json`)

| task | tag | steps | train/val | post pos cm step 0 | final | Clean | Rand | extension inputs zeroed | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| handover_block | b1 | 8000 | 11308/2854 | 1.684 | 1.634 | 1.135 | 1.884 | 1.688 | 2815152191c3 |
| handover_block | b2 | 8000 | 11308/2854 | 1.684 | 1.386 | 0.96 | 1.599 | 1.836 | fa5371afacf1 |
| stack_blocks_two | b2 | 8000 | 11822/2813 | 1.412 | 1.359 | 0.912 | 1.587 | 1.433 | bd7222c1fcb6 |


## Adaptive decoder + ProgressHead (`training/acl_<task>_<tag>.json`)

| task | tag | steps | trained | trainable params | frozen params | train/val | primary pos cm | partner pos cm | primary grip-state acc | partner grip-state acc | primary pos cm (predicted arm mode) | arm-mode acc | progress acc (demo labels) | trainable tensors unchanged | sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| handover_block | a1 | 8000 | all modules | 24057823 | 0 | 11308/2854 | 1.31 | 0.996 | 0.9872 | 0.9943 | 1.315 | 0.9622 | 0.9874 | 0 | 16d069ed0a4e |
| handover_block | a2 | 3000 | progress head only | 1114888 | 22942935 | 11308/2854 | 1.31 | 0.996 | 0.9872 | 0.9943 | 1.316 | 0.9569 | 0.9914 | 0 | 94ba11b5e07d |
| stack_blocks_two | a1 | 8000 | all modules | 24057823 | 0 | 11822/2813 | 1.234 | 0.386 | 0.9772 | 1.0 | 1.235 | 0.9641 | 0.9959 | 0 | b94dcdef74d9 |
| stack_blocks_two | a2 | 3000 | progress head only | 1114888 | 22942935 | 11822/2813 | 1.234 | 0.386 | 0.9772 | 1.0 | 1.235 | 0.9634 | 0.9904 | 0 | decf9ed94cc1 |


### ProgressHead on held-out ON-POLICY episodes (classes 0 NOT_STARTED, 1 IN_PROGRESS, 2 COMPLETED, 3 FAILED_OR_RECOVERY, 4 TASK_COMPLETED)

| task | tag | on-policy train samples | val samples | val acc | confusion label->pred | tensors changed / total per module group |
|---|---|---|---|---|---|---|
| handover_block | a2 | 627 | 145 | 0.9448 | {"1": {"1": 96, "3": 3, "4": 0}, "3": {"1": 3, "3": 38, "4": 2}, "4": {"1": 0, "3": 0, "4": 3}} | mode/stage/hist/slot: 0/7; base.queries+encoders: 0/16; base.transformer: 0/72; base.action_head(P): 0/2; partner/role/place: 0/9; head_partner(Q): 0/2; tqe: 0/5; vgc: 0/4; tph: 0/6; progress_head: 22/22 |
| stack_blocks_two | a2 | 750 | 183 | 0.9454 | {"1": {"0": 0, "1": 127, "2": 0, "3": 2, "4": 0}, "2": {"0": 0, "1": 0, "2": 21, "3": 0, "4": 0}, "3": {"0": 2, "1": 4, "2": 2, "3": 19, "4": 0}, "4": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 6}} | mode/stage/hist/slot: 0/7; base.queries+encoders: 0/16; base.transformer: 0/72; base.action_head(P): 0/2; partner/role/place: 0/9; head_partner(Q): 0/2; tqe: 0/5; vgc: 0/4; tph: 0/6; progress_head: 22/22 |


On-policy label report stack_blocks_two: episodes 117, labels {'1': 666, '2': 111, '3': 131, '4': 25}, episodes with a real failure 80, failure followed by task success 6, raw dirs ['onpolicy/raw/stack_blocks_two', 'onpolicy/raw2/stack_blocks_two']


On-policy label report handover_block: episodes 123, labels {'1': 566, '2': 0, '3': 186, '4': 20}, episodes with a real failure 75, failure followed by task success 2, raw dirs ['onpolicy/raw/handover_block', 'onpolicy/raw2/handover_block']


## Runtime arm-mode switching and gripper coordination (final2 cells; `FINAL_*_RESULTS.csv`)

| task | config | success | decodes by arm mode | arm-mode switches | episodes with single-arm and BIMANUAL modes | episodes using both arms | receiver holds after donor release / transfers | donor release predicted by decoder / transfers | verified grasps / attempts |
|---|---|---|---|---|---|---|---|---|---|
| handover_block | clean | 13/20 | {"BIMANUAL": 38, "RIGHT_ARM": 26, "LEFT_ARM": 1} | 39 | 13/20 | 13/20 | 13/13 | 13/13 | 26/47 |
| stack_blocks_two | clean | 18/20 | {"LEFT_ARM": 38, "RIGHT_ARM": 30, "BIMANUAL": 24} | 31 | 11/20 | 12/20 | 0/0 | 0/0 | 39/45 |
| handover_block | rand | 6/20 | {"BIMANUAL": 27, "RIGHT_ARM": 26, "LEFT_ARM": 3} | 31 | 12/20 | 14/20 | 8/8 | 8/8 | 22/63 |
| stack_blocks_two | rand | 15/20 | {"RIGHT_ARM": 30, "BIMANUAL": 40, "LEFT_ARM": 44} | 56 | 18/20 | 19/20 | 0/0 | 0/0 | 40/47 |


Runtime use: the arm-mode logits of the ProgressHead select the mode embedding of the joint decoder and decide whether the partner arm's synchronised chunk is executed; the donor gripper is released on the decoder's partner-gripper prediction and only after the receiver's grasp has been verified by the measured finger opening; the transfer counts only if the receiver still holds after the donor has opened.

Replanning interval (development seeds, a1, progress decisions off): K=5 (the full 5 strided control steps of a chunk) vs K=3: stack_blocks_two Randomized 10-11/12 vs 7/12, Clean 12/12 vs 11/12; handover_block Randomized 7-8/12 vs 4/12, Clean 8-9/12 vs 10/12. K=5 selected; K=2 was started and stopped for time (incomplete record).

Limitations: (1) the ProgressHead's grasp verdict (D1) is logged and trained but does not override a positive measured-finger verdict (on development seeds the finger test made no error, the head rejected one correct grasp); (2) the post-release verdict (D2) detected every failed final placement it was asked about on development seeds (4/4 stack rejections were true failures) but the subsequent re-grasp succeeded in only 1/4 (stack) and 0/6 (hand-over, perturbed training-domain rollouts): the hand-over receive stage has no table re-grasp behaviour; (3) round 3 of on-policy generation (with decisions on) and the K=2 cells were stopped for time and are incomplete.