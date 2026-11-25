# History of frozen versions

All numbers are native RoboTwin successes (simulator `eval_success`) from each stage's own frozen protocol, taken from the per-stage result tables kept in `results/history/` and `results/final_v21_clc/`. **Each row/stage used its own fresh seed set**; rows are not matched comparisons. Within a stage, "previous model" columns were re-run on that stage's seeds and are matched.

| Version (research label) | Training data | Seeds Clean / Rand, n per cell | Reports |
|---|---|---|---|
| Ten-task original V21-family reconstruction (`robotwin_safediff_original_10task`, commit bc8ccd23) | Clean demos only | 9000000+ / 9500000+, n = 10 (10 tasks) | `ORIGINAL_10TASK_FINAL_REPORT.md`, `results/history/original10task_*.csv` |
| Four-task study, frozen first final (`freeze/`; "full" and "baseline") | Clean + Randomized | 6000000+ / 6500000+, n = 20 | `V21_ACL_FINAL_REPORT_4TASK.md` (columns "earlier frozen full/baseline") |
| V21-ACL (adaptive decoder + progress head + multi-stage closed loop; `freeze2`, commit e1c92aa2) | Clean + Randomized | 8000000+ / 8500000+, n = 20 | `V21_ACL_FINAL_REPORT_4TASK.md`, `results/history/FINAL_*_RESULTS.csv` |
| V21-PIC (persistent instruction conditioning; `freeze3`, commit 65c3de48) | as above; only `stack_blocks_two` retrained | 8100000+ / 8600000+, n = 20 | `V21_PIC_FINAL_REPORT.md`, `results/history/PIC_FINAL_RESULTS.csv`, `PIC_LANGUAGE_DEPENDENCE.csv` |
| **V21-CLC (released; `freeze4`, commit ddc8b50d; results d69d2e08)** | as above; no new weights | 8200000+ / 8700000+, n = 20 | `V21_CLC_FINAL_REPORT.md`, `results/final_v21_clc/` |

"Baseline" in the four-task study = the ten-task original Clean-only models evaluated on the four-task seeds (`final_baseline`, run without any of the replacement checkpoints).

## Per-cell results (successes / n)

| Version | Config | click_bell | place_container_plate | stack_blocks_two | handover_block | Pooled |
|---|---|---|---|---|---|---|
| Ten-task original | Clean | 9/10 | 7/10 | 6/10 | 0/10 | 22/40 (36/100 over 10 tasks) |
| | Randomized | 1/10 | 0/10 | 0/10 | 0/10 | 1/40 (6/100 over 10 tasks) |
| Baseline (ten-task models, 4-task seeds) | Clean | 17/20 | 14/20 | 11/20 | 0/20 | 42/80 |
| | Randomized | 6/20 | 5/20 | 0/20 | 0/20 | 11/80 |
| Full retrained | Clean | 20/20 | 14/20 | 13/20 | 4/20 | 51/80 |
| | Randomized | 18/20 | 11/20 | 4/20 | 0/20 | 33/80 |
| ACL | Clean | 20/20 | 16/20 | 18/20 | 13/20 | 67/80 |
| | Randomized | 17/20 | 8/20 | 15/20 | 6/20 | 46/80 |
| PIC | Clean | 20/20 | 16/20 | 16/20 | 14/20 | 66/80 |
| | Randomized | 14/20 | 10/20 | 13/20 | 5/20 | 42/80 |
| PIC stage, previous (ACL) model on PIC seeds | Clean | 20/20 | 14/20 | 19/20 | 12/20 | 65/80 |
| | Randomized | 16/20 | 7/20 | 12/20 | 8/20 | 43/80 |
| **CLC (released)** | Clean | 20/20 | 18/20 | 18/20 | 15/20 | **71/80** |
| | Randomized | 19/20 | 14/20 | 13/20 | 9/20 | **55/80** |
| CLC stage, previous (PIC/freeze3) config on CLC seeds | Clean | 20/20 | 17/20 | 17/20 | 14/20 | 68/80 |
| | Randomized | 19/20 | 8/20 | 15/20 | 6/20 | 48/80 |

Notes
* The PIC stage left three tasks' configuration unchanged and retrained only `stack_blocks_two`; its Clean standard-success cells for `stack_blocks_two` were evaluated on non-identical seed sets (invalid scenes skipped differently), see the report. Its gain is in counterfactual-instruction goal completion (Clean 2 -> 13, Randomized 0 -> 6 of 20; analysis metric with privileged final block poses), not in standard success.
* `handover_block` Randomized, CLC stage: the CLC and previous runs share 19 of 20 seeds (one scene was skipped differently in each run). On the 19 common seeds: 8 vs 6 successes. The research report text mentions 18 common seeds; the per-episode records (`episodes_regsp.csv`, `episodes_previous_frozen_freeze3.csv`) give 19.
* Development evidence for the CLC configuration (not benchmark results): `results/final_v21_clc/CLC_CORRECTION_EFFECT_development.csv`, `docs/history/V21_CLC_FINAL_REPORT.md` section 3.
* The 4-task "full" and "baseline" cells were measured with an earlier executor and checkpoint set than the released one and are listed for context only.
* The seed ledger, the seed-leakage correction made during development, and the independence audits (`independence_final*.json`) are described in the research reports; the released final seeds (8200000+/8700000+) were never used for training, calibration, development or any earlier final.
