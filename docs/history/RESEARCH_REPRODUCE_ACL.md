# REPRODUCE (Machine D)

Environment: `python`, RoboTwin at `<RoboTwin>`, baseline worktree W=`outputs/robotwin_safediff_original_10task` (read-only), experiment dir N=`outputs/robotwin_v21_4task_counterfactual` (branch `robotwin/safediff-v21-4task-counterfactual`). Durable copy of the git-ignored artifacts: `outputs/backup_v21_4task_acl_20261116` (N/, W/ essentials, git bundle). One SAPIEN worker per GPU; never run SAPIEN and training on the same GPU.

```bash
cd $N
# 1 additional randomized view units and proposal-head refresh (handover_block, stack_blocks_two)
scripts/run_gen_da2.sh GPU TASK 120
CUDA_VISIBLE_DEVICES=G DA_EXTRA_ONLY=da_views2 python scripts/build_da_cache.py TASK
CUDA_VISIBLE_DEVICES=G DA_EXTRA=da_views2 DATAG=_obs2 python scripts/train_da_heads.py TASK
# 2 adaptive decoder + progress head (8000 steps)
CUDA_VISIBLE_DEVICES=G python scripts/train_acl.py TASK a1 8000
# 3 on-policy progress samples on training-domain seeds (real perturbed outcomes), labels, progress-head retraining (decoder frozen)
ACL_PERTURB=0.3,0.035,0.3 LABEL=op_r1 N_EP=40 CLEAN_SEED=5300000 RAND_SEED=5350000 ACL_SAVE_FEATS=$N/onpolicy/raw/TASK scripts/run_acl.sh G TASK:clean TASK:rand
ACL_PERTURB=0.3,0.035,0.5 LABEL=op_r2 N_EP=40 CLEAN_SEED=5300200 RAND_SEED=5350200 ACL_SAVE_FEATS=$N/onpolicy/raw2/TASK scripts/run_acl.sh G TASK:clean TASK:rand
python scripts/acl_onpolicy.py TASK onpolicy/raw/TASK onpolicy/raw2/TASK
CUDA_VISIBLE_DEVICES=G ACL_PROG_ONLY=1 ACL_RESUME=$N/ckpt_acl/TASK/acl_a1.pt ACL_ONPOLICY=$N/onpolicy/labelled/TASK ACL_OP_W=1.0 python scripts/train_acl.py TASK a2 3000
# 4 development (seeds 7200000+/7500000+ and 7200100+/7500100+)
LABEL=dev_c1 N_EP=12 scripts/run_acl2.sh G TASK:clean TASK:rand
LABEL=dev_c1b N_EP=20 CLEAN_SEED=7200100 RAND_SEED=7500100 scripts/run_acl2.sh G TASK:clean TASK:rand
# 5 freeze, final (seeds 8000000+/8500000+, n=20), audit, reports
python scripts/freeze2.py 1
scripts/run_final2.sh G TASK:clean TASK:rand
python scripts/audit_independence2.py
python scripts/make_reports.py
sha256sum -c freeze2/MANIFEST_FREEZE2.sha256 && sha256sum -c MANIFEST_FINAL.sha256
```

Note: simulator rendering/physics is not bit-reproducible on this machine (the same seed can give outcomes differing by about +-2 successes at n=12); reruns reproduce the rates within sampling noise, not the per-seed outcomes. `run_gen_da2.sh`'s episode count and the exact historical commands of steps 1-2 are recorded in `logs/` and `queue.jsonl`.
