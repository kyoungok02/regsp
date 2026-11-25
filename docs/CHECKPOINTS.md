# Checkpoints

**No weights are included in this repository and no public release location exists.** Do not rely on any download link; none is provided. If you hold the weights (or after the owner releases them), place them as below.

## Backbone

`REGSP_BACKBONE_CKPT` -> directory with `config.json` (type `safediff_vla`, architecture `temporal_decoder_grounded_grasp_v2`, `backbone_name = lerobot/smolvla_vlabench`), `model.safetensors`, `policy_preprocessor.json`, `policy_postprocessor.json`, normalizer `.safetensors` files, `tokenizer/`.
`model.safetensors` SHA-256: `b8cc22a667ea4abd8e0b1091722e981f3137bf840c16dee7823df40bbb1eb509` (472,169,389 parameters). Provenance: fresh-initialised grounded-grasp-v2 decoder, 20k steps, seed 1000, batch 4, on `lerobot/vlabench_unified` with the public frozen `lerobot/smolvla_vlabench` backbone (`scripts/train/train_backbone.py`). Loading this checkpoint with ReGSP on upstream `lerobot==0.7.0` gives a state-dict digest identical to loading it with the research fork.

## Per-task bundle (`REGSP_CKPT_ROOT/<task>/`)

File names and which tasks need them are defined in `regsp/bundle.py`; hashes and research-tree provenance in `configs/bundle_manifest.json`. Example (`handover_block`): `stageplan_head.pt`, `decoder.pt`, `bim_decoder.pt`, `proposal_{head,left,right}_camera.pt`, `anchor_head.pt`, `referent_scorer.pt`, `place_scorer.pt`, `progress.pt`.

| Task | Components not shared with the others |
|---|---|
| `click_bell`, `place_container_plate` | `referent_scorer.pt` |
| `stack_blocks_two` | `subgoal_scorer.pt` (replaces the plain scorer), `bim_decoder.pt`, `progress.pt` (instruction-conditioned) |
| `handover_block` | `referent_scorer.pt`, `place_scorer.pt`, `bim_decoder.pt`, `progress.pt`, proposal heads trained with the extra randomized views (`obs2`) |

Superseded files that the research-tree runner still loaded and then overwrote (original ten-task heads/scorer/anchor/decoders, the `scorer_cf_u1` cycle scorer of `stack_blocks_two`) are **not** part of the bundle; `tests/test_checkpoint_compat.py` shows that the resulting policies are weight-identical to the research-tree policies.

## Build and verify a bundle from the research layout

```bash
python scripts/utils/assemble_bundle.py --v21-root <outputs/robotwin_v21_4task_counterfactual> --orig-root <outputs/robotwin_safediff_original_10task> --out $REGSP_CKPT_ROOT [--copy]
python scripts/utils/assemble_bundle.py --verify $REGSP_CKPT_ROOT          # SHA-256 of every file
python scripts/utils/verify_bundle_load.py                                   # build the four policies, compare module digests
```

## Loading example

```python
import os
os.environ.update(REGSP_BACKBONE_CKPT="...", REGSP_CKPT_ROOT="...", REGSP_ROBOTWIN_ROOT="...")
from regsp.execution.factory import build_policy      # applies configs/final_v21_clc.json for the task
policy = build_policy("handover_block")               # -> ReGSPBimanualPolicy;  run(env, instruction, log)
```

Do not publish weights without the owner's explicit decision on the storage location (Git LFS was deliberately not set up).
