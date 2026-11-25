# Installation

## Verified configuration

| Item | Version |
|---|---|
| OS / GPU | Linux 7.0, NVIDIA RTX PRO 4500 Blackwell (32 GB), driver 615.71.09 |
| Python | 3.12.16 |
| PyTorch / torchvision | 2.7.1+cu128 / 0.22.1+cu128 |
| transformers | 5.22.2 |
| numpy / scipy | 1.26.4 / 1.17.1 |
| opencv / transforms3d / PyYAML | 4.13.0 / 0.4.2 / 6.0.3 |
| safetensors / huggingface-hub / einops / draccus | 0.8.0 / 1.33.0 / 0.8.2 / 0.11.6 |
| LeRobot | **upstream `lerobot==0.7.0`** (also imports with 0.6.1, but 0.6.1 cannot read the checkpoint's tokenizer artifact: use 0.7.0) |
| RoboTwin | commit `0aeea2d669c0f8516f4d5785f0aa33ba812c14b4` (MIT) |
| Simulator stack | sapien 3.0.0b1, mplib 0.2.1, toppra 0.6.11, pytorch3d 0.7.8, the cuRobo copy bundled in RoboTwin (`envs/curobo`) |

## How it was verified (honest account)

The verification environment is the machine's existing `robotwin` conda environment (which also contains the SafeDiff research fork of LeRobot as an editable install). To test against *upstream* LeRobot, upstream `lerobot==0.7.0` was unpacked with `pip install --no-deps --target <dir>` and put in front of `PYTHONPATH`; the editable fork was therefore shadowed (the loaded `lerobot.__file__` and `lerobot.__version__ == 0.7.0` were printed to confirm). A from-scratch virtual environment built only from `requirements.txt` was **not** created. The command sequence in the README is the intended clean install; the pins in `requirements.txt` are the versions that were actually present when the tests, the weight-equivalence check and the smoke episode ran.

## Notes

* `lerobot==0.7.0` requires `numpy>=2`, SAPIEN 3.0.0b1 requires `numpy<2`; ReGSP only needs LeRobot's policy/processor base classes, so install it with `--no-deps`.
* The backbone loads the VLM weights of `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` through the Hugging Face Hub at policy construction; they must be reachable or cached (`HF_HOME`).
* Never install the SafeDiff research fork of LeRobot in the same environment (both register the policy type `safediff_vla`).
* RoboTwin must be set up according to its own documentation (assets download, cuRobo build). ReGSP changes the working directory to `REGSP_ROBOTWIN_ROOT` when it builds an environment, as RoboTwin's own evaluation scripts require.
* Rendering: the default OIDN denoiser failed on the verification GPUs; `RGL_DENOISER=optix` (default in the provided scripts) selects OptiX.

## Environment variables

| Variable | Meaning |
|---|---|
| `REGSP_BACKBONE_CKPT` | `.../pretrained_model` of the frozen backbone |
| `REGSP_CKPT_ROOT` | weight bundle root |
| `REGSP_ROBOTWIN_ROOT` | RoboTwin checkout |
| `REGSP_CLEAN_ROOT`, `REGSP_WORK_ROOT`, `REGSP_CLEAN_DATA` | training/data-preparation directories (only for `scripts/data`, `scripts/train`) |
