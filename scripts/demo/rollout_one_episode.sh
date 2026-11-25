#!/bin/bash
# One native episode of the frozen ReGSP policy (requires the weight bundle, backbone checkpoint, RoboTwin; docs/INSTALL.md).
# usage: scripts/demo/rollout_one_episode.sh GPU OUT_DIR TASK [clean|rand]
# Note: the instruction text is re-sampled in every run (it is not fixed by the evaluation seed), so single-seed outcomes can differ between runs.
set -eu
G=$1; OUT=$2; T=$3; C=${4:-clean}; HERE=$(cd "$(dirname "$0")" && pwd)
if [ "$C" = clean ]; then CFG=demo_clean; S=8200000; else CFG=demo_randomized; S=8700000; fi
RGL_DENOISER=${RGL_DENOISER:-optix} CUDA_VISIBLE_DEVICES=$G ${PYTHON:-python} "$HERE/../evaluate/robotwin_eval.py" --task "$T" --config "$CFG" --n 1 --st_seed $S --out "$OUT"
