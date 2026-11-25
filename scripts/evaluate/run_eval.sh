#!/bin/bash
# ReGSP frozen native RoboTwin evaluation (V21-CLC, configs/final_v21_clc.json).
# usage: scripts/evaluate/run_eval.sh GPU OUT_ROOT TASK:CFG [TASK:CFG ...]       CFG = clean | rand
#        N_EP (default 20), CLEAN_SEED (default 8200000), RAND_SEED (default 8700000) may be overridden through the environment.
# Output: OUT_ROOT/<clean|rand>/<task>/{seed*.json, summary.json, run.log, DONE}.  A cell is restarted (max 20 process attempts, linear back-off) after
# simulator crashes; finished seeds are never re-run (seed*.json), skipped seeds (expert-invalid scenes) are recorded as skip*.json.
# Required: REGSP_BACKBONE_CKPT REGSP_CKPT_ROOT REGSP_ROBOTWIN_ROOT (docs/INSTALL.md); python must have regsp, upstream lerobot==0.7.0 and RoboTwin's dependencies.
set -u
G=$1; OUT=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python}
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 RGL_DENOISER=${RGL_DENOISER:-optix}   # OptiX denoiser: the default OIDN denoiser fails on the verification GPUs
N_EP=${N_EP:-20}; CLEAN_SEED=${CLEAN_SEED:-8200000}; RAND_SEED=${RAND_SEED:-8700000}
for TC in "$@"; do
  T=${TC%%:*}; C=${TC##*:}
  if [ "$C" = clean ]; then CFG=demo_clean; S=$CLEAN_SEED; else CFG=demo_randomized; S=$RAND_SEED; fi
  D=$OUT/$C/$T; mkdir -p "$D"
  for k in $(seq 1 20); do
    [ -f "$D/DONE" ] && break
    echo "$(date -u +%FT%TZ) process attempt $k gpu=$G" >> "$D/attempts.log"
    CUDA_VISIBLE_DEVICES=$G $PY "$HERE/robotwin_eval.py" --task "$T" --config "$CFG" --n "$N_EP" --st_seed "$S" --out "$D" >> "$D/run.log" 2>&1
    echo "$(date -u +%FT%TZ) process exit rc=$?" >> "$D/attempts.log"
    [ -f "$D/DONE" ] && break
    sleep $(( 20*k > 120 ? 120 : 20*k ))
  done
  [ -f "$D/DONE" ] || { echo "$(date -u +%FT%TZ) GAVE_UP after 20 restarts" >> "$D/attempts.log"; touch "$D/GAVE_UP"; }
done
