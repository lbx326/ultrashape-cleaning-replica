#!/usr/bin/env bash
# Quick demo on two HSSD meshes.
#
# Required environment:
#   HSSD_DIR          Path to a directory containing HSSD ``*.glb`` files.
#                     Must contain both the sofa and clock fixtures below
#                     (override by editing ``A`` / ``B``).
#
# Optional:
#   VLM_PYTHON_EXE    If set, Stage 2 runs in that interpreter.
#   QWEN3VL_MODEL_PATH  Path or HF id for Qwen3-VL. Defaults to the HF hub.
#   ULTRASHAPE_VAE_CONFIG / ULTRASHAPE_VAE_CKPT   Required for Stage 4 VAE.
#   CUDA_VISIBLE_DEVICES  Passed through (defaults to 0).
#
# See ``.env.example`` for the full set of variables.
set -euo pipefail

HSSD_DIR="${HSSD_DIR:?HSSD_DIR must point at a directory containing the demo GLBs}"
PYTHON="${PYTHON:-python}"
OUT="${OUT:-./demo_out}"
: "${CUDA_VISIBLE_DEVICES:=0}"; export CUDA_VISIBLE_DEVICES
mkdir -p "$OUT"

# Pick two varied meshes.
A=${DEMO_SOFA:-00366b86401aa16b702c21de49fd59b75ab9c57b.glb}   # sofa
B=${DEMO_CLOCK:-00258bed0c6e87a14e33c3eebffc48c898135698.glb}  # clock

for f in "$A" "$B"; do
  NAME="${f%.glb}"
  echo "===> $NAME"
  "$PYTHON" -m ultrashape_cleaning.clean_mesh \
      --input  "$HSSD_DIR/$f" \
      --output "$OUT/$NAME.clean.ply" \
      --save-report "$OUT/$NAME.report.json" \
      --render-dir "$OUT/renders" \
      --vlm-cache-dir "$OUT/.vlm_cache" \
      --resolution 512 --canonicalize geom --full
done

echo
echo "Done. Outputs in $OUT/"
ls -la "$OUT/"
