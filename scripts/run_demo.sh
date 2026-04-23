#!/usr/bin/env bash
# Quick demo on two HSSD meshes. Requires the reference cluster paths to be
# available; edit if you're elsewhere.
set -euo pipefail

ENV_UL=/moganshan/afs_a/lbx/env/ultrashape/bin/python
HSSD_DIR=/moganshan/afs_a/rsync/chuan/TRELLIS/datasets/hssd/raw
OUT=./demo_out
mkdir -p "$OUT"

# Pick two varied meshes.
A=00366b86401aa16b702c21de49fd59b75ab9c57b.glb   # sofa
B=00258bed0c6e87a14e33c3eebffc48c898135698.glb   # clock

for f in "$A" "$B"; do
  NAME="${f%.glb}"
  echo "===> $NAME"
  CUDA_VISIBLE_DEVICES=0 "$ENV_UL" -m ultrashape_cleaning.clean_mesh \
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
