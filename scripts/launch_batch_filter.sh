#!/bin/bash
# Launch 8 independent python workers, one per GPU, for batch_filter_datasets.
# Each worker is fully independent (no mp.Process); GPU pinning is via
# shell-level CUDA_VISIBLE_DEVICES so torch sees only one device.
#
# Usage:
#   bash scripts/launch_batch_filter.sh [num_workers] [batch_size] [extra args...]
#
# Examples:
#   bash scripts/launch_batch_filter.sh                   # 8 workers, batch=4
#   bash scripts/launch_batch_filter.sh 8 4 --no-vae      # skip VAE
#   bash scripts/launch_batch_filter.sh 8 4 --limit-per-dataset 5
#
# Logs land in $LOGDIR/batch_filter_w{N}.log
set -euo pipefail

cd "$(dirname "$0")/.."
# Space-separated GPU IDs to use. Override with e.g. GPU_IDS="1 2 3 4 5 6 7"
# to skip GPU 0. Default 0..7.
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
GPU_ARR=($GPU_IDS)
DEFAULT_NW=${#GPU_ARR[@]}

NUM_WORKERS="${1:-$DEFAULT_NW}"
BATCH_SIZE="${2:-4}"
shift 2 2>/dev/null || true
EXTRA="$*"

PY=/moganshan/afs_a/lbx/env/ultrashape/bin/python
LOGDIR="${LOGDIR:-/moganshan/afs_a/lbx/workspace/GeoRoute/logs}"
mkdir -p "$LOGDIR"

PIDS=()
for ((W=0; W<NUM_WORKERS; W++)); do
    GPU=${GPU_ARR[$((W % ${#GPU_ARR[@]}))]}
    LOG="$LOGDIR/batch_filter_w${W}.log"
    echo "[launcher] spawning worker $W on GPU $GPU -> $LOG"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
        "$PY" -u scripts/batch_filter_datasets.py \
            --worker-id $W --num-workers $NUM_WORKERS \
            --batch-size $BATCH_SIZE \
            $EXTRA \
            > "$LOG" 2>&1 &
    PIDS+=($!)
    # Stagger daemon spawns by 30s — concurrent VLM-daemon startup hit a
    # race that left some daemons silently on CPU. By the time the second
    # worker starts loading Qwen, the first is well past CUDA init.
    sleep 30
done

echo "[launcher] launched ${#PIDS[@]} workers, pids: ${PIDS[*]}"
echo "[launcher] tailing logs in $LOGDIR/batch_filter_w*.log"
echo "[launcher] waiting for completion..."
RC=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[launcher] worker pid $pid exited non-zero"
        RC=1
    fi
done

echo "[launcher] all workers exited, running aggregator"
"$PY" scripts/batch_filter_datasets.py --aggregate-only $EXTRA \
    | tee "$LOGDIR/batch_filter_aggregate.log"

exit $RC
