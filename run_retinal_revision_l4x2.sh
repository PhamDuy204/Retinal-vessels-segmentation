#!/usr/bin/env bash
set -euo pipefail

cd /home/duy/NCKH/Retinal-vessels-segmentation
source /home/duy/.venvs/modal/bin/activate
source /tmp/retinal_revision_a10_env.sh

GPU_TYPE="${MODAL_GPU_TYPE:-L4:2}"
MAX_CONTAINERS="${MODAL_MAX_CONTAINERS:-1}"

if [[ "$GPU_TYPE" != *":"* ]]; then
  echo "WARNING: MODAL_GPU_TYPE=$GPU_TYPE does not request multiple GPUs. Expected L4:2."
fi

python3 -m json.tool "$CONFIG" >/dev/null
python3 -m json.tool "$PENDING" >/dev/null

echo "Experiment: $EXP_ID"
echo "GPU type: $GPU_TYPE"
echo "Max containers: $MAX_CONTAINERS"
echo "Config: $CONFIG"
echo "Pending tasks: $PENDING"
echo "Log: $LOG"
echo

echo "Dry run..."
MODAL_GPU_TYPE="$GPU_TYPE" MODAL_MAX_CONTAINERS="$MAX_CONTAINERS" \
  modal run modal_statistics.py \
    --config "$CONFIG" \
    --pending-tasks "$PENDING" \
    --dry-run 2>&1 | tee "$LOG"

echo
echo "Real run..."
MODAL_GPU_TYPE="$GPU_TYPE" MODAL_MAX_CONTAINERS="$MAX_CONTAINERS" \
  modal run --timestamps modal_statistics.py \
    --config "$CONFIG" \
    --pending-tasks "$PENDING" \
    --wait 2>&1 | tee -a "$LOG"

echo
echo "GPU/task check:"
grep -E "Dataset tasks to run|GPU 0|GPU 1|View run|FAILED|Completed" "$LOG" || true
