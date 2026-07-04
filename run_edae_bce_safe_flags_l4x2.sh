#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/duy/NCKH/Retinal-vessels-segmentation"
cd "$REPO_ROOT"
source /home/duy/.venvs/modal/bin/activate

PROFILE="${MODAL_PROFILE:-phamdinhanhduy}"
GPU_TYPE="${MODAL_GPU_TYPE:-L4:2}"
MAX_CONTAINERS="${MODAL_MAX_CONTAINERS:-1}"
EXP_ID="Retinal_training_revision_edae_bce_safe_flags"
CONFIG="/tmp/${EXP_ID}.json"
PENDING="/tmp/${EXP_ID}.pending_tasks.json"
LOG="/tmp/${EXP_ID}.modal.log"

python3 - "$CONFIG" "$PENDING" "$EXP_ID" <<PYCONFIG
import json, sys
config_path, pending_path, exp_id = sys.argv[1:4]
datasets = [
    "CHASEDB_1_patches",
    "DRIVE_patches",
    "STARE_F1_patches",
    "STARE_F2_patches",
    "STARE_F3_patches",
    "STARE_F4_patches",
    "STARE_F5_patches",
]
config = {
    "experiment_id": exp_id,
    "models": ["edae_net"],
    "seeds": [42],
    "datasets": datasets,
    "common_train_args": {
        "batch_size": 4,
        "epochs": 100,
        "loss": "bce_loss",
        "learning_rate": 0.001,
        "patches": 500,
        "patch_size": 64,
        "train_type": "patch",
        "type_split": "window",
        "output_root": "outputs",
        "wandb_mode": "online",
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True,
        "compile_model": False,
        "eval_amp": False,
        "tf32": False,
        "fast_nondeterministic": False,
        "eval_batch_size": 64,
        "eval_every": 1,
    },
    "wandb_entity": "phamdinhanhduy-university-of-information-and-technology",
    "wandb_project": "Retinal-Vessels-Segmentation",
}
tasks = [
    {
        "experiment_id": exp_id,
        "model": "edae_net",
        "seed": 42,
        "dataset": ",".join(datasets),
    }
]
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
with open(pending_path, "w", encoding="utf-8") as f:
    json.dump({"tasks": tasks}, f, indent=2)
PYCONFIG

python3 -m json.tool "$CONFIG" >/dev/null
python3 -m json.tool "$PENDING" >/dev/null

echo "Profile: $PROFILE"
echo "Experiment: $EXP_ID"
echo "GPU type: $GPU_TYPE"
echo "Max containers: $MAX_CONTAINERS"
echo "Config: $CONFIG"
echo "Pending: $PENDING"
echo "Log: $LOG"
echo "Flags: compile_model=false, eval_amp=false, tf32=false, fast_nondeterministic=false, loss=bce_loss"
echo

echo "Dry run..."
MODAL_PROFILE="$PROFILE" MODAL_GPU_TYPE="$GPU_TYPE" MODAL_MAX_CONTAINERS="$MAX_CONTAINERS" \
  modal run modal_statistics.py \
    --config "$CONFIG" \
    --pending-tasks "$PENDING" \
    --dry-run 2>&1 | tee "$LOG"

echo
echo "Real run..."
MODAL_PROFILE="$PROFILE" MODAL_GPU_TYPE="$GPU_TYPE" MODAL_MAX_CONTAINERS="$MAX_CONTAINERS" \
  modal run --timestamps modal_statistics.py \
    --config "$CONFIG" \
    --pending-tasks "$PENDING" \
    --wait 2>&1 | tee -a "$LOG"

echo
echo "Quick collapse check:"
grep -E "Epoch|Recall: 0\.0000|Specificity: 1\.0000|Dataset tasks to run|GPU 0|GPU 1|FAILED|Completed" "$LOG" | tail -200 || true
