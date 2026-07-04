#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/duy/NCKH/Retinal-vessels-segmentation"
cd "$REPO_ROOT"
source /home/duy/.venvs/modal/bin/activate

PROFILE="${MODAL_THIRD_PROFILE:-modal3}"
GPU_TYPE="${MODAL_GPU_TYPE:-L4:2}"
MAX_CONTAINERS="${MODAL_MAX_CONTAINERS:-1}"
VOLUME_NAME="${MODAL_VOLUME_NAME:-retinal-vessels-statistics}"
WANDB_SECRET_NAME="${MODAL_WANDB_SECRET:-wandb-secret}"
EXP_ID="Retinal_training_revision"
CONFIG="/tmp/${EXP_ID}_gtdla_frnet_modal3.json"
PENDING="/tmp/${EXP_ID}_gtdla_frnet_modal3.pending_tasks.json"
LOG="/tmp/${EXP_ID}_gtdla_frnet_modal3.modal.log"

cat <<INFO
This script runs Modal account/profile: $PROFILE
GPU request: $GPU_TYPE
Max containers: $MAX_CONTAINERS
Experiment ID: $EXP_ID

Modal CLI needs token_id + token_secret, not a single API key.
If profile $PROFILE already exists, press Ctrl+C now if you do not want to overwrite it.
INFO

read -r -p "Press Enter to set/verify Modal token for profile '$PROFILE'..."
modal token set --profile "$PROFILE" --no-activate --verify

echo "Active profile remains: $(modal profile current 2>/dev/null || true)"
echo "Testing third profile..."
MODAL_PROFILE="$PROFILE" modal token info >/dev/null

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  read -r -s -p "Enter WANDB_API_KEY for Modal account '$PROFILE': " WANDB_API_KEY
  echo
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is empty; cannot create Modal secret." >&2
  exit 1
fi

SECRET_JSON="$(mktemp /tmp/wandb-secret.XXXXXX.json)"
cleanup() {
  rm -f "$SECRET_JSON"
}
trap cleanup EXIT
python3 - "$SECRET_JSON" "$WANDB_API_KEY" <<PYJSON
import json, sys
path, api_key = sys.argv[1], sys.argv[2]
with open(path, "w", encoding="utf-8") as f:
    json.dump({"WANDB_API_KEY": api_key}, f)
PYJSON
chmod 600 "$SECRET_JSON"
MODAL_PROFILE="$PROFILE" modal secret create --force "$WANDB_SECRET_NAME" --from-json "$SECRET_JSON"

MODAL_PROFILE="$PROFILE" modal volume create "$VOLUME_NAME" >/dev/null 2>&1 || true
if ! MODAL_PROFILE="$PROFILE" modal volume ls "$VOLUME_NAME" /data >/dev/null 2>&1; then
  echo "Volume '$VOLUME_NAME' in profile '$PROFILE' has no /data yet. Uploading local data directory..."
  MODAL_PROFILE="$PROFILE" modal volume put --force "$VOLUME_NAME" "$REPO_ROOT/data" /
else
  echo "Volume '$VOLUME_NAME' already has /data."
fi

python3 - "$CONFIG" "$PENDING" "$EXP_ID" <<PYCONFIG
import json, sys
config_path, pending_path, exp_id = sys.argv[1:4]
models = ["gtdla", "fr_net"]
seeds = [42, 123, 456, 789, 890]
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
    "models": models,
    "seeds": seeds,
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
        "tf32": True,
        "fast_nondeterministic": True,
        "eval_batch_size": 64,
        "eval_every": 1,
    },
    "wandb_entity": "phamdinhanhduy-university-of-information-and-technology",
    "wandb_project": "Retinal-Vessels-Segmentation",
}
dataset_group = ",".join(datasets)
tasks = [
    {
        "experiment_id": exp_id,
        "model": model,
        "seed": seed,
        "dataset": dataset_group,
    }
    for model in models
    for seed in seeds
]
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
with open(pending_path, "w", encoding="utf-8") as f:
    json.dump({"tasks": tasks}, f, indent=2)
PYCONFIG

python3 -m json.tool "$CONFIG" >/dev/null
python3 -m json.tool "$PENDING" >/dev/null

echo
cat <<INFO
Config: $CONFIG
Pending: $PENDING
Log: $LOG
Each Modal task = one model + one seed + all 7 datasets.
Inside each L4:2 container, train.py should show Dataset tasks to run: 7 and use GPU 0 + GPU 1.
INFO

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
echo "GPU/task check:"
grep -E "Dataset tasks to run|GPU 0|GPU 1|View run|FAILED|Completed" "$LOG" || true
