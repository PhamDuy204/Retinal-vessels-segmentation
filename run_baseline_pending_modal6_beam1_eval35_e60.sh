#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/duy/NCKH/Retinal-vessels-segmentation"
cd "$REPO_ROOT"

MODAL_BIN="/home/duy/.venvs/modal5/bin/modal"
BEAM_BIN="/home/duy/.venvs/beam/bin/beam"
MODAL_CONFIG="configs/baseline_modal6_pending_eval35_e60.json"
MODAL_PENDING="configs/baseline_modal6_pending_eval35_e60.pending_tasks.json"
BEAM_HANDLER="beam_train_baseline_pending_pod.py:pod"
MODAL_LOG="/tmp/baseline_modal6_pending_eval35_e60.modal.log"
BEAM_LOG="/tmp/baseline_beam1_pending_eval35_e60.beam.log"

read MODAL_TOKEN_ID MODAL_TOKEN_SECRET < <(python3 - <<'PYMODAL'
from pathlib import Path
import shlex
text = Path('/home/duy/Downloads/modal_6.txt').read_text(errors='ignore').strip()
parts = shlex.split(text)
tid = sec = ''
for i, p in enumerate(parts):
    if p == '--token-id' and i + 1 < len(parts):
        tid = parts[i + 1]
    if p == '--token-secret' and i + 1 < len(parts):
        sec = parts[i + 1]
print(tid, sec)
PYMODAL
)
BEAM_TOKEN="$(python3 - <<'PYBEAM'
from pathlib import Path
import re
text = Path('/home/duy/Downloads/beam_1.txt').read_text(errors='ignore')
match = re.search(r'KEYS?:\s*(\S+)', text)
print(match.group(1) if match else '')
PYBEAM
)"

if [[ -z "$MODAL_TOKEN_ID" || -z "$MODAL_TOKEN_SECRET" || -z "$BEAM_TOKEN" ]]; then
  echo "Missing Modal 6 or Beam token."
  exit 2
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  secret_json="$(mktemp /tmp/wandb-secret.XXXXXX.json)"
  python3 - "$secret_json" "$WANDB_API_KEY" <<'PYJSON'
import json, sys
path, key = sys.argv[1], sys.argv[2]
with open(path, 'w', encoding='utf-8') as f:
    json.dump({'WANDB_API_KEY': key}, f)
PYJSON
  chmod 600 "$secret_json"
  MODAL_TOKEN_ID="$MODAL_TOKEN_ID" MODAL_TOKEN_SECRET="$MODAL_TOKEN_SECRET" \
    "$MODAL_BIN" secret create --force wandb-secret --from-json "$secret_json" >/dev/null
  rm -f "$secret_json"
  if BEAM_TOKEN="$BEAM_TOKEN" "$BEAM_BIN" secret list | grep -q '^  WANDB_API_KEY'; then
    BEAM_TOKEN="$BEAM_TOKEN" "$BEAM_BIN" secret modify WANDB_API_KEY "$WANDB_API_KEY" >/dev/null
  else
    BEAM_TOKEN="$BEAM_TOKEN" "$BEAM_BIN" secret create WANDB_API_KEY "$WANDB_API_KEY" >/dev/null
  fi
fi

if ! MODAL_TOKEN_ID="$MODAL_TOKEN_ID" MODAL_TOKEN_SECRET="$MODAL_TOKEN_SECRET" "$MODAL_BIN" secret list | grep -q 'wandb-secret'; then
  echo "Modal 6 is missing wandb-secret. Re-run with WANDB_API_KEY set."
  exit 3
fi
if ! BEAM_TOKEN="$BEAM_TOKEN" "$BEAM_BIN" secret list | grep -q 'WANDB_API_KEY'; then
  echo "Beam is missing WANDB_API_KEY secret. Re-run with WANDB_API_KEY set."
  exit 3
fi

MODAL_TOKEN_ID="$MODAL_TOKEN_ID" MODAL_TOKEN_SECRET="$MODAL_TOKEN_SECRET" "$MODAL_BIN" volume create retinal-vessels-statistics >/dev/null 2>&1 || true
if ! MODAL_TOKEN_ID="$MODAL_TOKEN_ID" MODAL_TOKEN_SECRET="$MODAL_TOKEN_SECRET" "$MODAL_BIN" volume ls retinal-vessels-statistics /data >/dev/null 2>&1; then
  MODAL_TOKEN_ID="$MODAL_TOKEN_ID" MODAL_TOKEN_SECRET="$MODAL_TOKEN_SECRET" "$MODAL_BIN" volume put --force retinal-vessels-statistics "$REPO_ROOT/data" /
fi

echo "Modal dry run..."
MODAL_TOKEN_ID="$MODAL_TOKEN_ID" MODAL_TOKEN_SECRET="$MODAL_TOKEN_SECRET" \
MODAL_GPU_TYPE="L4:2" MODAL_MAX_CONTAINERS="1" MODAL_WANDB_SECRET="wandb-secret" \
  "$MODAL_BIN" run modal_statistics_wandb.py --config "$MODAL_CONFIG" --pending-tasks "$MODAL_PENDING" --dry-run 2>&1 | tee "$MODAL_LOG"

echo "Modal real run..."
MODAL_TOKEN_ID="$MODAL_TOKEN_ID" MODAL_TOKEN_SECRET="$MODAL_TOKEN_SECRET" \
MODAL_GPU_TYPE="L4:2" MODAL_MAX_CONTAINERS="1" MODAL_WANDB_SECRET="wandb-secret" \
  "$MODAL_BIN" run --timestamps modal_statistics_wandb.py --config "$MODAL_CONFIG" --pending-tasks "$MODAL_PENDING" --wait 2>&1 | tee -a "$MODAL_LOG" &
modal_pid=$!

echo "Beam real run..."
BEAM_TOKEN="$BEAM_TOKEN" "$BEAM_BIN" run "$BEAM_HANDLER" 2>&1 | tee "$BEAM_LOG" &
beam_pid=$!

wait "$modal_pid"
wait "$beam_pid"
