from __future__ import annotations

import os

print("PY_START", bool(os.environ.get("WANDB_API_KEY")), flush=True)
import torch
print("TORCH_OK", torch.__version__, torch.cuda.is_available(), flush=True)
import wandb
print("WANDB_IMPORT_OK", flush=True)
import eval
print("EVAL_IMPORT_OK", flush=True)
import train
print("TRAIN_IMPORT_OK", flush=True)
from data_preparation import get_all_training_set
print("DATA_PREP_IMPORT_OK", flush=True)
datasets = get_all_training_set("data", 4, 500, 64, "patch", "window", "edae_net", 0, False, False, None, 890, 2)
print("DATASETS_BUILT", len(datasets), [d["name"] for d in datasets[:3]], flush=True)
