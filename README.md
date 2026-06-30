# Retinal vessel segmentation

This repository trains retinal vessel segmentation models and includes a reproducible statistical runner for 9 models × 7 dataset experiments × 5 seeds (315 dataset tasks).

The statistical runner preserves the existing scientific protocol: 100 epochs, evaluation after every epoch on the existing test directory, no early stopping, Adam without weight decay, threshold `0.487`, the existing patch extraction and 750 sampled patches per image, and `abe_dice_loss.py` unchanged. The selected epoch is the single row with the lexicographically greatest `(AUC, Recall, F1, cDice)` tuple. Here `cDice` is the hard centerline/clDice score computed from the prediction and target skeletons at the fixed threshold.

## Installation and W&B login

Install the project requirements and log in once. `train.py` does not accept an API key argument; W&B uses `WANDB_API_KEY` or the saved login.

```bash
pip install -r requirements.txt
wandb login
```

Artifacts are disabled by default for the sweep. Add `--log-artifacts` to a direct `train.py` invocation or set `log_artifacts` in the statistics config only when model uploads are wanted.

## Local statistical runs

Review [configs/statistics.json](configs/statistics.json) first. It is the single source for the experiment ID, nine models, five seeds, seven dataset experiment names, shared training arguments, and W&B destination.

Print the full plan without training:

```bash
python3 run_statistics.py --dry-run
```

Run on two GPUs. Each `train.py` process distributes its seven datasets over both visible GPUs, then the orchestrator starts the next model/seed pair:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 run_statistics.py
```

Resume after an interruption; completed dataset tasks are skipped while missing, running, and failed tasks are rerun:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 run_statistics.py --resume
```

Useful selection controls are `--models`, `--seeds`, and `--max-runs`. The latter limits model/seed subprocesses, not individual dataset tasks.

```bash
python3 run_statistics.py --dry-run --models unet custom_unet --seeds 42 --max-runs 2
```

For the online W&B smoke matrix of `unet` and `our_net`, two datasets, two
seeds, and two epochs, use the prepared config:

```bash
# One GPU: all eight dataset tasks run sequentially.
CUDA_VISIBLE_DEVICES=0 python3 run_statistics.py \
  --config configs/wandb_smoke_2models_2seeds.json

# Two GPUs: the two datasets in each model/seed invocation run concurrently.
CUDA_VISIBLE_DEVICES=0,1 python3 run_statistics.py \
  --config configs/wandb_smoke_2models_2seeds.json
```

Inspect the commands first with `--dry-run`, or append `--resume` after an
interruption. Edit the config's `models`, `seeds`, or `datasets` arrays to grow
the matrix. Per-task logs are under the task directory shown below; the full
runner log is under `outputs/<experiment_id>/orchestration/`, and online runs
appear in the configured W&B project.

The arrays in the JSON config are the experiment matrix. For example, add
another seed or dataset using the exact registered dataset name:

```json
"seeds": [42, 123, 456],
"datasets": ["DRIVE_patches", "STARE_F1_patches", "CHASEDB_1_patches"]
```

Change `experiment_id` before starting a genuinely new experiment so its local
results and W&B groups remain separate. Always run `--dry-run` after editing the
config, then use `--resume` to skip only tasks whose `status.json` says
`completed`:

```bash
CUDA_VISIBLE_DEVICES=0 python3 run_statistics.py \
  --config configs/wandb_smoke_2models_2seeds.json --dry-run
CUDA_VISIBLE_DEVICES=0 python3 run_statistics.py \
  --config configs/wandb_smoke_2models_2seeds.json --resume
```

Useful live/local checks are:

```bash
# Complete orchestration output (choose the newest timestamped file).
ls -lt outputs/<experiment_id>/orchestration/
tail -f outputs/<experiment_id>/orchestration/run_statistics_<timestamp>.log

# One exact task's console, epoch metrics, selected epoch, and status.
tail -f outputs/<experiment_id>/model=our_net/dataset=DRIVE_patches/seed=42/stdout.log
cat outputs/<experiment_id>/model=our_net/dataset=DRIVE_patches/seed=42/epoch_metrics.csv
cat outputs/<experiment_id>/model=our_net/dataset=DRIVE_patches/seed=42/selected_result.json
cat outputs/<experiment_id>/model=our_net/dataset=DRIVE_patches/seed=42/status.json
```

`num_workers` is per GPU training process: `2` therefore means two loader
workers on one GPU and four total when two GPUs are concurrently active.
`pin_memory` and `persistent_workers` are useful with workers enabled.
`eval_batch_size` is the number of inference patches per forward pass; reduce
it from `64` to `32` or `16` if evaluation runs out of VRAM. Keep `batch_size=4`
fixed when comparing experiments: in this repository it also changes how
source-image patch groups are flattened/chunked and can therefore change the
number and contents of optimizer steps. TF32, evaluation AMP, and the fast
nondeterministic mode improve speed but are not bitwise reproducible; apply the
same flags to every model in a comparison.

Every online W&B run logs the per-epoch curves and, when training finishes,
adds `best_epoch`, every `best_<metric>`, the four-value `best_selection_key`,
and an `epoch_selection_table`. The table contains every evaluated epoch and an
`is_best_epoch` column, making the selected row easy to inspect or export for a
manual calculation. The equivalent local source is `epoch_metrics.csv`; the
chosen row is duplicated in `selected_result.json`.

Each dataset task writes to:

```text
outputs/<experiment_id>/model=<model>/dataset=<dataset>/seed=<seed>/
├── epoch_metrics.csv
├── selected_result.json
├── best.pt
├── status.json
└── stdout.log
```

Orchestration logs and `pending_tasks.json` are written below `outputs/<experiment_id>/orchestration/`.

## Summarization

Summaries are built only from `selected_result.json` files whose
`experiment_id` exactly matches the selected config. This prevents smoke runs
or older experiments under a broad `outputs/` root from being mixed into the
result. Multiple roots can merge local, Modal, and Vast results:

```bash
python3 summarize_statistics.py \
  --runs-dir outputs modal-results/outputs vast-results/outputs \
  --output-dir statistics_summary
```

The command checks the configured five unique seeds per exact model/dataset pair, reports both missing and duplicate seeds, and calculates sample standard deviations with `ddof=1`. It produces `runs.csv`, `summary.csv`, `paper_table.md`, `paper_table.tex`, `missing_runs.csv`, and `failed_runs.csv`.

## Modal execution

Install and authenticate the current Modal CLI, create one persistent volume, and upload the local dataset into its `/data` directory:

```bash
pip install modal
modal setup
modal volume create retinal-vessels-statistics
modal volume put retinal-vessels-statistics ./data /data
```

Create the W&B secret without putting the key in command history if your shell supports interactive input:

```bash
read -s WANDB_API_KEY
modal secret create wandb-secret WANDB_API_KEY="$WANDB_API_KEY"
unset WANDB_API_KEY
```

`modal_statistics.py` packages the code with `Image.add_local_dir`, excluding datasets, outputs, checkpoints, Git metadata, and virtual environments. Every Modal function handles exactly one model × dataset × seed task on an L4 by default. Set `MODAL_GPU_TYPE=A10` to use an A10, and `MODAL_MAX_CONTAINERS` to control concurrency.

Submit only tasks still pending locally. `spawn_map` submits the task batch, while Modal's `--detach` keeps it running after the local command exits:

```bash
modal run --detach modal_statistics.py \
  --pending-tasks outputs/retinal-vessels-statistics-v1/orchestration/pending_tasks.json
```

Modal execution can be controlled without editing Python. The example below
uses the two-epoch config, gives the cloud run a separate experiment ID, limits
the autoscaler to four L4 containers, and requests four CPU cores/16 GB RAM per
GPU task:

```bash
MODAL_GPU_TYPE=L4 \
MODAL_MAX_CONTAINERS=4 \
MODAL_CPU_CORES=4 \
MODAL_MEMORY_MB=16384 \
modal run modal_statistics.py \
  --config configs/wandb_smoke_2models_2seeds.json \
  --experiment-id wandb-unet-ournet-2seed-2epoch-modal \
  --wait
```

Use `--dry-run` to inspect the exact task matrix, `--max-tasks 1 --wait` for a
blocking one-task smoke test, or comma-separated filters such as
`--models our_net --seeds 42,123 --datasets DRIVE_patches`. Runtime controls
are `MODAL_GPU_TYPE` (`T4`, `L4`, `A10`, `L40S`, `A100`, ...),
`MODAL_MAX_CONTAINERS`, `MODAL_CPU_CORES`, `MODAL_MEMORY_MB`,
`MODAL_TASK_TIMEOUT`, and `MODAL_RETRIES`. Volume and secret names can be
overridden through `MODAL_VOLUME_NAME` and `MODAL_WANDB_SECRET`.

When summarizing an experiment ID overridden at Modal submission time, pass the
same ID explicitly so other runs under a broad root are ignored:

```bash
python3 summarize_statistics.py \
  --runs-dir modal-results \
  --config configs/wandb_smoke_2models_2seeds.json \
  --experiment-id wandb-unet-ournet-2seed-2epoch-modal \
  --output-dir modal-results/summary
```

To test the cloud setup cheaply, append `--max-tasks 1`. Each task commits the persistent volume after training. Download all cloud outputs and merge them with local results:

```bash
modal volume get retinal-vessels-statistics /outputs ./modal-results/outputs
python3 summarize_statistics.py \
  --runs-dir outputs ./modal-results/outputs \
  --output-dir statistics_summary
```

## Optimized `our_net` training

`--model our_net` is verified at startup against `models/our_net/our_net.py` and its expected approximately 0.96M trainable parameters. The original `loss/abe_dice_loss.py` is unchanged. Select the exact-safe copy explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --model our_net \
  --datasets DRIVE_patches \
  --seed 42 \
  --epochs 1 \
  --loss abe_dice_loss_optimized \
  --num-workers 2 \
  --pin-memory \
  --persistent-workers \
  --experiment-id ournet-smoke-optimized \
  --wandb-mode disabled
```

The optimized loss skips only the 5×5 and 7×7 `MultiScopeLoss` branches whose weights are exactly zero. Use `--loss abe_dice_loss` at any time to run the untouched original. W&B model watching and artifact upload are off unless `--wandb-watch` or `--log-artifacts` is supplied.

Profile the first ten optimizer steps plus evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --model our_net \
  --datasets DRIVE_patches \
  --seed 42 \
  --epochs 1 \
  --loss abe_dice_loss_optimized \
  --num-workers 2 --pin-memory --persistent-workers \
  --profile --profile-steps 10 \
  --experiment-id ournet-profile \
  --wandb-mode disabled
```

The task directory receives `training_profile_summary.json`, `training_profile_table.txt`, and `evaluation_profile_summary.json`. Run the reproducible benchmark with:

```bash
CUDA_VISIBLE_DEVICES=0 python3 benchmark_training.py \
  --model our_net \
  --dataset DRIVE_patches \
  --warmup 3 \
  --steps 10 \
  --micro-batch 8 \
  --pin-memory \
  --output-dir benchmark_results/our_net
```

Add `--full-epoch` to time the compute-equivalent number of optimizer steps instead of estimating epoch compute time from the median. Data-loading time is reported separately by the production profiler.

The default path remains deterministic. TF32 and compilation are the safe first performance options to benchmark:

```bash
python3 train.py ... --tf32 --compile-model
```

FP16 training with `--amp` is numerically unstable for `our_net` and several baselines on this GPU. Evaluation-only autocast is available separately through `--eval-amp`.

For quick iteration, the following options reduce evaluation frequency and enable faster nondeterministic cuDNN kernels:

```bash
python3 train.py ... \
  --tf32 --compile-model \
  --eval-batch-size 64 --eval-amp --eval-every 5 \
  --fast-nondeterministic
```

Do not use the fast nondeterministic path for the final reproducibility sweep without rerunning all methods under the same protocol. Leave `--eval-batch-size` at its default `0` to preserve the legacy inference grouping exactly. On the tested RTX 5060 Ti, copying each 3,000-patch group to the GPU once and skipping the unused edge tensor improved training from 90.5 to 144.9 patches/s without changing model blocks, patch order, or optimizer steps.

## Focused checks

```bash
python3 -m pytest -q
python3 -m compileall -q train.py run_statistics.py summarize_statistics.py \
  modal_statistics.py statistics_utils.py set_up_seed.py tests
```
