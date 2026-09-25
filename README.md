# SGMA-Net: A Lightweight Mamba-Attention Network for Retinal Vessel Segmentation

Official implementation of **SGMA-Net**, published in *Expert Systems with Applications* (2027).

**Paper:** https://doi.org/10.1016/j.eswa.2026.134242

**Author share link:** https://authors.elsevier.com/c/1nllC3PiGTb9E1

SGMA-Net combines **RHMA**, **SAG**, **MDSA**, **SGWL**, and the **MSPG-style multi-scope loss** for thin-vessel segmentation in low-contrast fundus images.

---

## Prepare

Install the dependencies:

```bash
pip install -r requirements.txt
```

The default SGMA-Net uses the official `mamba_ssm.Mamba2` implementation on Linux.

Place the datasets under `data/`:

```text
data/
├── DRIVE/
│   ├── training/{images,mask}/
│   └── test/{images,mask}/
├── CHASEDB_1/
│   ├── training/{images,mask}/
│   └── test/{images,mask}/
└── STARE_F1/ ... STARE_F5/
```

The paper protocol uses 64×64 patches, stride 32, and 500 sampled training patches per source image.

---

## Train

### Default optimized training

`train.py` now defaults to the validated fast SGMA-Net configuration:

- model: `our_net`
- loss: `main_loss`
- epochs: **60** (or stop after 20 consecutive epochs without a lower epoch-mean training loss)
- checkpoint: `best.pt` is the epoch with the lowest mean training loss
- learning rate: `0.0018`
- DataLoader: 0 workers, pinned memory, persistent workers disabled
- training AMP: **BF16 enabled** (wider dynamic range than FP16)
- native low-precision GroupNorm feature maps for `our_net`
- micro-batch size 48
- fused CUDA Adam enabled
- cuDNN autotuning / fast nondeterministic CUDA path enabled
- W&B logging disabled by default
- TTA disabled by default

Therefore, DRIVE can be started with:

```bash
python train.py --datasets DRIVE_patches
```

The equivalent explicit command is:

```bash
python train.py \
  --model our_net \
  --loss main_loss \
  --datasets DRIVE_patches \
  --train_type patch \
  --type_split window \
  --patches 500 \
  --patch_size 64 \
  --batch_size 4 \
  --epochs 60 \
  --early-stopping-patience 20 \
  --learning_rate 0.0018 \
  --num-workers 0 \
  --pin-memory \
  --no-persistent-workers \
  --amp \
  --amp-dtype bf16 \
  --amp-native-norm \
  --eval-amp \
  --eval-amp-dtype bf16 \
  --eval-start-epoch 50 \
  --eval-every 1 \
  --eval-batch-size 256 \
  --micro-batch-size 48 \
  --fused-adam \
  --fast-nondeterministic \
  --wandb-mode disabled
```

`--early-stopping-patience 20` is the default; set it to `0` to train through all epochs. Checkpoint selection still uses the minimum mean training loss when early stopping is disabled. If the selected epoch precedes `--eval-start-epoch`, it is evaluated once at the end using the existing BF16 evaluation settings. The AMP configuration and forward/backward path are unchanged.

Train on CHASE-DB1 by replacing the dataset with `CHASEDB_1_patches`.

### Windows model

On Windows, use `our_net_window`. It keeps the SGMA-Net architecture and
replaces only the Tri Dao Mamba2 block with the pure-PyTorch `mambapy`
backend, which does not require the Linux-only Mamba CUDA extensions.

`our_net_window` is forced to full FP32 for both training and evaluation
(`--no-amp --no-eval-amp --amp-dtype fp32`) even if AMP flags are supplied.

```bash
python train.py --model our_net_window --datasets DRIVE_patches
```

For a one-epoch validation smoke test:

```bash
python train.py   --model our_net_window   --datasets DRIVE_patches   --epochs 60   --stop-after-epoch 1   --eval-start-epoch 1   --wandb-mode disabled
```

Evaluation now uses explicit BF16 (override with `--eval-amp-dtype`).
The validated 60-epoch configuration keeps Mamba2 in FP32 under AMP and
computes the bottleneck MAB multiplicative/statistics/gating path in FP32
during BF16 training. FP16 evaluation also uses the protected MAB path.
Evaluation raises on non-finite predictions before reconstruction/metric updates.
Use `--no-eval-amp` for a full FP32 reference. Completed runs also retain
`last.pt` so the final epoch can be re-evaluated separately from `best.pt`.

> **Mixed-precision safety note:** BF16 is the default because the previous FP16 run on DRIVE produced a non-finite loss late in training, while BF16 keeps the same Tensor Core mixed-precision path with a much wider exponent range. If a dataset/model still produces `NaN`/non-finite loss, keep the other optimized runtime settings and fall back to full FP32 using the command below. FP16 remains available explicitly with `--amp-dtype fp16` for experiments where it is known to be stable.

```bash
python train.py \
  --datasets DRIVE_patches \
  --no-amp \
  --amp-dtype fp32 \
  --no-eval-amp \
  --no-amp-native-norm
```

That command changes only the mixed-precision path back to FP32; the optimized loader settings, micro-batching, fused Adam, patch/reconstruction optimizations, and other default speedups remain enabled.

Quick local smoke test:

```bash
python run_statistics.py --config configs/smoke_statistics.json
```

For the full controlled experiment matrix:

```bash
python run_statistics.py --config configs/statistics.json
```

### Weights & Biases (optional)

W&B logging is disabled by default. To upload a run, pass `--wandb-mode online`.
On the first machine/session that will upload runs to W&B, log in once and paste
your personal API token when prompted:

```bash
wandb login
```

Do **not** commit the token to this repository. You can also provide it through
the `WANDB_API_KEY` environment variable in CI/cloud environments.

To change the W&B project name for a direct training run:

```bash
python train.py \
  --datasets DRIVE_patches \
  --wandb-mode online \
  --wandb-project My-Retinal-Project
```

For `run_statistics.py`, set the project in the JSON config:

```json
{
  "wandb_project": "My-Retinal-Project",
  "wandb_entity": "your-wandb-entity"
}
```

For direct `train.py` runs, no extra flag is needed to keep W&B disabled.
For statistics configs, set:

```json
{
  "wandb_mode": "disabled"
}
```


---

## Loss

The canonical training objective is now:

```text
loss/main_loss.py
└── MainLoss
```

`main_loss` is the optimized implementation used by the default training command. Its multi-scope window sums avoid materializing large `unfold` tensors, while the numerically sensitive loss calculations are accumulated in FP32.

`loss/abe_dice_loss_optimized.py` is retained as a backward-compatible alias so historical configs that still reference `abe_dice_loss_optimized` continue to run.

---

## Ablation models

The architecture ablations reuse the same SGMA-Net implementation so block logic cannot drift:

```text
baseline
baseline_RHMA
baseline_RHMA_SAG
baseline_RHMA_SAG_MDSA
our_net                 # + SGWL
```

Use `main_loss` for the optimized full training objective. Historical loss implementations are retained in `loss/` for reproducibility of older runs.

---

## Core implementation

```text
models/our_net/
├── bottle_neck.py      # LGFI + RHMA (official Mamba + MHSA)
├── modules.py          # SAG + MDC + MDSA + SGWL
└── our_net.py          # SGMA-Net encoder/decoder

loss/
├── main_loss.py                    # default optimized objective
├── abe_dice_loss_optimized.py      # backward-compatible alias
└── abe_dice_loss.py                # historical implementation
```

All models use the same preprocessing pipeline:

```text
RGB -> grayscale -> z-score -> CLAHE -> unsharp masking
```

---

## Citation

If this code is useful in your research, please cite:

```bibtex
@article{nguyen-tat2027sgmanet,
  title   = {SGMA-Net: A lightweight mamba-attention network for thin-vessel segmentation in low-contrast fundus images with statistical feature refinement},
  author  = {Thien B. Nguyen-Tat and Duy Pham Dinh Anh,  Tan, Sang Hua},
  journal = {Expert Systems with Applications},
  volume  = {333},
  pages   = {134242},
  year    = {2027},
  doi     = {10.1016/j.eswa.2026.134242}
}
```

## Interactive DRIVE inference

The demo supports only **`our_net`** (Linux) and **`our_net_window`** (Windows).
The variants use different Mamba implementations, so their checkpoints are not interchangeable. The default is `our_net` (`--os linux`) even on Windows.
From the repository root:

```bash
pip install -r requirements.txt
streamlit run demo.py -- --checkpoints inference_models/drive_epoch58.safetensors --image_paths data/DRIVE/test/images
```

`--` passes the following options to `demo.py` rather than Streamlit:

| Option | Required | Meaning |
| --- | --- | --- |
| `--os` | No | `linux` selects `our_net`; `win` selects `our_net_window`. Defaults to `linux` on every operating system; pass `--os win` for `our_net_window`. |
| `--checkpoints` | No | One or more compatible `.pt` or `.safetensors` files, directories, comma-separated paths, or globs for the model menu. |
| `--image_paths` | No | Images, directories, comma-separated paths, or globs for the image menu; you can always upload an image instead. |

Without `--checkpoints`, the model selector becomes a checkpoint upload button:

```bash
streamlit run demo.py -- --os win --image_paths data/DRIVE/test/images
```
