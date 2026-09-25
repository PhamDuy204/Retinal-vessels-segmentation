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
- epochs: **60**
- learning rate: `0.0018`
- patch training: 750 sampled 64×64 patches/image in the current sampler, window split
- DataLoader: 0 workers, pinned memory, persistent workers disabled
- training AMP: **BF16 enabled** (wider dynamic range than FP16)
- native low-precision GroupNorm feature maps for `our_net`
- evaluation starts at epoch **50** and uses **BF16**, evaluation batch size 256; Mamba2 remains FP32 under AMP
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

Runtime audit: the current dataset sampler actually returns **750 patches/image**
regardless of `--patches 500`. This fix preserves that existing sampling behavior
and the evaluation patch grid; runtime comparisons must use 750 patches/image.
The paper's 500-patch protocol and this runtime setting are different.


Validated DRIVE checkpoints from stability_bneckmabfp32_60e_wandb_20260925
are stored under checkpoints/stability_bneckmabfp32_60e_wandb_20260925/.
The selected checkpoint is epoch 58 (best.pt).

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
  author  = {Nguyen-Tat, Thien B. and Pham Dinh Anh, Duy and Tan, Sang Hua},
  journal = {Expert Systems with Applications},
  volume  = {333},
  pages   = {134242},
  year    = {2027},
  doi     = {10.1016/j.eswa.2026.134242}
}
```

## Interactive DRIVE inference

Run on the development branch with the exported `safetensors` model or training `.pt` checkpoints (files,
directories, comma-separated paths, or shell globs) and optional image paths:

```bash
streamlit run demo.py -- --checkpoints inference_models/drive_epoch58.safetensors --image_paths data/DRIVE/test/images
```

Upload PNG/JPEG/TIFF/PPM/BMP/WebP or choose an image path, select a checkpoint,
then press **Predict**. The right panel shows a binary mask (0 background, 1 vessel);
the arrow toggles a vessel-focused Grad-CAM overlay on the original image;
the arrow reverses direction to return to the mask. The mask preview uses
dark teal vessels on a light background for readability. The Grad-CAM calculation
runs only when requested. Downloaded mask PNG contains literal pixel values 0/1.

The UI loads `.pt` state dicts with `weights_only=True` and supports exported
`our_net` `.safetensors` weights. Export other `.pt` checkpoints with
`python convert_checkpoint.py path/to/best.pt --output inference_models/model.safetensors`.
The shipped epoch 58 safetensors weights match the `.pt` tensors exactly.
The Mamba2 CUDA kernels would require custom ONNX operators or a slower scan implementation;
changing to safetensors improves weight portability but does not speed up the model forward. On the RTX 3060,
a warmed DRIVE 584×565 inference with a 48×48 overlapping patch grid took
~0.35–0.38 seconds end to end (first prediction is slower for CUDA warmup).
The 48×48 grid covers DRIVE with 169 patches; other image sizes choose a
slightly denser stride where needed so every edge has coverage. Across all
20 DRIVE test images, per-image F1 versus the previous 32×32 UI grid changed
by -0.0030 to +0.0038; the pixel disagreement was 0.43–0.56%. This UI speed
mode is for interactive viewing; paper evaluation still uses the denser 32×8 grid.

CPU fallback uses a reference Mamba2 scan with the same checkpoint weights:
it is functional but took ~23 seconds per DRIVE image with the new grid on the tested machine.
The <0.4 second target is achieved on the RTX 3060 after warmup, not CPU.
