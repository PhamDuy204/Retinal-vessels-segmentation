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

The default SGMA-Net uses the official `mamba_ssm.Mamba` implementation.

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
- epochs: **100** (or stop after 20 consecutive epochs without a lower epoch-mean training loss)
- checkpoint: `best.pt` is the epoch with the lowest mean training loss
- learning rate: `0.0018`
- patch training: 500 sampled 64×64 patches/image, window split
- DataLoader: 4 workers, pinned memory, persistent workers
- training AMP: **FP16 enabled**
- native low-precision GroupNorm feature maps for `our_net`
- evaluation AMP enabled, evaluation batch size 256
- micro-batch size 48
- fused CUDA Adam enabled
- cuDNN autotuning / fast nondeterministic CUDA path enabled
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
  --epochs 100 \
  --early-stopping-patience 20 \
  --learning_rate 0.0018 \
  --num-workers 4 \
  --pin-memory \
  --persistent-workers \
  --amp \
  --amp-dtype fp16 \
  --amp-native-norm \
  --eval-amp \
  --eval-batch-size 256 \
  --micro-batch-size 48 \
  --fused-adam \
  --fast-nondeterministic
```

`--early-stopping-patience 20` is the default; set it to `0` to train through all epochs. Checkpoint selection still uses the minimum mean training loss when early stopping is disabled. If the selected epoch precedes `--eval-start-epoch`, it is evaluated once at the end using the existing evaluation settings. Neither AMP nor the training forward/backward path is changed.

Train on CHASE-DB1 by replacing the dataset with `CHASEDB_1_patches`.

> **Mixed-precision safety note:** the default FP16 configuration is the fast configuration validated on DRIVE. Other datasets or numerical conditions can still produce `NaN`/non-finite loss under mixed precision. If this happens, keep the other optimized runtime settings but disable FP16 and run the model/evaluation in FP32 using the command below.

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
