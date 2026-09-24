# SGMA-Net: A Lightweight Mamba-Attention Network for Retinal Vessel Segmentation

Official implementation of **SGMA-Net**, published in *Expert Systems with Applications* (2027).

**Paper:** https://doi.org/10.1016/j.eswa.2026.134242

**Author share link:** https://authors.elsevier.com/c/1nllC3PiGTb9E1

SGMA-Net combines **RHMA**, **SAG**, **MDSA**, **SGWL**, and the **MSPG loss** for thin-vessel segmentation in low-contrast fundus images.

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

Train SGMA-Net on DRIVE:

```bash
python train.py \
  --model our_net \
  --loss our_loss \
  --datasets DRIVE_patches \
  --train_type patch \
  --type_split window \
  --patches 500 \
  --patch_size 64 \
  --batch_size 4 \
  --epochs 100
```

Train on CHASE-DB1 by replacing the dataset with `CHASEDB_1_patches`.

Quick local smoke test (1 epoch, 10 optimizer steps per dataset, W&B disabled):

```bash
python run_statistics.py --config configs/smoke_statistics.json
```

For the full controlled experiment matrix:

```bash
python run_statistics.py --config configs/statistics.json
```

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

Use `bce_dice_loss` for architecture-only ablations and `our_loss` for the full **BCE + Dice + MSPG** objective.

---

## Core implementation

```text
models/our_net/
├── bottle_neck.py      # LGFI + RHMA (official Mamba + MHSA)
├── modules.py          # SAG + MDC + MDSA + SGWL
└── our_net.py          # SGMA-Net encoder/decoder

loss/
├── bce_dice_loss.py
└── our_loss.py         # BCE + Dice + MSPG
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
