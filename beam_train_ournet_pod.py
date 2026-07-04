from beam import Image, Pod, Volume

image = Image(
    python_version="python3.11",
    commands=[
        "apt-get update -y && apt-get install -y libgl1 libglib2.0-0",
        "python -m pip install --upgrade pip",
        "python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.7.1+cu128 torchvision==0.22.1+cu128",
        "python -m pip install 'numpy<2.3' stringzilla==3.10.4 albumentations==2.0.8 scikit-image opencv-python-headless tqdm wandb torchmetrics ml-collections kornia einops ninja huggingface_hub packaging",
        "python -m pip install --no-deps https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1%2Bcu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
        "python -m pip install --no-deps https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5%2Bcu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
    ],
)

outputs = Volume(name="retinal-ournet-outputs", mount_path="/outputs")

train_cmd = """
set -euo pipefail
cd /mnt/code
python - <<'CHECK'
from pathlib import Path
print('cwd=/mnt/code')
print('data_drive_exists=', Path('data/DRIVE').exists())
print('config_exists=', Path('configs/ournet_beam_best_width64_ema995_ttaflip_eval30.json').exists())
CHECK
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export PYTHONFAULTHANDLER=1
python -X faulthandler - <<'SMOKE'
print('smoke_import_begin', flush=True)
import torch
print('torch_version=', torch.__version__, flush=True)
print('torch_cxx11_abi=', torch._C._GLIBCXX_USE_CXX11_ABI, flush=True)
from load_model import load_model_class
print('load_model_import_ok', flush=True)
model_class = load_model_class('our_net')
print('model_class_ok', model_class, flush=True)
model = model_class(3, 1, width=64).cuda().train()
print('model_cuda_ok params=', sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
x = torch.randn(1, 3, 64, 64, device='cuda')
y = model(x)
if isinstance(y, (tuple, list)):
    loss = sum(t.float().mean() for t in y)
else:
    loss = y.float().mean()
loss.backward()
torch.cuda.synchronize()
print('smoke_forward_backward_ok', float(loss.detach().cpu()), flush=True)
SMOKE
echo train_begin
python -u train.py \
  --model our_net \
  --model-width 64 \
  --seed 2026 \
  --experiment-id Retinal_training_revision_beam_ournet_width64_ema995_ttaflip_eval30_drive_seed2026_e60 \
  --datasets DRIVE_patches \
  --data-root data \
  --output-root /outputs \
  --wandb-mode disabled \
  --wandb-entity phamdinhanhduy-university-of-information-and-technology \
  --wandb-project Retinal-Vessels-Segmentation \
  --batch_size 4 \
  --epochs 60 \
  --loss abe_dice_loss_optimized \
  --learning_rate 0.001 \
  --patches 500 \
  --patch_size 64 \
  --train_type patch \
  --type_split window \
  --num-workers 0 \
  --pin-memory \
  --eval-batch-size 64 \
  --eval-every 5 \
  --eval-start-epoch 30 \
  --eval-tta-mode flips \
  --ema-decay 0.995 \
  --ema-update-every 1 \
  --fast-nondeterministic \
  --tf32 \
  --resume
""".strip()

pod = Pod(
    app="retinal-ournet-train",
    name="retinal-ournet-width64-eval30-a10g",
    gpu="A10G",
    cpu=4,
    memory="32Gi",
    image=image,
    volumes=[outputs],
    keep_warm_seconds=0,
    entrypoint=["bash", "-lc", train_cmd],
)
