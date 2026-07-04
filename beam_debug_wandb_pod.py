from beam import Image, Pod

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

cmd = '\nset +e\ncd /mnt/code\nexport PYTHONFAULTHANDLER=1\nnvidia-smi --query-gpu=name,memory.total --format=csv,noheader\npython -V\necho WANDB_KEY_PRESENT=${WANDB_API_KEY:+yes}\necho TRAIN_ONLINE_BEGIN\npython -u train.py \\\n  --model edae_net \\\n  --seed 456 \\\n  --experiment-id beam_debug_wandb_online \\\n  --datasets DRIVE_patches \\\n  --data-root data \\\n  --output-root /tmp/beam-debug-outputs-online \\\n  --wandb-mode online \\\n  --wandb-entity phamdinhanhduy-university-of-information-and-technology \\\n  --wandb-project Retinal-Vessels-Segmentation \\\n  --batch_size 4 \\\n  --epochs 1 \\\n  --loss bce_loss \\\n  --learning_rate 0.001 \\\n  --patches 10 \\\n  --patch_size 64 \\\n  --train_type patch \\\n  --type_split window \\\n  --num-workers 0 \\\n  --eval-batch-size 64 \\\n  --eval-every 1 \\\n  --eval-start-epoch 1 \\\n  --tf32 \\\n  --resume\nstatus=$?\necho TRAIN_ONLINE_STATUS=$status\nsleep 5\nexit $status\n'

pod = Pod(
    app="retinal-beam-debug-wandb",
    name="retinal-beam-debug-wandb-a10g",
    gpu="A10G",
    cpu=4,
    memory="32Gi",
    image=image,
    secrets=["WANDB_API_KEY"],
    keep_warm_seconds=0,
    entrypoint=["bash", "-lc", cmd],
)
