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

cmd = "\nset +e\ncd /mnt/code\nexport PYTHONFAULTHANDLER=1\nnvidia-smi --query-gpu=name,memory.total --format=csv,noheader\npython -V\npython - <<'PYCHK'\nimport sys, os\nprint('cwd', os.getcwd(), flush=True)\nprint('path0', sys.path[0], flush=True)\nimport torch\nprint('torch', torch.__version__, torch.cuda.is_available(), flush=True)\nimport train\nprint('import_train_ok', flush=True)\nPYCHK\necho HELP_BEGIN\npython -u train.py --help | head -40\nstatus=$?\necho HELP_STATUS=$status\necho TRAIN_BEGIN\npython -u train.py \\\n  --model edae_net \\\n  --seed 456 \\\n  --experiment-id beam_debug_eval35_e60 \\\n  --datasets DRIVE_patches \\\n  --data-root data \\\n  --output-root /tmp/beam-debug-outputs \\\n  --wandb-mode disabled \\\n  --batch_size 4 \\\n  --epochs 1 \\\n  --loss bce_loss \\\n  --learning_rate 0.001 \\\n  --patches 10 \\\n  --patch_size 64 \\\n  --train_type patch \\\n  --type_split window \\\n  --num-workers 0 \\\n  --eval-batch-size 64 \\\n  --eval-every 1 \\\n  --eval-start-epoch 35 \\\n  --tf32 \\\n  --resume\nstatus=$?\necho TRAIN_STATUS=$status\nsleep 5\nexit $status\n"

pod = Pod(
    app="retinal-beam-debug",
    name="retinal-beam-debug-a10g",
    gpu="A10G",
    cpu=4,
    memory="32Gi",
    image=image,
    keep_warm_seconds=0,
    entrypoint=["bash", "-lc", cmd],
)
