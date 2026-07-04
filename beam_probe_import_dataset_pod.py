from __future__ import annotations

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
outputs = Volume(name="retinal-baseline-outputs", mount_path="/outputs")
probe = """
set -uo pipefail
cd /mnt/code
export PYTHONFAULTHANDLER=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo PY_PROBE_BEFORE
python -X faulthandler -u beam_probe_payload.py
echo PROBE_DONE
"""
pod = Pod(
    app="retinal-baseline-pending",
    name="retinal-baseline-beam-probe-a10g",
    gpu="A10G",
    cpu=8,
    memory=49152,
    image=image,
    volumes=[outputs],
    secrets=["WANDB_API_KEY"],
    keep_warm_seconds=0,
    entrypoint=["bash", "-lc", probe],
)
