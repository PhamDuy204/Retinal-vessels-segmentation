from beam import Image, Pod

image = Image(python_version="python3.11", python_packages=[])

pod = Pod(
    app="retinal-ournet-smoke",
    name="retinal-ournet-smoke",
    gpu="RTX4090",
    cpu=2,
    memory="8Gi",
    image=image,
    keep_warm_seconds=0,
    entrypoint=[
        "bash",
        "-lc",
        "pwd; ls -la | head -40; test -d data/DRIVE && echo DATA_EXISTS || echo DATA_MISSING; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; sleep 5",
    ],
)
