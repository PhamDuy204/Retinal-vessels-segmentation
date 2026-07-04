from beam import Image, function

image = Image(python_version="python3.11", python_packages=[])

@function(gpu="RTX4090", cpu=2, memory="8Gi", image=image, timeout=600, retries=0, name="retinal-beam-smoke")
def smoke():
    import os
    import subprocess
    from pathlib import Path
    print("CWD", os.getcwd())
    print("FILES", sorted([p.name for p in Path('.').iterdir()])[:40])
    print("DATA_EXISTS", Path('data/DRIVE').exists())
    try:
        print(subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True))
    except Exception as exc:
        print("NVIDIA_SMI_FAILED", type(exc).__name__, exc)
    return {"ok": True, "cwd": os.getcwd(), "data_exists": Path('data/DRIVE').exists()}
