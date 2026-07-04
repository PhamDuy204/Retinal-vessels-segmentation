from beam import Image, Pod

image = Image(
    python_version="python3.11",
    commands=[
        "python -m pip install --upgrade pip",
        "python -m pip install wandb",
    ],
)

cmd = "\nset +e\ncd /mnt/code\nexport PYTHONFAULTHANDLER=1\nexport WANDB_DEBUG=true\npython - <<'PYWB'\nimport os, sys, traceback\nprint('WANDB_KEY_PRESENT=', bool(os.environ.get('WANDB_API_KEY')), flush=True)\ntry:\n    import wandb\n    print('wandb_version=', wandb.__version__, flush=True)\n    print('login_begin', flush=True)\n    ok = wandb.login(key=os.environ.get('WANDB_API_KEY'), relogin=True)\n    print('login_result=', ok, flush=True)\n    print('init_begin', flush=True)\n    run = wandb.init(entity='phamdinhanhduy-university-of-information-and-technology', project='Retinal-Vessels-Segmentation', name='beam_wandb_smoke', mode='online')\n    print('run_id=', run.id, flush=True)\n    wandb.log({'beam_wandb_smoke': 1})\n    wandb.finish()\n    print('wandb_done', flush=True)\nexcept BaseException as e:\n    print('EXC', type(e).__name__, str(e), flush=True)\n    traceback.print_exc()\n    sys.exit(7)\nPYWB\nstatus=$?\necho PYWB_STATUS=$status\nsleep 5\nexit $status\n"

pod = Pod(
    app="retinal-beam-wandb-smoke",
    name="retinal-beam-wandb-smoke",
    cpu=1,
    memory="2Gi",
    image=image,
    secrets=["WANDB_API_KEY"],
    keep_warm_seconds=0,
    entrypoint=["bash", "-lc", cmd],
)
