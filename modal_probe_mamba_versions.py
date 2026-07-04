from __future__ import annotations
from pathlib import Path
import modal

LOCAL_ROOT = Path(__file__).resolve().parent
image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('libgl1', 'libglib2.0-0')
    .pip_install_from_requirements(str(LOCAL_ROOT / 'requirements-modal.txt'))
    .pip_install('mamba-ssm', 'causal-conv1d', extra_options='--no-build-isolation')
)
app = modal.App('probe-mamba-versions')

@app.function(image=image, gpu='L4', timeout=1800, memory=16384, cpu=4)
def probe():
    import json, importlib, pkgutil, torch
    out = {'torch': torch.__version__, 'cuda': torch.version.cuda}
    try:
        import mamba_ssm
        out['mamba_ssm_file'] = getattr(mamba_ssm, '__file__', '')
        out['mamba_ssm_has_top_Mamba3'] = hasattr(mamba_ssm, 'Mamba3')
        import mamba_ssm.modules as mods
        out['modules'] = sorted([m.name for m in pkgutil.iter_modules(mods.__path__) if 'mamba' in m.name])
        for name in ['mamba_ssm.modules.mamba3', 'mamba_ssm.modules.mamba2', 'mamba_ssm']:
            try:
                mod = importlib.import_module(name)
                out[name] = {'ok': True, 'has_Mamba3': hasattr(mod, 'Mamba3'), 'has_Mamba2': hasattr(mod, 'Mamba2')}
            except Exception as e:
                out[name] = {'ok': False, 'err': repr(e)}
    except Exception as e:
        out['err'] = repr(e)
    return json.dumps(out, sort_keys=True)

@app.local_entrypoint()
def main():
    print(probe.remote())
