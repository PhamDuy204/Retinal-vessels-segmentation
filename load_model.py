import importlib
import os
import sys
import importlib.util
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Track the currently loaded model to avoid conflicts when switching
_current_model = None
_loaded_modules = {}  # Track which modules we loaded from which model

def load_model_class(model_name):
    global _current_model, _loaded_modules
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If switching to a different model, aggressively clear old model's modules
    if _current_model and _current_model != model_name:
        # Remove old model directory from sys.path
        old_model_dir = os.path.join(base_dir, 'models', _current_model)
        if old_model_dir in sys.path:
            sys.path.remove(old_model_dir)
        
        # Remove all modules that were loaded from the old model
        old_module_names = list(_loaded_modules.keys())
        for shortname in old_module_names:
            if shortname in sys.modules:
                del sys.modules[shortname]
        # Also remove any package imports (e.g., models.our_net.our_net, models.our_net.bottle_neck)
        keys_to_delete = [k for k in sys.modules.keys() if f'models.{_current_model}' in k]
        for k in keys_to_delete:
            del sys.modules[k]
        _loaded_modules.clear()
    
    # Ensure the specific model folder is on sys.path
    model_dir = os.path.join(base_dir, 'models', model_name)
    if os.path.isdir(model_dir) and model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    # Load all .py files in the model dir into sys.modules under both short names
    # and full module paths so pickle can find them either way
    for fname in os.listdir(model_dir):
        if not fname.endswith('.py') or fname.startswith('__'):
            continue
        shortname = fname[:-3]
        path = os.path.join(model_dir, fname)
        
        # Remove from sys.modules if it exists to force a fresh load
        if shortname in sys.modules:
            del sys.modules[shortname]
        
        # Also remove the full path version
        full_module_path = f"models.{model_name}.{shortname}"
        if full_module_path in sys.modules:
            del sys.modules[full_module_path]
        
        # Load module from file location
        spec = importlib.util.spec_from_file_location(shortname, path)
        if spec is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            # If executing the module fails, skip it (it may import heavy deps)
            continue
        
        # Register under BOTH short name AND full path
        sys.modules[shortname] = module
        sys.modules[full_module_path] = module
        _loaded_modules[shortname] = True

    # Clear the importlib cache to force reimport of package modules
    importlib.invalidate_caches()
    
    # Now import the model module as package module first, fallback to short name
    module_path = f"models.{model_name}.{model_name}"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        module = importlib.import_module(model_name)
    
    # Track the current model so we can clean up when switching
    _current_model = model_name
    
    return getattr(module, 'SegModel')


def load_loss_class(loss_name):
    module_path = f"loss.{loss_name}"
    module = importlib.import_module(module_path)
    name = loss_name.split('_')
    new_name =''
    for i in range(len(name)):
        new_name+=name[i][0].upper()+name[i][1:]
    return getattr(module,new_name)