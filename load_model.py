import importlib
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
def load_model_class(model_name):
    module_path = f"models.{model_name}.{model_name}"
    module = importlib.import_module(module_path)
    return module.SegModel