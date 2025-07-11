import importlib
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_model_class(model_name):
    module_path = f"models.{model_name}.{model_name}"
    module = importlib.import_module(module_path)
    return module.SegModel


def load_loss_class(loss_name):
    module_path = f"loss.{loss_name}"
    module = importlib.import_module(module_path)
    name = loss_name.split('_')
    new_name =''
    for i in range(len(name)):
        new_name+=name[i][0].upper()+name[i][1:]
    return getattr(module,new_name)