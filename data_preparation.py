import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.utils.data import random_split,DataLoader
from dataset import CustomDataset

def get_all_training_set(data_paths,batch_size=4):
    from transforms import get_train_transforms
    train_transforms=get_train_transforms()
    names= sorted([d for d in os.listdir(data_paths) if os.path.isdir(os.path.join(data_paths, d))])
    all_datasets=[]
    for i,name in enumerate(names):
        
        all_training_data=CustomDataset(os.path.join(data_paths,name,'training'),train_transforms)
        train_set,val_set = random_split(all_training_data,[0.75,0.25],generator = torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        all_datasets.append({
            'train_loader': train_loader,
            'val_loader': val_loader,
            'name': name
        })
    return all_datasets


def get_all_test_set(data_paths):
    from transforms import get_test_transforms
    test_transforms=get_test_transforms()
    names= os.listdir(data_paths)
    all_datasets=[]
    for name in names:
        all_datasets.append(
            CustomDataset(os.path.join(data_paths,name,'test'),test_transforms)
        )
    return all_datasets

    
    


