import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.utils.data import random_split,DataLoader
from dataset import CustomDataset

def get_all_training_set(data_paths,batch_size=1):
    from transforms import get_train_transforms,get_test_transforms
    train_transforms=get_train_transforms()
    test_transforms=get_test_transforms()
    names= sorted([d for d in os.listdir(data_paths) if os.path.isdir(os.path.join(data_paths, d))])
    all_datasets=[]
    for i,name in enumerate(names):
        
        train_set=CustomDataset(os.path.join(data_paths,name,'training'),train_transforms)
        val_set = CustomDataset(os.path.join(data_paths,name,'test'),test_transforms)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        all_datasets.append({
            'train_loader': train_loader,
            'val_loader': val_loader,
            'name': name
        })
    return all_datasets


    
    


