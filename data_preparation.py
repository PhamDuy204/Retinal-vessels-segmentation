import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.utils.data import random_split,DataLoader,ConcatDataset
from dataset import CustomTrainDataset,CustomTestDataset
def get_name(concat_datasets):
    lst_name = []
    for d in concat_datasets.datasets:
        lst_name.append(d.get_name())
    return list(set(lst_name))

def get_all_training_set(data_paths,batch_size=1,num_patches=500,patch_size=64):
    from transforms import get_train_transforms,get_train_patch_transforms
    names= sorted([d for d in os.listdir(data_paths) if os.path.isdir(os.path.join(data_paths, d))])
    all_custom_datasets=[]
    all_custom_patch_datasets=[]
    all_train_methods=[]

    for method in range(2):
        if method == 0:
            patches=False
            train_transforms = get_train_transforms()
        else:
            patches=True
            train_transforms = get_train_patch_transforms()
        for name in names:
            train_set=CustomTrainDataset(os.path.join(data_paths,name,'training'),train_transforms,with_patches=patches,
                                         num_patches=num_patches,patch_size=patch_size)
            if patches==False:
                val_set = CustomTrainDataset(os.path.join(data_paths,name,'test'),train_transforms,with_patches=patches,num_patches=num_patches,patch_size=patch_size)
            else:
                val_set = CustomTestDataset(os.path.join(data_paths,name,'test'),train_transforms)

            if patches==False:all_custom_datasets.append(ConcatDataset([train_set, val_set]))
            else:
                all_custom_patch_datasets.append(ConcatDataset([train_set, 
                                                                CustomTrainDataset(os.path.join(data_paths,name,'test'),train_transforms,with_patches=patches,num_patches=num_patches,patch_size=patch_size)]))
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=4)
            val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,pin_memory=True,num_workers=4)
            suffix = '_patches' if patches else ''
            all_train_methods.append({
                'train_loader': train_loader,
                'val_loader': val_loader,
                'name': name+suffix,
                'patches': patches
            })
    for i in range(len(all_custom_datasets)):
        train_set = ConcatDataset(all_custom_datasets[0:i]+all_custom_datasets[i+1:])
        val_set = all_custom_datasets[i]
        name  = get_name(val_set)[-1]
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=4)
        val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,pin_memory=True,num_workers=4)

        all_train_methods.append({
                'train_loader': train_loader,
                'val_loader': val_loader,
                'name': f'val on {name} and train on remaining datasets',
                'patches': False
            })
    for i in range(len(all_custom_patch_datasets)):
        train_transforms = get_train_patch_transforms()
        train_set = ConcatDataset(all_custom_patch_datasets[0:i]+all_custom_patch_datasets[i+1:])
        name = get_name(all_custom_patch_datasets[i])[-1]
        val_set =  CustomTestDataset(os.path.join(data_paths,name,'*'),train_transforms) 
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=4)
        val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,pin_memory=True,num_workers=4)
        all_train_methods.append({
                'train_loader': train_loader,
                'val_loader': val_loader,
                'name': f'val on {name} and train on remaining datasets with (patches) ',
                'patches': True
            })
    return all_train_methods


    
    


