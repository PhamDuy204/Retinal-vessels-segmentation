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

def get_all_training_set(data_paths,batch_size=1,num_patches=500,patch_size=64,training_type='normal'):
    from transforms import get_train_transforms,get_train_patch_transforms
    names= sorted([d for d in os.listdir(data_paths) if os.path.isdir(os.path.join(data_paths, d))])
    all_custom_train_datasets=[]
    all_custom_test_datasets=[]
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
            if patches and (name=='HRF'):continue
            train_set=CustomTrainDataset(os.path.join(data_paths,name,'training'),train_transforms,with_patches=patches,
                                         num_patches=num_patches,patch_size=patch_size)
            if patches==False:
                val_set = CustomTrainDataset(os.path.join(data_paths,name,'test'),train_transforms,with_patches=patches,num_patches=num_patches,patch_size=patch_size)
            else:
                val_set = CustomTestDataset(os.path.join(data_paths,name,'test'),train_transforms)

            if patches==False:
                all_custom_train_datasets.append(
                    train_set
                )
                all_custom_test_datasets.append(
                    val_set
                )
            else:
                all_custom_patch_datasets.append(ConcatDataset([train_set, 
                                                                CustomTrainDataset(os.path.join(data_paths,name,'test'),train_transforms,with_patches=patches,num_patches=num_patches,patch_size=patch_size)]))
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,)            
            val_loader   = DataLoader(val_set, batch_size=1, shuffle=False,)            
            suffix = '_patches' if patches else ''
            all_train_methods.append({
                'train_loader': train_loader,
                'val_loader': val_loader,
                'name': name+suffix,
                'patches': patches
            })
    for i in range(len(all_custom_train_datasets)):
        for j in range(len(all_custom_test_datasets)):
            if i == j: continue
            train_set =all_custom_train_datasets[i]
            val_set = all_custom_test_datasets[j]
            val_name = val_set.get_name()
            train_name  = train_set.get_name()
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,)        
            val_loader   = DataLoader(val_set, batch_size=1, shuffle=False,)
            all_train_methods.append({
                    'train_loader': train_loader,
                    'val_loader': val_loader,
                    'name': f'val_on_{train_name}_and_train_on_{val_name}',
                    'patches': False
                })
    for i in range(len(all_custom_patch_datasets)):
        train_transforms = get_train_patch_transforms()
        train_set = ConcatDataset(all_custom_patch_datasets[0:i]+all_custom_patch_datasets[i+1:])
        name = get_name(all_custom_patch_datasets[i])[-1]
        val_set =  CustomTestDataset(os.path.join(data_paths,name,'*'),train_transforms) 
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,)        
        val_loader   = DataLoader(val_set, batch_size=1, shuffle=False,)        
        all_train_methods.append({
                'train_loader': train_loader,
                'val_loader': val_loader,
                'name': f'val_on_{name}_and_train_on_remaining_datasets_with_patches',
                'patches': True
            })
    if training_type=='all':
        return all_train_methods
    elif training_type=='normal':
        return [method for method in all_train_methods if method['patches']==False]
    else:
        return [method for method in all_train_methods if method['patches']==True]


    
    


