import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.utils.data import random_split,DataLoader,ConcatDataset
from dataset import CustomTrainDataset,CustomTestDataset
from set_up_seed import seed_worker


def make_data_loader(
    dataset,
    batch_size,
    shuffle,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    seed=42,
    prefetch_factor=2,
):
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    kwargs = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(persistent_workers and num_workers > 0),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        **kwargs,
    )

def get_name(concat_datasets):
    lst_name = []
    for d in concat_datasets.datasets:
        lst_name.append(d.get_name())
    return list(set(lst_name))

def get_all_training_set(
    data_paths,
    batch_size=1,
    num_patches=500,
    patch_size=64,
    training_type='normal',
    type_split='random',
    model_name='',
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    seed=42,
    transform_seed=None,
    prefetch_factor=2,
):
    from transforms import get_train_transforms,get_train_patch_transforms,get_test_transforms,get_test_patch_transforms
    names= sorted([d for d in os.listdir(data_paths) if os.path.isdir(os.path.join(data_paths, d))])
    all_custom_train_datasets=[]
    all_custom_test_datasets=[]
    all_custom_train_patch_datasets=[]
    all_custom_test_patch_datasets=[]
    all_train_methods=[]

    for method in range(2):
        if method == 0:
            patches=False
            train_transforms = get_train_transforms()
            test_transforms = get_test_transforms()
        else:
            patches=True
            train_transforms = get_train_patch_transforms()
            test_transforms = get_test_patch_transforms()
        effective_transform_seed = seed if transform_seed is None else transform_seed
        if effective_transform_seed is not None:
            if hasattr(train_transforms, "set_random_seed"):
                train_transforms.set_random_seed(effective_transform_seed)
            if hasattr(test_transforms, "set_random_seed"):
                test_transforms.set_random_seed(effective_transform_seed)
        for name in names:
            if patches and (name=='HRF'):continue
            train_set=CustomTrainDataset(os.path.join(data_paths,name,'training'),train_transforms,with_patches=patches,
                                         num_patches=num_patches,patch_size=patch_size,type_split=type_split,model_name=model_name)
            if patches==False:
                val_set = CustomTrainDataset(os.path.join(data_paths,name,'test'),test_transforms,with_patches=patches,num_patches=num_patches,patch_size=patch_size,type_split=type_split,model_name=model_name)
            else:
                val_set = CustomTestDataset(os.path.join(data_paths,name,'test'),test_transforms,type_split=type_split,model_name=model_name)

            if patches==False:
                all_custom_train_datasets.append(
                    train_set
                )
                all_custom_test_datasets.append(
                    val_set
                )
            else:
                if name not in ['STARE_F2','STARE_F3','STARE_F4','STARE_F5','CHASEDB_F1','CHASEDB_F2','CHASEDB_F3','CHASEDB_F4','REVERSED_DRIVE','DRIVE_RANDOM_SEED42']:
                    all_custom_train_patch_datasets.append(
                        CustomTrainDataset(os.path.join(data_paths,name,'*'),train_transforms,with_patches=patches,
                                            num_patches=num_patches,patch_size=patch_size,type_split=type_split,model_name=model_name)
                    )
                    all_custom_test_patch_datasets.append(
                        CustomTestDataset(os.path.join(data_paths,name,'*'),test_transforms,type_split=type_split,model_name=model_name)
                    )
                
            train_loader = make_data_loader(
                train_set, batch_size, True, num_workers, pin_memory,
                persistent_workers, seed, prefetch_factor
            )
            val_loader = make_data_loader(
                val_set, 1, False, num_workers, pin_memory,
                persistent_workers, seed, prefetch_factor
            )
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
            train_loader = make_data_loader(
                train_set, batch_size, True, num_workers, pin_memory,
                persistent_workers, seed, prefetch_factor
            )
            val_loader = make_data_loader(
                val_set, 1, False, num_workers, pin_memory,
                persistent_workers, seed, prefetch_factor
            )
            all_train_methods.append({
                    'train_loader': train_loader,
                    'val_loader': val_loader,
                    'name': f'val_on_{val_name}_and_train_on_{train_name}',
                    'patches': False
                })
    for i in range(len(all_custom_train_patch_datasets)):
        for j in range(len(all_custom_train_patch_datasets)):
            if (i!=j) and (all_custom_train_patch_datasets[i].get_name()!='CHASEDB_1') and (all_custom_test_patch_datasets[j].get_name()!='CHASEDB_1'):
                train_set = all_custom_train_patch_datasets[i]
                val_set = all_custom_test_patch_datasets[j]
                val_name = val_set.get_name()
                train_name=train_set.get_name()
                train_loader = make_data_loader(
                    train_set, batch_size, True, num_workers, pin_memory,
                    persistent_workers, seed, prefetch_factor
                )
                val_loader = make_data_loader(
                    val_set, 1, False, num_workers, pin_memory,
                    persistent_workers, seed, prefetch_factor
                )
                all_train_methods.append({
                        'train_loader': train_loader,
                        'val_loader': val_loader,
                        'name': f'val_on_{val_name}_and_train_on_{train_name}_with_patches',
                        'patches': True
                    })
    if training_type=='all':
        return all_train_methods
    elif training_type=='normal':
        return [method for method in all_train_methods if method['patches']==False]
    else:
        return [method for method in all_train_methods if method['patches']==True]


    
    
