import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import numpy as np
import glob
from torch.utils.data import Dataset
from utils import *
from albumentations.pytorch import ToTensorV2

class CustomTrainDataset(Dataset):
    def __init__(self,root_path,img_transforms=None,with_patches = False,num_patches=500,patch_size=64,type_split='random'):
        self.image_paths =  sorted(glob.glob(root_path + '/images/*.jpg')+glob.glob(root_path + '/images/*.tif')\
                            + glob.glob(root_path + '/images/*.ppm'))
        self.mask_paths = sorted(glob.glob(root_path + '/mask/*.png')+glob.glob(root_path + '/mask/*.tif')\
                            + glob.glob(root_path + '/mask/*.ppm')+glob.glob(root_path + '/mask/*.gif'))

        self.image_transforms = img_transforms
        self.name = root_path.split('/')[-2]
        self.with_patches=with_patches
        self.num_patches=num_patches
        self.patch_size=patch_size
        self.type_split=type_split
    def get_name(self):
        return self.name
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self,index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = preprocessing_img(image_path)
        mask = mask = np.array(Image.open(mask_path),dtype=np.uint8)
        if (len(mask.shape)==3):mask=mask[:,:,0]
        mask = np.ceil(mask/255).astype(np.uint8)
        if self.image_transforms:
            t = self.image_transforms(image = image,mask=mask)
            image = t['image']
            mask  = t['mask']
            edge = ToTensorV2()(image=sobel_transform(image.clone().detach().cpu().numpy().transpose(1,2,0)))['image']
            if self.with_patches:
                # print(image.dtype)
                if self.type_split=='random':
                    patches_image,boxes = split_patch(image,self.num_patches,self.patch_size)
                    patches_mask,_ = split_patch(mask,self.num_patches,self.patch_size,boxes)
                    patches_edge,_ = split_patch(edge,self.num_patches,self.patch_size,boxes)
                else:
                    h_i,w_i=image.shape[-2:]
                    condition=int(h_i>w_i)
                    num_patch=(21+condition,21+(1-condition))

                    patches_image,_ = extract_patches_with_target_count(image,self.patch_size,num_patch)
                    if len(mask.shape)!=0:mask=mask.unsqueeze(0)
                    patches_mask,_ = extract_patches_with_target_count(mask,self.patch_size,num_patch)
                    patches_edge,_ = extract_patches_with_target_count(edge,self.patch_size,num_patch)
                return {
                    'image':patches_image,
                    'mask':patches_mask.long().squeeze(),
                    'edge':patches_edge
                }

        else:
            raise Exception('img_transforms is compulsory for dataset class')
        
        return {
            'image':image,
            'mask':mask.squeeze(),
            'edge':edge
        }
    
class CustomTestDataset(Dataset):
    def __init__(self,root_path,img_transforms=None,type_split='random'):
        self.image_paths =  sorted(glob.glob(root_path + '/images/*.jpg')+glob.glob(root_path + '/images/*.tif')\
                            + glob.glob(root_path + '/images/*.ppm'))
        self.mask_paths = sorted(glob.glob(root_path + '/mask/*.png')+glob.glob(root_path + '/mask/*.tif')\
                            + glob.glob(root_path + '/mask/*.ppm')+glob.glob(root_path + '/mask/*.gif'))

        self.image_transforms = img_transforms
        self.name = root_path.split('/')[-2]
        self.type_split=type_split
    def get_name(self):
        return self.name
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self,index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        image = preprocessing_img(image_path)
        mask = mask = np.array(Image.open(mask_path),dtype=np.uint8)
        if (len(mask.shape)==3):mask=mask[:,:,0]
        mask = np.ceil(mask/255).astype(np.uint8)
        if self.image_transforms:
            t = self.image_transforms(image = image,mask=mask)
            image = t['image']
            mask  = t['mask']
            if self.type_split=='random':
                image=mirror_padding_v2(image)
            edge = ToTensorV2()(image=sobel_transform(image.clone().detach().cpu().numpy().transpose(1,2,0)))['image']
        else:
            raise Exception('img_transforms is compulsory for dataset class')
        
        return {
            'image':image,
            'mask':mask.squeeze(),
            'edge':edge,
        }

