import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import numpy as np
import glob
from torch.utils.data import Dataset
from utils import *
from albumentations.pytorch import ToTensorV2
import kornia.augmentation as K
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
# import torchvision

import random

# class RandomResizedCropBoth:
#     def __init__(self, size, scale=(0.08, 1.0), ratio=(3/4, 4/3),p=0.5):
#         self.size = size
#         self.scale = scale
#         self.ratio = ratio
#         self.p=p

#     def __call__(self, image, mask,edge=None):
#         if np.random.rand()>self.p:
#             i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(
#                 image, scale=self.scale, ratio=self.ratio
#             )
#             image = F.resized_crop(image, i, j, h, w, self.size, InterpolationMode.BILINEAR)
#             mask  = F.resized_crop(mask,  i, j, h, w, self.size, InterpolationMode.NEAREST)
#             edge  = F.resized_crop(edge,  i, j, h, w, self.size, InterpolationMode.NEAREST)
#         return image, mask,edge
    
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
        self.index_patch=[[]for _ in range(len(self.image_paths))]
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
            edge = ToTensorV2()(image=sobel_transform(image.clone().detach().permute(1,2,0).mean(-1).unsqueeze(-1).cpu().numpy()))['image']
            if self.with_patches:
                # print(image.dtype)
                if self.type_split=='random':
                    patches_image,boxes = split_patch(image,self.num_patches,self.patch_size)
                    patches_mask,_ = split_patch(mask,self.num_patches,self.patch_size,boxes)
                    patches_edge,_ = split_patch(edge,self.num_patches,self.patch_size,boxes)
                else:
                    # print(image.shape)
                    image=mirror_padding_v2(image)
                    h,w=image.shape[-2:]
                    # print(image.shape)
                    # print(mask.shape)
                    if len(mask.shape)<3:mask=mask.unsqueeze(0)
                    mask=mirror_padding_v2(mask)
                    # print(mask.shape)
                    edge=mirror_padding_v2(edge)
                    # print(edge.shape)
                    # h_i,w_i=image.shape[-2:]
                    # condition=int(h_i>w_i)
                    num_patch=((h-self.patch_size)//8+1,(w-self.patch_size)//8+1)

                    patches_image,_ = extract_patches_with_target_count(image,self.patch_size,num_patch)
                    patches_mask,_ = extract_patches_with_target_count(mask,self.patch_size,num_patch)
            
                    patches_edge,_ = extract_patches_with_target_count(edge,self.patch_size,num_patch)

                    filter_=patches_mask.sum((-1,-2))>=1
                    # print(filter.shape)
                    patches_image=patches_image.unsqueeze(1)[filter_]
                    # print(patches_image.shape)
                    patches_mask=patches_mask[filter_]
                    patches_edge=patches_edge[filter_].unsqueeze(1)
                aug = K.AugmentationSequential(
                        K.RandomCrop((self.patch_size-self.patch_size//4, self.patch_size-self.patch_size//4), same_on_batch=True,p=0.5),
                        K.PadTo((self.patch_size, self.patch_size)),   # padding để khôi phục shape gốc
                        data_keys=["input", "mask"]
                    ).cuda()
                num_sample =torch.randperm(len(patches_image))[:2000]
                patches_image=patches_image[num_sample]
                patches_mask=patches_mask.long()[num_sample]
                
                if len(patches_mask.shape)<4:
                    patches_mask=patches_mask.unsqueeze(1)
                patches_edge=patches_edge[num_sample]
                patches_image,patches_mask=aug(patches_image.cuda(),patches_mask.cuda())
                return {
                    'image':patches_image,
                    'mask':patches_mask.squeeze(),
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

