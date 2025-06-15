import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import numpy as np
import glob
from torch.utils.data import Dataset
from utils import *
from albumentations.pytorch import ToTensorV2

class CustomDataset(Dataset):
    def __init__(self,root_path,img_transforms=None):
        self.image_paths =  sorted(glob.glob(root_path + '/images/*.jpg')+glob.glob(root_path + '/images/*.tif')\
                            + glob.glob(root_path + '/images/*.ppm'))
        self.mask_paths = sorted(glob.glob(root_path + '/mask/*.png')+glob.glob(root_path + '/mask/*.tif')\
                            + glob.glob(root_path + '/mask/*.ppm')+glob.glob(root_path + '/mask/*.gif'))

        self.image_transforms = img_transforms
        self.name = root_path.split('/')[-3]
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
        else:
            raise Exception('img_transforms is compulsory for dataset class')
        return {
            'image':image,
            'mask':mask,
            'edge':edge
        }
