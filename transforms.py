import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_train_transforms():
    return A.Compose([
        A.Resize(512,512,interpolation=cv2.INTER_AREA),
        A.Rotate(limit=45, p=0.7),
        A.HorizontalFlip(p=0.7),
        A.VerticalFlip(p=0.7),
        
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0)
        ], p=0.5),
        A.Normalize(mean=(0.5,),std=(0.5,)),
        ToTensorV2()
    ])
def get_test_transforms():
    return A.Compose([
        A.Resize(512,512,interpolation=cv2.INTER_AREA),
        A.Normalize(mean=(0.5,),std=(0.5,)),
        ToTensorV2()
    ])

def get_train_patch_transforms():
    return A.Compose([
        A.Rotate(limit=45, p=0.7),
        A.HorizontalFlip(p=0.7),
        A.VerticalFlip(p=0.7),
        
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0)
        ], p=0.5),
        A.Normalize(mean=(0,),std=(1,)),
        ToTensorV2()
    ])

def get_test_patch_transforms():
    return A.Compose([
        A.Normalize(mean=(0,),std=(1,)),
        ToTensorV2()
    ])

 