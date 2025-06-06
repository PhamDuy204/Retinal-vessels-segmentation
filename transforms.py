import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_train_transforms():
    return A.Compose([
        A.Resize(512,512,interpolation=cv2.INTER_AREA),
        A.Normalize(mean=(0.5),std=(0.5)),
        ToTensorV2()
    ])

def get_test_transforms():
    return A.Compose([
        A.Resize(512,512,interpolation=cv2.INTER_AREA),
        A.Normalize(mean=(0.5),std=(0.5)),
        ToTensorV2()
    ])    