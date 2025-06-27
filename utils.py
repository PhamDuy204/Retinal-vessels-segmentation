import cv2
import numpy as np
import torch
import math
import torch.nn.functional as F
import inspect
from PIL import Image
import kornia

def convert_gray(image,weigh=np.array([0.299, 0.587, 0.114])):
    image=image.astype(np.float64)
    gray_img = image*weigh
    return np.sum(gray_img,-1)

def sobel_transform(image):
    blur_img=cv2.GaussianBlur(image,(5,5),1)
    sb_x =np.abs(cv2.Sobel(blur_img,-1,1,0))
    sb_y =np.abs(cv2.Sobel(blur_img,-1,0,1))
    sb = (sb_x+sb_y)/2
    return sb

def apply_gamma_correction(image, gamma=1.0):
    image_normalized = image / 255.0
    gamma_corrected = np.power(image_normalized, gamma)
    gamma_corrected = np.uint8(gamma_corrected * 255)

    return gamma_corrected

def preprocessing_img(path):
    img=np.array(Image.open(path).convert('RGB'),dtype=np.uint8)
    gray_img = convert_gray(img).astype(np.uint8)
    clahe = cv2.createCLAHE(2,(8,8))
    clahe_img = clahe.apply(gray_img)
    out = apply_gamma_correction(clahe_img,1.5)
    return out

def get_small_vessel(mask,kernel=7):
    if type(mask) is not torch.Tensor:
        mask = torch.tensor(mask)
    mask = mask.type(torch.float32)
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0)
    floor = math.floor((kernel-1)/2)
    ceil = math.ceil((kernel-1)/2)
    pad_mask = F.pad(mask,(floor,ceil,floor,ceil))
    mean_filter = F.conv2d(pad_mask,torch.ones(1,1,kernel,kernel)/(kernel**2)).squeeze()
    mask=mask.squeeze()
    return torch.where(mean_filter<0.5,1.,0.)*mask

def compute_enahnce_img(img,mask,kernel=7):
    cp_img = img.clone().detach()
    small_vessel = get_small_vessel(mask,kernel)
    fill_value  = torch.sum((mask-small_vessel)*cp_img)/(torch.sum((mask-small_vessel))*3)
    return cp_img*(1-small_vessel) + small_vessel*fill_value

def check_model_forward_args(model):
    forward_fn = model.forward
    sig = inspect.signature(forward_fn)

    num_params = len(sig.parameters) - 1
    return num_params

def split_patch(image,num_patches=1000,size=64,boxes=None):
    if len(image.shape)<3:
        image.unsqueeze(0)
    H,W=image.shape[-2:]
    half_size = size//2
    image =image.type(torch.float32)
    x = torch.randint(half_size,W-half_size-1,(num_patches,))
    y = torch.randint(half_size,H-half_size-1,(num_patches,))

    x1 = (x - half_size)
    y1 = (y - half_size)
    x2 = (x + half_size)
    y2 = (y + half_size)
    if boxes is None :
        boxes = torch.stack([y1, x1, y2, x1, y2, x2,y1, x2], dim=1).unfold(-1,2,2).type(torch.float32)
    patches = kornia.geometry.transform.crop_and_resize(image.unsqueeze(0).repeat(num_patches,1,1,1), boxes, size=(size, size))
    return patches.squeeze(0),boxes

def mirror_padding(image):
    if len(image.shape)<3:
        image.unsqueeze(0)
    H,W=image.shape[-2:]
    image = F.pad(image,(0,int(H%2),0,int(W%2)),mode='reflect')
    return image

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

   