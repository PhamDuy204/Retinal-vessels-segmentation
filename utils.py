import cv2
import numpy as np
import torch
import math
import torch.nn.functional as F
import inspect
from PIL import Image
import kornia
from scipy.signal import wiener
import pywt

def convert_gray(image,weigh=np.array([0.299, 0.587, 0.114])):
    image=image.astype(np.float64)
    gray_img = image*weigh
    return np.sum(gray_img,-1)

def unsharp_mask(image, ksize=(5,5), sigma=1.0, amount=1.0):
    blur = cv2.GaussianBlur(image, ksize, sigma)
    mask = cv2.subtract(image, blur)
    return cv2.addWeighted(image, 1.0, mask, amount, 0).clip(0,255)

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
    r,g,b=img.transpose(2,0,1)
    new_r=wiener(apply_gamma_correction(r,2).astype(np.uint8).astype(np.float32),12)
    new_g=cv2.createCLAHE(8,(12,12)).apply(g.astype(np.uint8))
    new_b=wiener(cv2.createCLAHE(6,(12,12)).apply(b).astype(np.float32),12)
    new_img = np.array([new_r,new_g,new_b]).transpose(1,2,0).astype(np.uint8)
    out=convert_gray(new_img).clip(0,255).astype(np.uint8)
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

   