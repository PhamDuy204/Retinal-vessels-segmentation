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

def apply_gamma_correction(orimage, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(np.array(orimage.copy(), dtype = np.uint8), table)

def preprocessing_img(path):
    mean_=73.00342685729963
    std_=54.45611922239714
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))

    img=np.array(Image.open(path).convert('RGB'))
    gray=convert_gray(img)
    gray=(gray-mean_)/std_
    gray=((gray-np.min(gray))/(np.max(gray)-np.min(gray)))*255
    
    gray=clahe.apply(np.array(gray,dtype=np.uint8))
    return unsharp_mask(apply_gamma_correction(gray,1.2))

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
    # alway convert to even shape
    if len(image.shape)<3:
        image.unsqueeze(0)
    H,W=image.shape[-2:]
    image = F.pad(image,(0,int(W%2),0,int(H%2)),mode='reflect')
    return image

def mirror_padding_v2(image):
    # alway convert to 2^n*k shape
    shapes=np.array([448,512,640,768,896,1024,1152])
    if len(image.shape)<3:
        image.unsqueeze(0)
    H,W=image.shape[-2:]
    if H not in shapes:
        new_h_shape=shapes[shapes>=H][0]
    else:
        new_h_shape=H
    if W not in shapes:
        new_w_shape=shapes[shapes>=W][0]
    else:
        new_w_shape=W

    image = F.pad(image,(0,new_w_shape-W,0,new_h_shape-H),mode='reflect')
    return image

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def extract_patches_with_target_count(img, patch_size, target_patches_per_dim):
    while len(img.shape)<4:
        img=img.unsqueeze(0)

    if isinstance(patch_size, int):
        ph, pw = patch_size, patch_size
    else:
        ph, pw = patch_size

    _, _, H, W = img.shape
    sh = (H - ph) // (target_patches_per_dim[0] - 1) if target_patches_per_dim[0] > 1 else H
    sw = (W - pw) // (target_patches_per_dim[1] - 1) if target_patches_per_dim[1] > 1 else W

    patches = kornia.contrib.extract_tensor_patches(img, (ph, pw), stride=(sh, sw),allow_auto_padding=True).flatten(0,1)
    return patches, (sh, sw)
def reverse_to_original_image(patches, original_size,patch_size,stride):
    original_image = kornia.contrib.combine_tensor_patches(patches, original_size=original_size,window_size=patch_size,stride=stride,allow_auto_unpadding=True)
    return original_image

   