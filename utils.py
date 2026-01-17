import cv2
import numpy as np
import torch
import math
import torch.nn.functional as F
import inspect
from PIL import Image
import kornia
import torch.nn as nn
from io import BytesIO

def init_weights_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_weights_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    
def convert_gray(image,weigh=np.array([0.299, 0.587, 0.114])):
    image=image.astype(np.float64)
    gray_img = image*weigh
    return np.sum(gray_img,-1)

def unsharp_mask(image, ksize=(7,7), sigma=1.0, amount=2):
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
    if isinstance(path,str):
        img=np.array(Image.open(path).convert('RGB'))
    else:
        img=path

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))

    gray=convert_gray(img)
    gray=(gray-mean_)/std_
    gray=((gray-np.min(gray))/(np.max(gray)-np.min(gray)))*255
    
    gray=clahe.apply(np.array(gray,dtype=np.uint8))
    return unsharp_mask(gray)

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

    patches = kornia.contrib.extract_tensor_patches(img, (ph, pw), stride=(sh, sw),allow_auto_padding=False).flatten(0,1)
    return patches, (sh, sw)
def reverse_to_original_image(patches, original_size,patch_size,stride):
    original_image = kornia.contrib.combine_tensor_patches(patches, original_size=original_size,window_size=patch_size,stride=stride,allow_auto_unpadding=False)
    return original_image


def create_error_map(pred_mask, gt_mask):
    """
    pred_mask, gt_mask: (H, W) uint8 {0,1}
    return: (H, W, 3) uint8 RGB
    """
    h, w = pred_mask.shape
    error_map = np.zeros((h, w, 3), dtype=np.uint8)

    # True Positive (Green)
    tp = (pred_mask == 1) & (gt_mask == 1)
    error_map[tp] = [0, 255, 0]

    # False Positive (Red)
    fp = (pred_mask == 1) & (gt_mask == 0)
    error_map[fp] = [255, 0, 0]

    # False Negative (Blue)
    fn = (pred_mask == 0) & (gt_mask == 1)
    error_map[fn] = [0, 0, 255]

    # True Negative giữ màu đen
    return error_map


def overlay_error_map(image, error_map, alpha=0.6):
    """
    image: (H,W,3) uint8
    error_map: (H,W,3) uint8
    """
    overlay = image.copy()
    mask = np.any(error_map != 0, axis=-1)

    overlay[mask] = (
        (1 - alpha) * overlay[mask] +
        alpha * error_map[mask]
    ).astype(np.uint8)

    return overlay

def save_png_to_bytes(img):
    """
    img: np.ndarray (H,W) or (H,W,3) uint8
         or PIL.Image
    return: bytes PNG
    """
    buf = BytesIO()
    if isinstance(img, np.ndarray):
        Image.fromarray(img).save(buf, format="PNG")
    else:
        img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def create_zoom_inset(image_path, output_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # 1. Xác định vị trí vùng muốn cắt (x, y, width, height)
    # Bạn hãy điều chỉnh các con số này theo ý muốn
    crop_x, crop_y, crop_w, crop_h = 243, 419, 48, 30 
    
    # 2. Cắt vùng đó ra
    roi = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

    # 3. Phóng to vùng đã cắt (ví dụ phóng lên 2.5 lần)
    zoom_scale = 4
    zoomed_roi = cv2.resize(roi, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_LANCZOS4)

    # 4. Vẽ khung vàng cho vùng cắt trên ảnh gốc
    cv2.rectangle(img, (crop_x, crop_y), (crop_x+crop_w, crop_y+crop_h), (0, 255, 0), 2)

    # 5. Xác định vị trí đặt ảnh đã zoom (ví dụ: góc dưới bên phải)
    zh, zw = zoomed_roi.shape[:2]
    pos_x, pos_y = w - zw , h - zh# cách lề 20px
    
    # Vẽ khung vàng cho ảnh zoom
    cv2.rectangle(zoomed_roi, (0, 0), (zw-1, zh-1), (0, 255, 0), 3)

    # 6. Đè ảnh zoom lên ảnh gốc
    img[pos_y:pos_y+zh, pos_x:pos_x+zw] = zoomed_roi

    # (Tùy chọn) Vẽ đường gạch đứt nối giữa vùng gốc và vùng zoom
    # cv2.line(img, (crop_x+crop_w, crop_y+crop_h), (pos_x, pos_y), (0, 255, 255), 1)

    cv2.imwrite(output_path, img)

# # Chạy thử
# create_zoom_inset("anh_vong_mac_1.jpg", "ket_qua_1.jpg")