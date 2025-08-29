import numpy as np 
import cv2 
import torch 
import torch.nn as nn 


class Postprocessing(nn.Module): 
    def __init__(self): 
        super(Postprocessing, self).__init__() 
        pass 
    

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # lưu thông tin gốc
        original_device = X.device
        original_dtype = X.dtype

        # (N,1,H,W) -> (N,H,W)
        X_np = X.detach().cpu().numpy()
        arr = np.squeeze(X_np, axis=1)  # shape (N,H,W), float in [0,1]

        # to uint8 [0,255]
        arr_uint8 = (arr * 255.0).round().astype(np.uint8)
        N, H, W = arr_uint8.shape
        total_pix = H * W

        # --- compute per-image histograms in one shot using bincount+offset trick ---
        # flatten values and add offsets so that bincount groups by (image_index*256 + value)
        flat_vals = arr_uint8.ravel().astype(np.int64)
        # repeat offsets: for image i, repeat i*256 exactly H*W times
        offsets = np.repeat(np.arange(N, dtype=np.int64) * 256, total_pix)
        combined = flat_vals + offsets  # each entry in [0, N*256-1]
        hist = np.bincount(combined, minlength=N * 256).reshape(N, 256).astype(np.float64)
        # hist.shape == (N,256)

        # --- Otsu computation (vectorized across N) ---
        bins = np.arange(256, dtype=np.float64)
        # probabilities (or normalized histogram) not required strictly, we can work with counts and divide by total_pix where needed
        prob = hist / float(total_pix)  # (N,256)

        # cumulative sums along intensity axis
        omega1 = np.cumsum(prob, axis=1)  # weight of class 1 for thresholds t = 0..255
        mu1_cumsum = np.cumsum(prob * bins, axis=1)  # cumulative first moment
        # total mean per image
        mu_total = mu1_cumsum[:, -1][:, None]  # shape (N,1)

        # avoid division by zero; when omega1==0 or ==1 we will set between var to 0
        eps = 1e-12
        mu1 = mu1_cumsum / (omega1 + eps)  # shape (N,256)
        omega2 = 1.0 - omega1
        mu2 = (mu_total - mu1_cumsum) / (omega2 + eps)

        # between-class variance
        var_between = omega1 * omega2 * (mu1 - mu2) ** 2  # (N,256)

        # mask out invalid thresholds where omega1==0 or omega2==0
        invalid = (omega1 <= eps) | (omega2 <= eps)
        var_between[invalid] = 0.0

        # find threshold index (0..255) that maximizes var_between for each image
        thresholds = np.argmax(var_between, axis=1).astype(np.uint8)  # (N,)

        # --- build masks by broadcasting thresholds and comparing ---
        # thresholds[:,None,None] broadcast to (N,H,W)
        masks_uint8 = (arr_uint8 > thresholds[:, None, None]).astype(np.uint8) * 255  # 0/255

        # to tensor shape (N,1,H,W)
        output_tensor = torch.from_numpy(masks_uint8).unsqueeze(1)

        # return to original device and dtype, normalized to {0.0,1.0}
        output_tensor = output_tensor.to(original_device)
        output_tensor = output_tensor.to(original_dtype)
        return output_tensor / 255.0
