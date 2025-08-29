import torch
import torch.nn as nn

class Postprocessing(nn.Module):
    """
    Otsu's thresholding chạy song song trên GPU bằng PyTorch.
    Input: Tensor (N, 1, H, W) với giá trị trong khoảng [0, 1].
    Output: Tensor (N, 1, H, W) nhị phân {0.0, 1.0}.
    """
    def __init__(self, num_bins: int = 256):
        super(Postprocessing, self).__init__()
        self.num_bins = num_bins

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        device = X.device
        dtype = X.dtype
        N, _, H, W = X.shape

        # Scale về [0, num_bins-1] (giả lập uint8)
        X_scaled = (X * (self.num_bins - 1)).long().view(N, -1)  # (N, H*W)

        # Tính histogram cho từng ảnh trong batch
        # hist[i, :] = histogram của ảnh i
        hist = torch.stack([
            torch.bincount(X_scaled[i], minlength=self.num_bins)
            for i in range(N)
        ], dim=0).to(device=device, dtype=torch.float32)  # (N, 256)

        # Tổng số pixel
        total = hist.sum(dim=1, keepdim=True)  # (N, 1)

        # Xác suất cho từng mức xám
        prob = hist / total  # (N, 256)

        # Cumulative sum cho trọng số và mean
        omega = torch.cumsum(prob, dim=1)  # (N, 256)
        mu = torch.cumsum(prob * torch.arange(self.num_bins, device=device), dim=1)  # (N, 256)

        # Tổng mean (global mean)
        mu_total = mu[:, -1].unsqueeze(1)  # (N, 1)

        # Between-class variance
        sigma_b2 = (mu_total * omega - mu)**2 / (omega * (1 - omega) + 1e-8)  # (N, 256)

        # Tìm ngưỡng tốt nhất cho từng ảnh
        thresholds = torch.argmax(sigma_b2, dim=1)  # (N,)

        # Áp dụng threshold để tạo mask
        X_uint8 = (X * 255).to(torch.uint8)
        thresholds = thresholds.view(N, 1, 1, 1).to(X_uint8.device)
        masks = (X_uint8 >= thresholds).to(dtype)  # (N, 1, H, W)

        return masks
