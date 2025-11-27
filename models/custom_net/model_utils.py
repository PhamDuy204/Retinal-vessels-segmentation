import torch 
import torch.nn as nn 
import numpy as np 

class OtsuBinarize(nn.Module):
    def __init__(self):
        super(OtsuBinarize, self).__init__()

    def otsu_threshold(self, gray):
        """
        Computes Otsu's threshold for a grayscale image (numpy array, uint8, 0-255).
        Returns the threshold value (int, 0-255).
        """
        pixel_number = gray.shape[0] * gray.shape[1]
        mean_weight = 1.0 / pixel_number
        his, bins = np.histogram(gray, np.arange(0, 257))
        final_thresh = -1
        final_value = -1
        intensity_arr = np.arange(256)
        for t in bins[1:-1]:  # From 1 to 255
            pcb = np.sum(his[:t])
            pcf = np.sum(his[t:])
            if pcb == 0 or pcf == 0:
                continue
            Wb = pcb * mean_weight
            Wf = pcf * mean_weight
            mub = np.sum(intensity_arr[:t] * his[:t]) / float(pcb) if pcb > 0 else 0
            muf = np.sum(intensity_arr[t:] * his[t:]) / float(pcf) if pcf > 0 else 0
            value = Wb * Wf * (mub - muf) ** 2
            if value > final_value:
                final_thresh = t
                final_value = value
        if final_thresh == -1:
            final_thresh = 0  # Fallback if no threshold found (e.g., uniform image)
        return final_thresh

    def forward(self, logits):
        """
        Takes model logits (torch.Tensor, shape [B, 1, H, W]), applies sigmoid to get probabilities,
        then uses Otsu to find per-image threshold and binarize.
        During training, returns sigmoid output to ensure differentiability.
        During inference, returns binarized output using Otsu threshold.
        """
        probs = torch.sigmoid(logits)  # [B, 1, H, W]
        if self.training:
            return probs  # Return sigmoid output during training for differentiability
        else:
            binary = torch.zeros_like(probs)
            for i in range(probs.size(0)):
                img = probs[i, 0].detach().cpu().numpy()  # [H, W] float [0,1]
                gray = (img * 255).astype(np.uint8)  # Scale to 0-255 uint8
                thresh = self.otsu_threshold(gray) / 255.0  # Normalize threshold back to [0,1]
                binary[i, 0] = (probs[i, 0] > thresh).float()
            return binary.to(logits.device)