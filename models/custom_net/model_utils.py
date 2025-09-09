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