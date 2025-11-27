
import torch 
from models.dual_unet.dual_unet import SegModel

print("Starting model")
model = SegModel(1, 1)
print("Model started successfully")

# Create test input: batch_size=3, channels=3 (RGB), height=256, width=256
input_tensor = torch.rand(4, 1, 256, 256)

print(f"Input shape: {input_tensor.shape}")

# Perform forward pass without gradient computation
with torch.no_grad(): 
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    print("Model executed successfully!")