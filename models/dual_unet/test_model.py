"""
Test Script for Dual-Branch U-Net Model

This script provides a basic test to verify that the Dual-Branch U-Net model
can be instantiated and perform forward pass without errors. It serves as
a quick validation of the model architecture and can be used as a starting
point for more comprehensive testing.

The test creates a random input tensor simulating an RGB image and passes it
through the model to ensure all components work together correctly.

Usage:
    python test_model.py

Expected Output:
    - Model initialization confirmation
    - Input tensor shape
    - Output tensor shape
    - Success message

Example Output:
    Starting model
    Model started successfully
    Input shape: torch.Size([3, 3, 256, 256])
    Output shape: torch.Size([3, 1, 254, 254])
    Model executed successfully!
"""

import torch 
from dual_net import SegModel

print("Starting model")
model = SegModel(3, 1)
print("Model started successfully")

# Create test input: batch_size=3, channels=3 (RGB), height=256, width=256
input_tensor = torch.rand(3, 3, 256, 256)

print(f"Input shape: {input_tensor.shape}")

# Perform forward pass without gradient computation
with torch.no_grad(): 
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    print("Model executed successfully!")