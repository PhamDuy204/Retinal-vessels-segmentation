# Dual-Branch U-Net for Image Segmentation

A PyTorch implementation of a dual-branch U-Net architecture that combines depthwise separable convolutions with dynamic convolutions for efficient and adaptive image segmentation.

## Architecture Overview

The Dual-Branch U-Net introduces several key innovations:

1. **Dual-Branch Encoder**: Each encoder block processes features through two parallel paths:
   - **Depthwise Separable Convolution**: Efficient spatial feature extraction
   - **Dynamic Convolution**: Content-adaptive kernel selection

2. **Attention-Guided Decoder**: Uses attention mechanisms to selectively fuse encoder skip connections with upsampled decoder features

3. **Dynamic Layer Creation**: Automatically adapts to varying channel dimensions throughout the network

## Key Features

- **Efficiency**: Depthwise separable convolutions reduce computational cost
- **Adaptability**: Dynamic convolutions adapt to input content
- **Robustness**: Attention mechanisms improve feature fusion
- **Flexibility**: Dynamic layer creation handles varying tensor dimensions

## Network Architecture

```
Input (3 channels) → Dual-Branch U-Net → Output (1 channel segmentation mask)

Encoder Path:
├── EncoderBlock1: 3 → 64    (H/2 × W/2)
├── EncoderBlock2: 64 → 128  (H/4 × W/4)
├── EncoderBlock3: 128 → 256 (H/8 × W/8)
├── EncoderBlock4: 256 → 512 (H/16 × W/16)
└── Bottleneck: 512 → 1024   (H/32 × W/32)

Decoder Path:
├── DecoderBlock1: 512 + 1024 → 512 (H/16 × W/16)
├── DecoderBlock2: 256 + 512 → 256  (H/8 × W/8)
├── DecoderBlock3: 128 + 256 → 128  (H/4 × W/4)
└── DecoderBlock4: 64 + 128 → 64    (H/2 × W/2)

Final: 64 → 1 (Segmentation Mask)
```

## File Structure

```
test-model/
├── dual_net.py          # Main U-Net architecture
├── modules.py           # Encoder and Decoder blocks
├── attention.py         # Attention mechanism for decoder
├── depthwise.py         # Depthwise separable convolution
├── dyconv.py           # Dynamic convolution implementation
├── test_model.py       # Model testing script
├── requirements.txt    # Python dependencies
└── README.md          # This documentation
```

## Module Descriptions

### 1. `dual_net.py` - Main Architecture
- **SegModel**: Complete dual-branch U-Net implementation
- Manages encoder-decoder structure with skip connections
- Handles progressive channel scaling (64→128→256→512→1024)

### 2. `modules.py` - Core Building Blocks

#### EncoderBlock
- Dual-branch processing with depthwise and dynamic convolutions
- Feature fusion through concatenation and 1x1 convolution
- Downsampling for multi-scale feature extraction

#### DecoderBlock
- Attention-guided feature fusion
- Dynamic convolution layer creation
- Progressive upsampling with skip connections

### 3. `attention.py` - Attention Mechanism
- Dynamic channel alignment for tensors with different dimensions
- Spatial dimension matching through interpolation
- Feature-wise attention weighting for improved fusion

### 4. `depthwise.py` - Efficient Convolution
- Depthwise separable convolution implementation
- Reduces parameters and computational cost
- Maintains spatial feature extraction capability

### 5. `dyconv.py` - Adaptive Convolution

#### AttentionBlock
- Generates attention weights for kernel selection
- Temperature-controlled softmax for curriculum learning
- Global context extraction via adaptive pooling

#### Dynamic_conv2d
- Multiple parallel convolution kernels
- Content-based attention for kernel weighting
- Efficient grouped convolution implementation

## Usage

### Basic Usage

```python
import torch
from dual_net import SegModel

# Initialize model
model = SegModel(in_channels=3, out_channels=64)

# Create sample input (batch_size=1, channels=3, height=256, width=256)
input_tensor = torch.rand(1, 3, 256, 256)

# Forward pass
with torch.no_grad():
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")  # Expected: [1, 1, ~254, ~254]
```

### Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from dual_net import SegModel

# Initialize model and training components
model = SegModel(in_channels=3, out_channels=64)
criterion = nn.BCELoss()  # Binary cross-entropy for segmentation
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop (pseudo-code)
model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the test script:
   ```bash
   python test_model.py
   ```

## Requirements

- PyTorch >= 1.0
- torchvision
- numpy
- opencv-python (optional, for image preprocessing)

## Key Innovations

### 1. Dual-Branch Encoder
Combines the efficiency of depthwise separable convolutions with the adaptability of dynamic convolutions, providing both computational efficiency and feature richness.

### 2. Dynamic Layer Creation
Automatically creates convolution layers based on actual tensor dimensions, handling the complexity of varying channel sizes throughout the network.

### 3. Attention-Guided Fusion
Uses attention mechanisms to selectively combine encoder and decoder features, improving segmentation accuracy by focusing on relevant features.

### 4. Temperature-Controlled Learning
Implements curriculum learning in dynamic convolution through temperature scheduling, starting with soft attention and gradually becoming more focused.

## Model Parameters

With default settings (out_channels=64):
- Input channels: 3 (RGB)
- Base channels: 64
- Maximum channels: 1024 (at bottleneck)
- Dynamic kernels per layer: 4
- Attention temperature: Initially 1, decreases during training

## Performance Considerations

- **Memory**: Dynamic layer creation may increase memory usage during training
- **Computation**: Dual-branch processing requires more computation than standard U-Net
- **Flexibility**: Dynamic adaptation to tensor dimensions provides robustness across different input sizes

## Future Improvements

1. **Batch Normalization**: Add batch normalization to encoder blocks
2. **Dropout**: Include dropout layers for regularization
3. **Multi-Scale Loss**: Implement deep supervision with multi-scale losses
4. **Kernel Size Diversity**: Use different kernel sizes in dynamic convolution
5. **Channel Attention**: Add channel attention mechanisms

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- U-Net architecture: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Dynamic Convolution: Chen et al., "Dynamic Convolution: Attention over Convolution Kernels"
- Depthwise Separable Convolution: Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"

## Authorship 
- This code was implemented by TanSang and documented by Github Copilot. 