# ESAM2-BLS: PyTorch Implementation

This repository provides a PyTorch implementation of the **ESAM2-BLS (Enhanced Segment Anything Model 2 for Breast Lesion Segmentation)** model. The model architecture is based on the research paper: "ESAM2-BLS: Enhanced Segment Anything Model 2 for Efficient Breast Lesion Segmentation in Ultrasound Imaging". 

ESAM2-BLS is designed for the efficient and accurate segmentation of breast lesions in ultrasound images, a critical task for early breast cancer detection and diagnosis. It enhances the SAM2 (Segment Anything Model 2) architecture by incorporating specialized modules tailored for the unique characteristics of medical ultrasound imaging. 

## Model Architecture Overview

The ESAM2-BLS model leverages the strong visual understanding capabilities of the pre-trained SAM2 foundation model and introduces specific adaptations to address the challenges of ultrasound imaging, such as speckle noise, low contrast, and shadowing artifacts. 

The main components of the ESAM2-BLS architecture are:

1.  **Adapter-based Hiera Encoder**:
    *   Built upon SAM2's pre-trained Hiera Block for hierarchical feature extraction. 
    *   Integrates specialized **adapter modules** at each encoder layer. These adapters use channel attention, depthwise separable convolutions, and skip connections to fine-tune the model efficiently for ultrasound-specific image characteristics with minimal additional parameters. 

2.  **Multi-Scale Information Fusion (MSIF) BottleNeck Block**:
    *   A lightweight module that processes and fuses multi-scale features from the encoder. 
    *   Employs **Axial Dilated Depthwise Convolution (ADConv)** with varying dilation rates ($\rho = 1, 2, 3$) to capture details of lesions of different sizes and complexities efficiently. 

3.  **Lightweight Decoder**:
    *   Adopts a U-shaped structure, deviating from SAM2's original computationally intensive decoder. 
    *   Uses **depthwise separable convolutions** for feature processing. 
    *   Employs **bilinear interpolation** for upsampling, which is beneficial for noisy ultrasound images as it produces smoother feature maps and avoids artifacts. 
    *   Integrates **skip connections** from the encoder to fuse multi-scale features and recover high-frequency details. 

## Key Features

*   **Efficient Domain Adaptation**: Tailors the powerful SAM2 model to the nuances of breast ultrasound imaging using lightweight adapter modules. 
*   **Multi-Scale Feature Processing**: The MSIF BottleNeck Block and hierarchical encoder effectively capture lesion details across various scales. 
*   **Computational Efficiency**: The lightweight decoder and efficient convolution techniques make the model suitable for resource-constrained environments. 
*   **Robust Optimization**: Utilizes a joint optimization loss function combining Weighted Intersection over Union (IoU) Loss, Binary Cross-Entropy (BCE) Loss, and a Deep Supervision mechanism for improved accuracy and boundary delineation. 
*   **Advanced Training Strategy**: Employs the AdamW optimizer and a cosine annealing learning rate schedule. 

## Code Structure

The implementation is organized into several Python classes:

*   **Main Model**:
    *   `ESAM2_BLS`: The complete end-to-end model.
*   **Encoder Components**:
    *   `AdapterBasedHieraEncoder`: The main encoder structure.
    *   `HieraLayer`: A conceptual layer of the Hiera encoder.
    *   `AdapterModule`: The specialized adapter for ultrasound image characteristics.
    *   `HieraAttentionPlaceholder`: **Important:** A placeholder for SAM2's Hiera attention mechanism.
*   **MSIF BottleNeck Components**:
    *   `MSIFBottleNeckBlock`: The multi-scale information fusion bottleneck.
    *   `AxialDilatedDepthwiseConv`: The ADConv module used in MSIF and Decoder.
*   **Decoder Components**:
    *   `LightweightDecoder`: The U-shaped decoder structure.
    *   `DecoderBlock`: A single block within the decoder.
*   **Helper Modules**:
    *   `DepthwiseSeparableConv`: Efficient convolution block.
    *   `ChannelAttention`: Channel attention mechanism used in the adapter.
*   **Loss Functions**:
    *   `JointOptimizationLoss`: The complete loss function as described in the paper.
    *   `IoULoss`: Weighted Intersection over Union loss.
    *   `BCELossWrapper`: Binary Cross-Entropy loss.
    *   `DeepSupervisionLoss`: Loss for intermediate decoder layers.

## Requirements

*   Python 3.x
*   PyTorch (tested with version 3.5.2, but should be compatible with recent versions)
    *   `pip install torch torchvision torchaudio`

## Usage

### Model Instantiation

You can instantiate the `ESAM2_BLS` model with various architectural parameters. Default values are provided in the example.python
import torch
from esam2_bls_model import ESAM2_BLS # Assuming your code is in esam2_bls_model.py

# Example configuration (adjust as needed)
```python
example_img_channels = 3
example_num_classes = 1 # Binary segmentation
example_encoder_initial_channels = 32
example_encoder_stage_channels = [2, 4, 6, 8]
example_encoder_layers_per_stage = [1, 1, 1, 1]
example_encoder_pool_strides = [2, 2, 2, 2]
example_msif_bottleneck_channels_ratio = 0.5
example_msif_out_channels_ratio = 1.0
example_decoder_channels_list = [2, 4, 6, 8]
example_deep_supervision_levels = 3

model = ESAM2_BLS(
    img_channels=example_img_channels,
    num_classes=example_num_classes,
    encoder_initial_channels=example_encoder_initial_channels,
    encoder_stage_channels=example_encoder_stage_channels,
    encoder_layers_per_stage=example_encoder_layers_per_stage,
    encoder_pool_strides=example_encoder_pool_strides,
    msif_bottleneck_channels_ratio=example_msif_bottleneck_channels_ratio,
    msif_out_channels_ratio=example_msif_out_channels_ratio,
    decoder_channels_list=example_decoder_channels_list,
    deep_supervision_levels=example_deep_supervision_levels
)

if torch.cuda.is_available():
    model = model.cuda()
```

```
### Forward Pass

The model returns a list of output logits, corresponding to the predictions from different decoder layers for deep supervision. The last element in the list is the final prediction.

```python
batch_size = 2
img_size = 256
dummy_input = torch.randn(batch_size, example_img_channels, img_size, img_size)
if torch.cuda.is_available():
    dummy_input = dummy_input.cuda()

model.train() # or model.eval()
list_of_output_logits = model(dummy_input)

final_logits = list_of_output_logits[-1] # Main prediction
# Apply sigmoid for probabilities: final_probs = torch.sigmoid(final_logits)
```

### Loss Calculation

The `JointOptimizationLoss` class implements the composite loss function described in the paper.

```python
from esam2_bls_model import JointOptimizationLoss # Assuming loss functions are in the same file

# Lambda weights for JointOptimizationLoss are not specified in the paper, using examples.
# ds_weights are alpha_l from the paper for L=3 deep supervision levels.
criterion = JointOptimizationLoss(
    lambda1=1.0, lambda2=1.0, lambda3=1.0, # Example lambda values
    ds_weights=[0.2, 0.3, 0.5],
    num_ds_layers=example_deep_supervision_levels
)
if torch.cuda.is_available():
    criterion = criterion.cuda()

dummy_target = (torch.rand(batch_size, example_num_classes, img_size, img_size) > 0.5).float()
if torch.cuda.is_available():
    dummy_target = dummy_target.cuda()

loss = criterion(list_of_output_logits, dummy_target)
print(f"Calculated loss: {loss.item()}")
```

## Important Notes

*   **Hiera Attention Placeholder**: The `HieraAttentionPlaceholder` in the encoder is a simplified stand-in. For full replication of ESAM2-BLS, which is based on SAM2, this placeholder should be replaced with the actual Hiera attention mechanism from a SAM2 implementation.
*   **Pre-trained Weights**: This repository provides the model architecture. The ESAM2-BLS paper describes fine-tuning a *pre-trained* SAM2 model. Loading these pre-trained weights is a separate step not covered by this architectural code.
*   **Hyperparameters**: Many architectural hyperparameters (e.g., channel dimensions, adapter ratios, loss weighting coefficients $\lambda_1, \lambda_2, \lambda_3$) are configurable. The values in the example script are illustrative and may require tuning for optimal performance on specific datasets.
*   **SAM2 Base**: The accuracy and performance of this implementation will heavily depend on the correct integration with a base SAM2 Hiera encoder.

## Citation

If you use this model or refer to the ESAM2-BLS architecture in your research, please cite the original paper:
```bibtex
@article{2025,
  title={{ESAM2-BLS: Enhanced Segment Anything Model 2 for Efficient Breast Lesion Segmentation in Ultrasound Imaging}},
  author={X},
  journal={X},
  year={2025},
  volume={X},
  pages={X-X}
}
```