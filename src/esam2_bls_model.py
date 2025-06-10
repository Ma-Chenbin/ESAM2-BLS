import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Helper Modules ---

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution
    Described for use in the Adapter module and Lightweight Decoder.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                   padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = self.relu(x)
        return x


class AxialDilatedDepthwiseConv(nn.Module):
    """
    Axial Dilated Depthwise Convolution (ADConv)
    As described in Section 3.3 (MSIF BottleNeck Block) and used in Lightweight Decoder.
    Equations 3, 4, 5.
    """

    def __init__(self, channels, kernel_size=3, dilation_rate=1, bias=False):
        super().__init__()
        padding_h = dilation_rate * (kernel_size - 1) // 2
        padding_w = dilation_rate * (kernel_size - 1) // 2

        self.conv_horizontal = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size),
                                         padding=(0, padding_w), groups=channels,
                                         dilation=(1, dilation_rate), bias=bias)
        self.conv_vertical = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1),
                                       padding=(padding_h, 0), groups=channels,
                                       dilation=(dilation_rate, 1), bias=bias)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h = self.conv_horizontal(x)
        x_v = self.conv_vertical(x)
        x = x_h + x_v  # Equation 5
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    """
    Channel Attention Mechanism for Adapter Module (Section 3.2).
    A simple Squeeze-and-Excitation block.
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# --- Main Components ---

class AdapterModule(nn.Module):
    """
    Adapter Module for Hiera Encoder (Section 3.2, Figure 2 description, Equation 2).
    """

    def __init__(self, in_channels, adapter_intermediate_channels_ratio=0.25):
        super().__init__()
        adapter_channels = int(in_channels * adapter_intermediate_channels_ratio)
        if adapter_channels == 0: adapter_channels = 1  # Ensure at least 1 channel

        # Linear_down (using 1x1 Conv)
        self.linear_down = nn.Conv2d(in_channels, adapter_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # Components described for the adapter:
        # 1. Channel Attention
        self.channel_attention = ChannelAttention(adapter_channels)
        # 2. Depthwise Separable Convolutions (3x3 and 5x5)
        self.dwsc_3x3 = DepthwiseSeparableConv(adapter_channels, adapter_channels, kernel_size=3, padding=1)
        self.dwsc_5x5 = DepthwiseSeparableConv(adapter_channels, adapter_channels, kernel_size=5, padding=2)

        # Linear_up (using 1x1 Conv)
        self.linear_up = nn.Conv2d(adapter_channels, in_channels, kernel_size=1, bias=False)

        # Initialize weights for the up-projection to zero for identity initialization of the branch
        nn.init.zeros_(self.linear_up.weight)
        if self.linear_up.bias is not None:
            nn.init.zeros_(self.linear_up.bias)

    def forward(self, x_in):
        # Branch for adapter processing
        x = self.linear_down(x_in)
        x = self.relu(x)

        x = self.channel_attention(x)
        x_dwsc = self.dwsc_3x3(x)
        x_dwsc = self.dwsc_5x5(x_dwsc)  # Sequential application

        x = self.linear_up(x_dwsc)

        # Skip connection (Equation 2: Adapter(Xin) = Xin + Processed_Branch)
        return x_in + x


class HieraAttentionPlaceholder(nn.Module):
    """
    Placeholder for the SAM2 Hiera Block's Attention mechanism.
    A proper implementation would use a Transformer self-attention block from SAM2.
    """

    def __init__(self, dim):
        super().__init__()
        # This is a placeholder. A real implementation would use SAM2's Hiera attention.
        # For example, a simple convolutional attention or nn.Identity if not implementing full attention.
        # Using nn.Identity to signify it needs replacement.
        self.attn = nn.Identity()
        print("WARNING: HieraAttentionPlaceholder is using nn.Identity. Replace with SAM2's Hiera attention.")

    def forward(self, x):
        return self.attn(x)


class HieraLayer(nn.Module):
    """
    A single layer/stage of the Hiera Encoder (conceptual).
    Based on Equation 1: X_l = Adapter(Attention(Pooling(X_{l-1})))
    """

    def __init__(self, in_channels, out_channels, pool_kernel=2, pool_stride=2, use_adapter=True):
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride) if pool_stride > 1 else nn.Identity()
        # Assuming attention operates on pooled features. If pooling changes channels, adjust here.
        # For simplicity, assuming channels change after pooling or attention.
        # This part is highly dependent on the actual SAM2 Hiera architecture.
        # Let's assume out_channels is the channel dim for this stage.

        # Placeholder for channel adjustment if pooling doesn't do it
        self.channel_adjust = nn.Conv2d(in_channels, out_channels,
                                        kernel_size=1) if in_channels != out_channels and pool_stride == 1 else nn.Identity()

        self.attention = HieraAttentionPlaceholder(dim=out_channels)  # dim should match feature dim for attention

        self.adapter = AdapterModule(out_channels) if use_adapter else nn.Identity()

    def forward(self, x):
        x = self.pooling(x)
        if not isinstance(self.channel_adjust, nn.Identity):
            # If pooling changes spatial dim but not channels, and we need to change channels
            if x.shape[1] == self.channel_adjust.in_channels:  # type: ignore
                x = self.channel_adjust(x)
        # Ensure x has `out_channels` before attention and adapter
        # This is a simplification. Real Hiera has specific channel progression.
        if x.shape![1]= self.attention.attn.in_features if hasattr(self.attention.attn, 'in_features') else False:  # type: ignore
            # A more robust way to handle channel changes for a generic Hiera block is needed.
            # For now, we assume `out_channels` is the target for this stage.
            if x.shape![1]= self.adapter.linear_down.in_channels:  # type: ignore
                # This is a fallback, ideally channel dimensions are managed by Hiera structure
                # print(f"Adjusting channels in HieraLayer from {x.shape[1]} to {self.adapter.linear_down.in_channels}")
                temp_adjust = nn.Conv2d(x.shape[1], self.adapter.linear_down.in_channels, kernel_size=1,
                                        device=x.device)  # type: ignore
                x = temp_adjust(x)

        x = self.attention(x)
        x = self.adapter(x)
        return x


class AdapterBasedHieraEncoder(nn.Module):
    """
    Adapter-based Hiera Encoder (Section 3.2).
    Conceptually, a stack of HieraLayers.
    Outputs multi-scale features for skip connections.
    """

    def __init__(self, img_channels=3, initial_patch_embed_channels=64,
                 stage_channels=8,
                 num_layers_per_stage=[2, 2, 2, 2],
                 pool_strides=[1, 2, 2, 2]):  # First stage might not pool if patch embed does
        super().__init__()
        self.skip_features =

        # Initial patch embedding (typical for ViT-like encoders)
        # SAM2 Hiera might have a specific way of creating initial patches.
        # For simplicity, a Conv layer.
        self.patch_embed = nn.Conv2d(img_channels, initial_patch_embed_channels,
                                     kernel_size=7, stride=2, padding=3, bias=False)  # Example
        self.patch_embed_bn = nn.BatchNorm2d(initial_patch_embed_channels)
        self.patch_embed_relu = nn.ReLU(inplace=True)

        current_channels = initial_patch_embed_channels
        self.stages = nn.ModuleList()

        for i, (ch_out, num_layers, stride) in enumerate(zip(stage_channels, num_layers_per_stage, pool_strides)):
            stage_layers =
            # First layer of a stage might handle pooling/downsampling
            stage_layers.append(HieraLayer(current_channels, ch_out, pool_stride=stride if num_layers > 0 else 1))
            current_channels = ch_out
            for _ in range(num_layers - 1):
                stage_layers.append(
                    HieraLayer(current_channels, ch_out, pool_stride=1))  # No pooling within stage layers
            self.stages.append(nn.Sequential(*stage_layers))

    def forward(self, x):
        self.skip_features =
        x = self.patch_embed(x)
        x = self.patch_embed_bn(x)
        x = self.patch_embed_relu(x)
        # self.skip_features.append(x) # Optional: skip from patch_embed

        for stage in self.stages:
            x = stage(x)
            self.skip_features.append(x)

        # Return the final output of encoder and the list of skip features (reversed for decoder)
        return x, self.skip_features[::-1]


class MSIFBottleNeckBlock(nn.Module):
    """
    Multi-Scale Information Fusion (MSIF) BottleNeck Block (Section 3.3).
    Equations 6, 7, 8.
    """

    def __init__(self, in_channels, bottleneck_channels_ratio=0.5, out_channels_ratio=1.0):
        super().__init__()
        bottleneck_channels = int(in_channels * bottleneck_channels_ratio)
        out_channels_final = int(in_channels * out_channels_ratio)

        # Initial PConv for dimensionality reduction (Equation 6)
        self.pconv_initial = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn_initial = nn.BatchNorm2d(bottleneck_channels)
        self.relu_initial = nn.ReLU(inplace=True)

        # ADConv with different dilation rates (Equation 7)
        self.adconv_rho1 = AxialDilatedDepthwiseConv(bottleneck_channels, kernel_size=3, dilation_rate=1)
        self.adconv_rho2 = AxialDilatedDepthwiseConv(bottleneck_channels, kernel_size=3, dilation_rate=2)
        self.adconv_rho3 = AxialDilatedDepthwiseConv(bottleneck_channels, kernel_size=3, dilation_rate=3)

        # Final PConv for fusion (Equation 8)
        # Concatenated features will have 3 * bottleneck_channels
        self.pconv_final = nn.Conv2d(bottleneck_channels * 3, out_channels_final, kernel_size=1, bias=False)
        self.bn_final = nn.BatchNorm2d(out_channels_final)
        self.relu_final = nn.ReLU(inplace=True)

    def forward(self, x_in):
        x = self.pconv_initial(x_in)
        x = self.bn_initial(x)
        x = self.relu_initial(x)

        y1 = self.adconv_rho1(x)
        y2 = self.adconv_rho2(x)
        y3 = self.adconv_rho3(x)

        x_concat = torch.cat([y1, y2, y3], dim=1)

        x_out = self.pconv_final(x_concat)
        x_out = self.bn_final(x_out)
        x_out = self.relu_final(x_out)
        return x_out


class DecoderBlock(nn.Module):
    """
    A single block in the Lightweight Decoder (Section 3.4).
    Equations 9, 10, 11.
    """

    def __init__(self, in_channels, skip_channels, out_channels, use_depthwise_separable=True):
        super().__init__()
        self.bilinear_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Conv1x1 after concatenation (Equation 9)
        # Channels after concat: in_channels from previous decoder block + skip_channels from encoder
        self.conv1x1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # ADConv (rho=2) (Equation 10)
        self.adconv = AxialDilatedDepthwiseConv(out_channels, kernel_size=3, dilation_rate=2)

        # Final BatchNorm and GELU (Equation 11 implies GELU after ADConv's internal ReLU)
        # ADConv already has BN and ReLU. Let's add GELU as the final activation for the block.
        self.bn2 = nn.BatchNorm2d(out_channels)  # Redundant if ADConv has BN
        self.gelu = nn.GELU()

        # Alternative: use DepthwiseSeparableConv as mentioned as core component
        if use_depthwise_separable:
            self.main_conv = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1)
        else:  # Use ADConv as per Eq 10
            self.main_conv = self.adconv

    def forward(self, x_prev_decoder, x_skip_encoder):
        x = self.bilinear_upsample(x_prev_decoder)

        # Ensure spatial dimensions match for concatenation if upsampling isn't perfect
        if x.shape[2:] != x_skip_encoder.shape[2:]:
            x = F.interpolate(x, size=x_skip_encoder.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, x_skip_encoder], dim=1)

        x = self.conv1x1(x)  # Eq 9, part 1
        x = self.bn1(x)
        x = self.relu1(x)  # Eq 9, part 2 (implied activation)

        x = self.main_conv(x)  # Eq 10 (ADConv) or DWSC

        # Eq 11: GELU(BatchNorm(F2)) where F2 is output of ADConv
        # The BN here might be an additional one for the block's output.
        x = self.bn2(x)
        x = self.gelu(x)
        return x


class LightweightDecoder(nn.Module):
    """
    Lightweight Decoder with U-shaped structure (Section 3.4).
    Outputs multiple masks for deep supervision. L=3 decoder layers supervised.
    """

    def __init__(self, bottleneck_out_channels, encoder_skip_channels_list,
                 decoder_channels_list=, num_classes=1, deep_supervision_levels=3):
        super().__init__()
        self.deep_supervision_levels = deep_supervision_levels
        self.decoder_blocks = nn.ModuleList()

        current_channels = bottleneck_out_channels
        num_skip_features = len(encoder_skip_channels_list)

        for i in range(len(decoder_channels_list)):
            skip_ch = encoder_skip_channels_list[i] if i < num_skip_features else 0
            dec_ch = decoder_channels_list[i]
            self.decoder_blocks.append(
                DecoderBlock(current_channels, skip_ch, dec_ch)
            )
            current_channels = dec_ch

        # Final segmentation heads for deep supervision
        self.segmentation_heads = nn.ModuleList()
        # These heads take output from decoder blocks and produce segmentation masks.
        # "output of the l-th decoder layer through a 1x1 convolution and then bilinearly upsampling"
        for i in range(min(self.deep_supervision_levels, len(decoder_channels_list))):
            # Output from decoder_blocks[i] has decoder_channels_list[i] channels
            idx_from_end = len(decoder_channels_list) - 1 - i
            # Supervise from deeper layers first (closer to final output)
            # The number of channels for these heads will be `decoder_channels_list[i]`.
            self.segmentation_heads.append(
                nn.Conv2d(decoder_channels_list[i], num_classes, kernel_size=1)
            )

    def forward(self, x_bottleneck, skip_features_list, target_size):
        x = x_bottleneck
        intermediate_outputs =

        num_decoder_blocks = len(self.decoder_blocks)
        for i in range(num_decoder_blocks):
            skip = skip_features_list[i] if i < len(skip_features_list) else None
            if skip is None and i < num_decoder_blocks:  # Should not happen if lists are sized correctly
                raise ValueError(f"Missing skip feature for decoder block {i}")

            x = self.decoder_blocks[i](x, skip)

            # Collect outputs for deep supervision from the specified layers
            # If we have 3 decoder blocks, all are supervised.
            # The segmentation head is applied to the output of the block.
            if i < self.deep_supervision_levels:
                # If decoder_blocks are [block0, block1, block2 (final)],
                # then output of block0 is "Layer 1", output of block2 is "Layer 3".
                # So, seg_head processes output of block0, seg_head[1] for block1, etc.

                inter_mask_logits = self.segmentation_heads[i](x)
                inter_mask_logits = F.interpolate(inter_mask_logits, size=target_size, mode='bilinear',
                                                  align_corners=False)
                intermediate_outputs.append(inter_mask_logits)

        return intermediate_outputs  # List of masks, final one is the primary prediction


class ESAM2_BLS(nn.Module):
    """
    ESAM2-BLS Model (Enhanced Segment Anything Model 2 for Breast Lesion Segmentation).
    Overall architecture from Section 3.1 and Figure 1 description.
    """

    def __init__(self, img_channels=3, num_classes=1,
                 encoder_initial_channels=64,
                 encoder_stage_channels=,  # Channels for each Hiera stage output
                 encoder_layers_per_stage=[1, 1, 1, 1],  # Simplified Hiera structure
                 encoder_pool_strides=[2, 2, 2, 2],  # Strides for pooling in Hiera stages
                 msif_bottleneck_channels_ratio=0.5,  # Ratio for MSIF internal channels
                 msif_out_channels_ratio=1.0,  # Ratio for MSIF output channels relative to its input
                 decoder_channels_list=,  # Channels for each decoder block output
                 deep_supervision_levels=3):  # L=3
        super().__init__()
        self.num_classes = num_classes

        # Adapter-based Hiera Encoder (Section 3.2)
        self.encoder = AdapterBasedHieraEncoder(
            img_channels=img_channels,
            initial_patch_embed_channels=encoder_initial_channels,
            stage_channels=encoder_stage_channels,
            num_layers_per_stage=encoder_layers_per_stage,
            pool_strides=encoder_pool_strides
        )

        # MSIF BottleNeck Block (Section 3.3)
        # Input to MSIF is the output of the encoder
        msif_in_channels = encoder_stage_channels[-1]
        self.msif_bottleneck = MSIFBottleNeckBlock(
            in_channels=msif_in_channels,
            bottleneck_channels_ratio=msif_bottleneck_channels_ratio,
            out_channels_ratio=msif_out_channels_ratio
        )

        # Lightweight Decoder (Section 3.4)
        # Skip features from encoder stages (reversed: deepest first)
        # Encoder skip channels should match encoder_stage_channels
        # Decoder needs to know the channel count of the output from MSIF bottleneck
        bottleneck_actual_out_channels = int(msif_in_channels * msif_out_channels_ratio)

        # The skip connections for the decoder come from the Hiera encoder stages.
        # Their channel dimensions are defined by `encoder_stage_channels`.
        # The decoder typically upsamples from the bottleneck feature.
        # The skip connections are fused at each upsampling stage.
        # `encoder_skip_channels_list` should be `encoder_stage_channels` in reverse order,
        # but only those that match the number of decoder upsampling stages.
        # If decoder_channels_list has 3 stages, we need 3 skip connections.
        # These would typically be from the later (deeper) stages of the encoder.
        # Example: if encoder_stage_channels =
        # Skips for a 3-stage decoder might be  (from stages outputting 512, 256, 128 respectively,
        # assuming the 512-channel one is the input to bottleneck)
        # The `self.encoder.skip_features` is already reversed.
        # We need to select the correct ones.
        # Let's assume `encoder_stage_channels` are the channels of the skip connections.
        # The `skip_features_list` from encoder is already reversed (deepest first).
        # So, `encoder_stage_channels[::-1]` would be the list of skip channels.

        self.decoder = LightweightDecoder(
            bottleneck_out_channels=bottleneck_actual_out_channels,
            encoder_skip_channels_list=encoder_stage_channels[::-1][:len(decoder_channels_list)],
            # Match number of decoder blocks
            decoder_channels_list=decoder_channels_list,
            num_classes=num_classes,
            deep_supervision_levels=deep_supervision_levels
        )

    def forward(self, x):
        target_size = x.shape[2:]  # H, W of original input

        encoder_output, skip_features = self.encoder(x)
        # skip_features is a list, deepest first.
        # Example: if 4 stages, skip_features = [stage4_out, stage3_out, stage2_out, stage1_out]
        # If decoder has 3 blocks, it will use skip_features, skip_features[1], skip_features[1]

        bottleneck_output = self.msif_bottleneck(encoder_output)

        # List of masks (logits), from shallowest supervised to deepest supervised
        # The last mask in the list is considered the primary output.
        masks_logits_list = self.decoder(bottleneck_output, skip_features, target_size)

        return masks_logits_list


# --- Loss Functions (Section 3.5) ---

class IoULoss(nn.Module):
    """ IoU Loss (Equation 12) """

    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred_probs, target_mask):
        # pred_probs should be after sigmoid
        intersection = torch.sum(pred_probs * target_mask, dim=(1, 2, 3))
        union = torch.sum(pred_probs, dim=(1, 2, 3)) + torch.sum(target_mask, dim=(1, 2, 3)) - intersection

        iou = (intersection + self.epsilon) / (union + self.epsilon)
        return 1.0 - iou.mean()


class BCELossWrapper(nn.Module):
    """ BCE Loss (Equation 13) - using BCEWithLogitsLoss for stability """

    def __init__(self):
        super().__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, target_mask):
        return self.bce_with_logits(pred_logits, target_mask)


class DeepSupervisionLoss(nn.Module):
    """ Deep Supervision Loss (Equation 14) """

    def __init__(self, num_supervised_layers=3, weights=[0.2, 0.3, 0.5]):  # alpha_l values
        super().__init__()
        if len(weights) != num_supervised_layers:
            raise ValueError("Length of weights must match num_supervised_layers")
        self.num_supervised_layers = num_supervised_layers
        self.weights = weights  # [alpha_1 (shallowest),..., alpha_L (deepest)]
        self.iou_loss = IoULoss()
        self.bce_loss = BCELossWrapper()

    def forward(self, pred_logits_list, target_mask):
        total_ds_loss = 0.0
        if len(pred_logits_list) != self.num_supervised_layers:
            # This might happen if decoder outputs fewer layers than specified for DS
            # print(f"Warning: Number of predicted masks ({len(pred_logits_list)}) "
            #       f"does not match number of supervised layers ({self.num_supervised_layers}).")
            # Supervise available layers.
            num_to_supervise = min(len(pred_logits_list), self.num_supervised_layers)
        else:
            num_to_supervise = self.num_supervised_layers

        for i in range(num_to_supervise):
            # pred_logits_list is ordered from shallowest supervised to deepest.
            # weights are also ordered shallowest to deepest.
            logits = pred_logits_list[i]
            probs = torch.sigmoid(logits)

            loss_iou_i = self.iou_loss(probs, target_mask)
            loss_bce_i = self.bce_loss(logits, target_mask)  # BCEWithLogits takes logits

            # sum alpha_l * (L_IoU + L_BCE)
            layer_loss = self.weights[i] * (loss_iou_i + loss_bce_i)
            total_ds_loss += layer_loss

        return total_ds_loss


class JointOptimizationLoss(nn.Module):
    """ Total Loss Function (Equation 15) """

    def __init__(self, lambda1=1.0, lambda2=1.0, lambda3=1.0,
                 ds_weights=[0.2, 0.3, 0.5], num_ds_layers=3):
        super().__init__()
        self.lambda1 = lambda1  # Weight for main IoU loss (on final prediction)
        self.lambda2 = lambda2  # Weight for main BCE loss (on final prediction)
        self.lambda3 = lambda3  # Weight for Deep Supervision loss component

        self.main_iou_loss = IoULoss()
        self.main_bce_loss = BCELossWrapper()
        self.ds_loss_module = DeepSupervisionLoss(num_supervised_layers=num_ds_layers, weights=ds_weights)

    def forward(self, pred_logits_list, target_mask):
        # pred_logits_list is ordered shallowest to deepest supervised.
        # The last one is the primary prediction.
        final_logits = pred_logits_list[-1]
        final_probs = torch.sigmoid(final_logits)

        loss_main_iou = self.main_iou_loss(final_probs, target_mask)
        loss_main_bce = self.main_bce_loss(final_logits, target_mask)

        loss_ds = self.ds_loss_module(pred_logits_list, target_mask)

        total_loss = (self.lambda1 * loss_main_iou +
                      self.lambda2 * loss_main_bce +
                      self.lambda3 * loss_ds)
        return total_loss


if __name__ == '__main__':
    # These channel numbers are examples and should be configured based on SAM2 or experimentation.
    # ESAM2-BLS is based on SAM2, so encoder channels would ideally match SAM2's Hiera.
    # Typical ViT/Hiera large models might have stage_channels like  or similar.
    example_img_channels = 3
    example_num_classes = 1  # Binary segmentation (lesion vs. background)
    example_encoder_initial_channels = 32
    example_encoder_stage_channels =  # Output channels of each Hiera stage
    example_encoder_layers_per_stage = [1, 1, 1, 1]  # Number of HieraLayers within each stage
    example_encoder_pool_strides = [2, 2, 2, 2]  # Strides for pooling at start of each Hiera stage
    # First stride is for patch_embed, subsequent for HieraLayer pooling

    example_msif_bottleneck_channels_ratio = 0.5
    example_msif_out_channels_ratio = 1.0  # MSIF output channels = input_channels * ratio

    # Decoder channels should be chosen to progressively upsample.
    # Number of decoder blocks often matches number of pooling stages in encoder minus one.
    # If 4 encoder stages, 3 decoder upsampling blocks are common.
    example_decoder_channels_list = 2 # Output channels of each DecoderBlock
    example_deep_supervision_levels = 3  # L=3

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
    ).cuda()

    # Dummy input
    batch_size = 2
    img_size = 256  # Example image size
    dummy_input = torch.randn(batch_size, example_img_channels, img_size, img_size).cuda()
    dummy_target = (torch.rand(batch_size, example_num_classes, img_size, img_size) > 0.5).float().cuda()

    # Forward pass
    model.train()
    list_of_output_logits = model(dummy_input)

    print(f"Model produced {len(list_of_output_logits)} output masks (for deep supervision).")
    for i, logits in enumerate(list_of_output_logits):
        print(f"Mask {i} shape: {logits.shape}")

    # Loss calculation
    criterion = JointOptimizationLoss(
        lambda1=1.0, lambda2=1.0, lambda3=1.0,  # Example lambda values
        ds_weights=[0.2, 0.3, 0.5],  # alpha_l
        num_ds_layers=example_deep_supervision_levels
    ).cuda()

    loss = criterion(list_of_output_logits, dummy_target)
    print(f"Calculated loss: {loss.item()}")

    # Optimizer and Scheduler (as mentioned in Section 3.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)  # Example T_max

    # Example training step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    print("Performed one optimization step.")

    # Evaluation mode
    model.eval()
    with torch.no_grad():
        list_of_output_logits_eval = model(dummy_input)
        final_prediction_logits = list_of_output_logits_eval[-1]  # Last one is the main prediction
        final_prediction_probs = torch.sigmoid(final_prediction_logits)
    print(f"Evaluation mode: final prediction probabilities shape: {final_prediction_probs.shape}")