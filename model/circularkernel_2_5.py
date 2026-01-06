import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

# Import circular convolution - UPDATE THIS PATH AS NEEDED
from conv.CircleConv3x3 import CircleConv3x3

def emergency_fix_all_circular_weights(model):
    """Fix circular convolution weight initialization to prevent NaN"""
    for name, module in model.named_modules():
        if hasattr(module, 'w_transform_matrix') and hasattr(module, 'weight'):
            # 1. Analyze transformation matrix eigenvalues
            eigenvals = torch.linalg.eigvals(module.w_transform_matrix)
            max_eigenval_mag = torch.abs(eigenvals).max().item()
            
            # 2. Compute safe scale to prevent amplification explosion
            safe_scale = 0.01 / np.sqrt(max_eigenval_mag)
            
            # 3. Apply scaled He initialization
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            scaled_std = safe_scale * np.sqrt(2.0 / fan_in)
            
            with torch.no_grad():
                nn.init.normal_(module.weight, mean=0.0, std=scaled_std)
                if hasattr(module, 'bias') and module.bias is not None:
                    bound = safe_scale / np.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)

class FireModule(nn.Module):
    """Fire module from SqueezeNet adapted for U-Net with CIRCULAR CONVOLUTIONS
    Consists of squeeze layer (1x1 conv) followed by expand layer (parallel 1x1 and 3x3 convs)"""
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(FireModule, self).__init__()
        
        # Squeeze layer - 1x1 convolution to reduce channels (UNCHANGED)
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_channels)
        self.squeeze_relu = nn.ReLU(inplace=True)
        
        # Expand layer - parallel 1x1 and 3x3 convolutions
        # Each produces half of expand_channels, then concatenated
        self.expand_1x1 = nn.Conv2d(squeeze_channels, expand_channels // 2, kernel_size=1)  # UNCHANGED
        self.expand_1x1_bn = nn.BatchNorm2d(expand_channels // 2)
        
        # TRANSFORMED: 3x3 conv → CircleConv3x3
        self.expand_3x3 = CircleConv3x3(squeeze_channels, expand_channels // 2, kernel_size=3, padding=1)
        self.expand_3x3_bn = nn.BatchNorm2d(expand_channels // 2)
        
        self.expand_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Squeeze phase
        x = self.squeeze(x)
        x = self.squeeze_bn(x)
        x = self.squeeze_relu(x)
        
        # Expand phase - parallel convolutions
        expand_1x1 = self.expand_1x1(x)
        expand_1x1 = self.expand_1x1_bn(expand_1x1)
        
        expand_3x3 = self.expand_3x3(x)
        expand_3x3 = self.expand_3x3_bn(expand_3x3)
        
        # Concatenate and apply ReLU
        out = torch.cat([expand_1x1, expand_3x3], dim=1)
        out = self.expand_relu(out)
        
        return out

class GatedCircularConv2d(nn.Module):
    """Gated Convolution with CIRCULAR KERNELS as described in Yu et al. 2018
    Formula: O = φ(Features) ⊙ σ(Gating)
    where Features = Wf * I, Gating = Wg * I"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(GatedCircularConv2d, self).__init__()
        
        if kernel_size == 3:
            # TRANSFORMED: Use circular convolutions for 3x3
            self.feature_conv = CircleConv3x3(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            self.gate_conv = CircleConv3x3(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            # UNCHANGED: Keep regular convolutions for non-3x3
            self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
            self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        
    def forward(self, x):
        # Compute feature and gating branches
        features = self.feature_conv(x)    # Wf * I
        gating = self.gate_conv(x)         # Wg * I
        
        # Apply activations as per paper: φ(Features) ⊙ σ(Gating)
        activated_features = torch.relu(features)  # φ(Features) - using ReLU as φ
        gates = torch.sigmoid(gating)              # σ(Gating) - soft gates (0-1)
        
        # Element-wise multiplication (gating operation)
        output = activated_features * gates        # φ(Features) ⊙ σ(Gating)
        
        return output

class GatedConvBlock(nn.Module):
    """Gated convolution block with BatchNorm - using circular convolutions"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(GatedConvBlock, self).__init__()
        self.gated_conv = GatedCircularConv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.gated_conv(x)  # Already includes ReLU activation and gating
        x = self.bn(x)
        return x

class SqueezeExcitation(nn.Module):
    """Squeeze and Excitation module for channel attention (UNCHANGED)"""
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # Squeeze: Global average pooling
        squeeze = self.global_avg_pool(x).view(batch_size, channels)
        # Excitation: FC layers
        excitation = self.fc1(squeeze)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation).view(batch_size, channels, 1, 1)
        # Apply attention
        return x * excitation

class SqueezeEncoderDecoderBranch(nn.Module):
    """Encoder-Decoder branch with skip connections using Fire Modules with CIRCULAR CONVOLUTIONS
    Outputs 512x512x1 depth map for fusion and exposes encoder features"""
    def __init__(self, input_channels, base_channels=64):
        super(SqueezeEncoderDecoderBranch, self).__init__()
        
        # Encoder - 3 fire modules with progressive downsampling
        # Fire module parameters: squeeze_channels = in_channels // 8, expand_channels = out_channels
        self.fire1 = FireModule(input_channels, max(input_channels // 8, 4), base_channels)
        self.fire2 = FireModule(base_channels, base_channels // 8, base_channels * 2)
        self.fire3 = FireModule(base_channels * 2, base_channels // 4, base_channels * 4)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Decoder - learnable upsampling with ConvTranspose2d (UNCHANGED)
        self.upsample3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        
        # Decoder fire modules (after skip connections)
        self.fire_dec3 = FireModule(base_channels * 4, base_channels // 2, base_channels * 2)  # *4 due to skip connection
        self.fire_dec2 = FireModule(base_channels * 2, base_channels // 4, base_channels)      # *2 due to skip connection
        self.fire_dec1 = FireModule(base_channels, base_channels // 8, base_channels)
        
        # Final output layer to produce single channel output (UNCHANGED - 1x1 conv)
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)
        
    def forward(self, x):
        # Encoder path with 3 fire modules
        enc1 = self.fire1(x)             # 512x512 -> 512x512x64
        enc1_pool = self.pool(enc1)      # 512x512x64 -> 256x256x64
        
        enc2 = self.fire2(enc1_pool)     # 256x256x64 -> 256x256x128
        enc2_pool = self.pool(enc2)      # 256x256x128 -> 128x128x128
        
        enc3 = self.fire3(enc2_pool)     # 128x128x128 -> 128x128x256
        
        # Decoder path with learnable upsampling and skip connections
        dec3_up = self.upsample3(enc3)   # 128x128x256 -> 256x256x128
        dec3_skip = torch.cat([dec3_up, enc2], dim=1)  # 256x256x(128+128) = 256x256x256
        dec3 = self.fire_dec3(dec3_skip) # 256x256x256 -> 256x256x128
        
        dec2_up = self.upsample2(dec3)   # 256x256x128 -> 512x512x64
        dec2_skip = torch.cat([dec2_up, enc1], dim=1)  # 512x512x(64+64) = 512x512x128
        dec2 = self.fire_dec2(dec2_skip) # 512x512x128 -> 512x512x64
        
        dec1 = self.fire_dec1(dec2)      # 512x512x64 -> 512x512x64
        
        # Final single-channel output (will be used for fusion in 4th branch)
        output = self.final_conv(dec1)   # 512x512x64 -> 512x512x1
        
        # Return both output and encoder features for fusion guidance
        encoder_features = {
            'enc1': enc1,   # 512x512x64
            'enc2': enc2,   # 256x256x128  
            'enc3': enc3    # 128x128x256
        }
        
        return output, encoder_features

class SparseDepthBranch(SqueezeEncoderDecoderBranch):
    """Branch for processing sparse depth input (Z-matrix from fringe projection profilometry)
    Output: 512x512x1 depth map for fusion"""
    def __init__(self, base_channels=64):
        super(SparseDepthBranch, self).__init__(input_channels=1, base_channels=base_channels)

class GrayscaleBranch(SqueezeEncoderDecoderBranch):
    """Branch for processing grayscale image input
    Output: 512x512x1 depth map for fusion"""
    def __init__(self, base_channels=64):
        super(GrayscaleBranch, self).__init__(input_channels=1, base_channels=base_channels)

class RelativeDepthBranch(SqueezeEncoderDecoderBranch):
    """Branch for processing relative depth map from Depth Anything V2
    Output: 512x512x1 depth map for fusion"""
    def __init__(self, base_channels=64):
        super(RelativeDepthBranch, self).__init__(input_channels=1, base_channels=base_channels)

class FusionBranch(nn.Module):
    """Fourth branch with U-Net architecture using CIRCULAR gated convolutions
    Incorporates SE-processed encoder features from three parallel branches at deepest level only
    Predicts residual depth that gets added to sparse depth input
    Input: Concatenated 4-channel tensor (512x512x4) + deepest encoder features from parallel branches
    Output: Dense depth = sparse_depth + residual_prediction (512x512x1)"""
    def __init__(self, base_channels=64, dropout_prob=0.0):
        super(FusionBranch, self).__init__()
        
        # Store dropout probability for fine-tuning adjustments
        self.dropout_prob = dropout_prob
        
        # Encoder - 3 gated convolution blocks with progressive downsampling (CIRCULAR)
        self.enc_block1 = GatedConvBlock(4, base_channels)  # 4 input channels from concatenation
        self.enc_block2 = GatedConvBlock(base_channels, base_channels * 2)
        self.enc_block3 = GatedConvBlock(base_channels * 2, base_channels * 4)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout layers for regularization during fine-tuning (default prob=0.0 for synthetic pre-training)
        self.dropout_enc1 = nn.Dropout2d(p=dropout_prob)
        self.dropout_enc2 = nn.Dropout2d(p=dropout_prob)
        self.dropout_enc3 = nn.Dropout2d(p=dropout_prob)
        self.dropout_dec3 = nn.Dropout2d(p=dropout_prob)
        self.dropout_dec2 = nn.Dropout2d(p=dropout_prob)
        self.dropout_dec1 = nn.Dropout2d(p=dropout_prob)
        
        # SE module for processing deepest encoder features from parallel branches (like paper's abstract features)
        self.se_enc3 = SqueezeExcitation(base_channels * 4)    # For 256-channel features at deepest level
        
        # Decoder - learnable upsampling with ConvTranspose2d (UNCHANGED)
        self.upsample3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        
        # Decoder gated conv blocks (after skip connections) (CIRCULAR)
        self.dec_block3 = GatedConvBlock(base_channels * 4, base_channels * 2)  # *4 due to skip connection (128+128)
        self.dec_block2 = GatedConvBlock(base_channels * 2, base_channels)      # *2 due to skip connection (64+64)
        self.dec_block1 = GatedConvBlock(base_channels, base_channels)
        
        # Final output layer to produce single channel residual depth map (UNCHANGED - 1x1 conv)
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)
    
    def set_dropout_prob(self, prob):
        """Update dropout probability for fine-tuning on real data"""
        self.dropout_prob = prob
        self.dropout_enc1.p = prob
        self.dropout_enc2.p = prob
        self.dropout_enc3.p = prob
        self.dropout_dec3.p = prob
        self.dropout_dec2.p = prob
        self.dropout_dec1.p = prob
    
    def forward(self, sparse_depth_input, sparse_branch_output, grayscale_branch_output, relative_depth_branch_output,
                sparse_encoder_features, grayscale_encoder_features, relative_depth_encoder_features):
        """
        Args:
            sparse_depth_input: Original sparse depth input (512x512x1)
            sparse_branch_output: Output from sparse depth branch (512x512x1)
            grayscale_branch_output: Output from grayscale branch (512x512x1)
            relative_depth_branch_output: Output from relative depth branch (512x512x1)
            sparse_encoder_features: Dict with 'enc1', 'enc2', 'enc3' from sparse branch
            grayscale_encoder_features: Dict with 'enc1', 'enc2', 'enc3' from grayscale branch
            relative_depth_encoder_features: Dict with 'enc1', 'enc2', 'enc3' from relative depth branch
        
        Returns:
            Dense depth map = sparse_depth_input + residual_prediction (512x512x1)
        """
        # Concatenate all inputs along channel dimension
        fusion_input = torch.cat([
            sparse_depth_input,           # Original sparse depth
            sparse_branch_output,         # Processed sparse depth
            grayscale_branch_output,      # Depth from grayscale
            relative_depth_branch_output  # Refined relative depth
        ], dim=1)  # 512x512x4
        
        # Encoder path with 3 gated conv blocks + dropout for regularization
        fusion_enc1 = self.enc_block1(fusion_input)  # 512x512x4 -> 512x512x64
        fusion_enc1 = self.dropout_enc1(fusion_enc1)  # Apply dropout
        fusion_enc1_pool = self.pool(fusion_enc1)    # 512x512x64 -> 256x256x64
        
        fusion_enc2 = self.enc_block2(fusion_enc1_pool)  # 256x256x64 -> 256x256x128
        fusion_enc2 = self.dropout_enc2(fusion_enc2)     # Apply dropout
        fusion_enc2_pool = self.pool(fusion_enc2)        # 256x256x128 -> 128x128x128
        
        fusion_enc3 = self.enc_block3(fusion_enc2_pool)  # 128x128x128 -> 128x128x256
        fusion_enc3 = self.dropout_enc3(fusion_enc3)     # Apply dropout
        
        # Apply SE to deepest encoder features from parallel branches and add to fusion features
        # (Following paper's approach of adding abstract guidance features at one specific point)
        se_sparse_enc3 = self.se_enc3(sparse_encoder_features['enc3'])
        se_grayscale_enc3 = self.se_enc3(grayscale_encoder_features['enc3'])
        se_relative_enc3 = self.se_enc3(relative_depth_encoder_features['enc3'])
        
        # Add SE-processed deepest features (like paper's abstract guidance ZI, ZS)
        fusion_enc3 = fusion_enc3 + se_sparse_enc3 + se_grayscale_enc3 + se_relative_enc3
        
        # Decoder path with learnable upsampling and skip connections + dropout
        dec3_up = self.upsample3(fusion_enc3)            # 128x128x256 -> 256x256x128
        dec3_skip = torch.cat([dec3_up, fusion_enc2], dim=1)  # 256x256x(128+128) = 256x256x256
        dec3 = self.dec_block3(dec3_skip)                # 256x256x256 -> 256x256x128
        dec3 = self.dropout_dec3(dec3)                   # Apply dropout
        
        dec2_up = self.upsample2(dec3)                   # 256x256x128 -> 512x512x64
        dec2_skip = torch.cat([dec2_up, fusion_enc1], dim=1)  # 512x512x(64+64) = 512x512x128
        dec2 = self.dec_block2(dec2_skip)                # 512x512x128 -> 512x512x64
        dec2 = self.dropout_dec2(dec2)                   # Apply dropout
        
        dec1 = self.dec_block1(dec2)                     # 512x512x64 -> 512x512x64
        dec1 = self.dropout_dec1(dec1)                   # Apply dropout
        
        # Predict residual depth
        residual_depth = self.final_conv(dec1)           # 512x512x64 -> 512x512x1
        
        # Final dense depth = sparse depth + residual prediction (like the paper)
        dense_depth = sparse_depth_input + residual_depth
        
        return dense_depth

class SqueezeUNetFusionCircular2_5(nn.Module):
    """Complete 2.5M parameter SqueezeUNet fusion model with CIRCULAR CONVOLUTIONS"""
    def __init__(self, base_channels=64, dropout_prob=0.0):
        super(SqueezeUNetFusionCircular2_5, self).__init__()
        
        # Initialize all four branches with circular convolutions
        self.sparse_branch = SparseDepthBranch(base_channels)
        self.grayscale_branch = GrayscaleBranch(base_channels)
        self.relative_depth_branch = RelativeDepthBranch(base_channels)
        self.fusion_branch = FusionBranch(base_channels, dropout_prob)
        
        # CRITICAL: Apply weight initialization fix to prevent NaN issues
        self.apply_weight_fix()
    
    def apply_weight_fix(self):
        """Apply emergency weight initialization fix to all circular convolutions"""
        emergency_fix_all_circular_weights(self)
        return self
    
    def forward(self, sparse_input, grayscale_input, relative_depth_input):
        # Forward pass through the three parallel branches
        sparse_output, sparse_encoder_features = self.sparse_branch(sparse_input)
        grayscale_output, grayscale_encoder_features = self.grayscale_branch(grayscale_input)
        relative_depth_output, relative_depth_encoder_features = self.relative_depth_branch(relative_depth_input)
        
        # Forward pass through fusion branch with SE-processed encoder guidance
        final_output = self.fusion_branch(
            sparse_input, sparse_output, grayscale_output, relative_depth_output,
            sparse_encoder_features, grayscale_encoder_features, relative_depth_encoder_features
        )
        
        return final_output

# Example usage and testing
if __name__ == "__main__":
    # Initialize model with circular convolutions
    model = SqueezeUNetFusionCircular2_5()
    
    # Test with dummy inputs (batch_size=2, channels=1, height=512, width=512)
    batch_size = 2
    sparse_input = torch.randn(batch_size, 1, 512, 512)
    grayscale_input = torch.randn(batch_size, 1, 512, 512)
    relative_depth_input = torch.randn(batch_size, 1, 512, 512)
    
    print("Testing circular convolution model...")
    try:
        # Forward pass
        final_output = model(sparse_input, grayscale_input, relative_depth_input)
        
        print(f"✅ Model works: {final_output.shape}")
        print(f"✅ No NaN: {not torch.isnan(final_output).any()}")
        print(f"✅ Output range: [{final_output.min().item():.3f}, {final_output.max().item():.3f}]")
        
        # Count parameters
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        total_params = count_parameters(model)
        print(f"✅ Total parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
        
        # Verify circular convolutions were applied
        circular_count = 0
        total_conv_count = 0
        for name, module in model.named_modules():
            if isinstance(module, CircleConv3x3):
                circular_count += 1
            if isinstance(module, (nn.Conv2d, CircleConv3x3)):
                total_conv_count += 1
        
        print(f"✅ Circular convolutions: {circular_count}/{total_conv_count} convolution layers")
        print("✅ Transformation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
