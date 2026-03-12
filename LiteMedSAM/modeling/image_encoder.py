"""Image encoder for LiteMedSAM with PraNet-V2 architecture and exposure correction."""

import torch
import torch.nn as nn
from .common import ExposureCorrection, PartialDecoder, FeatureFusion


class ResidualBlock(nn.Module):
    """Residual block with optional downsampling.
    
    Architecture:
    - Conv 3x3 -> BatchNorm -> ReLU
    - Conv 3x3 -> BatchNorm
    - Residual connection
    - ReLU activation
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (if dimensions change)
        self.skip_proj = None
        if stride != 1 or in_channels != out_channels:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """Forward pass with residual connection.
        
        Args:
            x: Input tensor [B, in_channels, H, W]
            
        Returns:
            Output tensor [B, out_channels, H/stride, W/stride]
        """
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.skip_proj is not None:
            identity = self.skip_proj(x)
        
        # Residual addition
        out = out + identity
        out = self.relu(out)
        
        return out


class ImageEncoder(nn.Module):
    """Enhanced image encoder with PraNet-V2 architecture and exposure correction.
    
    Architecture:
    - Stem: Initial 7x7 convolution + MaxPool
    - 4 residual layers with progressive downsampling
    - Exposure correction branch at each stage
    - Partial Decoder (PD) for coarse mask generation
    - Feature fusion combining spatial and exposure features
    
    Output:
    - Multi-scale spatial features (F2, F3, F4)
    - Multi-scale exposure correction features
    - Coarse masks at each scale
    - Dual supervision masks (background and foreground)
    """
    
    def __init__(self, in_channels=1, out_channels=256, base_channels=32):
        super(ImageEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        
        # Initial convolution - reduce to base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )  # Output: H/4, W/4, base_channels
        
        # Stage 1 - basic residual blocks at 1/4 resolution
        self.layer1 = self._make_layer(
            base_channels,
            base_channels,
            num_blocks=2,
            stride=1
        )
        
        # Stage 2 - increase channels and reduce to 1/8 resolution
        self.layer2 = self._make_layer(
            base_channels,
            base_channels * 2,
            num_blocks=2,
            stride=2
        )
        
        # Stage 3 - further increase channels and reduce to 1/16 resolution
        self.layer3 = self._make_layer(
            base_channels * 2,
            base_channels * 4,
            num_blocks=2,
            stride=2
        )
        
        # Stage 4 - final enhancement at 1/16 resolution
        self.layer4 = self._make_layer(
            base_channels * 4,
            base_channels * 4,
            num_blocks=2,
            stride=1
        )
        
        # Final projection to embedding dimension
        self.final_proj = nn.Sequential(
            nn.Conv2d(base_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # ============ Exposure Correction Modules ============
        # Extract exposure correction features at each stage
        self.exposure_f2 = ExposureCorrection(base_channels)
        self.exposure_f3 = ExposureCorrection(base_channels * 2)
        self.exposure_f4 = ExposureCorrection(base_channels * 4)
        self.exposure_final = ExposureCorrection(base_channels * 4)  # x4 has base_channels*4 channels before projection
        
        # ============ Partial Decoder (PD) Modules ============
        # Generate coarse masks from each stage features
        self.pd_f2 = PartialDecoder(base_channels, out_channels=1)
        self.pd_f3 = PartialDecoder(base_channels * 2, out_channels=1)
        self.pd_f4 = PartialDecoder(base_channels * 4, out_channels=1)
        self.pd_final = PartialDecoder(out_channels, out_channels=1)
        
        # ============ Feature Fusion Modules ============
        # Fuse spatial features with exposure correction features
        self.fusion_f2 = FeatureFusion(base_channels, exposure_channels=3)
        self.fusion_f3 = FeatureFusion(base_channels * 2, exposure_channels=3)
        self.fusion_f4 = FeatureFusion(base_channels * 4, exposure_channels=3)
        self.fusion_final = FeatureFusion(base_channels * 4, exposure_channels=3)  # x4 has base_channels*4 channels before projection
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a residual block layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_blocks: Number of blocks in this layer
            stride: Stride for first block (for downsampling)
            
        Returns:
            Sequential module containing the blocks
        """
        layers = []
        
        # First block (may downsample)
        layers.append(
            ResidualBlock(in_channels, out_channels, stride=stride)
        )
        
        # Remaining blocks (stride=1)
        for _ in range(1, num_blocks):
            layers.append(
                ResidualBlock(out_channels, out_channels, stride=1)
            )
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Extract multi-scale features with exposure correction and coarse masks.
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            features: Encoded features [B, out_channels, H/16, W/16]
            coarse_mask: Final coarse mask [B, 1, H/16, W/16] from PartialDecoder
            dual_masks: Dict with first dual supervision masks from PartialDecoder
                - bg: [None, bg_mask_2, bg_mask_3, bg_mask_4] (stages 2,3,4 only)
                - fg: [None, fg_mask_2, fg_mask_3, fg_mask_4] (stages 2,3,4 only)
            skip_connections: List of intermediate features for skip connections
                Note: These intermediate features already contain fused exposure correction information
        """
        # Stem
        x0 = self.stem(x)  # 1/4 resolution
        
        # ============ Stage 1 ============
        x1 = self.layer1(x0)  # 1/4 resolution
        exp_x1 = self.exposure_f2(x1)
        x1_fused = self.fusion_f2(x1, exp_x1)
        coarse_mask_1, _, _ = self.pd_f2(x1_fused)  # No dual masks from stage 1
        
        # ============ Stage 2 ============
        x2 = self.layer2(x1_fused)  # 1/8 resolution
        exp_x2 = self.exposure_f3(x2)
        x2_fused = self.fusion_f3(x2, exp_x2)
        coarse_mask_2, bg_mask_2, fg_mask_2 = self.pd_f3(x2_fused)  # First dual masks from stage 2
        
        # ============ Stage 3 ============
        x3 = self.layer3(x2_fused)  # 1/16 resolution
        exp_x3 = self.exposure_f4(x3)
        x3_fused = self.fusion_f4(x3, exp_x3)
        coarse_mask_3, bg_mask_3, fg_mask_3 = self.pd_f4(x3_fused)  # First dual masks from stage 3
        
        # ============ Stage 4 ============
        x4 = self.layer4(x3_fused)  # 1/16 resolution
        exp_x4 = self.exposure_final(x4)
        x4_fused = self.fusion_final(x4, exp_x4)
        
        # Final projection
        features = self.final_proj(x4_fused)  # 1/16 resolution
        coarse_mask_4, bg_mask_4, fg_mask_4 = self.pd_final(features)  # First dual masks from encoder stage 4
        
        # Store only the final coarse mask from PartialDecoder
        coarse_masks = coarse_mask_4
        
        # Store first dual supervision masks (from PartialDecoder on stages 2, 3, 4)
        # Stage 1 has no dual masks from PartialDecoder
        dual_masks = {
            'bg': [None, bg_mask_2, bg_mask_3, bg_mask_4],
            'fg': [None, fg_mask_2, fg_mask_3, fg_mask_4]
        }
        
        # Store skip connections for decoder (already contain fused exposure correction features)
        skip_connections = [x1_fused, x2_fused, x3_fused, x4_fused]
        
        return {
            'features': features,
            'coarse_masks': coarse_masks,
            'dual_masks': dual_masks,
            'skip_connections': skip_connections
        }
