"""Common utilities for LiteMedSAM model with PraNet-V2 and exposure correction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Tuple


class MLPBlock(nn.Module):
    """Multi-layer perceptron block with two linear layers and activation."""
    
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        """Initialize MLPBlock.
        
        Args:
            embedding_dim: Input and output dimension
            mlp_dim: Hidden dimension for the MLP
            act: Activation function (default: GELU)
        """
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP block.
        
        Args:
            x: Input tensor [*, embedding_dim]
            
        Returns:
            Output tensor [*, embedding_dim]
        """
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    """Layer normalization for 2D feature maps (for convolutional layers).
    
    Normalizes over channel dimension for BCHW format tensors.
    """
    
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        """Initialize LayerNorm2d.
        
        Args:
            num_channels: Number of channels (C dimension)
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LayerNorm2d.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Normalized tensor [B, C, H, W]
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ExposureCorrection(nn.Module):
    """Exposure correction module for learning illumination-invariant features.
    
    Inspired by WTNet for handling exposure variations in medical images.
    Extracts exposure correction features that can be combined with spatial features.
    """
    
    def __init__(self, in_channels: int, out_channels: int = 3):
        """Initialize ExposureCorrection.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of exposure correction outputs (RGB curves)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Exposure feature extraction
        self.exposure_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract exposure correction features.
        
        Args:
            x: Input features [B, C, H, W]
            
        Returns:
            exposure_feat: Exposure correction features [B, 3, H, W]
            illumination_map: Illumination intensity map [B, 1, H, W]
        """
        exposure_feat = self.exposure_conv(x)
        
        # Compute illumination intensity as mean of exposure features
        illumination = exposure_feat.mean(dim=1, keepdim=True)
        illumination = torch.sigmoid(illumination)
        
        return exposure_feat, illumination


class DualSupervisedReverseAttention(nn.Module):
    """Dual-Supervised Reverse Attention (DSRA) module from PraNet-V2.
    
    Performs reverse attention mechanism with dual supervision using background
    and foreground masks for better feature refinement.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int = None):
        """Initialize DSRA module.
        
        Args:
            in_channels: Input channel dimension
            out_channels: Output channel dimension (default: same as input)
        """
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Attention generation from background
        self.bg_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Attention generation from foreground
        self.fg_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, bg_mask: torch.Tensor, fg_mask: torch.Tensor) -> torch.Tensor:
        """Apply dual-supervised reverse attention.
        
        Args:
            x: Input features [B, C, H, W]
            bg_mask: Background mask for supervision [B, 1, H, W]
            fg_mask: Foreground mask for supervision [B, 1, H, W]
            
        Returns:
            Refined features [B, out_channels, H, W]
        """
        # Generate attention maps from masks
        bg_attn = self.bg_attention(x + (bg_mask - 0.5))
        fg_attn = self.fg_attention(x + (fg_mask - 0.5))
        
        # Reverse attention: highlight foreground, suppress background
        reverse_attn = fg_attn * (1.0 - bg_attn)
        
        # Apply attention to feature
        attended_x = x * reverse_attn
        
        # Combine with original features
        out = self.refine(x + attended_x)
        
        return out


class PartialDecoder(nn.Module):
    """Partial Decoder for generating coarse masks from encoder features.
    
    Also called Progressive Decoder (PD) in PraNet-V2. Generates binary masks
    at each encoding stage for use as guidance in the full decoder.
    Produces masks at the same resolution as input features.
    """
    
    def __init__(self, in_channels: int, out_channels: int = 1):
        """Initialize PartialDecoder.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (default: 1 for binary mask)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Decode without upsampling - keep same resolution as input
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate coarse masks and background/foreground predictions.
        
        Args:
            x: Input features [B, C, H, W]
            
        Returns:
            coarse_mask: Binary coarse mask [B, 1, H, W] (same resolution as input)
            bg_mask: Background mask [B, 1, H, W]
            fg_mask: Foreground mask [B, 1, H, W]
        """
        # Decode to mask at same resolution as input (no upsampling)
        logits = self.decode(x)
        coarse_mask = torch.sigmoid(logits)
        
        # Generate dual supervision masks
        bg_mask = 1.0 - coarse_mask
        fg_mask = coarse_mask
        
        return coarse_mask, bg_mask, fg_mask


class FeatureFusion(nn.Module):
    """Feature fusion module combining spatial and exposure correction features.
    
    Uses element-wise multiplication (Hadamard product) to fuse complementary
    information from spatial features and exposure correction features.
    """
    
    def __init__(self, spatial_channels: int, exposure_channels: int = 3):
        """Initialize FeatureFusion.
        
        Args:
            spatial_channels: Number of spatial feature channels
            exposure_channels: Number of exposure correction channels
        """
        super().__init__()
        self.spatial_channels = spatial_channels
        self.exposure_channels = exposure_channels
        
        # Project exposure features to match spatial channels
        self.exposure_proj = nn.Sequential(
            nn.Conv2d(exposure_channels, spatial_channels, kernel_size=1),
            nn.BatchNorm2d(spatial_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion weights
        self.fusion_weight = nn.Parameter(torch.ones(1, 1, 1, 1) * 0.5)
    
    def forward(
        self, 
        spatial_feat: torch.Tensor, 
        exposure_feat: torch.Tensor
    ) -> torch.Tensor:
        """Fuse spatial and exposure correction features.
        
        Args:
            spatial_feat: Spatial features [B, C, H, W]
            exposure_feat: Exposure correction features [B, 3, H, W]
            
        Returns:
            Fused features [B, C, H, W]
        """
        # Project exposure features
        exposure_proj = self.exposure_proj(exposure_feat)
        
        # Element-wise multiplication fusion
        # Exposure features modulate the spatial features
        fused = spatial_feat * (1.0 + self.fusion_weight * exposure_proj)
        
        return fused
