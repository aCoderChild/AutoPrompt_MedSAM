"""Image encoder for LiteMedSAM with multi-scale feature extraction."""

import torch
import torch.nn as nn


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


class LiteImageEncoder(nn.Module):
    """Lightweight image encoder with multi-scale feature extraction.
    
    Architecture:
    - Stem: Initial 7x7 convolution + MaxPool
    - 4 residual layers with progressive downsampling
    - Final projection to embedding dimension
    
    Output:
    - Features at 1/16 resolution with embedding dimension
    - Skip connections for potential decoder use
    """
    
    def __init__(self, in_channels=1, out_channels=256, base_channels=32):
        super(LiteImageEncoder, self).__init__()
        
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
        """Extract multi-scale features from image.
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            features: Encoded features [B, out_channels, H/16, W/16]
            skip_connections: List of intermediate features for skip connections
                - skip_connections[0]: features at 1/4 resolution
                - skip_connections[1]: features at 1/8 resolution
                - skip_connections[2]: features at 1/16 resolution (before proj)
        """
        # Stem
        x0 = self.stem(x)  # 1/4 resolution
        
        # Encoder stages with skip connections
        x1 = self.layer1(x0)  # 1/4 resolution
        x2 = self.layer2(x1)  # 1/8 resolution
        x3 = self.layer3(x2)  # 1/16 resolution
        x4 = self.layer4(x3)  # 1/16 resolution
        
        # Final projection
        features = self.final_proj(x4)  # 1/16 resolution
        
        # Store skip connections for decoder
        skip_connections = [x1, x2, x3, x4]
        
        return features, skip_connections
