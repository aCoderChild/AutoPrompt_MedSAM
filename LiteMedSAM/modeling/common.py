"""Common utilities for LiteMedSAM model."""

import torch
import torch.nn as nn
from typing import Type


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
