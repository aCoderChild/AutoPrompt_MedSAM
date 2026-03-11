"""Prompt encoder for flexible multi-modal prompt support."""

import torch
import torch.nn as nn


class PromptEncoder(nn.Module):
    """Flexible prompt encoder supporting multiple prompt types.
    
    Supports:
    - Bounding box prompts: 4 coordinates (x_min, y_min, x_max, y_max)
    - Point prompts: 2 coordinates (x, y)
    - Mask prompts: Binary mask [B, 1, H, W]
    - Learnable prompts: Default trainable tokens
    """
    
    def __init__(self, embed_dim=256):
        super(PromptEncoder, self).__init__()
        self.embed_dim = embed_dim
        
        # ============ Bounding Box Encoding ============
        # Normalize bbox coordinates and encode to embedding
        self.bbox_embed = nn.Sequential(
            nn.Linear(4, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # ============ Point Encoding ============
        # Encode point coordinates to embedding
        self.point_embed = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # ============ Mask Encoding ============
        # Extract features from binary mask using convolution
        self.mask_embed = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Project mask features to embedding dimension
        self.mask_proj = nn.Sequential(
            nn.Linear(64, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # ============ Learnable Prompt ============
        # Default learnable token when no prompt is provided
        self.learnable_prompt = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
    
    def forward(self, bbox=None, points=None, mask=None):
        """Encode prompts to embedding space.
        
        Args:
            bbox: Bounding box coordinates [B, 4] (x_min, y_min, x_max, y_max) or None
            points: Point coordinates [B, 2] (x, y) or None
            mask: Binary mask [B, 1, H, W] or None
            
        Returns:
            prompt_embed: Prompt embedding [B, embed_dim]
        """
        
        if bbox is not None:
            # Encode bounding box
            return self._encode_bbox(bbox)
        
        elif points is not None:
            # Encode point prompt
            return self._encode_points(points)
        
        elif mask is not None:
            # Encode mask prompt
            return self._encode_mask(mask)
        
        else:
            # Return learnable prompt token
            return self.learnable_prompt
    
    def _encode_bbox(self, bbox):
        """Encode bounding box prompt.
        
        Args:
            bbox: Bounding box [B, 4]
            
        Returns:
            embedding: [B, embed_dim]
        """
        # Normalize box coordinates if needed
        bbox_normalized = bbox.clone()
        return self.bbox_embed(bbox_normalized)
    
    def _encode_points(self, points):
        """Encode point prompt.
        
        Args:
            points: Point coordinates [B, 2]
            
        Returns:
            embedding: [B, embed_dim]
        """
        # Normalize point coordinates if needed
        points_normalized = points.clone()
        return self.point_embed(points_normalized)
    
    def _encode_mask(self, mask):
        """Encode mask prompt from binary mask.
        
        Args:
            mask: Binary mask [B, 1, H, W]
            
        Returns:
            embedding: [B, embed_dim]
        """
        # Extract features using convolutions
        batch_size = mask.shape[0]
        mask_feat = self.mask_embed(mask)  # [B, 64, 1, 1]
        mask_feat = mask_feat.view(batch_size, -1)  # [B, 64]
        
        # Project to embedding dimension
        embedding = self.mask_proj(mask_feat)  # [B, embed_dim]
        
        return embedding
