"""Mask decoder with progressive upsampling and feature fusion."""

import torch
import torch.nn as nn


class LiteDecoder(nn.Module):
    """Lightweight decoder with progressive upsampling and feature fusion.
    
    Architecture:
    - Prompt fusion: Combine image features with prompt embedding
    - Progressive upsampling: 4x upsampling (1/16 -> 1/8 -> 1/4 -> 1/)
    - Refinement blocks: Feature refinement at each scale
    - Auxiliary output: Intermediate supervision at 1/4 resolution
    
    Design focuses on:
    - Lightweight computation
    - Multi-scale feature fusion
    - Flexible prompt integration
    """
    
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=2):
        super(LiteDecoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # ============ Prompt Fusion ============
        # Fuse image features with prompt embedding
        self.prompt_fusion = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # ============ Refinement Block 1 ============
        # Feature refinement at 1/16 resolution
        self.refine_block1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # ============ Upsampling Path ============
        # Progressive 2x upsampling (2x -> 4x -> 8x -> 16x)
        
        # 1/16 -> 1/8 upsampling
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # ============ Refinement Block 2 ============
        # Feature refinement at 1/8 resolution
        self.refine_block2 = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # 1/8 -> 1/4 upsampling
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True)
        )
        
        # ============ Refinement Block 3 ============
        # Feature refinement at 1/4 resolution
        self.refine_block3 = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True)
        )
        
        # 1/4 -> 1 upsampling
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(inplace=True)
        )
        
        # ============ Output Layers ============
        # Main output layer
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim // 8, hidden_dim // 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 8, output_dim, kernel_size=1)
        )
        
        # Auxiliary output for intermediate supervision (at 1/4 resolution)
        self.aux_out = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, hidden_dim // 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 8, output_dim, kernel_size=1)
        )
    
    def forward(self, x, prompt_embed):
        """Decode segmentation from features and prompt.
        
        Args:
            x: Image features [B, input_dim, H/16, W/16]
            prompt_embed: Encoded prompt embeddings [B, embed_dim]
            
        Returns:
            logits: Segmentation logits [B, output_dim, H, W]
            aux_logits: Auxiliary output at 1/4 resolution for intermediate supervision
        """
        
        # ============ Fuse Prompt with Image Features ============
        batch_size = x.shape[0]
        
        # Expand prompt to spatial dimensions
        prompt_expanded = prompt_embed.view(batch_size, -1, 1, 1)
        prompt_expanded = prompt_expanded.expand_as(x)
        
        # Fuse prompt with image features (weighted addition)
        fused = x + prompt_expanded * 0.5
        
        # ============ Decode Path ============
        # Refinement and progressive upsampling
        refined1 = self.refine_block1(self.prompt_fusion(fused))  # 1/16 resolution
        
        up1 = self.upsample1(refined1)  # 1/8 resolution
        refined2 = self.refine_block2(up1)  # 1/8 resolution
        
        up2 = self.upsample2(refined2)  # 1/4 resolution
        refined3 = self.refine_block3(up2)  # 1/4 resolution
        
        # ============ Auxiliary Output ============
        # Output at 1/4 resolution for intermediate supervision
        aux_logits = self.aux_out(refined3)
        
        # ============ Final Output ============
        up3 = self.upsample3(refined3)  # Full resolution (1x)
        logits = self.out_conv(up3)  # Output segmentation
        
        return logits, aux_logits
