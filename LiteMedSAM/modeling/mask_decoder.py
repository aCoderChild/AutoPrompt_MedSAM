"""Mask decoder with DSRA modules and exposure correction (PraNet-V2 inspired)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import DualSupervisedReverseAttention, FeatureFusion


class LiteDecoder(nn.Module):
    """Enhanced decoder with DSRA modules and exposure correction (PraNet-V2 inspired).
    
    Architecture:
    - Prompt fusion: Combine features with prompt embedding
    - Multi-scale guidance: Use coarse masks from encoder as supervision
    - DSRA modules: Dual-Supervised Reverse Attention for feature refinement
    - Exposure correction decoding: Integrate exposure features with spatial features
    - Progressive upsampling: 4x upsampling (1/16 -> 1/8 -> 1/4 -> 1/)
    - Dual supervision: Background and foreground supervision at each level
    
    Design focuses on:
    - Multi-scale feature guidance from encoder
    - Dual supervision for robust mask learning
    - Exposure correction feature integration
    - Hierarchical decoding with DSRA refinement
    """
    
    def __init__(
        self,
        input_dim=256,
        hidden_dim=128,
        output_dim=2,
        base_channels=32
    ):
        super(LiteDecoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.base_channels = base_channels
        
        # ============ Prompt Fusion ============
        # Fuse image features with prompt embedding
        self.prompt_fusion = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # ============ DSRA Module 1 (1/16 resolution) ============
        self.dsra1 = DualSupervisedReverseAttention(hidden_dim, hidden_dim)
        
        # ============ Upsampling to 1/8 ============
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # ============ Feature fusion with skip connection ============
        self.skip_fusion1 = FeatureFusion(hidden_dim // 2, exposure_channels=3)
        
        # ============ DSRA Module 2 (1/8 resolution) ============
        self.dsra2 = DualSupervisedReverseAttention(hidden_dim // 2, hidden_dim // 2)
        
        # ============ Upsampling to 1/4 ============
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True)
        )
        
        # ============ Feature fusion with skip connection ============
        self.skip_fusion2 = FeatureFusion(hidden_dim // 4, exposure_channels=3)
        
        # ============ DSRA Module 3 (1/4 resolution) ============
        self.dsra3 = DualSupervisedReverseAttention(hidden_dim // 4, hidden_dim // 4)
        
        # ============ Upsampling to full resolution ============
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(inplace=True)
        )
        
        # ============ Exposure Correction Decoding ============
        # Decode exposure correction features to contribute to final output
        self.exposure_decode = nn.Sequential(
            nn.Conv2d(3, hidden_dim // 8, kernel_size=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(inplace=True)
        )
        
        # ============ Output Layers ============
        # Foreground (salient object) prediction
        self.fg_out = nn.Sequential(
            nn.Conv2d(hidden_dim // 8, hidden_dim // 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 8, output_dim, kernel_size=1)
        )
        
        # Background prediction for complementary supervision
        self.bg_out = nn.Sequential(
            nn.Conv2d(hidden_dim // 8, hidden_dim // 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 8, 1, kernel_size=1)
        )
        
        # Auxiliary outputs for intermediate supervision at 1/4 resolution
        self.aux_fg_out = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, hidden_dim // 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 8, output_dim, kernel_size=1)
        )
        
        self.aux_bg_out = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, hidden_dim // 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 8, 1, kernel_size=1)
        )
    
    def forward(
        self,
        x,
        prompt_embed,
        encoder_outputs=None,
        confidence=None
    ):
        """Decode segmentation from features and prompt with multi-scale guidance.
        
        Args:
            x: Image features [B, input_dim, H/16, W/16]
            prompt_embed: Encoded prompt embeddings [B, embed_dim]
            encoder_outputs: Dict containing:
                - coarse_masks: Final coarse mask from PartialDecoder [B, 1, H/16, W/16]
                - dual_masks: Dict with 'bg' and 'fg' masks from PartialDecoder (stages 2,3,4 only)
                  - Format: {'bg': [None, bg_2, bg_3, bg_4], 'fg': [None, fg_2, fg_3, fg_4]}
                  - Stage 1 has None (no PartialDecoder dual masks at stage 1)
                  - DSRA modules will generate additional attention masks internally
                - skip_connections: Skip connections from encoder (with fused exposure features)
            confidence: Confidence score of prompt
            
        Returns:
            outputs: Dict containing:
                - logits: Segmentation logits [B, output_dim, H, W]
                - bg_logits: Background logits [B, 1, H, W]
                - aux_logits: Auxiliary output at 1/4 resolution
                - aux_bg_logits: Auxiliary background output at 1/4 resolution
        """
        
        # Default encoder outputs if not provided
        if encoder_outputs is None:
            encoder_outputs = {
                'coarse_masks': None,
                'dual_masks': {'bg': [None, None, None, None], 'fg': [None, None, None, None]},
                'skip_connections': [None, None, None, None]
            }
        
        batch_size = x.shape[0]
        
        # ============ Fuse Prompt with Image Features ============
        # Expand prompt to spatial dimensions
        prompt_expanded = prompt_embed.view(batch_size, -1, 1, 1)
        prompt_expanded = prompt_expanded.expand_as(x)
        
        # Fuse prompt with image features
        fused = x + prompt_expanded * 0.5
        refined = self.prompt_fusion(fused)
        
        # ============ Decoder Level 1 (1/16 resolution) ============
        # Get dual supervision masks for DSRA
        # Use mask[3] or mask[2] (both at 1/16 after removing upsampling from PartialDecoder)
        bg_mask_1 = encoder_outputs['dual_masks']['bg'][3] if encoder_outputs['dual_masks']['bg'][3] is not None else torch.zeros_like(refined[:, :1])
        fg_mask_1 = encoder_outputs['dual_masks']['fg'][3] if encoder_outputs['dual_masks']['fg'][3] is not None else torch.ones_like(refined[:, :1])
        
        # Apply DSRA with dual supervision
        refined = self.dsra1(refined, bg_mask_1, fg_mask_1)
        
        # ============ Upsampling to 1/8 ============
        up1 = self.upsample1(refined)  # 1/8 resolution
        
        # Fuse with skip connection and exposure features if available
        skip_1 = encoder_outputs['skip_connections'][1] if encoder_outputs['skip_connections'][1] is not None else torch.zeros_like(up1)
        # Exposure features are now fused into skip_connections during encoding, so we generate fallback zeros if needed
        exp_feat_1 = torch.zeros(
            batch_size, 3, up1.shape[2], up1.shape[3], device=up1.device, dtype=up1.dtype
        )
        up1 = up1 + skip_1
        up1 = self.skip_fusion1(up1, exp_feat_1)
        
        # ============ Decoder Level 2 (1/8 resolution) ============
        bg_mask_2 = encoder_outputs['dual_masks']['bg'][1] if encoder_outputs['dual_masks']['bg'][1] is not None else torch.zeros_like(up1[:, :1])
        fg_mask_2 = encoder_outputs['dual_masks']['fg'][1] if encoder_outputs['dual_masks']['fg'][1] is not None else torch.ones_like(up1[:, :1])
        
        up1 = self.dsra2(up1, bg_mask_2, fg_mask_2)
        
        # ============ Upsampling to 1/4 ============
        up2 = self.upsample2(up1)  # 1/4 resolution
        
        # Fuse with skip connection and exposure features
        skip_0 = encoder_outputs['skip_connections'][0] if encoder_outputs['skip_connections'][0] is not None else torch.zeros_like(up2)
        # Exposure features are now fused into skip_connections during encoding, so we generate fallback zeros if needed
        exp_feat_0 = torch.zeros(
            batch_size, 3, up2.shape[2], up2.shape[3], device=up2.device, dtype=up2.dtype
        )
        up2 = up2 + skip_0
        up2 = self.skip_fusion2(up2, exp_feat_0)
        
        # ============ Decoder Level 3 (1/4 resolution) ============
        bg_mask_0 = encoder_outputs['dual_masks']['bg'][0] if encoder_outputs['dual_masks']['bg'][0] is not None else torch.zeros_like(up2[:, :1])
        fg_mask_0 = encoder_outputs['dual_masks']['fg'][0] if encoder_outputs['dual_masks']['fg'][0] is not None else torch.ones_like(up2[:, :1])
        
        up2 = self.dsra3(up2, bg_mask_0, fg_mask_0)
        
        # ============ Auxiliary Output (1/4 resolution) ============
        # For intermediate supervision
        aux_fg_logits = self.aux_fg_out(up2)
        aux_bg_logits = self.aux_bg_out(up2)
        
        # ============ Upsampling to Full Resolution ============
        up3 = self.upsample3(up2)  # Full resolution (1x)
        
        # ============ Decode Exposure Correction Features ============
        # Exposure features are now fused during encoding into skip_connections
        # Create zero feature placeholder for final exposure decoding
        exp_feat_final = torch.zeros(
            batch_size, 3, 16, 16, device=up3.device, dtype=up3.dtype
        )
        # Upsample exposure features to match full resolution
        exp_feat_final = F.interpolate(
            exp_feat_final, size=(up3.shape[2], up3.shape[3]), 
            mode='bilinear', align_corners=False
        )
        exp_decoded = self.exposure_decode(exp_feat_final)
        
        # Wise-multiplication fusion: modulate spatial features with exposure
        up3_fused = up3 * (1.0 + exp_decoded * 0.3)
        
        # ============ Final Output ============
        # Foreground (salient object) prediction
        fg_logits = self.fg_out(up3_fused)
        
        # Background prediction
        bg_logits = self.bg_out(up3_fused)
        
        return {
            'logits': fg_logits,
            'bg_logits': bg_logits,
            'aux_logits': aux_fg_logits,
            'aux_bg_logits': aux_bg_logits
        }
