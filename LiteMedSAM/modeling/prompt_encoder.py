"""Prompt encoder with coarse mask guidance and exposure correction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptEncoder(nn.Module):
    """Enhanced prompt encoder supporting coarse mask guidance and exposure correction.
    
    Incorporates:
    - Coarse mask from encoder's Partial Decoder as primary prompt
    - Multi-modal external prompts (bbox, points, masks) as secondary
    - Integration with exposure correction features
    - Adaptive prompt weighting based on mask confidence
    """
    
    def __init__(self, embed_dim=256):
        super(PromptEncoder, self).__init__()
        self.embed_dim = embed_dim
        
        # ============ Coarse Mask Encoder ============
        # Process coarse mask from partial decoder
        self.coarse_mask_encode = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.coarse_proj = nn.Sequential(
            nn.Linear(64, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # ============ Exposure Correction Integration ============
        # Process illumination information
        self.exposure_encode = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.exposure_proj = nn.Sequential(
            nn.Linear(32, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, embed_dim // 2)
        )
        
        # ============ Confidence-based Weighting ============
        # Adaptive weight based on coarse mask confidence
        self.confidence_estimator = nn.Linear(embed_dim // 2, 1)
        
        # ============ Bounding Box Encoding ============
        # External prompt support (secondary)
        self.bbox_embed = nn.Sequential(
            nn.Linear(4, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # ============ Point Encoding ============
        self.point_embed = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # ============ Mask Prompt Encoding ============
        self.mask_embed = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.mask_proj = nn.Sequential(
            nn.Linear(64, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # ============ Learnable Prompt ============
        self.learnable_prompt = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
    
    def forward(
        self,
        coarse_mask=None,
        illumination_map=None,
        bbox=None,
        points=None,
        mask=None
    ):
        """Encode prompts combining coarse mask guidance with optional external prompts.
        
        Args:
            coarse_mask: Coarse mask from encoder's PD [B, 1, H, W] (primary prompt)
            illumination_map: Illumination map from exposure correction [B, 1, H, W]
            bbox: Bounding box coordinates [B, 4] (secondary prompt)
            points: Point coordinates [B, 2] (secondary prompt)
            mask: Binary mask [B, 1, H, W] (secondary prompt)
            
        Returns:
            prompt_embed: Combined prompt embedding [B, embed_dim]
            confidence: Confidence score of prompt
        """
        
        embeddings = []
        confidence_score = None
        
        # ============ Primary Prompt: Coarse Mask ============
        if coarse_mask is not None:
            batch_size = coarse_mask.shape[0]
            coarse_feat = self.coarse_mask_encode(coarse_mask)  # [B, 64, 1, 1]
            coarse_feat = coarse_feat.view(batch_size, -1)  # [B, 64]
            coarse_embed = self.coarse_proj(coarse_feat)  # [B, embed_dim]
            embeddings.append(coarse_embed)
            
            # Estimate confidence based on coarse mask
            # Use middle portion of embedding for confidence
            confidence_input = coarse_embed[:, :self.embed_dim//2]
            confidence_score = torch.sigmoid(self.confidence_estimator(confidence_input))
        
        # ============ Exposure Correction Integration ============
        if illumination_map is not None:
            batch_size = illumination_map.shape[0]
            exp_feat = self.exposure_encode(illumination_map)  # [B, 32, 1, 1]
            exp_feat = exp_feat.view(batch_size, -1)  # [B, 32]
            exp_embed = self.exposure_proj(exp_feat)  # [B, embed_dim//2]
            # Pad exposure embedding to match embed_dim
            exp_embed = F.pad(exp_embed, (0, self.embed_dim - exp_embed.shape[-1]))
            embeddings.append(exp_embed * 0.3)  # Weight exposure contribution
        
        # ============ Secondary Prompts ============
        if bbox is not None and coarse_mask is None:
            bbox_embed = self.bbox_embed(bbox)
            embeddings.append(bbox_embed)
        
        elif points is not None and coarse_mask is None:
            points_embed = self.point_embed(points)
            embeddings.append(points_embed)
        
        elif mask is not None and coarse_mask is None:
            batch_size = mask.shape[0]
            mask_feat = self.mask_embed(mask)  # [B, 64, 1, 1]
            mask_feat = mask_feat.view(batch_size, -1)  # [B, 64]
            mask_embed = self.mask_proj(mask_feat)  # [B, embed_dim]
            embeddings.append(mask_embed)
        
        # ============ Fallback to Learnable Prompt ============
        if len(embeddings) == 0:
            return self.learnable_prompt, torch.tensor(0.5, device=torch.device('cpu'))
        
        # ============ Combine Embeddings ============
        if len(embeddings) == 1:
            prompt_embed = embeddings[0]
        else:
            # Average multiple embeddings with confidence-based weighting
            prompt_embed = torch.stack(embeddings, dim=0).mean(dim=0)
        
        if confidence_score is None:
            confidence_score = torch.tensor(0.8, device=prompt_embed.device)
        
        return prompt_embed, confidence_score
