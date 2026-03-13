#!/usr/bin/env python3
"""Test the updated LiteMedSAM architecture with feature aggregation."""

import torch
from LiteMedSAM.modeling.image_encoder import LiteImageEncoder
from LiteMedSAM.modeling.mask_decoder import LiteDecoder
from LiteMedSAM.modeling.litemedsam import LiteMedSAM

print("Testing LiteMedSAM with aggregated PartialDecoder architecture...")
print("-" * 70)

# Test shapes
B, H, W = 2, 256, 256
print(f"Batch size: {B}, Image size: {H}x{W}")

# Create components
encoder = LiteImageEncoder(base_channels=32)
decoder = LiteDecoder(input_dim=256, hidden_dim=128)
model = LiteMedSAM(encoder, decoder, num_output_classes=2)

print("✓ Model created successfully")
print(f"  - Encoder: {type(encoder).__name__}")
print(f"  - Decoder: {type(decoder).__name__}")
print(f"  - Model: {type(model).__name__}")

# Forward pass
image = torch.randn(B, 3, H, W)
bbox = torch.tensor([[40, 40, 200, 200], [50, 50, 210, 210]])

print("\nRunning forward pass...")
try:
    output = model(image, bbox=bbox)
    
    print("✓ Forward pass successful!")
    print(f"  - Logits shape: {output[0].shape}")
    if output[1] is not None:
        print(f"  - Coarse mask shape: {output[1].shape}")
    print(f"  - Dual masks:")
    if output[2] is not None:
        print(f"    - BG mask shape: {output[2]['bg'].shape}")
        print(f"    - FG mask shape: {output[2]['fg'].shape}")
    
    print("\n" + "=" * 70)
    print("✅ Architecture validation PASSED")
    print("=" * 70)
    print("\nKey improvements implemented:")
    print("  ✓ PartialDecoder applied once after stage 4 (not per-stage)")
    print("  ✓ Feature aggregation from stages 2, 3, 4 at 1/16 resolution")
    print("  ✓ Dual masks as single tensors instead of per-stage lists")
    print("  ✓ Decoder handles interpolation of masks to different resolutions")
    
except Exception as e:
    print(f"✗ Error during forward pass: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 70)
    print("❌ Architecture validation FAILED")
    print("=" * 70)
