"""
Inference script for U-Net on 3D medical images.
"""

import argparse
import os
import numpy as np
import torch
from pathlib import Path


def load_model(checkpoint_path, device):
    """Load pre-trained U-Net model."""
    from train_unet import UNet
    
    model = UNet(in_channels=1, out_channels=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_3d_image(image_path):
    """Load and preprocess a 3D medical image."""
    data = np.load(image_path)
    image = data['image']  # Shape: (D, H, W)
    
    # Normalize each slice
    image = (image - image.mean()) / (image.std() + 1e-8)
    
    return image


@torch.no_grad()
def inference_3D(model, data_root, pred_save_dir, save_overlay=False,
                png_save_dir=None, num_workers=2, device='cuda'):
    """Run inference on 3D images, processing slice-by-slice."""
    os.makedirs(pred_save_dir, exist_ok=True)
    if save_overlay and png_save_dir:
        os.makedirs(png_save_dir, exist_ok=True)
    
    # Get list of 3D images
    image_files = list(Path(data_root).glob('*.npz'))
    print(f"Found {len(image_files)} 3D images for inference")
    
    for idx, image_file in enumerate(image_files):
        print(f"Processing {idx+1}/{len(image_files)}: {image_file.name}")
        
        # Load 3D image
        image_3d = preprocess_3d_image(str(image_file))
        num_slices = image_3d.shape[0]
        
        # Process each slice
        predictions_3d = np.zeros(image_3d.shape, dtype=np.uint8)
        
        for slice_idx in range(num_slices):
            slice_2d = image_3d[slice_idx]
            slice_tensor = torch.from_numpy(slice_2d).float().unsqueeze(0).unsqueeze(0)
            slice_tensor = slice_tensor.to(device)
            
            # Inference
            output = model(slice_tensor)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]
            predictions_3d[slice_idx] = pred
        
        # Save 3D prediction
        pred_path = os.path.join(pred_save_dir, f'{image_file.stem}_pred.npz')
        np.savez(pred_path, pred=predictions_3d)
        
        print(f"Saved 3D prediction: {pred_path}")


def main():
    parser = argparse.ArgumentParser(description='3D inference with U-Net')
    parser.add_argument('-checkpoint', required=True, help='Path to trained model')
    parser.add_argument('-data_root', required=True, help='Path to 3D images')
    parser.add_argument('-pred_save_dir', required=True, help='Path to save predictions')
    parser.add_argument('--save_overlay', action='store_true', help='Save overlay images')
    parser.add_argument('-png_save_dir', help='Path to save overlay images')
    parser.add_argument('-num_workers', type=int, default=2, help='Number of workers')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(args.checkpoint, device)
    inference_3D(model, args.data_root, args.pred_save_dir,
                args.save_overlay, args.png_save_dir, args.num_workers, device)
    
    print("3D inference completed!")


if __name__ == '__main__':
    main()
