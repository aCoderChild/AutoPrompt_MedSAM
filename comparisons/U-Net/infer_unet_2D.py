"""
Inference script for U-Net on 2D medical images.
"""

import argparse
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import cv2


def load_model(checkpoint_path, device):
    """Load pre-trained U-Net model."""
    # Import model from train script
    from train_unet import UNet
    
    model = UNet(in_channels=1, out_channels=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    """Load and preprocess a single 2D image."""
    # Load image (assuming npz format from preprocessing)
    if image_path.endswith('.npz'):
        data = np.load(image_path)
        image = data['image']
    else:
        # Load from standard image formats
        image = np.array(Image.open(image_path).convert('L'))
    
    # Normalize
    image = (image - image.mean()) / (image.std() + 1e-8)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).float()
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    
    return image_tensor, image


@torch.no_grad()
def inference_2D(model, data_root, pred_save_dir, save_overlay=False, 
                png_save_dir=None, num_workers=2, device='cuda'):
    """Run inference on 2D images."""
    os.makedirs(pred_save_dir, exist_ok=True)
    if save_overlay and png_save_dir:
        os.makedirs(png_save_dir, exist_ok=True)
    
    # Get list of images
    image_files = list(Path(data_root).glob('*.npz')) + \
                 list(Path(data_root).glob('*.png')) + \
                 list(Path(data_root).glob('*.jpg'))
    
    print(f"Found {len(image_files)} images for inference")
    
    for idx, image_file in enumerate(image_files):
        print(f"Processing {idx+1}/{len(image_files)}: {image_file.name}")
        
        # Preprocess
        image_tensor, original_image = preprocess_image(str(image_file))
        image_tensor = image_tensor.to(device)
        
        # Inference
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]
        
        # Save prediction
        pred_path = os.path.join(pred_save_dir, f'{image_file.stem}_pred.npz')
        np.savez(pred_path, pred=pred)
        
        # Save overlay if requested
        if save_overlay and png_save_dir:
            overlay = create_overlay(original_image, pred)
            overlay_path = os.path.join(png_save_dir, f'{image_file.stem}_overlay.png')
            Image.fromarray(overlay).save(overlay_path)


def create_overlay(image, prediction, alpha=0.5):
    """Create overlay of prediction on original image."""
    # Normalize image to 0-255
    image_normalized = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    
    # Create RGB overlay
    overlay = cv2.cvtColor(image_normalized, cv2.COLOR_GRAY2BGR)
    pred_colored = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    pred_colored[prediction == 1] = [0, 255, 0]  # Green for segmentation
    
    # Blend
    overlay = cv2.addWeighted(overlay, 1-alpha, pred_colored, alpha, 0)
    return overlay


def main():
    parser = argparse.ArgumentParser(description='2D inference with U-Net')
    parser.add_argument('-checkpoint', required=True, help='Path to trained model')
    parser.add_argument('-data_root', required=True, help='Path to input images')
    parser.add_argument('-pred_save_dir', required=True, help='Path to save predictions')
    parser.add_argument('--save_overlay', action='store_true', help='Save overlay images')
    parser.add_argument('-png_save_dir', help='Path to save overlay images')
    parser.add_argument('-num_workers', type=int, default=2, help='Number of workers')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(args.checkpoint, device)
    inference_2D(model, args.data_root, args.pred_save_dir, 
                args.save_overlay, args.png_save_dir, args.num_workers, device)
    
    print("Inference completed!")


if __name__ == '__main__':
    main()
