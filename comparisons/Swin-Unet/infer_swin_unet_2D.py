"""
2D Inference script for Swin-Unet.
"""

import argparse
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import cv2


def load_model(checkpoint_path, device):
    from train_swin_unet import SwinUnet
    model = SwinUnet(in_channels=1, out_channels=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    if image_path.endswith('.npz'):
        data = np.load(image_path)
        image = data['image']
    else:
        image = np.array(Image.open(image_path).convert('L'))
    
    image = (image - image.mean()) / (image.std() + 1e-8)
    image_tensor = torch.from_numpy(image).float()
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    
    return image_tensor, image


@torch.no_grad()
def inference_2D(model, data_root, pred_save_dir, save_overlay=False, 
                png_save_dir=None, num_workers=2, device='cuda'):
    os.makedirs(pred_save_dir, exist_ok=True)
    if save_overlay and png_save_dir:
        os.makedirs(png_save_dir, exist_ok=True)
    
    image_files = list(Path(data_root).glob('*.npz')) + \
                 list(Path(data_root).glob('*.png')) + \
                 list(Path(data_root).glob('*.jpg'))
    
    print(f"Found {len(image_files)} images")
    
    for idx, image_file in enumerate(image_files):
        print(f"Processing {idx+1}/{len(image_files)}: {image_file.name}")
        
        image_tensor, original_image = preprocess_image(str(image_file))
        image_tensor = image_tensor.to(device)
        
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]
        
        pred_path = os.path.join(pred_save_dir, f'{image_file.stem}_pred.npz')
        np.savez(pred_path, pred=pred)
        
        if save_overlay and png_save_dir:
            overlay = create_overlay(original_image, pred)
            overlay_path = os.path.join(png_save_dir, f'{image_file.stem}_overlay.png')
            Image.fromarray(overlay).save(overlay_path)


def create_overlay(image, prediction, alpha=0.5):
    image_normalized = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    overlay = cv2.cvtColor(image_normalized, cv2.COLOR_GRAY2BGR)
    pred_colored = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    pred_colored[prediction == 1] = [0, 255, 0]
    overlay = cv2.addWeighted(overlay, 1-alpha, pred_colored, alpha, 0)
    return overlay


def main():
    parser = argparse.ArgumentParser(description='2D inference with Swin-Unet')
    parser.add_argument('-checkpoint', required=True, help='Path to trained model')
    parser.add_argument('-data_root', required=True, help='Path to input images')
    parser.add_argument('-pred_save_dir', required=True, help='Path to save predictions')
    parser.add_argument('--save_overlay', action='store_true')
    parser.add_argument('-png_save_dir', help='Path to save overlays')
    parser.add_argument('-num_workers', type=int, default=2)
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(args.checkpoint, device)
    inference_2D(model, args.data_root, args.pred_save_dir, 
                args.save_overlay, args.png_save_dir, args.num_workers, device)
    print("Inference completed!")


if __name__ == '__main__':
    main()
