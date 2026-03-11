"""
2D Inference script for nnU-Net model on medical images.
Reference implementation for using nnUNetv2.
"""

import argparse
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import cv2


def preprocess_image(image_path, grey=False):
    """Load and preprocess a single 2D image."""
    if image_path.endswith('.npz'):
        data = np.load(image_path)
        image = data['image']
    else:
        if grey:
            image = np.array(Image.open(image_path).convert('L'))
        else:
            image = np.array(Image.open(image_path))
    
    # Normalize
    image = (image - image.mean()) / (image.std() + 1e-8)
    
    return image


def run_nnunet_inference(checkpoint_path, image_path, grey=False):
    """Run nnU-Net inference on a single image."""
    # This is a template - actual nnU-Net inference uses their API
    # Example of how to integrate nnUNetv2
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        
        # Initialize predictor with model weights
        predictor.initialize_from_trained_model_folder(checkpoint_path)
        
        # Load and preprocess image
        image = preprocess_image(image_path, grey)
        
        # Add channel dimension if needed
        if image.ndim == 2:
            image = image[np.newaxis]
        
        # Run prediction
        prediction = predictor.predict_single_npy_array(image)
        
        return prediction
    
    except ImportError:
        print("nnUNetv2 not installed. Install with: pip install nnunetv2")
        return None


@torch.no_grad()
def inference_2D(checkpoint_path, data_root, pred_save_dir, grey=False,
                save_overlay=False, png_save_dir=None, num_workers=2):
    """Run 2D inference on multiple images."""
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
        
        try:
            # Run nnU-Net inference
            pred = run_nnunet_inference(checkpoint_path, str(image_file), grey)
            
            if pred is None:
                continue
            
            # Save prediction
            pred_path = os.path.join(pred_save_dir, f'{image_file.stem}_pred.npz')
            np.savez(pred_path, pred=pred)
            
            # Save overlay if requested
            if save_overlay and png_save_dir:
                original_image = preprocess_image(str(image_file), grey)
                overlay = create_overlay(original_image, pred)
                overlay_path = os.path.join(png_save_dir, f'{image_file.stem}_overlay.png')
                Image.fromarray(overlay).save(overlay_path)
        
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")
            continue


def create_overlay(image, prediction, alpha=0.5):
    """Create overlay of prediction on original image."""
    # Normalize image to 0-255
    image_normalized = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    
    # Create RGB overlay
    overlay = cv2.cvtColor(image_normalized, cv2.COLOR_GRAY2BGR)
    pred_colored = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    pred_colored[prediction > 0] = [0, 255, 0]  # Green for segmentation
    
    # Blend
    overlay = cv2.addWeighted(overlay, 1-alpha, pred_colored, alpha, 0)
    return overlay


def main():
    parser = argparse.ArgumentParser(description='2D inference with nnU-Net')
    parser.add_argument('-checkpoint', required=True, help='Path to nnU-Net model checkpoint')
    parser.add_argument('-data_root', required=True, help='Path to input images')
    parser.add_argument('-pred_save_dir', required=True, help='Path to save predictions')
    parser.add_argument('--grey', action='store_true', help='Images are grayscale')
    parser.add_argument('--save_overlay', action='store_true', help='Save overlay images')
    parser.add_argument('-png_save_dir', help='Path to save overlay images')
    parser.add_argument('-num_workers', type=int, default=2, help='Number of workers')
    
    args = parser.parse_args()
    
    inference_2D(args.checkpoint, args.data_root, args.pred_save_dir,
                args.grey, args.save_overlay, args.png_save_dir, args.num_workers)
    
    print("Inference completed!")


if __name__ == '__main__':
    main()
