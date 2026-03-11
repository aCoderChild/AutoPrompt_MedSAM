"""
2D Inference script for MedSAM (zero-shot or fine-tuned).
"""

import argparse
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import cv2


def load_medsam_model(checkpoint_path, device):
    """Load MedSAM model."""
    try:
        from segment_anything import sam_model_registry
        
        model_type = "vit_b"  # Can be vit_b, vit_l, vit_h
        model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        model.to(device)
        model.eval()
        return model
    
    except ImportError:
        print("segment-anything not installed. Install with: pip install segment-anything")
        return None


def preprocess_image(image_path):
    """Load and preprocess image."""
    if image_path.endswith('.npz'):
        data = np.load(image_path)
        image = data['image']
    else:
        image = np.array(Image.open(image_path).convert('L'))
    
    # Normalize to [0, 255]
    if image.max() > 1.0:
        image = image.astype(np.float32)
    else:
        image = (image * 255).astype(np.uint8)
    
    return image


def get_bounding_box_from_label(label):
    """Extract bounding box from ground truth label."""
    if not np.any(label):
        return None
    
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return np.array([x_min, y_min, x_max, y_max])


@torch.no_grad()
def inference_medsam_2D(model, image, bbox=None, device='cuda'):
    """Run MedSAM inference on a 2D image."""
    # Ensure image is proper format
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Set image for predictor
    model.set_image(image)
    
    # Use bounding box prompt if provided
    if bbox is not None:
        masks, _, _ = model.predict(
            point_coords=None,
            point_labels=None,
            box=bbox,
            multimask_output=False
        )
        return masks[0]
    else:
        # Return placeholder if no prompt
        return np.zeros_like(image)


@torch.no_grad()
def inference_2D(checkpoint_path, data_root, pred_save_dir, save_overlay=False,
                png_save_dir=None, device='cuda'):
    """Run 2D inference with MedSAM."""
    os.makedirs(pred_save_dir, exist_ok=True)
    if save_overlay and png_save_dir:
        os.makedirs(png_save_dir, exist_ok=True)
    
    model = load_medsam_model(checkpoint_path, device)
    if model is None:
        return
    
    image_files = list(Path(data_root).glob('*.npz')) + \
                 list(Path(data_root).glob('*.png')) + \
                 list(Path(data_root).glob('*.jpg'))
    
    print(f"Found {len(image_files)} images")
    
    for idx, image_file in enumerate(image_files):
        print(f"Processing {idx+1}/{len(image_files)}: {image_file.name}")
        
        try:
            # Load image
            image = preprocess_image(str(image_file))
            
            # Try to load label for bounding box
            label_path = image_file.parent / f"{image_file.stem}_label.npz"
            bbox = None
            if label_path.exists():
                label_data = np.load(label_path)
                bbox = get_bounding_box_from_label(label_data['label'])
            
            # Run inference
            pred = inference_medsam_2D(model, image, bbox, device)
            
            # Save prediction
            pred_path = os.path.join(pred_save_dir, f'{image_file.stem}_pred.npz')
            np.savez(pred_path, pred=pred.astype(np.uint8))
            
            # Save overlay if requested
            if save_overlay and png_save_dir:
                overlay = create_overlay(image, pred)
                overlay_path = os.path.join(png_save_dir, f'{image_file.stem}_overlay.png')
                Image.fromarray(overlay).save(overlay_path)
        
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")
            continue


def create_overlay(image, prediction, alpha=0.5):
    """Create overlay of prediction on original image."""
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    
    # Create RGB overlay
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Ensure prediction is binary
    pred_binary = (prediction > 0.5).astype(np.uint8)
    
    pred_colored = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    pred_colored[pred_binary == 1] = [0, 255, 0]
    
    overlay = cv2.addWeighted(overlay, 1-alpha, pred_colored, alpha, 0)
    return overlay


def main():
    parser = argparse.ArgumentParser(description='2D inference with MedSAM')
    parser.add_argument('-checkpoint', required=True, help='Path to MedSAM checkpoint')
    parser.add_argument('-data_root', required=True, help='Path to input images')
    parser.add_argument('-pred_save_dir', required=True, help='Path to save predictions')
    parser.add_argument('--save_overlay', action='store_true', help='Save overlays')
    parser.add_argument('-png_save_dir', help='Path to save overlays')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    inference_2D(args.checkpoint, args.data_root, args.pred_save_dir,
                args.save_overlay, args.png_save_dir, device)
    
    print("Inference completed!")


if __name__ == '__main__':
    main()
