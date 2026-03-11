"""
3D Inference script for MedSAM (slice-by-slice).
"""

import argparse
import os
import numpy as np
import torch
from pathlib import Path


def load_medsam_model(checkpoint_path, device):
    """Load MedSAM model."""
    try:
        from segment_anything import sam_model_registry
        
        model_type = "vit_b"
        model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        model.to(device)
        model.eval()
        return model
    
    except ImportError:
        print("segment-anything not installed")
        return None


def preprocess_3d_image(image_path):
    """Load 3D medical image."""
    data = np.load(image_path)
    image_3d = data['image']
    image_3d = (image_3d - image_3d.mean()) / (image_3d.std() + 1e-8)
    return image_3d


def get_bounding_box_3d(label_3d, slice_idx):
    """Extract bounding box from label for specific slice."""
    label_slice = label_3d[slice_idx]
    if not np.any(label_slice):
        return None
    
    rows = np.any(label_slice, axis=1)
    cols = np.any(label_slice, axis=0)
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return np.array([x_min, y_min, x_max, y_max])


@torch.no_grad()
def inference_medsam_2D_slice(model, slice_2d, bbox=None, device='cuda'):
    """Run MedSAM on a single 2D slice."""
    # Normalize to [0, 255]
    slice_normalized = ((slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8) * 255).astype(np.uint8)
    
    model.set_image(slice_normalized)
    
    if bbox is not None:
        try:
            masks, _, _ = model.predict(
                point_coords=None,
                point_labels=None,
                box=bbox,
                multimask_output=False
            )
            return masks[0]
        except:
            return np.zeros_like(slice_2d)
    else:
        return np.zeros_like(slice_2d)


@torch.no_grad()
def inference_3D(checkpoint_path, data_root, pred_save_dir, device='cuda'):
    """Run 3D inference with MedSAM (slice-by-slice)."""
    os.makedirs(pred_save_dir, exist_ok=True)
    
    model = load_medsam_model(checkpoint_path, device)
    if model is None:
        return
    
    image_files = list(Path(data_root).glob('*.npz'))
    print(f"Found {len(image_files)} 3D images")
    
    for idx, image_file in enumerate(image_files):
        print(f"Processing {idx+1}/{len(image_files)}: {image_file.name}")
        
        try:
            # Load 3D image
            image_3d = preprocess_3d_image(str(image_file))
            num_slices = image_3d.shape[0]
            
            # Try to load labels for bounding boxes
            label_3d = None
            label_path = image_file.parent / f"{image_file.stem}_label.npz"
            if label_path.exists():
                label_data = np.load(label_path)
                label_3d = label_data['label']
            
            # Process each slice
            predictions_3d = np.zeros(image_3d.shape, dtype=np.uint8)
            
            for slice_idx in range(num_slices):
                slice_2d = image_3d[slice_idx]
                
                # Get bounding box if available
                bbox = None
                if label_3d is not None:
                    bbox = get_bounding_box_3d(label_3d, slice_idx)
                
                # Run inference
                pred = inference_medsam_2D_slice(model, slice_2d, bbox, device)
                predictions_3d[slice_idx] = (pred > 0.5).astype(np.uint8)
            
            # Save 3D prediction
            pred_path = os.path.join(pred_save_dir, f'{image_file.stem}_pred.npz')
            np.savez(pred_path, pred=predictions_3d)
            print(f"Saved: {pred_path}")
        
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")
            continue


def main():
    parser = argparse.ArgumentParser(description='3D inference with MedSAM')
    parser.add_argument('-checkpoint', required=True, help='Path to MedSAM checkpoint')
    parser.add_argument('-data_root', required=True, help='Path to 3D images')
    parser.add_argument('-pred_save_dir', required=True, help='Path to save predictions')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    inference_3D(args.checkpoint, args.data_root, args.pred_save_dir, device)
    
    print("3D inference completed!")


if __name__ == '__main__':
    main()
