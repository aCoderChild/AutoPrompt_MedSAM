"""
3D Inference script for nnU-Net model on medical images.
"""

import argparse
import os
import numpy as np
import torch
from pathlib import Path


def preprocess_3d_image(image_path):
    """Load and preprocess a 3D medical image."""
    data = np.load(image_path)
    image = data['image']  # Shape: (D, H, W)
    
    # Normalize
    image = (image - image.mean()) / (image.std() + 1e-8)
    
    return image


def run_nnunet_inference_3d(checkpoint_path, image_3d):
    """Run nnU-Net inference on a 3D volume."""
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
        
        predictor.initialize_from_trained_model_folder(checkpoint_path)
        
        # Add channel dimension if needed
        if image_3d.ndim == 3:
            image_3d = image_3d[np.newaxis]
        
        # Run prediction
        prediction = predictor.predict_single_npy_array(image_3d)
        
        return prediction
    
    except ImportError:
        print("nnUNetv2 not installed. Install with: pip install nnunetv2")
        return None


def inference_3D(checkpoint_path, data_root, pred_save_dir, png_save_dir=None, num_workers=2):
    """Run 3D inference on medical images."""
    os.makedirs(pred_save_dir, exist_ok=True)
    
    image_files = list(Path(data_root).glob('*.npz'))
    print(f"Found {len(image_files)} 3D images for inference")
    
    for idx, image_file in enumerate(image_files):
        print(f"Processing {idx+1}/{len(image_files)}: {image_file.name}")
        
        try:
            # Load 3D image
            image_3d = preprocess_3d_image(str(image_file))
            
            # Run nnU-Net inference
            pred = run_nnunet_inference_3d(checkpoint_path, image_3d)
            
            if pred is None:
                continue
            
            # Save 3D prediction
            pred_path = os.path.join(pred_save_dir, f'{image_file.stem}_pred.npz')
            np.savez(pred_path, pred=pred)
            
            print(f"Saved 3D prediction: {pred_path}")
        
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")
            continue


def main():
    parser = argparse.ArgumentParser(description='3D inference with nnU-Net')
    parser.add_argument('-checkpoint', required=True, help='Path to nnU-Net model')
    parser.add_argument('-data_root', required=True, help='Path to 3D images')
    parser.add_argument('-pred_save_dir', required=True, help='Path to save predictions')
    parser.add_argument('-png_save_dir', help='Path to save overlay images')
    parser.add_argument('-num_workers', type=int, default=2, help='Number of workers')
    
    args = parser.parse_args()
    
    inference_3D(args.checkpoint, args.data_root, args.pred_save_dir,
                args.png_save_dir, args.num_workers)
    
    print("3D inference completed!")


if __name__ == '__main__':
    main()
