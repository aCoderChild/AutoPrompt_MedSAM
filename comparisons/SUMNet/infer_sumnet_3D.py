"""Ensemble inference for SUMNet on 3D medical images (slice-by-slice).

SUMNet combines 8 semantic segmentation models for 3D volumetric segmentation.
Each slice is processed independently and predictions are aggregated.
"""

import argparse
import os
import numpy as np
import torch
from pathlib import Path
from collections import Counter

try:
    from mmseg.apis import init_segmentor, inference_segmentor
except ImportError:
    raise ImportError(
        "MMSegmentation not found. Install with: "
        "pip install mmengine mmsegmentation"
    )


SUMMNET_MODELS = [
    'fcn',
    'upernet', 
    'dnlnet',
    'deeplabv3',
    'ocrnet',
    'psanet',
    'danet',
    'ccnet'
]


class EnsembleSUMNet3D:
    """Ensemble SUMNet for 3D medical image segmentation (slice-based)."""
    
    def __init__(self, model_configs, checkpoints, device='cuda'):
        """
        Initialize ensemble with model configs and checkpoints.
        
        Args:
            model_configs: List of paths to MMSegmentation config files
            checkpoints: List of paths to model checkpoints
            device: 'cuda' or 'cpu'
        """
        self.models = []
        self.device = device
        
        assert len(model_configs) == len(checkpoints) == 8, \
            f"Expected 8 models, got {len(model_configs)}"
        
        for config, checkpoint, name in zip(model_configs, checkpoints, SUMMNET_MODELS):
            print(f"Loading {name}...")
            try:
                model = init_segmentor(
                    config,
                    checkpoint,
                    device=device
                )
                self.models.append(model)
                print(f"  ✓ {name} loaded")
            except Exception as e:
                print(f"  ✗ Failed to load {name}: {e}")
                raise
    
    def inference_3d(self, image_3d_path):
        """
        Run ensemble inference on 3D image (slice-by-slice).
        
        Args:
            image_3d_path: Path to 3D image (npz format with shape D,H,W)
            
        Returns:
            ensemble_pred_3d: 3D ensemble predictions
            confidence_3d: 3D confidence maps
        """
        # Load 3D image
        data = np.load(image_3d_path)
        image_3d = data['image']
        
        num_slices = image_3d.shape[0]
        pred_3d = []
        conf_3d = []
        
        print(f"Processing {num_slices} slices...")
        
        for slice_idx in range(num_slices):
            if (slice_idx + 1) % 10 == 0:
                print(f"  Slice {slice_idx + 1}/{num_slices}")
            
            slice_2d = image_3d[slice_idx]
            
            # Create temporary 2D file or convert to appropriate format
            all_predictions = []
            
            for model in self.models:
                pred = inference_segmentor(model, slice_2d)
                all_predictions.append(pred)
            
            # Aggregate slice predictions
            ensemble_pred, confidence = self._majority_voting(all_predictions)
            pred_3d.append(ensemble_pred)
            conf_3d.append(confidence)
        
        return np.stack(pred_3d), np.stack(conf_3d)
    
    @staticmethod
    def _majority_voting(predictions):
        """
        Aggregate predictions through majority voting.
        
        Args:
            predictions: List of 8 prediction arrays
            
        Returns:
            ensemble_pred: Majority vote result
            confidence: Voting confidence (0-1)
        """
        pred_stack = np.stack(predictions, axis=0)  # (8, H, W)
        
        ensemble_pred = np.zeros_like(pred_stack[0])
        confidence = np.zeros_like(pred_stack[0], dtype=np.float32)
        
        for h in range(pred_stack.shape[1]):
            for w in range(pred_stack.shape[2]):
                votes = pred_stack[:, h, w]
                counter = Counter(votes)
                most_common_class, count = counter.most_common(1)[0]
                ensemble_pred[h, w] = most_common_class
                confidence[h, w] = count / len(predictions)
        
        return ensemble_pred.astype(np.uint8), confidence


def save_results_3d(seg_pred, confidence, output_path):
    """Save 3D segmentation and confidence maps."""
    os.makedirs(output_path, exist_ok=True)
    
    # Save segmentation as NPY
    np.save(os.path.join(output_path, 'segmentation_3d.npy'), seg_pred)
    
    # Save confidence map
    np.save(os.path.join(output_path, 'confidence_3d.npy'), confidence)
    
    # Save as npz
    npz_path = os.path.join(output_path, 'predictions.npz')
    np.savez(npz_path, segmentation=seg_pred, confidence=confidence)
    
    print(f"  Saved to {output_path}")
    print(f"    - Segmentation shape: {seg_pred.shape}")
    print(f"    - Confidence shape: {confidence.shape}")


def main():
    parser = argparse.ArgumentParser(
        description='SUMNet ensemble inference on 3D medical images'
    )
    parser.add_argument(
        '-model_configs',
        nargs=8,
        required=True,
        help='Paths to 8 MMSegmentation config files'
    )
    parser.add_argument(
        '-checkpoints',
        nargs=8,
        required=True,
        help='Paths to 8 model checkpoints'
    )
    parser.add_argument(
        '-data_root',
        required=True,
        help='Path to input 3D images (npz format)'
    )
    parser.add_argument(
        '-pred_save_dir',
        required=True,
        help='Path to save predictions'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference'
    )
    
    args = parser.parse_args()
    
    # Initialize ensemble
    print("Initializing SUMNet Ensemble for 3D...")
    ensemble = EnsembleSUMNet3D(
        model_configs=args.model_configs,
        checkpoints=args.checkpoints,
        device=args.device
    )
    print(f"Ensemble ready: 8 models loaded\n")
    
    # Get image files
    image_files = sorted(Path(args.data_root).glob('*.npz'))
    print(f"Found {len(image_files)} 3D images\n")
    
    for idx, image_file in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {image_file.name}")
        
        try:
            # Run ensemble inference
            ensemble_pred_3d, confidence_3d = ensemble.inference_3d(str(image_file))
            
            # Save results
            output_dir = os.path.join(args.pred_save_dir, image_file.stem)
            save_results_3d(ensemble_pred_3d, confidence_3d, output_dir)
            print()
            
        except Exception as e:
            print(f"  Error: {e}\n")
            continue
    
    print("3D ensemble inference completed.")
    print(f"Results saved to: {args.pred_save_dir}")


if __name__ == '__main__':
    main()
