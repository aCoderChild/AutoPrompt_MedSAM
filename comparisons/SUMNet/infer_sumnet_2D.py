"""Ensemble inference for SUMNet on 2D medical images.

SUMNet combines 8 semantic segmentation models:
FCN, UPerNet, DNLNet, DeepLabV3, OCRNet, PSANet, DANet, CCNet

Ensemble predictions are aggregated through majority voting.
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


class EnsembleSUMNet:
    """Ensemble SUMNet model combining 8 segmentation architectures."""
    
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
    
    def inference(self, image):
        """
        Run ensemble inference on image.
        
        Args:
            image: Input image path or array
            
        Returns:
            ensemble_pred: Ensemble consensus predictions
            confidence: Confidence map (voting agreement)
        """
        all_predictions = []
        
        for model in self.models:
            print(f"Running model {len(all_predictions) + 1}/8...")
            pred = inference_segmentor(model, image)
            all_predictions.append(pred)
        
        # Aggregate predictions through majority voting
        ensemble_pred, confidence = self._majority_voting(all_predictions)
        
        return ensemble_pred, confidence
    
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
        # Stack predictions
        pred_stack = np.stack(predictions, axis=0)  # (8, H, W)
        
        # Majority voting
        ensemble_pred = np.zeros_like(pred_stack[0])
        confidence = np.zeros_like(pred_stack[0], dtype=np.float32)
        
        for h in range(pred_stack.shape[1]):
            for w in range(pred_stack.shape[2]):
                votes = pred_stack[:, h, w]
                counter = Counter(votes)
                most_common_class, count = counter.most_common(1)[0]
                ensemble_pred[h, w] = most_common_class
                confidence[h, w] = count / len(predictions)  # 0.125-1.0
        
        return ensemble_pred.astype(np.uint8), confidence
    
    @staticmethod
    def _weighted_averaging(predictions, weights=None):
        """
        Alternative: Weighted average of predictions.
        Useful for soft voting with model-specific confidence weights.
        
        Args:
            predictions: List of probability maps
            weights: Model weights (default: equal)
            
        Returns:
            ensemble_pred: Weighted average predictions
        """
        if weights is None:
            weights = np.ones(len(predictions)) / len(predictions)
        
        weighted_avg = np.zeros_like(predictions[0], dtype=np.float32)
        for pred, w in zip(predictions, weights):
            weighted_avg += pred.astype(np.float32) * w
        
        return np.argmax(weighted_avg, axis=-1)


def save_results(seg_pred, confidence, output_path, overlay_path=None, original_image=None):
    """Save segmentation and confidence maps."""
    os.makedirs(output_path, exist_ok=True)
    
    # Save segmentation as NPY
    np.save(os.path.join(output_path, 'segmentation.npy'), seg_pred)
    
    # Save confidence map
    np.save(os.path.join(output_path, 'confidence.npy'), confidence)
    
    # Save as PNG for visualization
    try:
        from PIL import Image
        Image.fromarray((seg_pred * 255).astype(np.uint8)).save(
            os.path.join(output_path, 'segmentation.png')
        )
        Image.fromarray((confidence * 255).astype(np.uint8)).save(
            os.path.join(output_path, 'confidence.png')
        )
        print(f"  Saved to {output_path}")
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description='SUMNet ensemble inference on 2D images'
    )
    parser.add_argument(
        '-model_configs',
        nargs=8,
        required=True,
        help='Paths to 8 MMSegmentation config files (FCN, UPerNet, DNLNet, DeepLabV3, OCRNet, PSANet, DANet, CCNet)'
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
        help='Path to input images (npz format)'
    )
    parser.add_argument(
        '-pred_save_dir',
        required=True,
        help='Path to save predictions'
    )
    parser.add_argument(
        '--save_overlay',
        action='store_true',
        help='Save overlay visualizations'
    )
    parser.add_argument(
        '-png_save_dir',
        default=None,
        help='Path to save PNG overlays'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference'
    )
    
    args = parser.parse_args()
    
    # Initialize ensemble
    print("Initializing SUMNet Ensemble...")
    ensemble = EnsembleSUMNet(
        model_configs=args.model_configs,
        checkpoints=args.checkpoints,
        device=args.device
    )
    print(f"Ensemble ready: 8 models loaded\n")
    
    # Get image files
    image_files = sorted(Path(args.data_root).glob('*.npz'))
    print(f"Found {len(image_files)} images\n")
    
    for idx, image_file in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {image_file.name}")
        
        try:
            # Run ensemble inference
            ensemble_pred, confidence = ensemble.inference(str(image_file))
            
            # Save results
            output_dir = os.path.join(args.pred_save_dir, image_file.stem)
            save_results(ensemble_pred, confidence, output_dir)
            print()
            
        except Exception as e:
            print(f"  Error: {e}\n")
            continue
    
    print("Ensemble inference completed.")
    print(f"Results saved to: {args.pred_save_dir}")


if __name__ == '__main__':
    main()
