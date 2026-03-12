"""
Training script for LiteMedSAM with PraNet-V2 + WTNet exposure correction.

This script trains the lightweight medical image segmentation model with:
- Multi-scale supervision from Partial Decoder coarse masks
- Dual supervision (background + foreground) in decoder
- Exposure correction feature integration
- Hybrid loss combining Dice + Cross-Entropy
- Learning rate scheduling and gradient clipping
- Checkpoint saving and resumption
- Tensorboard logging

Usage:
    python train_LiteMedSAM.py \
        --data_root data/Kvasir-SEG \
        --output_dir checkpoints \
        --batch_size 32 \
        --epochs 300 \
        --lr 1e-4 \
        --cuda_id 0
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter

from LiteMedSAM.modeling import LiteMedSAM


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(output_dir):
    """Setup logging configuration."""
    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ============================================================================
# DATASET
# ============================================================================
class MedicalImageSegmentationDataset(Dataset):
    """Medical image segmentation dataset with bounding box annotations.
    
    Expects directory structure:
        data_root/
        ├── images/
        │   ├── img1.png
        │   ├── img2.png
        │   └── ...
        └── masks/
            ├── img1.png
            ├── img2.png
            └── ...
    
    Optionally supports JSON bbox annotations:
        {
            "img1.png": [[x_min, y_min, x_max, y_max], ...],
            "img2.png": [[x_min, y_min, x_max, y_max], ...],
            ...
        }
    """
    
    def __init__(self, data_root, image_size=512, bbox_file=None):
        """Initialize dataset.
        
        Args:
            data_root: Root directory containing 'images' and 'masks' subdirs
            image_size: Resize images to this size
            bbox_file: Optional JSON file with bbox annotations
        """
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.images_dir = self.data_root / 'images'
        self.masks_dir = self.data_root / 'masks'
        
        # Get image filenames
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.endswith(('.png', '.jpg', '.jpeg', '.tiff'))
        ])
        
        # Load bboxes if provided
        self.bboxes = {}
        if bbox_file and os.path.exists(bbox_file):
            with open(bbox_file, 'r') as f:
                self.bboxes = json.load(f)
        
        logging.info(f"Loaded dataset with {len(self.image_files)} images")
        if self.bboxes:
            logging.info(f"Loaded bboxes for {len(self.bboxes)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get image and mask pair.
        
        Returns:
            dict with:
                - 'image': Normalized image [1, H, W]
                - 'mask': Binary mask [1, H, W]
                - 'bbox': Bounding box [4] if available
                - 'filename': Original filename
        """
        filename = self.image_files[idx]
        image_path = self.images_dir / filename
        mask_path = self.masks_dir / filename
        
        # Load image and mask
        try:
            import cv2
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                raise ValueError(f"Failed to load {filename}")
            
            # Resize
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            
            # Normalize
            image = image.astype(np.float32) / 255.0
            mask = (mask > 127).astype(np.float32)  # Binary mask
            
            # Convert to tensors
            image_tensor = torch.from_numpy(image).unsqueeze(0)  # [1, H, W]
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)    # [1, H, W]
            
            # Get bbox if available
            bbox = None
            if filename in self.bboxes and len(self.bboxes[filename]) > 0:
                bbox_list = self.bboxes[filename]
                # Use first bbox
                x_min, y_min, x_max, y_max = bbox_list[0]
                bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
            
            return {
                'image': image_tensor,
                'mask': mask_tensor,
                'bbox': bbox,
                'filename': filename
            }
        
        except Exception as e:
            logging.error(f"Error loading {filename}: {e}")
            # Return dummy data
            return {
                'image': torch.zeros(1, self.image_size, self.image_size),
                'mask': torch.zeros(1, self.image_size, self.image_size),
                'bbox': None,
                'filename': filename
            }


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """Calculate Dice loss.
        
        Args:
            pred: Predictions [B, C, H, W] or [B, H, W]
            target: Target [B, 1, H, W] or [B, H, W]
        
        Returns:
            Dice loss scalar
        """
        # Flatten
        if pred.dim() == 4:
            pred = pred.squeeze(1)  # [B, H, W]
        if target.dim() == 4:
            target = target.squeeze(1)  # [B, H, W]
        
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        return dice.mean()


class CombinedLoss(nn.Module):
    """Combined Dice + CE loss for training."""
    
    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.BCELoss()
    
    def forward(self, pred, target, bg_pred=None):
        """Calculate combined loss.
        
        Args:
            pred: Main prediction [B, 1, H, W] or [B, H, W]
            target: Target mask [B, 1, H, W]
            bg_pred: Background prediction (optional) [B, 1, H, W]
        
        Returns:
            Combined loss scalar
        """
        # Ensure correct shapes
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
        
        # Clamp to (0, 1) for BCE
        pred_clamped = torch.clamp(pred, 1e-6, 1 - 1e-6)
        
        # Main losses
        dice = self.dice_loss(pred_clamped, target)
        ce = self.ce_loss(pred_clamped, target)
        
        loss = self.dice_weight * dice + self.ce_weight * ce
        
        # Background loss (if provided)
        if bg_pred is not None:
            if bg_pred.dim() == 4:
                bg_pred = bg_pred.squeeze(1)
            bg_target = 1.0 - target
            bg_ce = self.ce_loss(torch.clamp(bg_pred, 1e-6, 1 - 1e-6), bg_target)
            loss = loss + self.ce_weight * bg_ce
        
        return loss


# ============================================================================
# TRAINING
# ============================================================================
class LiteMedSAMTrainer:
    """Trainer for LiteMedSAM model."""
    
    def __init__(self, model, device, output_dir, lr=1e-4, logger=None):
        """Initialize trainer.
        
        Args:
            model: LiteMedSAM model
            device: torch device
            output_dir: Output directory for checkpoints
            lr: Learning rate
            logger: Logger instance
        """
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
        # Loss function
        self.loss_fn = CombinedLoss(dice_weight=0.5, ce_weight=0.5)
        
        # Optimizer: use lower LR for base encoder, higher for new modules
        base_lr = lr * 0.1  # 10x lower for pretrained features
        new_lr = lr
        
        base_params = [
            model.image_encoder.stem.parameters(),
            model.image_encoder.layer1.parameters(),
            model.image_encoder.layer2.parameters(),
            model.image_encoder.layer3.parameters(),
            model.image_encoder.layer4.parameters(),
            model.image_encoder.final_proj.parameters(),
        ]
        
        optimizer_params = [
            {'params': list(p for params in base_params for p in params), 'lr': base_lr},
            {'params': model.image_encoder.exposure_f2.parameters(), 'lr': new_lr},
            {'params': model.image_encoder.exposure_f3.parameters(), 'lr': new_lr},
            {'params': model.image_encoder.exposure_f4.parameters(), 'lr': new_lr},
            {'params': model.image_encoder.exposure_final.parameters(), 'lr': new_lr},
            {'params': model.image_encoder.pd_f2.parameters(), 'lr': new_lr},
            {'params': model.image_encoder.pd_f3.parameters(), 'lr': new_lr},
            {'params': model.image_encoder.pd_f4.parameters(), 'lr': new_lr},
            {'params': model.image_encoder.pd_final.parameters(), 'lr': new_lr},
            {'params': model.image_encoder.fusion_f2.parameters(), 'lr': new_lr},
            {'params': model.image_encoder.fusion_f3.parameters(), 'lr': new_lr},
            {'params': model.image_encoder.fusion_f4.parameters(), 'lr': new_lr},
            {'params': model.image_encoder.fusion_final.parameters(), 'lr': new_lr},
            {'params': model.prompt_encoder.parameters(), 'lr': new_lr},
            {'params': model.mask_decoder.parameters(), 'lr': new_lr},
        ]
        
        self.optimizer = optim.Adam(optimizer_params)
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=300, eta_min=1e-6)
        
        # Tensorboard
        self.writer = SummaryWriter(str(self.output_dir / 'logs'))
        
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train_epoch(self, train_loader, epoch, num_epochs):
        """Train one epoch.
        
        Args:
            train_loader: Training dataloader
            epoch: Current epoch number
            num_epochs: Total number of epochs
        
        Returns:
            Average loss for epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            image = batch['image'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            try:
                outputs = self.model(image)
                logits = outputs['logits']
                bg_logits = outputs.get('bg_logits', None)
                
                # Ensure logits are [B, 1, H, W] or [B, H, W]
                if logits.shape[1] == 2:  # 2-class output
                    logits = logits[:, 1:2, :, :]
                
                # Resize logits to match mask size if needed
                if logits.shape[-2:] != mask.shape[-2:]:
                    logits = torch.nn.functional.interpolate(
                        logits, size=mask.shape[-2:], 
                        mode='bilinear', align_corners=False
                    )
                
                # Apply sigmoid to logits
                pred = torch.sigmoid(logits)
                
                # Calculate loss
                loss = self.loss_fn(pred, mask, bg_logits)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Logging
                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = total_loss / num_batches
                    self.logger.info(
                        f"Epoch [{epoch+1}/{num_epochs}] "
                        f"Batch [{batch_idx+1}/{len(train_loader)}] "
                        f"Loss: {loss.item():.6f} "
                        f"Avg Loss: {avg_loss:.6f}"
                    )
                    self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        self.scheduler.step()
        
        return avg_loss
    
    def validate(self, val_loader, epoch):
        """Validate model on validation set.
        
        Args:
            val_loader: Validation dataloader
            epoch: Current epoch number
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                try:
                    outputs = self.model(image)
                    logits = outputs['logits']
                    bg_logits = outputs.get('bg_logits', None)
                    
                    if logits.shape[1] == 2:
                        logits = logits[:, 1:2, :, :]
                    
                    if logits.shape[-2:] != mask.shape[-2:]:
                        logits = torch.nn.functional.interpolate(
                            logits, size=mask.shape[-2:],
                            mode='bilinear', align_corners=False
                        )
                    
                    pred = torch.sigmoid(logits)
                    loss = self.loss_fn(pred, mask, bg_logits)
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                except Exception as e:
                    self.logger.error(f"Error in validation batch: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        self.writer.add_scalar('val/loss', avg_loss, self.global_step)
        
        self.logger.info(f"Validation Loss at Epoch {epoch+1}: {avg_loss:.6f}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            loss: Current loss
        """
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Latest checkpoint
        latest_path = checkpoint_dir / 'latest.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'global_step': self.global_step
        }, latest_path)
        
        # Best checkpoint
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = checkpoint_dir / 'best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': loss,
                'global_step': self.global_step
            }, best_path)
            self.logger.info(f"Saved best checkpoint at epoch {epoch+1}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        
        Returns:
            Starting epoch
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['loss']
        self.global_step = checkpoint.get('global_step', 0)
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'] + 1


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train LiteMedSAM')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, 
                        default='data/Kvasir-SEG',
                        help='Root directory of dataset')
    parser.add_argument('--bbox_file', type=str,
                        default='data/Kvasir-SEG/kavsir_bboxes.json',
                        help='Path to bbox JSON file (optional)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation set ratio')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Input image size')
    
    # Training arguments
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    # Hardware arguments
    parser.add_argument('--cuda_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(args.output_dir)
    logger.info(f"Device: {device}")
    logger.info(f"Arguments: {args}")
    
    # Create model
    logger.info("Creating LiteMedSAM model...")
    model = LiteMedSAM(
        in_channels=1,
        out_channels=2,
        embed_dim=256,
        base_channels=32
    )
    model = model.to(device)
    model.get_model_summary()
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = MedicalImageSegmentationDataset(
        data_root=args.data_root,
        image_size=args.image_size,
        bbox_file=args.bbox_file if os.path.exists(args.bbox_file) else None
    )
    
    # Train/val split
    num_samples = len(dataset)
    num_val = int(num_samples * args.val_split)
    num_train = num_samples - num_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [num_train, num_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Trainer
    trainer = LiteMedSAMTrainer(
        model=model,
        device=device,
        output_dir=args.output_dir,
        lr=args.lr,
        logger=logger
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch, args.epochs)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.6f}")
        
        # Validate
        val_loss = trainer.validate(val_loader, epoch)
        
        # Save checkpoint
        trainer.save_checkpoint(epoch, val_loss)
        
        # Log to tensorboard
        trainer.writer.add_scalar('train/epoch_loss', train_loss, epoch)
        trainer.writer.add_scalar('val/epoch_loss', val_loss, epoch)
        trainer.writer.add_scalar('lr', 
                                  trainer.optimizer.param_groups[0]['lr'], 
                                  epoch)
    
    logger.info("Training completed!")
    logger.info(f"Best checkpoint saved to {args.output_dir}/checkpoints/best.pth")
    
    trainer.writer.close()


if __name__ == '__main__':
    main()
