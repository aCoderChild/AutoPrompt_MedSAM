"""
Training script for U-Net model on medical image segmentation.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class MedicalImageDataset(Dataset):
    """Dataset loader for medical images in npz format."""
    
    def __init__(self, data_path):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to directory containing npz files
        """
        self.data_files = list(Path(data_path).glob('*.npz'))
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        """Load and return a sample."""
        data = np.load(self.data_files[idx])
        image = torch.from_numpy(data['image']).float()
        label = torch.from_numpy(data['label']).long()
        return image, label


class UNet(nn.Module):
    """
    U-Net implementation for medical image segmentation.
    
    Architecture:
    - 5 levels of encoder-decoder with skip connections
    - Each level has 2 convolutional blocks
    """
    
    def __init__(self, in_channels=1, out_channels=2, features=64):
        """
        Initialize U-Net.
        
        Args:
            in_channels: Number of input channels (default: 1 for grayscale)
            out_channels: Number of output classes
            features: Number of features in first layer
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, features)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self.conv_block(features, features * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self.conv_block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = self.conv_block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(features * 8, features * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, 2, 2)
        self.dec4 = self.conv_block(features * 16, features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 2, 2)
        self.dec3 = self.conv_block(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 2, 2)
        self.dec2 = self.conv_block(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 2, 2)
        self.dec1 = self.conv_block(features * 2, features)
        
        # Final output layer
        self.final = nn.Conv2d(features, out_channels, 1)
    
    @staticmethod
    def conv_block(in_channels, out_channels):
        """Create a convolutional block with ReLU and BatchNorm."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder
        dec4 = self.dec4(torch.cat([self.upconv4(bottleneck), enc4], 1))
        dec3 = self.dec3(torch.cat([self.upconv3(dec4), enc3], 1))
        dec2 = self.dec2(torch.cat([self.upconv2(dec3), enc2], 1))
        dec1 = self.dec1(torch.cat([self.upconv1(dec2), enc1], 1))
        
        # Final output
        return self.final(dec1)


def train(args):
    """Train U-Net model."""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize model
    model = UNet(in_channels=1, out_channels=2)
    model.to(device)
    
    # Create dataloader
    dataset = MedicalImageDataset(args.input)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.max_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.max_epochs}], "
                      f"Batch [{batch_idx+1}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 
                      os.path.join(args.output, 'unet_best.pt'))
            print(f"Best model saved with loss: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train U-Net for medical image segmentation')
    parser.add_argument('-i', '--input', required=True, help='Path to input dataset')
    parser.add_argument('-o', '--output', required=True, help='Path to save trained model')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=500, help='Maximum number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
