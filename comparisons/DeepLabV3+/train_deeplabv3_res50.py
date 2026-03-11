"""
Training script for DeepLabV3+ model on medical images.
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
        self.data_files = list(Path(data_path).glob('*.npz'))
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])
        image = torch.from_numpy(data['image']).float()
        label = torch.from_numpy(data['label']).long()
        return image, label


class DeepLabV3AtrousBlock(nn.Module):
    """Atrous convolution block (dilated convolution)."""
    
    def __init__(self, in_channels, out_channels, dilation):
        super(DeepLabV3AtrousBlock, self).__init__()
        self.atrous_conv = nn.Conv2d(
            in_channels, out_channels, 3,
            padding=dilation, dilation=dilation
        )
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return torch.relu(self.bn(self.atrous_conv(x)))


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""
    
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        
        # Multiple atrous convolutions with different dilation rates
        self.aspp_conv1 = DeepLabV3AtrousBlock(in_channels, out_channels, 1)
        self.aspp_conv2 = DeepLabV3AtrousBlock(in_channels, out_channels, 6)
        self.aspp_conv3 = DeepLabV3AtrousBlock(in_channels, out_channels, 12)
        self.aspp_conv4 = DeepLabV3AtrousBlock(in_channels, out_channels, 18)
        
        # Image pooling branch
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Apply ASPP
        aspp1 = self.aspp_conv1(x)
        aspp2 = self.aspp_conv2(x)
        aspp3 = self.aspp_conv3(x)
        aspp4 = self.aspp_conv4(x)
        
        # Image pooling
        pool = self.image_pool(x)
        pool = torch.nn.functional.interpolate(pool, (h, w), mode='bilinear', align_corners=False)
        
        # Concatenate
        aspp_combined = torch.cat([aspp1, aspp2, aspp3, aspp4, pool], dim=1)
        
        return self.project(aspp_combined)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ architecture for medical image segmentation.
    Encoder-decoder with ASPP module.
    """
    
    def __init__(self, in_channels=1, out_channels=2, features=64):
        super(DeepLabV3Plus, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, 7, stride=2, padding=3),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Encoder stages
        self.conv1 = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, features * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(inplace=True)
        )
        
        # ASPP module
        self.aspp = ASPP(features * 2, 256)
        
        # Decoder
        self.decoder_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(256 + features, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Final output
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.final_conv = nn.Conv2d(256, out_channels, 1)
    
    def forward(self, x):
        # Encoder
        h, w = x.shape[2], x.shape[3]
        
        enc = self.encoder(x)  # 1/4 resolution
        enc1 = self.conv1(enc)
        enc2 = self.conv2(enc1)  # 1/8 resolution
        
        # ASPP
        aspp_out = self.aspp(enc2)  # 1/8 resolution
        
        # Decoder
        aspp_up = self.decoder_upsample(aspp_out)  # 1/2 resolution
        
        # Combine with encoder features
        combined = torch.cat([aspp_up, enc1], dim=1)
        dec = self.decoder_conv(combined)
        
        # Final output
        out = self.final_upsample(dec)
        out = torch.nn.functional.interpolate(out, (h, w), mode='bilinear', align_corners=False)
        out = self.final_conv(out)
        
        return out


def train(args):
    """Train DeepLabV3+ model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize model
    model = DeepLabV3Plus(in_channels=1, out_channels=2)
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
    
    best_loss = float('inf')
    
    for epoch in range(args.max_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.max_epochs}], "
                      f"Batch [{batch_idx+1}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 
                      os.path.join(args.output, 'deeplabv3plus_best.pt'))


def main():
    parser = argparse.ArgumentParser(description='Train DeepLabV3+ for medical image segmentation')
    parser.add_argument('-i', '--input', required=True, help='Path to input dataset')
    parser.add_argument('-o', '--output', required=True, help='Path to save trained model')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=500, help='Maximum number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--compile', action='store_true', help='Compile model')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
