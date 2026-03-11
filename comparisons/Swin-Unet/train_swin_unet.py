"""
Training script for Swin-Unet model.
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


class SwinUnetBlock(nn.Module):
    """Simplified Swin Transformer block."""
    
    def __init__(self, dim, num_heads=8):
        super(SwinUnetBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        # Attention block
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # MLP block
        normed = self.norm2(x)
        x = x + self.mlp(normed)
        return x


class SwinUnet(nn.Module):
    """
    Swin-Unet: Transformer-based U-shaped architecture for medical image segmentation.
    """
    
    def __init__(self, in_channels=1, out_channels=2, hidden_dim=96):
        super(SwinUnet, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, 4, stride=4)
        
        # Encoder
        self.encoder1 = SwinUnetBlock(hidden_dim)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = SwinUnetBlock(hidden_dim * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = SwinUnetBlock(hidden_dim * 4)
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 2, 2)
        self.decoder2 = SwinUnetBlock(hidden_dim * 2)
        
        self.upconv1 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 2, 2)
        self.decoder1 = SwinUnetBlock(hidden_dim)
        
        # Final projection
        self.final_conv = nn.Conv2d(hidden_dim, out_channels, 1)
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, hidden_dim, H//4, W//4)
        b, c, h, w = x.shape
        
        # Encoder
        enc1 = x.view(b, c, -1).permute(0, 2, 1)  # (B, HW, C)
        enc1 = self.encoder1(enc1)
        enc1 = enc1.permute(0, 2, 1).view(b, c, h, w)
        
        x = self.pool1(enc1)  # (B, hidden_dim, h//2, w//2)
        b2, c2, h2, w2 = x.shape
        x = x.view(b2, c2, -1).permute(0, 2, 1)
        x = self.encoder2(x)
        enc2 = x.permute(0, 2, 1).view(b2, c2, h2, w2)
        
        x = self.pool2(enc2)
        b3, c3, h3, w3 = x.shape
        x = x.view(b3, c3, -1).permute(0, 2, 1)
        x = self.bottleneck(x)
        x = x.permute(0, 2, 1).view(b3, c3, h3, w3)
        
        # Decoder
        x = self.upconv2(x)
        x = x + enc2
        b2, c2, h2, w2 = x.shape
        x = x.view(b2, c2, -1).permute(0, 2, 1)
        x = self.decoder2(x)
        x = x.permute(0, 2, 1).view(b2, c2, h2, w2)
        
        x = self.upconv1(x)
        x = x + enc1
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        x = self.decoder1(x)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        
        # Final output
        output = self.final_conv(x)
        return output


def train(args):
    """Train Swin-Unet model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize model
    model = SwinUnet(in_channels=1, out_channels=2)
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
                      os.path.join(args.output, 'swin_unet_best.pt'))


def main():
    parser = argparse.ArgumentParser(description='Train Swin-Unet for medical image segmentation')
    parser.add_argument('-i', '--input', required=True, help='Path to input dataset')
    parser.add_argument('-o', '--output', required=True, help='Path to save trained model')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=500, help='Maximum number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
