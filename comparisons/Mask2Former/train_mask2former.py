"""
Training script for Mask2Former model.
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


class Mask2FormerSimplified(nn.Module):
    """
    Simplified Mask2Former-like architecture for medical image segmentation.
    Features transformer and learnable queries.
    """
    
    def __init__(self, in_channels=1, out_channels=2, hidden_dim=256, num_queries=100):
        super(Mask2FormerSimplified, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # Backbone (simplified feature extractor)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Learnable queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        )
        
        # Decoder head
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, 2, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)  # (B, hidden_dim, H//4, W//4)
        
        # Get learnable queries
        queries = self.query_embed.weight.unsqueeze(0).expand(x.size(0), -1, -1)
        
        # Flatten spatial dimensions for transformer
        b, c, h, w = features.shape
        features_flat = features.view(b, c, -1).permute(0, 2, 1)  # (B, HW, C)
        
        # Apply transformer encoder
        transformer_out = self.transformer_encoder(features_flat)
        
        # Reshape features back
        features = transformer_out.permute(0, 2, 1).view(b, c, h, w)
        
        # Decode
        output = self.decoder(features)
        return output


def train(args):
    """Train Mask2Former model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize model
    model = Mask2FormerSimplified(in_channels=1, out_channels=2)
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
                      os.path.join(args.output, 'mask2former_best.pt'))


def main():
    parser = argparse.ArgumentParser(description='Train Mask2Former for medical image segmentation')
    parser.add_argument('-i', '--input', required=True, help='Path to input dataset')
    parser.add_argument('-o', '--output', required=True, help='Path to save trained model')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=500, help='Maximum number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
