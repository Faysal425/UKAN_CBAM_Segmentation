#!/usr/bin/env python3
"""Training script for UKAN-CBAM."""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.ukan_cbam import UKAN_CBAM
from src.ukan_cbam.data.datasets import BinarySegmentationDataset
from src.ukan_cbam.losses import DiceBCELoss
from src.ukan_cbam.metrics import DiceIoUMetric
from src.ukan_cbam.utils.train_utils import init_weights, set_seed

def main():
    parser = argparse.ArgumentParser(description='Train UKAN-CBAM model')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='config file')
    parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(42)

    # Create model
    model = UKAN_CBAM(**config['model'])
    model.apply(init_weights)
    
    # Create datasets
    train_dataset = BinarySegmentationDataset(split='train', **config['dataset'])
    val_dataset = BinarySegmentationDataset(split='val', **config['dataset'])
    
    train_loader = DataLoader(train_dataset, shuffle=True, **config['dataset'])
    val_loader = DataLoader(val_dataset, shuffle=False, **config['dataset'])
    
    # Loss and optimizer
    criterion = DiceBCELoss(**config['training']['loss'])
    optimizer = AdamW(model.parameters(), **config['training']['optimizer'])
    scheduler = CosineAnnealingLR(optimizer, **config['training']['scheduler'])
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Training UKAN-CBAM with {sum(p.numel() for p in model.parameters())} parameters")
    
    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate metrics
                dice = DiceIoUMetric()(outputs, masks)
                val_dice += dice
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Dice: {val_dice/len(val_loader):.4f}")

if __name__ == '__main__':
    main()
