#!/usr/bin/env python3
"""Evaluation script for UKAN-CBAM."""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.ukan_cbam import UKAN_CBAM
from src.ukan_cbam.data.datasets import BinarySegmentationDataset
from src.ukan_cbam.metrics import MultiMetric
from src.ukan_cbam.utils.train_utils import set_seed

def main():
    parser = argparse.ArgumentParser(description='Evaluate UKAN-CBAM model')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='model checkpoint')
    parser.add_argument('--split', type=str, default='test', help='dataset split to evaluate')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(42)

    # Create model
    model = UKAN_CBAM(**config['model'])
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Create dataset
    dataset = BinarySegmentationDataset(split=args.split, **config['dataset'])
    dataloader = torch.utils.data.DataLoader(dataset, **config['dataset'])
    
    # Evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    metric = MultiMetric()
    
    total_metrics = {
        'dice': 0.0,
        'iou': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'accuracy': 0.0
    }
    
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            batch_metrics = metric(outputs, masks)
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]
    
    # Calculate average metrics
    num_batches = len(dataloader)
    for key in total_metrics:
        total_metrics[key] /= num_batches
    
    print("Evaluation Results:")
    for key, value in total_metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == '__main__':
    main()
