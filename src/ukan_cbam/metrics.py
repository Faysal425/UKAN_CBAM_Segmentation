"""Metrics for UKAN-CBAM."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceIoUMetric:
    def __init__(self, smooth=1e-6):
        self.smooth = smooth

    def __call__(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()
        
        # Flatten
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        
        union = preds.sum() + targets.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return dice, iou

class PrecisionRecallMetric:
    def __init__(self, threshold=0.5, smooth=1e-6):
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = (preds > self.threshold).float()
        
        # Flatten
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        tp = (preds * targets).sum()
        fp = ((1 - targets) * preds).sum()
        fn = (targets * (1 - preds)).sum()
        
        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        recall = (tp + self.smooth) / (tp + fn + self.smooth)
        
        return precision, recall

class F1ScoreMetric:
    def __init__(self, threshold=0.5, smooth=1e-6):
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = (preds > self.threshold).float()
        
        # Flatten
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        tp = (preds * targets).sum()
        fp = ((1 - targets) * preds).sum()
        fn = (targets * (1 - preds)).sum()
        
        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        recall = (tp + self.smooth) / (tp + fn + self.smooth)
        
        f1 = 2 * (precision * recall) / (precision + recall + self.smooth)
        
        return f1

class AccuracyMetric:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = (preds > self.threshold).float()
        
        # Flatten
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        correct = (preds == targets).float().sum()
        total = targets.numel()
        
        return correct / total

class MultiMetric:
    def __init__(self, threshold=0.5, smooth=1e-6):
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = (preds > self.threshold).float()
        
        # Flatten
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        tp = (preds * targets).sum()
        fp = ((1 - targets) * preds).sum()
        fn = (targets * (1 - preds)).sum()
        tn = ((1 - preds) * (1 - targets)).sum()
        
        dice = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)
        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        recall = (tp + self.smooth) / (tp + fn + self.smooth)
        f1 = 2 * (precision * recall) / (precision + recall + self.smooth)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        return {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }
