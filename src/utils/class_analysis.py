"""
Utility script to analyze dataset class distribution.
Run this to verify what classes exist in your dataset and their balance.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm
from collections import Counter
import torch


def analyze_dataset_classes(dataset, dataset_name="Dataset"):
    """
    Analyze all masks in dataset to find unique classes and distribution.
    
    Args:
        dataset: PyTorch dataset with masks
        dataset_name: Name for display purposes
    
    Returns:
        tuple: (sorted list of unique classes, Counter of class pixel counts)
    """
    all_classes = set()
    class_counts = Counter()
    
    print(f"\nAnalyzing {dataset_name} ({len(dataset)} samples)...")
    for i in tqdm(range(len(dataset)), desc="Processing"):
        _, mask = dataset[i]
        
        # Handle both numpy arrays and torch tensors
        if isinstance(mask, np.ndarray):
            unique = np.unique(mask)
        else:  # torch tensor
            unique = mask.unique().cpu().numpy()
        
        all_classes.update(unique)
        
        # Count pixels per class
        if isinstance(mask, np.ndarray):
            values, counts = np.unique(mask, return_counts=True)
        else:
            values, counts = mask.unique(return_counts=True)
            values = values.cpu().numpy()
            counts = counts.cpu().numpy()
        
        for val, count in zip(values, counts):
            class_counts[int(val)] += int(count)
    
    all_classes = sorted(list(all_classes))
    
    # Display results
    print(f"\n{'='*60}")
    print(f"{dataset_name} Analysis Results")
    print(f"{'='*60}")
    print(f"Unique classes found: {all_classes}")
    print(f"Number of classes: {len(all_classes)}")
    print(f"\nClass distribution (pixel-wise):")
    print(f"{'-'*60}")
    
    total_pixels = sum(class_counts.values())
    for cls in all_classes:
        count = class_counts[cls]
        pct = 100 * count / total_pixels
        print(f"  Class {cls:2d}: {count:12,} pixels ({pct:6.2f}%)")
    
    print(f"{'-'*60}")
    print(f"  Total:   {total_pixels:12,} pixels (100.00%)")
    print(f"{'='*60}\n")
    
    # Calculate imbalance ratio
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    print(f"⚠ Class imbalance ratio: {imbalance_ratio:.1f}:1")
    if imbalance_ratio > 10:
        print(f"  → HIGH imbalance detected! Focal loss will help significantly.")
    elif imbalance_ratio > 5:
        print(f"  → Moderate imbalance. Focal loss recommended.")
    else:
        print(f"  → Relatively balanced classes.")
    
    return all_classes, class_counts


if __name__ == "__main__":
    from src.data.dataset import SegmentationDataset
    from src.data.transforms import get_train_transforms, get_val_transforms
    
    print("\n" + "="*60)
    print("Dataset Class Distribution Analysis")
    print("="*60)
    
    # Analyze training set
    train_dataset = SegmentationDataset(
        root_dir='data',
        split='train',
        transform=get_val_transforms(512)  # Use val transforms to avoid augmentation
    )
    train_classes, train_counts = analyze_dataset_classes(train_dataset, "Training Set")
    
    # Analyze validation set
    val_dataset = SegmentationDataset(
        root_dir='data',
        split='val',
        transform=get_val_transforms(512)
    )
    val_classes, val_counts = analyze_dataset_classes(val_dataset, "Validation Set")
    
    # Check consistency
    if train_classes == val_classes:
        print(f"✓ Train and validation sets have the same classes")
    else:
        print(f"⚠ WARNING: Train and val sets have different classes!")
        print(f"  Train only: {set(train_classes) - set(val_classes)}")
        print(f"  Val only: {set(val_classes) - set(train_classes)}")
