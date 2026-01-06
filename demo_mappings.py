import os
import yaml
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.data.dataset import SegmentationDataset
import albumentations as A

def main():
    # Load config
    config_path = 'configs/config_mask.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup dataset
    root_dir = config['data']['root_dir']
    sources = config['data']['sources']
    class_mapping = config['data']['class_mapping']
    image_size = config['data']['image_size']
    
    # We don't want complex augmentations for the demo, just resizing
    transform = A.Compose([
        A.Resize(image_size, image_size),
    ])
    
    dataset = SegmentationDataset(
        root_dir=root_dir,
        split='val',
        transform=transform,
        class_mapping=class_mapping,
        sources=sources
    )
    
    if len(dataset) < 5:
        print(f"Warning: Dataset only has {len(dataset)} images. Showing all.")
        indices = list(range(len(dataset)))
    else:
        indices = random.sample(range(len(dataset)), 20)
    
    # Final classes based on mapping
    # 0: Background, 1: Facade, 2: Window, 3: Others/Vents
    # We can infer this from the mapping values
    unique_mapped_classes = sorted(list(set(class_mapping.values())))
    print(f"Unique mapped classes: {unique_mapped_classes}")
    
    fig, axes = plt.subplots(len(indices), 2, figsize=(12, 4 * len(indices)))
    
    for i, idx in enumerate(indices):
        image, mask, window_count = dataset[idx]
        unique_vals = np.unique(mask)
        print(f"Sample {idx}: Unique mapped values: {unique_vals}")
        
        # Original image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Image {idx} (GT Window Count: {window_count.item()})")
        axes[i, 0].axis('off')
        
        # Mapped mask
        # Create a display mask where 255 is mapped to a visible "error/ignore" value (e.g. 9)
        display_mask = mask.copy()
        display_mask[display_mask == 255] = 9
        
        im = axes[i, 1].imshow(display_mask, cmap='tab10', vmin=0, vmax=9)
        axes[i, 1].set_title(f"Mapped Mask (Values: {unique_vals})")
        axes[i, 1].axis('off')
        
        if i == 0:
            cbar = plt.colorbar(im, ax=axes[i, 1], orientation='vertical')
            cbar.set_label('Class ID (9 = Unmapped/Ignore)')

    plt.tight_layout()
    output_path = 'demo_mappings.png'
    plt.savefig(output_path)
    print(f"Saved demo visualization to {output_path}")

if __name__ == "__main__":
    main()
