import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

mask_path = r"C:\Users\nickk\Documents\Repo\SegmentationHub\wandb\run-20251229_201507-jjzvty76\files\media\images\mask\val_prediction_0_d0ea6df2d17994eb78d2.mask.png"
output_path = "debug_mask_viz21.png"

if os.path.exists(mask_path):
    mask = Image.open(mask_path)
    print(f"Format: {mask.mode}")
    mask_np = np.array(mask)
    print(f"Unique values: {np.unique(mask_np)}")
    print(f"Shape: {mask_np.shape}")
    
    # Apply a color map to make it visible
    # We use 'jet' or 'tab20' which gives distinct colors for categorical data
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_np, cmap='tab20')
    plt.colorbar()
    plt.title(f"Visualized Mask (Values: {np.unique(mask_np)})")
    plt.savefig(output_path)
    print(f"✅ Saved visible mask to {output_path}")
    print("Open this file to see the actual labels!")
else:
    print("File not found")
