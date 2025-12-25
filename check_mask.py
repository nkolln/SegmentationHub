
import numpy as np
from PIL import Image
import os

mask_path = r"c:\Users\nickk\Documents\Repo\SegmentationHub\data\raw\base\cmp_b0001.png"

if os.path.exists(mask_path):
    mask = Image.open(mask_path)
    print(f"Format: {mask.mode}")
    mask_np = np.array(mask)
    print(f"Unique values: {np.unique(mask_np)}")
    print(f"Shape: {mask_np.shape}")
else:
    print("File not found")
