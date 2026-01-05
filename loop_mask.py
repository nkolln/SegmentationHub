import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configuration
folder = "run-20260104_004434-enk2prcd"
mask_dir = f"wandb/{folder}/files/media/images/mask/"

# Color mapping matching the dataset.py logic
# Based on label_names.txt MINUS 1, plus merging logic:
# mask = np.where((mask == 9) | (mask == 2) | (mask == 5), 2, mask)
LABEL_NAMES = {
    0: "Background",
    1: "Facade",
    2: "Structural Details (Windows)",
    3: "Door",
    4: "Cornice",
    5: "Molding",
    6: "Balcony",
    7: "Blind",
    8: "Deco",
    9: "Sill",
    10: "Pillar",
    11: "Shop"
}

# Distinct colors for categorical data
COLORS = plt.cm.tab20(np.linspace(0, 1, 20))

if not os.path.exists(mask_dir):
    print(f"Directory not found: {mask_dir}")
    exit()

dirs = os.listdir(mask_dir)

for d in dirs:
    if "legible" in d or not d.endswith(".png"):
        continue
        
    mask_path = os.path.join(mask_dir, d)
    output_path = os.path.join(mask_dir, f"legible_{d}")
    
    mask = Image.open(mask_path)
    mask_np = np.array(mask)
    unique_vals = np.unique(mask_np)
    
    print(f"Processing {d} | Unique values: {unique_vals}")
    
    plt.figure(figsize=(12, 10))
    # Use vmin/vmax to ensure 0 always maps to the same color across images
    im = plt.imshow(mask_np, cmap='tab20', vmin=0, vmax=19)
    
    # Create Legend based on the unique values actually present in THIS image
    legend_patches = []
    for val in unique_vals:
        label = LABEL_NAMES.get(val, f"Unknown ({val})")
        color = COLORS[val % 20]
        legend_patches.append(mpatches.Patch(color=color, label=f"{val}: {label}"))
    
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title(f"Visualized Mask: {d}\nRun: {folder}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"✅ Saved visible mask with legend to: {output_path}")

print("\nDone! Check the 'legible_' files in the wandb directory.")