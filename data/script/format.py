import os
import zipfile
import requests
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image

# Configuration
DATASET_URL = "http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")

# CMP Original Classes (RGB values in their label pngs) to Your Target Classes
# Target IDs: 0=Background, 1=Paintable Wall, 2=Window, 3=Door, 4=Obstacle
COLOR_MAP = {
    (0, 0, 0): 0,       # Background -> Background
    (1, 1, 1): 1,       # Wall -> Paintable Wall
    (2, 2, 2): 4,       # Molding -> Obstacle
    (3, 3, 3): 4,       # Cornice -> Obstacle
    (4, 4, 4): 4,       # Pillar -> Obstacle
    (5, 5, 5): 2,       # Window -> Window
    (6, 6, 6): 3,       # Door -> Door
    (7, 7, 7): 4,       # Sill -> Obstacle
    (8, 8, 8): 4,       # Blind -> Obstacle
    (9, 9, 9): 4,       # Balcony -> Obstacle
    (10, 10, 10): 4,    # Shop -> Obstacle
    (11, 11, 11): 4,    # Deco -> Obstacle
}

def download_dataset():
    """Download and extract CMP dataset"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "cmp_base.zip"
    
    if not zip_path.exists():
        print(f"Downloading dataset from {DATASET_URL}...")
        response = requests.get(DATASET_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as bar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                bar.update(len(data))
    
    # Extract
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(RAW_DIR)

def process_mask(mask_path):
    """Convert RGB/Indexed PNG mask to Target Class ID mask"""
    # CMP labels are often saved as indexed PNGs or low-value RGBs
    # Load as PIL to handle palette seamlessly, convert to RGB to be safe
    mask_img = Image.open(mask_path).convert('RGB')
    mask_np = np.array(mask_img)
    
    # Create empty output mask
    output_mask = np.zeros(mask_np.shape[:2], dtype=np.uint8)
    
    # Map colors to class IDs
    # Note: CMP 'base' dataset often uses simple values like (1,1,1) for class 1
    # We check the first channel since R=G=B in this dataset
    b_channel = mask_np[:, :, 0] 
    
    # Remap based on our dictionary logic
    # 1 (Wall) -> 1 (Paintable)
    output_mask[b_channel == 1] = 1 
    # 5 (Window) -> 2 (Window)
    output_mask[b_channel == 5] = 2
    # 6 (Door) -> 3 (Door)
    output_mask[b_channel == 6] = 3
    
    # Everything else that isn't background (0) is an obstacle
    obstacles = [2, 3, 4, 7, 8, 9, 10, 11]
    for obs_id in obstacles:
        output_mask[b_channel == obs_id] = 4
        
    return output_mask

def prepare_data():
    download_dataset()
    
    # Setup paths
    base_path = RAW_DIR / "base"
    images_dir = PROCESSED_DIR / "images"
    masks_dir = PROCESSED_DIR / "masks"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get file lists (images usually .jpg, labels .png)
    image_files = sorted(list(base_path.glob("*.jpg")))
    
    print("Processing images and masks...")
    valid_files = []
    
    for img_path in tqdm(image_files):
        # Find corresponding png label
        basename = img_path.stem
        mask_path = base_path / f"{basename}.png"
        
        if not mask_path.exists():
            continue
            
        # Copy/Resize Image
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(images_dir / f"{basename}.jpg"), img)
        
        # Process and Resize Mask
        # We process first to get IDs, then resize with NEAREST to preserve IDs
        mask = process_mask(mask_path)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(masks_dir / f"{basename}.png"), mask)
        
        valid_files.append(basename)
        
    # Create Splits
    print("Creating train/val/test splits...")
    train_files, test_files = train_test_split(valid_files, test_size=0.2, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)
    
    # Save splits
    with open(SPLITS_DIR / "train.txt", "w") as f:
        f.write("\n".join(train_files))
    with open(SPLITS_DIR / "val.txt", "w") as f:
        f.write("\n".join(val_files))
    with open(SPLITS_DIR / "test.txt", "w") as f:
        f.write("\n".join(test_files))
        
    print(f"Done! Processed {len(valid_files)} images.")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

if __name__ == "__main__":
    prepare_data()