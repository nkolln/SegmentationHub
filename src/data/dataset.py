import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class SegmentationDataset(Dataset):
    """
    Custom Dataset for Semantic Segmentation.
    """
    def __init__(self, root_dir, split='train', transform=None, class_mapping=None):
        """
        Args:
            root_dir (str): Root directory of the dataset (e.g. data/)
            split (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Data augmentation pipeline.
            class_mapping (dict): Dictionary mapping original classes to target classes.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.class_mapping = class_mapping
        
        self.images = []
        self.masks = []
        self._load_data()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = np.array(Image.open(image_path).convert("RGB"))
        # Mask is usually single channel
        mask = np.array(Image.open(mask_path)) 
        
        # Labels are 1-based (1-12). Shift to 0-based (0-11).
        # Background is 1 -> 0
        mask = mask.astype(np.int64) - 1
        
        # Clip to ensure no negative values just in case (e.g. if 0 existed)
        mask = np.clip(mask, 0, None)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask
        
    def _load_data(self):
        base_path = os.path.join(self.root_dir, 'raw', 'base')
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Base path not found: {base_path}")
            
        # Find all jpg images
        all_files = sorted([f for f in os.listdir(base_path) if f.endswith('.jpg')])
        
        # Simple deterministic split (first 80% train, next 20% val)
        split_idx = int(0.8 * len(all_files))
        
        if self.split == 'train':
            files = all_files[:split_idx]
        elif self.split == 'val':
            files = all_files[split_idx:]
        else:
            files = all_files # fallback/test
            
        for f in files:
            img_path = os.path.join(base_path, f)
            # Mask replaces .jpg with .png
            mask_name = f.replace('.jpg', '.png')
            mask_path = os.path.join(base_path, mask_name)
            
            if os.path.exists(mask_path):
                self.images.append(img_path)
                self.masks.append(mask_path)
