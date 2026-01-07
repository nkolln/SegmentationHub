import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class SegmentationDataset(Dataset):
    """
    Custom Dataset for Semantic Segmentation.
    """
    def __init__(self, root_dir, split='train', transform=None, class_mapping=None, sources=['base'], fold=0, num_folds=None, seed=42, test_split=0.0):
        """
        Args:
            root_dir (str): Root directory of the dataset (e.g. data/)
            split (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Data augmentation pipeline.
            class_mapping (dict): Dictionary mapping original classes to target classes.
            sources (list): List of source directories in root_dir/raw/ to load from.
            fold (int): Current fold index (0 to num_folds-1).
            num_folds (int): Total number of folds for K-fold validation. If None, uses default 80/20 split.
            seed (int): Seed for deterministic shuffling.
            test_split (float): Fraction of data to reserve as a hold-out test set.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.class_mapping = class_mapping
        self.sources = sources
        self.fold = fold
        self.num_folds = num_folds
        self.seed = seed
        self.test_split = test_split
        
        self.images = []
        self.masks = []
        self._load_data()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path,refined_mask_path = self.masks[idx]

        mask_path = np.random.choice([mask_path,refined_mask_path],p=[0.3,0.7])
        
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path)).astype(np.int64)
        
        # Apply class mapping if provided
        if self.class_mapping:
            new_mask = np.zeros_like(mask)
            for src_cls, tgt_cls in self.class_mapping.items():
                new_mask[mask == int(src_cls)] = int(tgt_cls)
            mask = new_mask
        else:
            # Default behavior if no mapping: assume 1-indexed and map 0 to ignore_index
            # This is safer than the previous -1 shift and clip
            ignore_index = 255
            new_mask = np.full_like(mask, ignore_index)
            mask_indices = mask > 0
            new_mask[mask_indices] = mask[mask_indices] - 1
            mask = new_mask

        # Parse XML for window count
        window_count = 0.0
        xml_path = image_path.replace('.jpg', '.xml')
        if os.path.exists(xml_path):
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    if obj.find('labelname').text == 'window':
                        window_count += 1.0
            except Exception:
                pass # Fail silently, count remains 0.0 (or we could log warning)
                
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask, torch.tensor(window_count, dtype=torch.float32), image_path
        
    def _load_data(self):
        for source in self.sources:
            base_path = os.path.join(self.root_dir, 'raw', source)
            refined_path = os.path.join(self.root_dir, 'refined_test', source)

            if not os.path.exists(base_path):
                print(f"Warning: Source path not found: {base_path}. Skipping.")
                continue
                
            # Find all jpg images
            all_files = sorted([f for f in os.listdir(base_path) if f.endswith('.jpg')])
            
            # Deterministic Shuffle
            import random
            random.seed(self.seed)
            random.shuffle(all_files)
            
            # 1. First, separate hold-out test set if requested
            if self.test_split > 0:
                test_size = int(len(all_files) * self.test_split)
                test_files = all_files[:test_size]
                remaining_files = all_files[test_size:]
                
                if self.split == 'test':
                    files = test_files
                    print(f"  > Source '{source}': reserved {len(test_files)} files as hold-out test set.")
                else:
                    all_files = remaining_files
            
            # 2. Split remaining data for K-Fold or Train/Val
            if self.split != 'test':
                if self.num_folds is not None:
                    # K-fold splitting on remaining files
                    fold_size = len(all_files) // self.num_folds
                    val_start = self.fold * fold_size
                    # Ensure the last fold takes any remainder
                    val_end = (self.fold + 1) * fold_size if self.fold < self.num_folds - 1 else len(all_files)
                    
                    if self.split == 'train':
                        files = all_files[:val_start] + all_files[val_end:]
                    elif self.split == 'val':
                        files = all_files[val_start:val_end]
                    elif self.split == 'all':
                        files = all_files
                    else:
                        files = all_files # fallback
                else:
                    # Default 80/20 split on remaining files
                    split_idx = int(0.8 * len(all_files))
                    if self.split == 'train':
                        files = all_files[:split_idx]
                    elif self.split == 'val':
                        files = all_files[split_idx:]
                    elif self.split == 'all':
                        files = all_files
                    else:
                        files = all_files # fallback
                
            for f in files:
                img_path = os.path.join(base_path, f)
                # Mask replaces .jpg with .png
                mask_name = f.replace('.jpg', '.png')
                mask_path = os.path.join(base_path, mask_name)
                refined_mask_path = os.path.join(refined_path, mask_name)
                
                if os.path.exists(refined_mask_path) and os.path.exists(mask_path):
                    self.images.append(img_path)
                    self.masks.append((mask_path,refined_mask_path))
                elif os.path.exists(refined_mask_path):
                    self.images.append(img_path)
                    self.masks.append((refined_mask_path))
                elif os.path.exists(mask_path):
                    self.images.append(img_path)
                    self.masks.append((mask_path))
            
            print(f"  > Source '{source}': found {len(all_files)} files, using {len(files)} for split '{self.split}'")
        
        print(f"Loaded {len(self.images)} images for split: {self.split}")
