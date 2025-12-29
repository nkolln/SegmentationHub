import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size):
    """Enhanced data augmentation for better generalization."""
    return A.Compose([
        A.Resize(image_size, image_size),
        
        # Geometric augmentations - ENHANCED
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),  # NEW: Vertical symmetry
        A.RandomRotate90(p=0.5),  # NEW: 90-degree rotations (0/90/180/270)
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),  # Increased scale
        A.GridDistortion(distort_limit=0.2, p=0.3),
        
        # Color & Noise augmentations - ENHANCED
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),  # Stronger
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.4),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        # Blur augmentation - NEW (improves robustness)
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        
        # Cutout - REDUCED (was too aggressive)
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
