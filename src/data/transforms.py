import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size):
    """Enhanced data augmentation for better generalization on building facades."""
    return A.Compose([
        A.Resize(image_size, image_size),
        
        # Geometric augmentations - FACADE-APPROPRIATE
        A.HorizontalFlip(p=0.5),
        # REMOVED: VerticalFlip and RandomRotate90 - buildings have strong vertical orientation
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.GridDistortion(distort_limit=0.2, p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.3),  # Better for building perspective
        
        # Color & Noise augmentations
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.4),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.3),  # FIXED: using std_range instead of var_limit
        
        # Blur augmentation
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        
        # Cutout - FIXED parameters for newer albumentations
        A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.2),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
