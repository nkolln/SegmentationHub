import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        # Geometric augmentations (help with overfitting spatial Features)
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.GridDistortion(distort_limit=0.2, p=0.3),
        
        # Color & Noise augmentations (help with lighting variance)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.GaussNoise(p=0.2),
        
        # Cutout / CoarseDropout (forces model to not rely on single features)
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
