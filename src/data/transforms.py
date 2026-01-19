import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size):
    """Standard training augmentation for building facades."""
    return A.Compose([
        A.Resize(image_size, image_size),
        
        # Geometric - facade appropriate
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.GridDistortion(distort_limit=0.2, p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        
        # Color & lighting
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.4),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
        
        # Blur
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        
        # Cutout
        A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.2),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_robust_facade_transforms(image_size):
    """
    Robust augmentation pipeline specifically designed for Dutch/European facades.
    
    Key additions over standard transforms:
    - Shadow simulation (common in street-level facade photos)
    - Weather effects (rain, fog - common in Netherlands)
    - Strong perspective/architectural distortions
    - Exposure variation (backlit buildings, overcast days)
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        
        # === GEOMETRIC: Architectural-aware ===
        A.HorizontalFlip(p=0.5),
        
        # Strong perspective for street-level photos of tall buildings
        A.Perspective(scale=(0.02, 0.12), p=0.4),
        
        # Simulate camera tilt/position variation
        A.ShiftScaleRotate(
            shift_limit=0.15, 
            scale_limit=0.25,  # Stronger zoom variation
            rotate_limit=12,   # Slight tilt, buildings are vertical
            border_mode=0,     # Constant fill
            p=0.6
        ),
        
        # Lens distortion (common in wide-angle facade photos)
        A.OneOf([
            A.GridDistortion(distort_limit=0.25, p=1.0),
            A.OpticalDistortion(distort_limit=0.15, shift_limit=0.15, p=1.0),
        ], p=0.3),
        
        # === SHADOWS & LIGHTING: Critical for facades ===
        # Simulate shadows cast by neighboring buildings, trees, sun angle
        A.RandomShadow(
            shadow_roi=(0, 0.3, 1, 1),  # Shadows typically on lower portions
            num_shadows_limit=(1, 3),
            shadow_dimension=5,
            p=0.4
        ),
        
        # Strong brightness/contrast for backlit buildings, overcast
        A.RandomBrightnessContrast(
            brightness_limit=0.4,
            contrast_limit=0.4,
            p=0.7
        ),
        
        # Local exposure variation (sun hitting part of building)
        A.RandomToneCurve(scale=0.15, p=0.3),
        
        # === WEATHER EFFECTS: Dutch climate ===
        A.OneOf([
            # Fog/mist effect
            A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.1, p=1.0),
            # Rain effect (simulated via slight blur + noise)
            A.Compose([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(std_range=(0.03, 0.08), p=1.0),
            ]),
            # Overcast flat lighting
            A.RandomGamma(gamma_limit=(70, 90), p=1.0),
        ], p=0.25),
        
        # === COLOR VARIATIONS ===
        # Building materials vary widely: brick, stone, stucco, paint
        A.HueSaturationValue(
            hue_shift_limit=20,   # Slight color cast from lighting
            sat_shift_limit=30,   # Some facades are very saturated
            val_shift_limit=20,
            p=0.5
        ),
        
        # White balance variation (different times of day)
        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.15,
            hue=0.05,
            p=0.3
        ),
        
        # === QUALITY DEGRADATION ===
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),      # Camera shake
            A.GaussianBlur(blur_limit=5, p=1.0),    # Out of focus
            A.Defocus(radius=(2, 4), p=1.0),        # Depth of field
        ], p=0.25),
        
        # Compression artifacts (common in scraped web images)
        A.ImageCompression(quality_range=(60, 95), p=0.2),
        
        # Noise
        A.GaussNoise(std_range=(0.02, 0.12), p=0.35),
        
        # === OCCLUSIONS: Simulate real-world obstructions ===
        # Trees, parked cars, signs blocking parts of facades
        A.CoarseDropout(
            num_holes_range=(2, 6),
            hole_height_range=(20, 60),
            hole_width_range=(20, 60),
            fill_value=0,
            p=0.25
        ),
        
        # Rectangular occlusions (cars, signs)
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(30, 80),
            hole_width_range=(50, 120),
            fill_value=128,  # Grey fill simulates occluding objects
            p=0.15
        ),
        
        # === FINAL ===
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
