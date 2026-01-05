import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
try:
    from sklearn.model_selection import KFold
except ImportError:
    KFold = None
from src.utils.config import load_config
from src.data.dataset import SegmentationDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.segformer import Segformer
from src.models.unet_baseline import UNet
from src.models.unet_viable import ResNetUNet
from src.training.trainer import Trainer

def create_model(config):
    model_name = config['model']['name']
    if model_name == "unet_viable":
        print("Initializing ResNet34-UNet...")
        model = ResNetUNet(num_classes=config['model']['num_classes'], pretrained=config['model'].get('pretrained', True))
    elif model_name == "segformer_hf":
        from src.models.segformer_hf import SegformerHF
        print("Initializing HuggingFace Segformer...")
        repo = config['model'].get('pretrained_repo', "nvidia/mit-b0")
        model = SegformerHF(num_classes=config['model']['num_classes'], pretrained_repo=repo)
    elif model_name == "segformer_manual":
        from src.models.segformer_manual import SegformerManual
        print("Initializing Manual Segformer (MiT-B0)...")
        model = SegformerManual(num_classes=config['model']['num_classes'])
    elif model_name == "deeplabv3plus":
        from src.models.deeplabv3 import DeepLabV3Plus
        encoder = config['model'].get('encoder_name', 'resnet101')
        print(f"Initializing DeepLabV3+ with {encoder} encoder...")
        model = DeepLabV3Plus(num_classes=config['model']['num_classes'], encoder_name=encoder)
    elif model_name == "unetplusplus":
        from src.models.deeplabv3 import UNetPlusPlus
        encoder = config['model'].get('encoder_name', 'efficientnet-b4')
        print(f"Initializing UNet++ with {encoder} encoder...")
        model = UNetPlusPlus(num_classes=config['model']['num_classes'], encoder_name=encoder)
    elif model_name == "mask2former":
        from src.models.mask2former import Mask2FormerHF
        repo = config['model'].get('pretrained_repo', "facebook/mask2former-swin-tiny-cityscapes-semantic")
        print(f"Initializing Mask2Former ({repo})...")
        model = Mask2FormerHF(num_classes=config['model']['num_classes'], pretrained_repo=repo)
    elif model_name == "dinov2":
        from src.models.dinov2 import DinoV2Seg
        variant = config['model'].get('encoder_name', 'dinov2_vits14')
        head_type = config['model'].get('head_type', 'simple')
        print(f"Initializing DINOv2 Segmentation ({variant}) with {head_type} head...")
        model = DinoV2Seg(num_classes=config['model']['num_classes'], model_type=variant, head_type=head_type)
    else:
        # Fallback
        print(f"Initializing UNet (fallback for unknown model: {model_name})...")
        model = UNet(n_channels=3, n_classes=config['model']['num_classes'])
    return model

def main():
    parser = argparse.ArgumentParser(description="Train Segmentation Model")
    parser.add_argument('--config', type=str, default='configs/config_mask.yaml', help='Path to config file')
    parser.add_argument('--dry-run', action='store_true', help='Verify basic pipeline functionality')
    parser.add_argument('--kfold', type=int, default=1, help='Number of folds for k-fold cross-validation (1 for normal train/val)')
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Override for dry run to make it fast
    if args.dry_run:
        config['training']['num_epochs'] = 1
        config['training']['batch_size'] = 2
        config['logging']['use_wandb'] = False

    # Data
    sources = config['data'].get('sources', ['base'])
    print(f"Loading data from sources: {sources}")

    if args.kfold > 1:
        if KFold is None:
            print("Error: scikit-learn is required for k-fold cross-validation. Please install it: pip install scikit-learn")
            return
            
        print(f"Running {args.kfold}-fold cross-validation...")
        
        # We create two full datasets to get the respective transforms, then Subset them
        train_full = SegmentationDataset(
            root_dir=config['data']['root_dir'], 
            split='all',
            transform=get_train_transforms(config['data']['image_size']),
            sources=sources
        )
        val_full = SegmentationDataset(
            root_dir=config['data']['root_dir'], 
            split='all',
            transform=get_val_transforms(config['data']['image_size']),
            sources=sources
        )
        
        if len(train_full) == 0:
            print("Error: No data found for specified sources.")
            return

        kf = KFold(n_splits=args.kfold, shuffle=True, random_state=config['training'].get('seed', 42))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_full)):
            print(f"\n--- Fold {fold+1}/{args.kfold} ---")
            
            train_dataset = Subset(train_full, train_idx)
            val_dataset = Subset(val_full, val_idx)
            
            print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

            train_loader = DataLoader(
                train_dataset, 
                batch_size=config['training']['batch_size'],
                shuffle=True, 
                num_workers=config['data']['num_workers']
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=config['training']['batch_size'],
                shuffle=False, 
                num_workers=config['data']['num_workers']
            )

            # Re-initialize model for each fold
            model = create_model(config)
            
            # Unique run name for each fold
            base_name = config.get('experiment_name', 'run')
            run_name = f"{base_name}_fold{fold+1}"
            
            trainer = Trainer(model, train_loader, val_loader, config, run_name=run_name)
            
            if args.dry_run:
                print(f"Dry run for Fold {fold+1} complete.")
                break # Only one fold for dry run
            else:
                trainer.train()
                
    else:
        # Standard train/val split logic
        train_dataset = SegmentationDataset(
            root_dir=config['data']['root_dir'], 
            split='train',
            transform=get_train_transforms(config['data']['image_size']),
            sources=sources
        )
        val_dataset = SegmentationDataset(
            root_dir=config['data']['root_dir'], 
            split='val',
            transform=get_val_transforms(config['data']['image_size']),
            sources=sources
        )

        # Dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['training']['batch_size'],
            shuffle=True, 
            num_workers=config['data']['num_workers']
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['training']['batch_size'],
            shuffle=False, 
            num_workers=config['data']['num_workers']
        )

        model = create_model(config)
        trainer = Trainer(model, train_loader, val_loader, config)
        
        if args.dry_run:
            print("Dry run initialized successfully. Pipeline is ready.")
        else:
            trainer.train()

if __name__ == '__main__':
    main()
