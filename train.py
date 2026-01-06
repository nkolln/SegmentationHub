import argparse
import torch
from torch.utils.data import DataLoader
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
        from src.models.unet_viable import ResNetUNet
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
        print(f"Initializing Mask2Former from config...")
        model = Mask2FormerHF(num_classes=config['model']['num_classes'], config=config)
    elif model_name == "dinov2":
        from src.models.dinov2 import DinoV2Seg
        variant = config['model'].get('encoder_name', 'dinov2_vits14')
        head_type = config['model'].get('head_type', 'simple')
        print(f"Initializing DINOv2 Segmentation ({variant}) with {head_type} head...")
        model = DinoV2Seg(num_classes=config['model']['num_classes'], model_type=variant, head_type=head_type)
    elif model_name == "aim":
        from src.models.aim import AIMSegmentationModel
        variant = config['model'].get('variant', 'aim-base')
        print(f"Initializing AIM Segmentation ({variant})...")
        model = AIMSegmentationModel(
            num_classes=config['model']['num_classes'], 
            model_type=variant,
            image_size=config['data']['image_size']
        )
    else:
        from src.models.unet_baseline import UNet
        print(f"Initializing UNet (fallback for unknown model: {model_name})...")
        model = UNet(n_channels=3, n_classes=config['model']['num_classes'])
    return model

def run_fold(fold, config, args):
    print(f"\n--- Starting training for Fold {fold} ---")
    
    # Data
    sources = config['data'].get('sources', ['base'])
    num_folds = config['data'].get('k_folds', None)
    class_mapping = config['data'].get('class_mapping', None)
    seed = config['training'].get('seed', 42)
    test_split = config['data'].get('test_split', 0.0)

    train_dataset = SegmentationDataset(
        root_dir=config['data']['root_dir'], 
        split='train',
        transform=get_train_transforms(config['data']['image_size']),
        sources=sources,
        fold=fold,
        num_folds=num_folds,
        class_mapping=class_mapping,
        seed=seed,
        test_split=test_split
    )
    val_dataset = SegmentationDataset(
        root_dir=config['data']['root_dir'], 
        split='val',
        transform=get_val_transforms(config['data']['image_size']),
        sources=sources,
        fold=fold,
        num_folds=num_folds,
        class_mapping=class_mapping,
        seed=seed,
        test_split=test_split
    )

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

    # Model
    model = create_model(config)

    # Trainer
    trainer = Trainer(model, train_loader, val_loader, config, fold=fold)
    
    if args.dry_run:
        print(f"Dry run for fold {fold} complete.")
        return None
    else:
        trainer.train()
        return trainer.best_val_iou

def main():
    parser = argparse.ArgumentParser(description="Train Segmentation Model")
    parser.add_argument('--config', type=str, default='configs/config_mask.yaml', help='Path to config file')
    parser.add_argument('--dry-run', action='store_true', help='Verify basic pipeline functionality')
    parser.add_argument('--fold', type=int, default=None, help='Run a specific fold (0-indexed)')
    args = parser.parse_args()

    config = load_config(args.config)
    
    if args.dry_run:
        config['training']['num_epochs'] = 1
        config['training']['batch_size'] = 2
        config['logging']['use_wandb'] = False

    num_folds = config['data'].get('k_folds', 1)
    
    if args.fold is not None:
        # Run specific fold
        run_fold(args.fold, config, args)
    elif num_folds > 1:
        # Run all folds in a loop
        fold_results = []
        for fold in range(num_folds):
            best_iou = run_fold(fold, config, args)
            if best_iou is not None:
                fold_results.append(best_iou)
        
        if fold_results:
            avg_iou = sum(fold_results) / len(fold_results)
            import numpy as np
            std_iou = np.std(fold_results)
            print(f"\n" + "="*30)
            print(f"🏁 K-FOLD RESULTS COMPLETED")
            print(f"Average Best IoU: {avg_iou:.4f} ± {std_iou:.4f}")
            print(f"Individual Folds: {[f'{r:.4f}' for r in fold_results]}")
            print("="*30)
            
            # Optional: Log aggregate to WandB if it's overall active
            if config['logging'].get('use_wandb', False):
                import wandb
                # We initialize a separate run for aggregation summary
                wandb.init(project=config['project_name'], name=f"{config['experiment_name']}_summary")
                wandb.log({
                    'kfold_avg_iou': avg_iou,
                    'kfold_std_iou': std_iou,
                    'kfold_results': fold_results
                })
                wandb.finish()
    else:
        # Standard training (no folds or k_folds=1)
        run_fold(0, config, args)

if __name__ == '__main__':
    main()
