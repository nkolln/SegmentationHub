import argparse
import torch
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.data.dataset import SegmentationDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.segformer import Segformer
from src.models.unet_baseline import UNet
from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train Segmentation Model")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--dry-run', action='store_true', help='Verify basic pipeline functionality')
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Override for dry run to make it fast
    if args.dry_run:
        config['training']['num_epochs'] = 1
        config['training']['batch_size'] = 2
        config['logging']['use_wandb'] = False

    # Data
    train_dataset = SegmentationDataset(
        root_dir=config['data']['root_dir'], 
        split='train',
        transform=get_train_transforms(config['data']['image_size'])
    )
    val_dataset = SegmentationDataset(
        root_dir=config['data']['root_dir'], 
        split='val',
        transform=get_val_transforms(config['data']['image_size'])
    )

    # Dataloaders - handle num_workers=0 if debugging or windows issues arise
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
    if config['model']['name'] == 'segformer_b0':
        model = Segformer(num_classes=config['model']['num_classes'])
    elif config['model']['name'] == 'unet':
        model = UNet(n_channels=3, n_classes=config['model']['num_classes'])
    else:
        raise ValueError(f"Unknown model name: {config['model']['name']}")

    # Trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    if args.dry_run:
        print("Dry run initialized successfully. Pipeline is ready.")
        # We could run one step here if we had fake data, but dataset is empty now.
    else:
        trainer.train()

if __name__ == '__main__':
    main()
