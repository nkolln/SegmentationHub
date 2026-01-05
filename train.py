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

def main():
    parser = argparse.ArgumentParser(description="Train Segmentation Model")
    parser.add_argument('--config', type=str, default='configs/config_mask.yaml', help='Path to config file')
    parser.add_argument('--dry-run', action='store_true', help='Verify basic pipeline functionality')
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
    model_name = config['model']['name']
    if model_name == "unet_viable":
        print("Initializing ResNet34-UNet...")
        model = ResNetUNet(num_classes=config['model']['num_classes'], pretrained=config['model'].get('pretrained', True))
    elif model_name == "segformer_hf":
        from src.models.segformer_hf import SegformerHF
        print("Initializing HuggingFace Segformer...")
        # e.g. "nvidia/mit-b0" or "nvidia/segformer-b0-finetuned-ade-512-512"
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
    elif model_name == "mask2former_og":
        from src.models.mask2former_og import Mask2FormerHF
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

    # Trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    if args.dry_run:
        print("Dry run initialized successfully. Pipeline is ready.")
        # We could run one step here if we had fake data, but dataset is empty now.
    else:
        trainer.train()

if __name__ == '__main__':
    main()
