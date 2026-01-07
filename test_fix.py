import torch
from src.models.dinov2 import DinoV2Seg
from src.training.trainer import Trainer
from src.utils.config import load_config
import os

def test():
    # Load a config (any config with dinov2)
    config_path = 'configs/config_dinov2_high.yaml'
    if not os.path.exists(config_path):
        # Fallback to a basic config structure
        config = {
            'model': {
                'num_classes': 12,
                'encoder_name': 'dinov2_vits14',
                'head_type': 'multiscale'
            },
            'training': {
                'learning_rate': 1e-4,
                'num_epochs': 1,
                'batch_size': 1,
                'weight_decay': 0.01,
                'use_amp': False
            },
            'loss': {},
            'logging': {
                'use_wandb': False,
                'output_dir': 'test_outputs'
            },
            'project_name': 'test',
            'experiment_name': 'test'
        }
    else:
        config = load_config(config_path)
        config['logging']['use_wandb'] = False
        config['training']['num_epochs'] = 1
        config['training']['batch_size'] = 1

    model = DinoV2Seg(num_classes=config['model']['num_classes'], 
                      model_type=config['model'].get('encoder_name', 'dinov2_vits14'), 
                      head_type=config['model'].get('head_type', 'multiscale'))
    
    # Mock data
    img = torch.randn(1, 3, 770, 770)
    mask = torch.randint(0, config['model']['num_classes'], (1, 770, 770))
    
    class MockDataset(torch.utils.data.Dataset):
        def __len__(self): return 1
        def __getitem__(self, idx): return img[0], mask[0]
        
    train_loader = torch.utils.data.DataLoader(MockDataset(), batch_size=1)
    val_loader = torch.utils.data.DataLoader(MockDataset(), batch_size=1)
    
    trainer = Trainer(model, train_loader, val_loader, config)
    
    print("Testing train_epoch...")
    trainer.train_epoch(0)
    print("train_epoch successful!")
    
    print("Testing validate...")
    trainer.validate(0)
    print("validate successful!")

if __name__ == "__main__":
    test()
