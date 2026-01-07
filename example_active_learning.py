import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.data.dataset import SegmentationDataset
from src.data.transforms import get_val_transforms
from src.utils.active_learning import ActiveLearningManager
from src.utils.config import load_config
from src.models.dinov2_counting import DinoV2SegCounting

def example_selection():
    # 1. Load Configuration
    config = load_config('configs/config_mask.yaml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. Setup "Unlabeled" Dataset
    # Using the 'val' split as a proxy for unlabeled data
    dataset = SegmentationDataset(
        root_dir=config['data']['root_dir'],
        split='val',
        transform=get_val_transforms(config['data']['image_size']),
        sources=config['data'].get('sources', ['base'])
    )
    
    loader = DataLoader(dataset, batch_size=config['training'].get('batch_size', 2), shuffle=False)
    
    # 3. Initialize Model (Using Mask2Former)
    print("Initializing Mask2Former for feature extraction...")
    model_path = 'outputs/mask2former/mask2formert12_new_head_clamp_main/best_model.pth'
    
    from src.models.mask2former import Mask2FormerHF
    
    if os.path.exists(model_path):
        print(f"Loading checkpoint from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Determine which config to use for model initialization
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            print("Using configuration found in checkpoint for model initialization.")
            model_config = checkpoint['config']
        else:
            print("Checkpoint does not contain config, using global config.")
            model_config = config
            
        # Initialize model with the correct config
        model = Mask2FormerHF(num_classes=model_config['model']['num_classes'], config=model_config)
        
        # Extract state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        print("Loading state dict into model...")
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Weights not found at {model_path}, using pretrained/random weights with global config.")
        model = Mask2FormerHF(num_classes=config['model']['num_classes'], config=config)
    
    # 4. Run Selection
    manager = ActiveLearningManager(model=model, device=device)
    
    print("Computing embeddings for selection...")
    # Passing the loader to get embeddings and their corresponding filenames
    embeddings, filenames = manager.get_embeddings(loader)
    
    print(f"Analyzing {len(filenames)} samples...")
    selection_results = manager.analyze_diversity(embeddings)
    
    # 5. Output Results and Visualize Outliers
    print("\n--- Active Learning Selection Results ---")
    print(f"Mean Diversity: {selection_results['mean_diversity']:.4f}")
    
    outliers = selection_results['outlier_indices']
    print("\nTop samples recommended for labeling (most unique):")
    
    # Setup visualization directory
    os.makedirs('outputs/active_learning_viz', exist_ok=True)
    
    for i, idx in enumerate(outliers):
        fname = filenames[idx] if idx < len(filenames) else f"sample_{idx}"
        print(f" - {fname}")
        
        # Get data for visualization
        image, gt_mask, _, _ = dataset[idx]
        
        # Model Prediction
        model.eval()
        with torch.no_grad():
            input_tensor = image.unsqueeze(0).to(device)
            outputs = model(input_tensor)
            
            # Post-process for Mask2Former
            if hasattr(model, 'post_process'):
                target_sizes = [input_tensor.shape[-2:]]
                pred_mask = model.post_process(outputs, target_sizes)[0]
            else:
                # Fallback for other models
                pred_mask = torch.argmax(outputs, dim=1)[0]
        
        # Plotting
        try:
            import matplotlib.pyplot as plt
            
            # De-normalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = image.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * std + mean).clip(0, 1)
            
            gt_mask_np = gt_mask.cpu().numpy()
            pred_mask_np = pred_mask.detach().cpu().numpy()
            
            # Filter ignore index for better visualization
            gt_mask_np = np.where(gt_mask_np == 255, 0, gt_mask_np)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(image_np)
            axes[0].set_title(f"Image: {os.path.basename(fname)}")
            axes[0].axis('off')
            
            axes[1].imshow(gt_mask_np, cmap='tab20', interpolation='nearest')
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis('off')
            
            axes[2].imshow(pred_mask_np, cmap='tab20', interpolation='nearest')
            axes[2].set_title("Model Prediction")
            axes[2].axis('off')
            
            plt.tight_layout()
            save_path = f'outputs/active_learning_viz/outlier_{i}_{os.path.basename(fname)}.png'
            plt.savefig(save_path)
            plt.close()
            print(f"   Saved visualization to {save_path}")
            
        except Exception as e:
            print(f"   Error during visualization for {fname}: {e}")

if __name__ == "__main__":
    example_selection()
