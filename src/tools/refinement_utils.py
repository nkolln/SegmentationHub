import torch
import os
from src.utils.config import load_config
from train import create_model
from src.data.transforms import get_val_transforms

def load_best_model(config_path, fold=0, checkpoint_path=None):
    """
    Load the best model for a given fold.
    """
    config = load_config(config_path)
    model = create_model(config)
    
    if checkpoint_path is None:
        # Default to the experiment name and output dir in config
        output_dir = config['logging']['output_dir']
        experiment_name = config['experiment_name']
        checkpoint_path = os.path.join(output_dir, experiment_name, f"best_model_fold{fold}.pth")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if checkpoint is the model state dict or the whole trainer state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    return model, config, device

def run_inference(model, image, device, transform=None):
    """
    Run inference on a single image.
    Args:
        model: Loaded model
        image: PIL Image or numpy array
        device: torch device
        transform: Optional transform to apply to image
    Returns:
        prediction: torch.Tensor (H, W) with class indices
    """
    if transform:
        input_tensor = transform(image=image)['image'].unsqueeze(0).to(device)
    else:
        # Fallback if no transform is provided - simple conversion
        from torchvision import transforms
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = t(image).unsqueeze(0).to(device)
        
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # Handle different model output formats
        if hasattr(model, 'post_process'):
            # For Mask2FormerHF
            target_sizes = [(image.shape[0], image.shape[1]) if hasattr(image, 'shape') else (image.height, image.width)]
            # target_sizes = torch.tensor([[image.height, image.width]], device=device)
            # Actually Mask2FormerHF.post_process expects list of tuples or tensor
            predictions = model.post_process(outputs, target_sizes=target_sizes)
            prediction = predictions[0]
        else:
            # For standard CNNs
            prediction = torch.argmax(outputs, dim=1).squeeze(0)
            
    return prediction
