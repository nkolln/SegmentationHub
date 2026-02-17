
import argparse
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import shutil
import glob
from torch.utils.data import DataLoader, Dataset

from src.utils.config import load_config
from train import create_model
from src.data.transforms import get_val_transforms

class InferenceDataset(Dataset):
    """
    Simple dataset for inference. 
    Accepts a list of image paths and applies standard validation transforms.
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Keep original size for later resizing if needed
        original_size = image.shape[:2] # H, W
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, img_path, original_size

def run_inference(args):
    """
    Run inference on all images in source_dir using the specified model.
    """
    # 1. Setup paths
    if not os.path.exists(args.source_dir):
        raise ValueError(f"Source directory does not exist: {args.source_dir}")
        
    output_images_dir = os.path.join(args.output_dir, "images")
    output_masks_dir = os.path.join(args.output_dir, "masks")
    output_vis_dir = os.path.join(args.output_dir, "vis")
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    if args.visualize:
        os.makedirs(output_vis_dir, exist_ok=True)

    # 2. Load Config & Model
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model structure
    model = create_model(config)
    model.to(device)
    model.eval()
    
    # Load weights
    if args.checkpoint:
        print(f"Loading weights from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Handle different checkpoint formats (e.g. state_dict vs full checkpoint)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
    else:
        print("WARNING: No checkpoint provided! Using random weights (garbage output).")

    # 3. Prepare Data
    # Find all common image formats
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.source_dir, ext)))
        # Case insensitive search fix for Linux/Windows differences usually requires regex, 
        # but glob is case-insensitive on Windows. 
        
    if len(image_paths) == 0:
        print(f"No images found in {args.source_dir}")
        return

    print(f"Found {len(image_paths)} images to process.")

    if args.limit:
        print(f"Limiting to first {args.limit} images.")
        image_paths = image_paths[:args.limit]

    transform = get_val_transforms(config['data']['image_size'])
    dataset = InferenceDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 4. Inference Loop
    print("Starting inference...")
    with torch.no_grad():
        for batch_images, batch_paths, batch_sizes in tqdm(dataloader):
            batch_images = batch_images.to(device)
            # batch_sizes is a list of tuples [(h1, h2...), (w1, w2...)] which is awkward
            # Convert to list of (h, w) tuples
            orig_sizes = []
            for i in range(len(batch_paths)):
                h = batch_sizes[0][i].item()
                w = batch_sizes[1][i].item()
                orig_sizes.append((h, w))
            
            # Forward pass
            outputs = model(batch_images)
            
            # Post-processing
            # Handle different return types (dict for Mask2Former, Tensor for others)
            preds = []
            if isinstance(outputs, dict) and 'mask_logits' in outputs:
                # Mask2Former / DINOv3-M2F path
                 if hasattr(model, 'post_process_semantic_segmentation'):
                    # Use the model's native post-processing (e.g. for M2F)
                    preds = model.post_process_semantic_segmentation(outputs, orig_sizes)
                 else:
                    # Fallback if method missing (shouldn't happen with our M2F)
                     raise NotImplementedError("Model dict output not supported without post_process_semantic_segmentation")
            else:
                # Standard UNet/Segformer path (outputs logits tensor)
                logits = outputs
                # Resize to original size
                for i, logit in enumerate(logits):
                    h, w = orig_sizes[i]
                    logit = logit.unsqueeze(0) # (1, C, H, W)
                    logit = torch.nn.functional.interpolate(logit, size=(h, w), mode='bilinear', align_corners=False)
                    pred = torch.argmax(logit, dim=1).squeeze(0) # (H, W)
                    preds.append(pred)
            
            # Save results
            for i, pred_mask in enumerate(preds):
                img_path = batch_paths[i]
                filename = os.path.basename(img_path)
                image_name_no_ext = os.path.splitext(filename)[0]
                
                # Copy original image to output structure
                shutil.copy(img_path, os.path.join(output_images_dir, filename))
                
                # Save mask as Indexed PNG (uint8)
                # Shift by +1 to match Dataset expectation (1-indexed, 0=ignore)
                mask_np = pred_mask.cpu().numpy().astype(np.uint8) + 1
                mask_path = os.path.join(output_masks_dir, f"{image_name_no_ext}.png")
                
                # Use PIL to save as P mode (paletted) so distinct values show as colors
                # but valid integer indices are preserved for training
                im = Image.fromarray(mask_np)
                im = im.convert("P")
                
                # Simple rainbow palette for 256 classes
                # 0=Black (Background), others=Random/Distinct
                palette = [0, 0, 0] # Class 0
                np.random.seed(42)
                for _ in range(255): 
                    palette.extend(np.random.randint(0, 255, 3))
                im.putpalette(palette)
                
                im.save(mask_path)
                
                # Optional: Visualization overlay
                if args.visualize:
                    # Load original image for overlay
                    orig_img = cv2.imread(img_path)
                    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                    
                    # Create colormap (random colors for classes)
                    # Use the standard simple colormap or generate one
                    num_classes = config['model']['num_classes']
                    colors = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
                    
                    colored_mask = np.zeros_like(orig_img)
                    for cls_id in range(num_classes):
                        colored_mask[mask_np == cls_id] = colors[cls_id]
                        
                    # Blend
                    overlay = cv2.addWeighted(orig_img, 0.6, colored_mask, 0.4, 0)
                    cv2.imwrite(os.path.join(output_vis_dir, filename), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Done! Results saved to {args.output_dir}")
    print(f"- Images: {len(os.listdir(output_images_dir))}")
    print(f"- Masks: {len(os.listdir(output_masks_dir))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate pseudo-labels with trained model")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained weights (.pth)')
    parser.add_argument('--source_dir', type=str, required=True, help='Directory containing images to predict')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save pseudo-labeled dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Inference batch size')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images to process')
    parser.add_argument('--visualize', action='store_true', help='Generate overlay visualizations')
    
    args = parser.parse_args()
    run_inference(args)
