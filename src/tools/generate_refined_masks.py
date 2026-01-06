import torch
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob

from src.tools.refinement_utils import load_best_model, run_inference
from src.data.transforms import get_val_transforms

def main():
    parser = argparse.ArgumentParser(description="Generate refined masks using the best model")
    parser.add_argument('--config', type=str, default='configs/config_mask.yaml', help='Path to config file')
    parser.add_argument('--fold', type=int, default=0, help='Fold to use for inference')
    parser.add_argument('--checkpoint', type=str, default=None, help='Explicit path to checkpoint')
    parser.add_argument('--source_dir', type=str, default='data/raw/extended', help='Directory containing images to refine')
    parser.add_argument('--output_dir', type=str, default='refined_masks', help='Directory to save refined masks')
    args = parser.parse_args()

    # Load model
    model, config, device = load_best_model(args.config, args.fold, args.checkpoint)
    
    # Setup transforms
    image_size = config['data']['image_size']
    transform = get_val_transforms(image_size)
    
    # Find images
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.source_dir, ext)))
    
    if not image_paths:
        print(f"No images found in {args.source_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating refined masks for {len(image_paths)} images in {args.output_dir}")

    for img_path in tqdm(image_paths):
        # Load image
        image_pil = Image.open(img_path).convert('RGB')
        image_np = np.array(image_pil)
        
        # Run inference
        prediction = run_inference(model, image_np, device, transform)
        
        # Convert prediction to 8-bit image
        prediction_np = prediction.cpu().numpy().astype(np.uint8)
        pred_pil = Image.fromarray(prediction_np)
        
        # Save prediction
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)
        save_path = os.path.join(args.output_dir, f"{name}.png")
        pred_pil.save(save_path)

    print("Generation complete!")

if __name__ == "__main__":
    main()
