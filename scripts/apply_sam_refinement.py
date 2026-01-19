"""
Apply SAM (Segment Anything Model) to a folder of images for refinement.
Generates automatic masks for each image to aid in labeling.

Usage:
    python scripts/apply_sam_refinement.py --input data/coreset_refinement --model-type huge
"""

import argparse
import os
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import SamModel, SamProcessor

def apply_sam_refinement(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load SAM model
    print(f"Loading SAM ({args.model_type})...")
    model_name = "facebook/sam-vit-huge" if args.model_type == "huge" else "facebook/sam-vit-base"
    model = SamModel.from_pretrained(model_name).to(device)
    processor = SamProcessor.from_pretrained(model_name)
    
    input_dir = Path(args.input)
    output_dir = input_dir / "sam_predictions"
    output_dir.mkdir(exist_ok=True)
    
    # Find images
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = [f for f in input_dir.glob('*') if f.suffix.lower() in extensions]
    
    print(f"Found {len(images)} images in {input_dir}")
    print(f"Saving predictions to {output_dir}")
    
    for img_path in tqdm(images, desc="Running SAM"):
        try:
            raw_image = Image.open(img_path).convert("RGB")
            
            # 1. Grid of points for automatic mask generation
            # SAM processor handles this if we don't provide points? 
            # Transformers implementation requires points or boxes usually.
            # To simulate "automatic mask generator", we can generate a grid of points.
            
            width, height = raw_image.size
            n_points_per_side = 32
            
            # Simple grid
            x = np.linspace(0, width, n_points_per_side)
            y = np.linspace(0, height, n_points_per_side)
            xv, yv = np.meshgrid(x, y)
            grid_points = np.stack([xv.flatten(), yv.flatten()], axis=1).tolist()
            
            # Batch points slightly to avoid OOM
            batch_size = 64
            all_masks = []
            all_scores = []
            
            # Prepare inputs with all points
            # Note: Transformers implementation expects shape (Batch, NumPoints, 2)
            inputs = processor(raw_image, input_points=[grid_points], return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Post-process
            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), 
                inputs["original_sizes"].cpu(), 
                inputs["reshaped_input_sizes"].cpu()
            ) 
            # masks is a list of tensors
            # Shape: (Batch, NumPoints, 3, H, W) -> 3 mask proposals per point
            
            scores = outputs.iou_scores.cpu() # (Batch, NumPoints, 3)
            
            # Select best mask per point
            batch_masks = masks[0] # (NumPoints, 3, H, W)
            batch_scores = scores[0] # (NumPoints, 3)
            
            best_mask_indices = torch.argmax(batch_scores, dim=1) # (NumPoints,)
            
            # Gather best masks
            final_masks = []
            for i in range(len(best_mask_indices)):
                idx = best_mask_indices[i]
                if batch_scores[i, idx] > args.conf_threshold:
                    final_masks.append(batch_masks[i, idx])
            
            if not final_masks:
                continue
                
            final_masks = torch.stack(final_masks) # (N_filtered, H, W)
            
            # Create a composite visualization
            # Just overlay all masks with random colors
            vis_img = np.array(raw_image).copy()
            overlay = np.zeros_like(vis_img)
            
            # Convert masks to numpy
            masks_np = final_masks.numpy()
            
            np.random.seed(42)
            colors = np.random.randint(0, 255, (len(masks_np), 3), dtype=np.uint8)
            
            for i, mask in enumerate(masks_np):
                color = colors[i]
                # mask is boolean (H, W)
                overlay[mask > 0] = color
            
            # Blend
            alpha = 0.5
            mask_indices = np.any(overlay > 0, axis=2)
            vis_img[mask_indices] = (vis_img[mask_indices] * (1 - alpha) + overlay[mask_indices] * alpha).astype(np.uint8)
            
            # Save visualization
            vis_path = output_dir / f"{img_path.stem}_vis.png"
            Image.fromarray(vis_img).save(vis_path)
            
            # Optional: Save individual masks or encoded format?
            # For now just confirming it ran.
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run SAM refinement on images")
    parser.add_argument('--input', type=str, required=True, help='Input directory with images')
    parser.add_argument('--model-type', type=str, default='huge', choices=['huge', 'base'], help='SAM model type')
    parser.add_argument('--conf-threshold', type=float, default=0.8, help='Confidence threshold for masks')
    
    args = parser.parse_args()
    apply_sam_refinement(args)

if __name__ == "__main__":
    main()
