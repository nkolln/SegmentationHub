import torch
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor
import os
import sys
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import SegmentationDataset

def parse_args():
    parser = argparse.ArgumentParser(description="SAM HF Refinement")
    parser.add_argument("--root_dir", default="data", help="Root directory of the dataset")
    parser.add_argument("--sources", nargs="+", default=["base"], help="Input sources in root_dir/raw/")
    parser.add_argument("--output_dir", default="data/refined_masks_hf", help="Directory to save refined masks")
    parser.add_argument("--viz_dir", default="visualizations", help="Subfolder for visualizations")
    parser.add_argument("--model_name", default="facebook/sam-vit-huge", help="HF SAM model name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run SAM on")
    parser.add_argument("--target_class", type=int, default=None, help="Specific class index to refine (if None, refines all)")
    parser.add_argument("--max_items", type=int, default=None, help="Max items to process (for quick testing)")
    return parser.parse_args()

def get_boxes_from_mask(mask, target_class=None):
    """
    Extract bounding boxes from a segmentation mask for all classes.
    Returns: List of (class_val, [x_min, y_min, x_max, y_max])
    """
    all_boxes = []
    
    if target_class is not None:
        classes_to_process = [target_class]
    else:
        # Ignore 255 (the default ignore_index in dataset.py)
        # We process everything else, including 0 (if it's a valid class)
        classes_to_process = np.unique(mask)
        classes_to_process = classes_to_process[classes_to_process != 255]
        
    for cls in classes_to_process:
        binary_mask = (mask == cls).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        for i in range(1, num_labels):  # Skip background for THIS class
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            if w < 2 or h < 2:
                continue
                
            all_boxes.append((int(cls), [float(x), float(y), float(x + w), float(y + h)]))
            
    return all_boxes

def get_color_mask(mask, num_classes=12):
    """Colorize a mask for visualization."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Simple color palette
    cmap = plt.get_cmap('tab20')
    
    for cls in np.unique(mask):
        if cls == 255: continue
        color = np.array(cmap(cls % 20)[:3]) * 255
        color_mask[mask == cls] = color.astype(np.uint8)
        
    return color_mask

def create_viz(image_rgb, baseline_mask, refined_mask, output_path):
    """Create a side-by-side visualization."""
    h, w = image_rgb.shape[:2]
    
    viz_baseline = get_color_mask(baseline_mask)
    viz_refined = get_color_mask(refined_mask)
    
    # Overlay on image (optional, but side-by-side is clearer for "baseline vs refined")
    # Let's do a 1x3 grid
    combined = np.hstack([
        image_rgb,
        viz_baseline,
        viz_refined
    ])
    
    Image.fromarray(combined).save(output_path)

def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading HF SAM model {args.model_name}...")
    model = SamModel.from_pretrained(args.model_name).to(device)
    processor = SamProcessor.from_pretrained(args.model_name)

    os.makedirs(args.output_dir, exist_ok=True)
    viz_full_dir = os.path.join(args.output_dir, args.viz_dir)
    os.makedirs(viz_full_dir, exist_ok=True)

    print(f"Initializing dataset from {args.root_dir}...")
    dataset = SegmentationDataset(
        root_dir=args.root_dir,
        sources=args.sources,
        split='all',
        transform=None 
    )

    num_to_process = args.max_items if args.max_items else len(dataset)
    for i in tqdm(range(num_to_process), desc="Processing items"):
        image_np, mask_np, _, image_path = dataset[i]
        
        raw_image = Image.fromarray(image_np)
        w, h = raw_image.size
        
        # Extract boxes with their original class labels
        labeled_boxes = get_boxes_from_mask(mask_np, target_class=args.target_class)
        
        if not labeled_boxes:
            continue
            
        # Separate labels and boxes for the processor
        box_labels = [lb[0] for lb in labeled_boxes]
        input_boxes = [lb[1] for lb in labeled_boxes]
        
        # Process in batch
        inputs = processor(raw_image, input_boxes=[input_boxes], return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            inputs["original_sizes"].cpu(), 
            inputs["reshaped_input_sizes"].cpu()
        )
        
        scores = outputs.iou_scores.cpu() 
        img_masks = masks[0] 
        
        final_mask = np.ones((h, w), dtype=np.uint8)
        # final_mask.fill(255) # Start with background/ignore
        
        for box_idx in range(len(input_boxes)):
            best_mask_idx = torch.argmax(scores[0, box_idx])
            best_mask = img_masks[box_idx, best_mask_idx].numpy()
            
            # Use the original class label
            cls_val = box_labels[box_idx]
            final_mask[best_mask] = cls_val + 1 
            
        # Save refined mask
        item_name = os.path.basename(image_path).replace('.jpg', '')
        mask_output_path = os.path.join(args.output_dir, f"{item_name}_refined.png")
        raw_mask_path = dataset.masks[i]
        raw_pil = Image.open(raw_mask_path)

        out_img = Image.fromarray(final_mask)

        # If the original had a palette, apply it to the new one
        if raw_pil.mode == 'P':
            out_img.putpalette(raw_pil.getpalette())

        out_img.save(mask_output_path)
        
        # Create and save visualization
        viz_output_path = os.path.join(viz_full_dir, f"{item_name}_viz.png")
        create_viz(image_np, mask_np, final_mask, viz_output_path)

    print(f"Finished. Refined masks saved to {args.output_dir}")
    print(f"Visualizations saved to {viz_full_dir}")

if __name__ == "__main__":
    main()
