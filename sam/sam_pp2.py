import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
import cv2
import os
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path

# --- CONFIGURATION ---
# Define colors for visualization (B, G, R)
CLASS_COLORS = {
    1: (0, 255, 0),   # Green
    2: (255, 0, 0),   # Blue
    3: (0, 0, 255),   # Red
}
# ---------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SAM Post-processing for Facade Datasets")
    parser.add_argument("--input_dirs", nargs="+", default=["data/raw/base", "data/raw/extended"], help="Input directories")
    parser.add_argument("--output_dir", required=True, help="Directory to save refined masks")
    parser.add_argument("--vis_dir", default=None, help="Directory to save debug visualizations (optional)")
    parser.add_argument("--sam_checkpoint", default="sam/sam_vit_h_4b8939.pth", help="Path to SAM checkpoint")
    parser.add_argument("--model_type", default="vit_h", help="SAM model type")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--target_label", default="window", help="Label name to process from XML")
    return parser.parse_args()

def get_boxes_from_xml(xml_path, img_width, img_height, target_label="window"):
    """
    Parses CMP-style XML files for bounding boxes.
    Uses exact label matching logic as seen in the main dataset.
    """
    boxes = []
    try:
        # 1. Read and wrap content if missing root (CMP format doesn't always have one)
        with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
        
        if not content.startswith("<root>"):
            content = f"<root>{content}</root>"
        
        root = ET.fromstring(content)
        
        # 2. Extract boxes1
        for obj in root.findall('object'):
            name_node = obj.find('labelname')
            if name_node is None:
                continue
            
            # Exact match: logic matches SegmentationDataset.__getitem__
            if name_node.text.strip().lower() != target_label.lower():
                continue
                
            points = obj.find('points')
            if points is None:
                continue
            
            xs = [float(x.text) for x in points.findall('x')]
            ys = [float(y.text) for y in points.findall('y')]
            
            if len(xs) < 2 or len(ys) < 2:
                continue
            
            # Convert normalized (0-1) to pixel coordinates
            x_min = min(xs) * img_width
            x_max = max(xs) * img_width
            y_min = min(ys) * img_height
            y_max = max(ys) * img_height
            
            boxes.append([x_min, y_min, x_max, y_max])
            
    except Exception as e:
        # print(f"Skipping {xml_path}: {e}")
        return []
    
    return boxes

def create_vis_image(image, mask, boxes):
    """
    Creates a visualization inline with WandB style:
    - Original Image
    - Semi-transparent mask overlay
    - Bounding boxes drawn
    """
    # 1. Create a copy for drawing
    vis = image.copy()
    
    # 2. Draw boxes (Yellow)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # 3. Create Mask Overlay (Green)
    # Create a colored mask
    color_mask = np.zeros_like(image)
    color_mask[mask > 0] = [0, 255, 0] # Green for windows
    
    # 4. Blend: Image * 0.6 + Mask * 0.4
    # This matches the "Unnormalized" look + alpha blend
    mask_indices = mask > 0
    if mask_indices.any():
        vis[mask_indices] = cv2.addWeighted(vis[mask_indices], 0.6, color_mask[mask_indices], 0.4, 0)
    
    return vis

def main():
    args = parse_args()
    
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = args.vis_dir if args.vis_dir else os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print(f"Loading SAM model {args.model_type}...")
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(args.device)
    predictor = SamPredictor(sam)
    
    # Collect all images
    all_images = []
    for d in args.input_dirs:
        all_images.extend(list(Path(d).glob("*.jpg"))[:5])
    
    print(f"Found {len(all_images)} images to process.")

    for img_path in tqdm(all_images, desc="Refining Labels"):
        xml_path = img_path.with_suffix(".xml")
        if not xml_path.exists():
            continue
            
        image = cv2.imread(str(img_path))
        if image is None: continue
        
        h, w = image.shape[:2]
        
        # Get Rough Boxes
        rough_boxes = get_boxes_from_xml(xml_path, w, h, target_label=args.target_label)
        if not rough_boxes:
            continue
        
        # Run SAM
        predictor.set_image(image)
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        for box in rough_boxes:
            # 1. Parse the Stencil Coordinates
            x1, y1, x2, y2 = map(int, box)
            
            # Clamp to image boundaries (safety)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1: continue

            # 2. Ask SAM to predict "inside this box"
            input_box = np.array([x1, y1, x2, y2])
            
            masks, scores, _ = predictor.predict(
                box=input_box,
                multimask_output=True
            )
            
            # 3. Apply the Stencil
            best_processed_mask = None
            max_fill = -1
            box_area = (x2 - x1) * (y2 - y1)

            for i, raw_mask in enumerate(masks):
                # Cut off anything outside the box
                clipped_mask = np.zeros_like(raw_mask)
                clipped_mask[y1:y2, x1:x2] = raw_mask[y1:y2, x1:x2]
                
                current_fill = np.sum(clipped_mask) / box_area
                
                # We simply want the mask that covers the most area
                # without any minimum requirement
                if current_fill > max_fill:
                    max_fill = current_fill
                    best_processed_mask = clipped_mask

            # 4. MODIFIED Safety Net
            # Only fall back to a full rectangle if SAM returned NOTHING (0 pixels).
            # If SAM returned even a small sliver (max_fill > 0), we use it.
            
            if best_processed_mask is not None and max_fill > 0.0:
                # Use the organic shape from SAM
                final_mask[best_processed_mask > 0] = 255
            else:
                # SAM failed completely (found nothing inside the box).
                # Only THEN do we force a rectangle so we don't lose the label.
                final_mask[y1:y2, x1:x2] = 255

        # --- SAVE MASK ---
        mask_filename = img_path.stem + ".png"
        cv2.imwrite(os.path.join(args.output_dir, mask_filename), final_mask)

        # --- GENERATE VISUALIZATION (Inline with your snippet) ---
        # We create a side-by-side: [Original with Boxes] | [Original with Refined Mask]
        vis_img = create_vis_image(image, final_mask, rough_boxes)
        
        # Stack vertically or horizontally. Let's do Horizontal.
        combined_vis = np.hstack([image, vis_img])
        
        cv2.imwrite(os.path.join(vis_dir, f"vis_{img_path.stem}.jpg"), combined_vis)

if __name__ == "__main__":
    main()