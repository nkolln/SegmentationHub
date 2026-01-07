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
    """ Robust XML parser that handles dirty files. """
    boxes = []
    try:
        with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Fix missing root tags
        if "<object>" in content and "<root>" not in content:
            start_index = content.find("<object>")
            clean_content = content[start_index:]
            # Remove anything after the last </object> if there is junk
            if "</object>" in clean_content:
                last_tag = clean_content.rfind("</object>") + 9
                clean_content = clean_content[:last_tag]
            wrapped_content = f"<root>{clean_content}</root>"
            root = ET.fromstring(wrapped_content)
        else:
            tree = ET.parse(xml_path)
            root = tree.getroot()

    except Exception as e:
        # print(f"Skipping unreadable XML {xml_path}: {e}")
        return []

    for obj in root.findall('object'):
        name_node = obj.find('labelname')
        if name_node is None: continue
        
        # Normalize string
        label_name = name_node.text.strip().lower()
        if target_label.lower() not in label_name:
            continue
            
        points = obj.find('points')
        if points is None: continue
        
        try:
            xs = [float(x.text) for x in points.findall('x')]
            ys = [float(y.text) for y in points.findall('y')]
        except ValueError: continue
        
        if len(xs) < 2 or len(ys) < 2: continue
        
        x_min = min(xs) * img_width
        x_max = max(xs) * img_width
        y_min = min(ys) * img_height
        y_max = max(ys) * img_height
        
        boxes.append([x_min, y_min, x_max, y_max])
    
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
            # box format: [x_min, y_min, x_max, y_max]
            input_box = np.array(box)
            
            # FIX 1: Add a center point prompt to "anchor" the mask
            # This tells SAM: "The object MUST contain this center pixel"
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            input_point = np.array([[center_x, center_y]])
            input_label = np.array([1]) # 1 means foreground
            
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=True
            )
            
            # FIX 2: Smart Selection Strategy
            # Instead of taking the highest score, we take the mask that best fills the box.
            # Windows usually fill the box. If SAM picks a tiny reflection, this logic discards it.
            
            best_idx = 0
            best_score = -1
            
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            
            for i, mask_candidate in enumerate(masks):
                # Calculate how much of the mask is INSIDE the box
                # (Intersection over Union logic, simplified for box containment)
                mask_area = np.sum(mask_candidate)
                
                # Ratio: How much of the box is filled?
                fill_ratio = mask_area / (box_area + 1e-6)
                
                # Heuristic: Windows usually fill 50% to 100% of the box.
                # If fill_ratio is tiny (<0.2), it's probably just a reflection/glare.
                # If fill_ratio is huge (>1.1), it spilled over the box (bad).
                
                # We prioritize the mask closest to filling 80-90% of the box
                # score_penalty = abs(0.9 - fill_ratio)
                
                # Or simpler: Pick the one with the highest SAM score 
                # BUT filter out the ones that are obviously too small.
                
                if fill_ratio < 0.2: 
                    continue # Skip tiny fragments
                    
                if scores[i] > best_score:
                    best_score = scores[i]
                    best_idx = i
            
            # Apply the winner
            chosen_mask = masks[best_idx]
            final_mask[chosen_mask] = 255

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