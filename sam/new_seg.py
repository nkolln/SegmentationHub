import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
import cv2
import os
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path
import sys

# Add project root to path to allow importing src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import SegmentationDataset

def parse_args():
    parser = argparse.ArgumentParser(description="SAM Post-processing for Facade Datasets")
    parser.add_argument("--root_dir", default="data", help="Root directory of the dataset")
    parser.add_argument("--sources", nargs="+", default=["base", "extended"], help="Input sources in root_dir/raw/")
    parser.add_argument("--output_dir", required=True, help="Directory to save refined masks")
    parser.add_argument("--sam_checkpoint", default="sam/sam_vit_h_4b8939.pth", help="Path to SAM checkpoint")
    parser.add_argument("--model_type", default="vit_h", help="SAM model type")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run SAM on")
    parser.add_argument("--target_label", default="window", help="Label name to process from XML")
    return parser.parse_args()

def get_boxes_from_xml(xml_path, img_width, img_height, target_label="window"):
    if not os.path.exists(xml_path):
        return []
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
        
    for obj in root.findall('object'):
        label_name = obj.find('labelname').text
        if label_name != target_label:
            continue
            
        points = obj.find('points')
        xs = [float(x.text) for x in points.findall('x')]
        ys = [float(y.text) for y in points.findall('y')]
        
        if len(xs) < 2 or len(ys) < 2:
            continue
            
        # Normalized to pixel coordinates
        x_min = min(xs) * img_width
        x_max = max(xs) * img_width
        y_min = min(ys) * img_height
        y_max = max(ys) * img_height
        
        boxes.append([x_min, y_min, x_max, y_max])

    return boxes

def main():
    args = parse_args()

    # 1. Load SAM (The "Teacher" model)
    print(f"Loading SAM model {args.model_type} from {args.sam_checkpoint}...")
    if not os.path.exists(args.sam_checkpoint):
        print(f"Warning: Checkpoint {args.sam_checkpoint} not found. Please ensure it exists.")
        
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(args.device)
    predictor = SamPredictor(sam)

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load Dataset
    print(f"Initializing dataset from {args.root_dir}...")
    dataset = SegmentationDataset(
        root_dir=args.root_dir,
        sources=args.sources,
        split='all',
        transform=None # We want raw images for SAM
    )

    # 3. Process Images
    for i in tqdm(range(len(dataset)), desc="Processing items"):
        # __getitem__ returns: image, mask, window_count, image_path
        image_rgb, mask, _, image_path = dataset[i]

        print(mask)
        input('a')
        
        # image_rgb is PIL or numpy? SegmentationDataset uses Image.open().convert("RGB")
        # and returns as numpy if no transform, or depends on albumentations.
        # Let's ensure it's a numpy array for SAM.
        image_np = np.array(image_rgb)
        
        h, w = image_np.shape[:2]
        predictor.set_image(image_np)
        
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        for box in mask:
            # box format: [x_min, y_min, x_max, y_max]
            input_box = np.array(box)
            
            # Ask SAM: "What is the object INSIDE this box?"
            masks, scores, logits = predictor.predict(
                box=input_box,
                multimask_output=True
            )
            
            # SAM returns 3 guesses. Usually the one with the highest score is best.
            best_mask = masks[np.argmax(scores)]
            
            # Add to final mask (assuming binary for now, can be adjusted for multiclass)
            final_mask[best_mask] = 255 # Or specific class index if needed
            
        # Save refined mask
        item_name = os.path.basename(image_path).replace('.jpg', '')
        mask_filename = item_name + "_mask.png"
        output_path = os.path.join(args.output_dir, mask_filename)
        
        # cv2 expects BGR for imwrite
        # final_mask is single channel, so BGR/RGB doesn't matter for saving
        cv2.imwrite(output_path, final_mask)

if __name__ == "__main__":
    main()
