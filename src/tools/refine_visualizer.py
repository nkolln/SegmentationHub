import os
import argparse
import shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob

def create_comparison_image(img_path, base_mask_path, refined_mask_path, output_path):
    """
    Creates a side-by-side comparison image and saves it.
    """
    img = Image.open(img_path).convert('RGB')
    
    refined_mask = Image.open(refined_mask_path)
    
    # Try to load base mask if it exists
    base_mask = None
    if base_mask_path and os.path.exists(base_mask_path):
        base_mask = Image.open(base_mask_path)
    
    num_plots = 3 if base_mask else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    
    # Image
    axes[0].imshow(img)
    axes[0].set_title(f"Image: {os.path.basename(img_path)}")
    axes[0].axis('off')
    
    # Base Mask
    curr_idx = 1
    if base_mask:
        axes[curr_idx].imshow(base_mask, cmap='tab10', vmin=0, vmax=9)
        axes[curr_idx].set_title("Base Mask")
        axes[curr_idx].axis('off')
        curr_idx += 1
    
    # Refined Mask
    axes[curr_idx].imshow(refined_mask, cmap='tab10', vmin=0, vmax=9)
    axes[curr_idx].set_title("Refined Mask (Model Prediction)")
    axes[curr_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Human-in-the-loop mask refinement visualizer")
    parser.add_argument('--refined_dir', type=str, default='refined_masks', help='Directory containing refined masks')
    parser.add_argument('--source_dir', type=str, default='data/raw/extended', help='Directory containing source images and base masks')
    parser.add_argument('--approved_dir', type=str, default='data/raw/approved', help='Directory to save approved masks')
    parser.add_argument('--viz_path', type=str, default='current_review.png', help='Path to save temp comparison image')
    args = parser.parse_args()

    os.makedirs(args.approved_dir, exist_ok=True)
    
    refined_masks = glob.glob(os.path.join(args.refined_dir, "*.png"))
    if not refined_masks:
        print(f"No refined masks found in {args.refined_dir}. Run generate_refined_masks.py first.")
        return

    print(f"Found {len(refined_masks)} masks to review.")
    print("Controls: [a] Approve, [s] Skip/Reject, [q] Quit")
    
    for rm_path in refined_masks:
        filename = os.path.basename(rm_path)
        name, _ = os.path.splitext(filename)
        
        # Find corresponding source image
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            possible_path = os.path.join(args.source_dir, name + ext)
            if os.path.exists(possible_path) and possible_path != rm_path:
                img_path = possible_path
                break
        
        if not img_path:
            print(f"Warning: Could not find source image for {filename}. Skipping.")
            continue
            
        # Find corresponding base mask (it's often a .png with same name)
        base_mask_path = os.path.join(args.source_dir, name + ".png")
        if not os.path.exists(base_mask_path):
            base_mask_path = None

        # Create visualization
        create_comparison_image(img_path, base_mask_path, rm_path, args.viz_path)
        
        print(f"\nReviewing: {filename}")
        print(f"Visualization saved to {args.viz_path}. Please open it to review.")
        
        choice = input("Approve [a], Skip [s], Quit [q]: ").lower()
        
        if choice == 'a':
            target_path = os.path.join(args.approved_dir, filename)
            shutil.copy(rm_path, target_path)
            print(f"Approved! Mask saved to {target_path}")
        elif choice == 'q':
            print("Quitting review.")
            break
        else:
            print("Skipped.")

    if os.path.exists(args.viz_path):
        os.remove(args.viz_path)
    print("Review session finished.")

if __name__ == "__main__":
    main()
