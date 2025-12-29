import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def main():
    # 1. Load Model and Processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    # 2. Load an image
    raw_image = Image.open(r"C:\Users\nickk\Documents\Repo\SegmentationHub\wandb\run-20251229_201507-jjzvty76\files\media\images\val_prediction_0_186da3d2a586cfc8302c.png").convert("RGB")
    
    # 3. Define a point prompt (x, y) coordinates
    input_points = [[[450, 600]]] # 2D array: [batch_size, num_points, 2]
    
    # 4. Process image and prompt
    inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
    
    # 5. Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 6. Post-process masks
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"].cpu(), 
        inputs["reshaped_input_sizes"].cpu()
    )
    scores = outputs.iou_scores
    
    # 7. Visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    
    # Show the best mask (highest score)
    best_mask_idx = torch.argmax(scores[0, 0, :])
    show_mask(masks[0][0][best_mask_idx], plt.gca())
    show_points(np.array(input_points[0]), np.array([1]), plt.gca())
    
    plt.axis('off')
    plt.title(f"SAM Segmentation Result (Score: {scores[0, 0, best_mask_idx]:.3f})")
    
    # Save the result
    output_path = "sam_output.png"
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    print(f"Result saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
