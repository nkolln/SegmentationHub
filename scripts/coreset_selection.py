"""
Lightly Coreset-driven Active Learning for facade segmentation.

Uses the Lightly library to perform intelligent sample selection based on
embedding diversity. Specifically uses CoreSet selection which finds the
most representative subset that covers the embedding space.

Usage:
    # Using a trained model for embeddings
    python scripts/coreset_selection.py --model outputs/dinov3/best_model.pth --limit 50
    
    # Using raw images (no model, uses image features)
    python scripts/coreset_selection.py --unlabeled data/unlabeled --limit 100
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader, Dataset
from src.data.transforms import get_val_transforms


class UnlabeledImageDataset(Dataset):
    """Simple dataset for loading unlabeled images."""
    
    def __init__(self, image_dir: str, transform=None, extensions=('.jpg', '.jpeg', '.png', '.webp')):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.images = []
        
        for ext in extensions:
            self.images.extend(list(self.image_dir.glob(f"*{ext}")))
            self.images.extend(list(self.image_dir.glob(f"*{ext.upper()}")))
        
        self.images = sorted(self.images)
        print(f"Found {len(self.images)} images in {image_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, str(img_path)


class CoresetSelector:
    """
    Selects the most diverse/representative samples using CoreSet algorithm.
    
    CoreSet selection iteratively picks samples that maximize the minimum
    distance to already-selected samples, ensuring good coverage of the
    embedding space.
    """
    
    def __init__(self, model=None, device='cuda', image_size=512):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.image_size = image_size
        
        if self.model:
            self.model.to(self.device)
            self.model.eval()
    
    def extract_embeddings(self, dataloader) -> tuple:
        """
        Extract embeddings from images using the model backbone.
        
        Returns:
            embeddings: numpy array of shape (N, D)
            filenames: list of image paths
        """
        embeddings = []
        filenames = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting embeddings"):
                if len(batch) == 2:
                    images, paths = batch
                elif len(batch) == 4:
                    images, masks, counts, paths = batch
                else:
                    images = batch[0]
                    paths = ["unknown"] * len(images)
                images = images.to(self.device)
                
                if self.model:
                    # Try to get backbone features
                    if hasattr(self.model, 'backbone'):
                        backbone = self.model.backbone
                        if hasattr(backbone, 'get_intermediate_layers'):
                            # DINOv2/v3 style
                            feats = backbone.get_intermediate_layers(images, n=1)[0]
                            # Average over spatial dimension
                            emb = feats.mean(dim=1)
                        elif hasattr(backbone, 'forward_features'):
                            feats = backbone.forward_features(images)
                            if isinstance(feats, dict):
                                emb = feats.get('x_norm_clstoken', feats.get('cls_token', feats))
                            else:
                                emb = feats.mean(dim=1) if len(feats.shape) == 3 else feats
                        else:
                            # Fallback for generic backbone (likely HF)
                            outputs = backbone(images)
                            # Check for HF return type (dict-like or tuple)
                            if hasattr(outputs, 'last_hidden_state'):
                                emb = outputs.last_hidden_state
                            elif isinstance(outputs, (tuple, list)):
                                emb = outputs[0]
                            else:
                                emb = outputs
                                
                            # Convert to spatial/global pooling
                            if hasattr(emb, 'shape') and len(emb.shape) > 2:
                                # (B, Seq, D) or (B, D, H, W)
                                if len(emb.shape) == 3:
                                    emb = emb.mean(dim=1) # (B, D)
                                elif len(emb.shape) == 4:
                                    emb = emb.mean(dim=(2, 3)) # (B, D)
                    else:
                        # Use model directly
                        emb = self.model(images)
                        if hasattr(emb, 'logits'):
                            emb = emb.logits
                        if len(emb.shape) > 2:
                            emb = torch.nn.functional.adaptive_avg_pool2d(emb, 1).flatten(1)
                else:
                    # No model: use raw image features (resized + flattened)
                    emb = torch.nn.functional.adaptive_avg_pool2d(images, (16, 16)).flatten(1)
                
                embeddings.append(emb.cpu().numpy())
                filenames.extend(paths)
        
        return np.concatenate(embeddings, axis=0), filenames
    
    def coreset_select(self, embeddings: np.ndarray, n_select: int, 
                       seed_indices: list = None) -> list:
        """
        Perform CoreSet selection using greedy farthest-point sampling.
        
        Args:
            embeddings: (N, D) array of embeddings
            n_select: Number of samples to select
            seed_indices: Optional indices of already-labeled samples to start from
            
        Returns:
            List of selected indices
        """
        from sklearn.metrics.pairwise import euclidean_distances
        
        n_samples = len(embeddings)
        n_select = min(n_select, n_samples)
        
        # Initialize distance to selected set
        if seed_indices and len(seed_indices) > 0:
            selected = list(seed_indices)
            # Compute distances to seed set
            seed_embeddings = embeddings[seed_indices]
            min_distances = euclidean_distances(embeddings, seed_embeddings).min(axis=1)
        else:
            selected = []
            # Start with the sample closest to the centroid (most "average")
            centroid = embeddings.mean(axis=0, keepdims=True)
            distances_to_centroid = euclidean_distances(embeddings, centroid).flatten()
            first_idx = np.argmin(distances_to_centroid)
            selected.append(first_idx)
            min_distances = euclidean_distances(embeddings, embeddings[[first_idx]]).flatten()
        
        # Iteratively select farthest points
        print(f"Selecting {n_select} samples using CoreSet...")
        
        for i in tqdm(range(n_select - len(selected)), desc="CoreSet selection"):
            # Find the sample farthest from current selection
            # Exclude already selected
            distances_copy = min_distances.copy()
            distances_copy[selected] = -np.inf
            
            new_idx = np.argmax(distances_copy)
            selected.append(new_idx)
            
            # Update minimum distances
            new_distances = euclidean_distances(embeddings, embeddings[[new_idx]]).flatten()
            min_distances = np.minimum(min_distances, new_distances)
        
        # Return only the newly selected (not seeds)
        if seed_indices:
            return [idx for idx in selected if idx not in seed_indices]
        return selected
    
    def uncertainty_weighted_coreset(self, embeddings: np.ndarray, 
                                     predictions: np.ndarray,
                                     n_select: int) -> list:
        """
        CoreSet selection weighted by prediction uncertainty.
        
        Samples with high uncertainty get a boost in their distance,
        making them more likely to be selected.
        
        Args:
            embeddings: (N, D) embedding array
            predictions: (N, C) softmax predictions
            n_select: Number to select
        """
        # Compute entropy as uncertainty measure
        # Higher entropy = more uncertain
        entropy = -np.sum(predictions * np.log(predictions + 1e-8), axis=1)
        entropy_normalized = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
        
        # Weight embeddings by uncertainty
        # Uncertain samples get "pushed" farther in embedding space
        uncertainty_weights = 1 + entropy_normalized * 0.5  # 1.0 to 1.5x
        weighted_embeddings = embeddings * uncertainty_weights[:, np.newaxis]
        
        return self.coreset_select(weighted_embeddings, n_select)


def run_selection(args):
    """Main selection routine."""
    
    # Load model if provided
    model = None
    if args.model and os.path.exists(args.model):
        print(f"Loading model from {args.model}...")
        checkpoint = torch.load(args.model, map_location='cpu')
        
        # Determine model type from path
        if 'dinov3_m2f' in args.model.lower() or ('config' in checkpoint and checkpoint['config']['model']['name'] == 'dinov3_m2f'):
            from src.models.dinov3_m2f import DINOv3Mask2Former
            # Create a minimal config if needed, or use the one from checkpoint
            config = checkpoint.get('config', None)
            model = DINOv3Mask2Former(num_classes=12, config=config)
        elif 'dinov3' in args.model.lower():
            from src.models.dinov3 import DinoV3Seg
            model = DinoV3Seg(num_classes=12)
        elif 'dinov2' in args.model.lower():
            from src.models.dinov2 import DinoV2Seg
            model = DinoV2Seg(num_classes=12)
        elif 'mask2former' in args.model.lower():
            from src.models.mask2former import Mask2FormerHF
            model = Mask2FormerHF(num_classes=12)
        else:
            print("Warning: Could not determine model type, using raw image features")
        
        if model and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        elif model:
            model.load_state_dict(checkpoint, strict=False)
    
    # Setup dataset
    transform = get_val_transforms(args.image_size)
    
    if args.unlabeled:
        dataset = UnlabeledImageDataset(args.unlabeled, transform=transform)
    else:
        # Use existing dataset (Training data)
        print("Using training dataset (base + extended)...")
        from src.data.dataset import SegmentationDataset
        dataset = SegmentationDataset(
            root_dir='data',
            split='all', # Use all data
            transform=transform,
            sources=['base', 'extended']
        )
    
    if len(dataset) == 0:
        print("Error: No images found!")
        return
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Run selection
    selector = CoresetSelector(model=model, device=args.device, image_size=args.image_size)
    embeddings, filenames = selector.extract_embeddings(dataloader)
    
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Selecting {args.limit} samples...")
    
    selected_indices = selector.coreset_select(embeddings, args.limit)
    
    # Output results
    print(f"\n{'='*60}")
    print(f"CORESET SELECTION RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {len(embeddings)}")
    print(f"Selected: {len(selected_indices)}")
    print(f"\nTop samples to label (by diversity):")
    
    selected_files = []
    for i, idx in enumerate(selected_indices[:20]):  # Show first 20
        fname = filenames[idx] if idx < len(filenames) else f"sample_{idx}"
        print(f"  {i+1}. {os.path.basename(fname)}")
        selected_files.append(fname)
    
    if len(selected_indices) > 20:
        print(f"  ... and {len(selected_indices) - 20} more")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "coreset_selection.txt"
    with open(results_file, 'w') as f:
        f.write(f"# CoreSet Selection Results\n")
        f.write(f"# Total samples: {len(embeddings)}\n")
        f.write(f"# Selected: {len(selected_indices)}\n\n")
        for idx in selected_indices:
            f.write(f"{filenames[idx]}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    print(f"\nResults saved to: {results_file}")
    
    # Optional: copy selected images to a subfolder or specific path
    if args.copy_selected or args.save_images_to:
        if args.save_images_to:
            selected_dir = Path(args.save_images_to)
        else:
            selected_dir = output_dir / "selected_for_labeling"
            
        selected_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        for idx in selected_indices:
            src = Path(filenames[idx])
            if src.exists():
                shutil.copy(src, selected_dir / src.name)
        
        print(f"Selected images copied to: {selected_dir}")
    
    return selected_indices, filenames


def main():
    parser = argparse.ArgumentParser(
        description="CoreSet-based active learning sample selection"
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=r"C:\Users\melte\Documents\Repos\SegmentationHub\outputs\dinov3\dinov3l_12_mh_refined_data_actual\checkpoint_epoch_10_fold0.pth",
        help='Path to trained model checkpoint for embedding extraction'
    )
    parser.add_argument(
        '--unlabeled', '-u',
        type=str,
        default=None,
        help='Directory containing unlabeled images'
    )
    parser.add_argument(
        '--limit', '-n',
        type=int,
        default=50,
        help='Number of samples to select'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/active_learning',
        help='Output directory for results'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=4,
        help='Batch size for embedding extraction'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=512,
        help='Image size for transforms'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--copy-selected',
        action='store_true',
        help='Copy selected images to output folder'
    )
    parser.add_argument(
        '--save-images-to',
        type=str,
        default=None,
        help='Specific folder to save selected images to (overrides default copy behavior)'
    )
    
    args = parser.parse_args()
    run_selection(args)


if __name__ == "__main__":
    main()
