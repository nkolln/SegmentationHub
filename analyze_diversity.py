import os
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from src.data.dataset import SegmentationDataset
from src.data.transforms import get_val_transforms
from src.utils.active_learning import ActiveLearningManager
from src.utils.config import load_config
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def main():
    parser = argparse.ArgumentParser(description="Analyze Dataset Diversity")
    parser.add_argument('--config', type=str, default='configs/config_mask.yaml')
    parser.add_argument('--output', type=str, default='outputs/diversity_analysis')
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output, exist_ok=True)

    # 1. Setup Dataset
    # We use all sources to see the overall diversity
    dataset = SegmentationDataset(
        root_dir=config['data']['root_dir'],
        split='all', # Custom split to load everything
        transform=get_val_transforms(config['data']['image_size']),
        sources=config['data'].get('sources', ['base', 'extended'])
    )
    
    # Patch _load_data for 'all' split if needed
    if not hasattr(dataset, 'images') or len(dataset.images) == 0:
        print("Dataset empty for 'all' split, falling back to basic loading...")
    
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    # 2. Compute Embeddings
    # For now, we use a pretrained ResNet or similar if no model is provided
    import torchvision.models as models
    backbone = models.resnet18(pretrained=True)
    backbone = torch.nn.Sequential(*(list(backbone.children())[:-1])) # Remove FC
    
    manager = ActiveLearningManager(model=backbone)
    embeddings, _ = manager.get_embeddings(loader)

    print(f"Computed embeddings for {len(embeddings)} images. Shape: {embeddings.shape}")

    # 3. Analyze Diversity
    stats = manager.analyze_diversity(embeddings)
    print(f"Mean Diversity Score: {stats['mean_diversity']:.4f}")
    
    # 4. Visualization (PCA/TSNE)
    print("Generating visualizations...")
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], alpha=0.5)
    plt.title("Dataset Diversity PCA")
    plt.savefig(os.path.join(args.output, 'pca_diversity.png'))
    
    # Optional: Plot outliers
    outliers = stats['outlier_indices']
    plt.scatter(embeddings_pca[outliers, 0], embeddings_pca[outliers, 1], color='red', label='Highly Diverse/Outliers')
    plt.legend()
    plt.savefig(os.path.join(args.output, 'pca_with_outliers.png'))
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
