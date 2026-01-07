# Active Learning Guide

Active learning helps you select the most informative samples from an unlabeled dataset for annotation, maximizing the performance gain of your model with the least amount of labeling effort.

## Core Components

1.  **`src/utils/active_learning.py`**: Contains the `ActiveLearningManager` which handles embedding computation and diversity analysis.
2.  **`analyze_diversity.py`**: A CLI tool to visualize your dataset's diversity using PCA or t-SNE.

## Workflow

### 1. Analyze Dataset Diversity
Before starting active learning, it's useful to see how diverse your current dataset is.

```powershell
.\.venv\Scripts\python analyze_diversity.py --config configs/config_mask.yaml
```

This will generate:
- `outputs/diversity_analysis/pca_diversity.png`: A 2D PCA plot of your image embeddings.
- `outputs/diversity_analysis/pca_with_outliers.png`: Highlights samples that are most "different" from the rest.

### 2. Selecting New Samples for Labeling
Use the `ActiveLearningManager` in your custom scripts to rank unlabeled images.

```python
from src.utils.active_learning import ActiveLearningManager
from src.training.trainer import Trainer

# Load your trained model
# ...

manager = ActiveLearningManager(model=model, device='cuda')
embeddings, filenames = manager.get_embeddings(unlabeled_loader)
stats = manager.analyze_diversity(embeddings)

# outliers are the most diverse samples
indices_to_label = stats['outlier_indices']
files_to_label = [filenames[i] for i in indices_to_label]
```

## Tips
- **Use a trained backbone**: For better results, use the encoder from your best-performing model instead of a generic pre-trained one.
- **Batch Selection**: Select samples in batches (e.g., top 100 most diverse) and add them to your training set.
- **Iteration**: Repeat the process after retraining the model with the new labels.
