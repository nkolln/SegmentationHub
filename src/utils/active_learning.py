import torch
import numpy as np
from tqdm import tqdm
from lightly.data import LightlyDataset
from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import NTXentLoss
import os

class ActiveLearningManager:
    def __init__(self, model=None, device='cuda'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model = model
        if self.model:
            self.model.to(self.device)
            self.model.eval()

    def get_embeddings(self, dataloader):
        """Compute embeddings for all images in the dataloader."""
        embeddings = []
        filenames = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing embeddings"):
                # Handle both standard datasets and LightlyDataset
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                    if len(batch) > 1 and isinstance(batch[-1], list):
                        fns = batch[-1]
                    else:
                        fns = None
                else:
                    images = batch
                    fns = None
                    
                images = images.to(self.device)
                
                # If we have a model, use it for embeddings
                if self.model:
                    # Try to get features from encoder
                    if hasattr(self.model, 'encoder'):
                        emb = self.model.encoder(images)
                    elif hasattr(self.model, 'backbone'):
                        emb = self.model.backbone(images)
                    else:
                        emb = self.model(images)
                    
                    # Handle HuggingFace dictionary-like outputs (e.g. Mask2Former)
                    if isinstance(emb, dict) or (hasattr(emb, 'to_dict') and not isinstance(emb, torch.Tensor)):
                        # For Mask2Former, we want the global context/embeddings
                        # If it has transformer_decoder_last_hidden_state, it's a good candidate
                        if hasattr(emb, 'transformer_decoder_last_hidden_state'):
                            emb = emb.transformer_decoder_last_hidden_state # (B, num_queries, hidden_dim)
                        elif hasattr(emb, 'encoder_last_hidden_state'):
                            emb = emb.encoder_last_hidden_state
                        elif 'pixel_values' in emb: # Some models return pixel_values for some reason
                            pass # Keep searching
                            
                    # Flatten if necessary
                    if isinstance(emb, torch.Tensor):
                        if len(emb.shape) > 2:
                            # If it's (B, N, C), average over N
                            if len(emb.shape) == 3:
                                emb = emb.mean(dim=1)
                            else:
                                emb = torch.nn.functional.adaptive_avg_pool2d(emb, (1, 1)).flatten(1)
                    else:
                        # Fallback for complex objects: use a simple pooling of the image if model failed to return tensor
                        emb = torch.nn.functional.adaptive_avg_pool2d(images, (8, 8)).flatten(1)
                else:
                    # Fallback or placeholder: use flattened images (not ideal but works for testing)
                    emb = torch.nn.functional.adaptive_avg_pool2d(images, (8, 8)).flatten(1)
                
                if isinstance(emb, torch.Tensor):
                    embeddings.append(emb.detach().cpu().numpy())
                else:
                    embeddings.append(emb) # Already a numpy array?
                if fns:
                    filenames.extend(fns)

        return np.concatenate(embeddings, axis=0), filenames

    def analyze_diversity(self, embeddings):
        """
        Analyze dataset diversity using Coreset or similar.
        Returns indices of the most 'diverse' samples.
        """
        # Placeholder for complex active learning logic
        # In a real scenario, we'd use Coreset selection
        from sklearn.metrics import pairwise_distances
        
        # Simple diversity metric: Average distance to nearest neighbor
        distances = pairwise_distances(embeddings)
        np.fill_diagonal(distances, np.inf)
        nn_distances = distances.min(axis=1)
        
        return {
            'mean_diversity': np.mean(nn_distances),
            'std_diversity': np.std(nn_distances),
            'outlier_indices': np.argsort(nn_distances)[-10:] # Top 10 most 'unique' items
        }
