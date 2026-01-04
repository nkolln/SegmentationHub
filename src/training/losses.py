import numpy as np
from scipy.ndimage import distance_transform_edt as eucl_dist
import torch
import torch.nn as nn
import torch.nn.functional as F
def one_hot(proba, num_classes):
    # proba: (B, C, H, W) or (B, H, W)
    if proba.dim() == 4:
        # Already (B, C, H, W)?
        pass
    else:
        # One-hot encoding
        proba = torch.eye(num_classes, device=proba.device)[proba.long()]
        proba = proba.permute(0, 3, 1, 2).float()
    return proba

class BoundaryLoss(nn.Module):
    def __init__(self, num_classes=4):
        super(BoundaryLoss, self).__init__()
        self.num_classes = num_classes

    def _one_hot(self, targets, num_classes):
        # targets: (B, H, W)
        B, H, W = targets.size()
        one_hot = torch.zeros((B, num_classes, H, W), device=targets.device)
        one_hot.scatter_(1, targets.unsqueeze(1).long(), 1)
        return one_hot
    
    def _compute_distance_map(self, target_one_hot):
        """
        Compute Signed Distance Map for each class in the batch.
        target_one_hot: (B, C, H, W) numpy array
        Returns: (B, C, H, W) tensors
        """
        # Convert to numpy for scipy
        target_np = target_one_hot.cpu().numpy()
        
        distance_maps = np.zeros_like(target_np)
        
        for b in range(target_np.shape[0]):
            for c in range(target_np.shape[1]):
                mask_c = target_np[b, c, :, :]
                
                # If mask is empty, distance is everywhere (approx with max)
                if mask_c.sum() == 0:
                    # Treat everything as background?
                    # EDT of background (0) is 0? No.
                    # If empty, distance to boundary is infinite?
                    # Let's verify standard behavior.
                    # Usually we want phi to be 0 for bg?
                    # Or just skip empty classes?
                    # Standard Boundary Loss impl:
                    # dist = edt(inverted) + edt(mask) - 1 ??
                    # Simple Signed Distance:
                    # -edt(mask) for inside
                    # +edt(inverted) for outside
                    
                    # If no object: inside is empty. Outside is full.
                    # EDT(inverted) is distances.
                    pos = eucl_dist(1 - mask_c)
                    distance_maps[b, c] = pos
                elif mask_c.sum() == mask_c.size:
                    # Full object
                    # Inside is full. Outside is empty.
                    neg = eucl_dist(mask_c)
                    distance_maps[b, c] = -neg
                else:
                    pos = eucl_dist(1 - mask_c)
                    neg = eucl_dist(mask_c)
                    # Signed distance: -inside, +outside
                    distance_maps[b, c] = pos - (neg - 1)
                    
        return torch.from_numpy(distance_maps).float().to(target_one_hot.device)

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, H, W)
        """
        probs = F.softmax(logits, dim=1)
        
        # Prepare targets
        target_one_hot = self._one_hot(targets, self.num_classes)
        
        # Compute Distance Maps (Expensive! Optimization possible: precompute in Dataset)
        # For now, do it on CPU here.
        with torch.no_grad():
            phi = self._compute_distance_map(target_one_hot)
            
        # Multiplying probabilities by signed distance
        # If deeply inside (phi < 0), prob should be 1. term is negative (good)
        # If deeply outside (phi > 0), prob should be 0. term is positive (bad if prob > 0)
        loss = torch.mean(probs * phi)
        
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W) - raw output of the model
        targets: (B, H, W) - ground truth class indices
        """
        num_classes = logits.shape[1]
        
        # Apply Softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        # targets is (B, H, W) -> (B, C, H, W)
        true_1_hot = torch.eye(num_classes, device=logits.device)[targets.long()]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        
        # Flatten
        probs_flat = probs.contiguous().view(probs.shape[0], num_classes, -1)
        true_1_hot_flat = true_1_hot.contiguous().view(true_1_hot.shape[0], num_classes, -1)
        
        # Calculate intersection and union
        intersection = (probs_flat * true_1_hot_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + true_1_hot_flat.sum(dim=2)
        
        # Dice score per class: 2*I / (U + smooth)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Average over classes and batch
        return 1.0 - dice_score.mean()

from .focal_loss import FocalLoss

class CombinedLoss(nn.Module):
    """
    Highly flexible loss class that combines multiple loss functions based on configuration.
    Example config:
    {
        'cross_entropy': 0.5,
        'dice': 0.5,
        'focal': {'weight': 0.0, 'alpha': 0.25, 'gamma': 2.0},
        'boundary': 0.1
    }
    """
    def __init__(self, loss_config, num_classes=4, weight=None, ignore_index=-100):
        super(CombinedLoss, self).__init__()
        self.loss_weights = {}
        self.losses = nn.ModuleDict()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        for loss_name, config in loss_config.items():
            if isinstance(config, (int, float)):
                weight_val = float(config)
                params = {}
            elif isinstance(config, dict):
                weight_val = float(config.get('weight', 0.0))
                params = {k: v for k, v in config.items() if k != 'weight'}
            else:
                continue

            if weight_val <= 0:
                continue

            self.loss_weights[loss_name] = weight_val

            if loss_name == 'cross_entropy':
                self.losses[loss_name] = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
            elif loss_name == 'dice':
                self.losses[loss_name] = DiceLoss(ignore_index=ignore_index, **params)
            elif loss_name == 'focal':
                self.losses[loss_name] = FocalLoss(ignore_index=ignore_index if ignore_index != -100 else 255, **params)
            elif loss_name == 'boundary':
                self.losses[loss_name] = BoundaryLoss(num_classes=num_classes)
            else:
                print(f"⚠️ Warning: Unknown loss type '{loss_name}' encountered in config. Skipping.")
                del self.loss_weights[loss_name]

        if not self.losses:
            print("⚠️ Warning: No valid losses found in config. Defaulting to CrossEntropy.")
            self.losses['cross_entropy'] = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
            self.loss_weights['cross_entropy'] = 1.0

    def forward(self, logits, targets):
        total_loss = 0
        loss_components = {}

        for loss_name, weight in self.loss_weights.items():
            loss_val = self.losses[loss_name](logits, targets)
            total_loss += weight * loss_val
            loss_components[f"loss_{loss_name}"] = loss_val.item()

        return total_loss
