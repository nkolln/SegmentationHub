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

class CombinedLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, dice_weight=0.5, ce_weight=0.5, boundary_weight=0.0):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.boundary = BoundaryLoss()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.boundary_weight = boundary_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        if self.boundary_weight > 0:
            boundary_loss = self.boundary(logits, targets)
            total_loss += self.boundary_weight * boundary_loss
            
        return total_loss
