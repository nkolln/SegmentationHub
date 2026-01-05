import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Numerically stable Focal Loss for addressing class imbalance.
    
    Focal loss focuses training on hard examples by down-weighting easy examples.
    This is especially useful when some classes are much rarer than others.
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255):
        """
        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples
            gamma: Focusing parameter for modulating loss. Higher = more focus on hard examples
            ignore_index: Class index to ignore in loss computation
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, H, W) - Raw logits from model
            targets: (B, H, W) - Ground truth class indices
        
        Returns:
            Scalar focal loss value
        """
        # Get softmax probabilities (more stable than using exp(-ce_loss))
        probs = F.softmax(inputs, dim=1)  # (B, C, H, W)
        
        # Flatten for easier processing
        B, C, H, W = inputs.shape
        inputs_flat = inputs.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        targets_flat = targets.view(-1)  # (B*H*W,)
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        
        # Create mask for valid pixels (not ignore_index)
        valid_mask = targets_flat != self.ignore_index
        
        if valid_mask.sum() == 0:
            # No valid pixels, return zero loss
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Filter out ignored pixels
        inputs_valid = inputs_flat[valid_mask]
        targets_valid = targets_flat[valid_mask]
        probs_valid = probs_flat[valid_mask]
        
        # Safety clip to prevent out-of-bounds index errors
        targets_valid = torch.clamp(targets_valid, 0, C - 1)
        
        # Get probability of true class for each pixel
        pt = probs_valid[torch.arange(len(targets_valid)), targets_valid]
        
        # Clamp to prevent numerical issues
        pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)
        
        # Compute focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute cross-entropy: -log(pt)
        ce_loss = -torch.log(pt)
        
        # Apply focal loss formula: alpha * (1-pt)^gamma * CE
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()
