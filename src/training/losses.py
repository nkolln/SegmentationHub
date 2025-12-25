import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, weight=None, ignore_index=-100, dice_weight=0.5, ce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
