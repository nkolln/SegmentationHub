import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

class Mask2FormerHF(nn.Module):
    """
    Mask2Former model using HuggingFace Transformers.
    Note: Mask2Former is a universal segmenter (instance/semantic/panoptic).
    """
    def __init__(self, num_classes, pretrained_repo="facebook/mask2former-swin-tiny-cityscapes-semantic"):
        super().__init__()
        self.num_classes = num_classes
        
        # Load model
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_repo, 
            num_queries=100, # Default for semantic
            num_labels=num_classes, 
            ignore_mismatched_sizes=True
        )
        
    def forward(self, x):
        # Mask2Former returns a Mask2FormerModelOutput
        # During training it returns losses if labels provided, 
        # but here we use it in a standard segmentation trainer.
        outputs = self.model(pixel_values=x)
        
        # In semantic mode, we typically want class logits per pixel.
        # Mask2Former outputs masks and class predictions separately.
        # The library provides a 'post_process_semantic_segmentation' but 
        # we need differentiable logits for the loss function.
        
        # For simplicity and compatibility with standard CE/Dice loss:
        # We use the class-logits and mask-logits to create final segmentation logits.
        mask_cls_logits = outputs.class_queries_logits # (B, num_queries, num_classes + 1)
        mask_pred_logits = outputs.masks_queries_logits # (B, num_queries, H/4, W/4)
        
        # Upsample mask logits to input size
        mask_pred_logits = nn.functional.interpolate(
            mask_pred_logits, 
            size=x.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        # For stability, we use log_softmax on class logits and log_sigmoid on mask logits
        # mask_cls_logits: (B, Q, C+1) -> (B, Q, C)
        mask_cls_probs = F.softmax(mask_cls_logits, dim=-1)[:, :, :-1] # (B, Q, C)
        mask_pred_probs = mask_pred_logits.sigmoid() # (B, Q, H, W)
        
        # Compute final per-pixel probabilities: (B, C, H, W)
        probs = torch.einsum("bqc,bqhw->bchw", mask_cls_probs, mask_pred_probs)
        
        # Clip probabilities to avoid log(0) and log(1) issues, then convert to logits
        probs = torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)
        logits = torch.log(probs)
        
        return logits
