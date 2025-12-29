import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class SegformerHF(nn.Module):
    """
    Segformer model using HuggingFace Transformers.
    """
    def __init__(self, num_classes, pretrained_repo="nvidia/mit-b0"):
        super().__init__()
        self.num_classes = num_classes
        
        # Load model with ignore_mismatched_sizes to handle num_classes change
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_repo, 
            num_labels=num_classes, 
            ignore_mismatched_sizes=True
        )
        
    def forward(self, x):
        # HF Segformer returns a SemanticSegmenterOutput
        outputs = self.model(pixel_values=x)
        logits = outputs.logits # Shape: (B, num_classes, H/4, W/4)
        
        # Upsample to input size
        # We need to access original image size. 
        # Usually we want (B, C, H, W).
        # HF documentation says: "The model outputs logits of shape (batch_size, config.num_labels, height/4, width/4)."
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=x.shape[-2:], # (H, W)
            mode="bilinear", 
            align_corners=False
        )
        
        return upsampled_logits
