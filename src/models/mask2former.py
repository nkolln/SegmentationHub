import torch
import torch.nn as nn
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

class Mask2FormerHF(nn.Module):
    """
    Mask2Former model using HuggingFace Transformers.
    """
    def __init__(self, num_classes, config=None):
        super().__init__()
        self.num_classes = num_classes
        
        pretrained_repo = config['model'].get('pretrained_repo', "facebook/mask2former-swin-tiny-cityscapes-semantic") if config else "facebook/mask2former-swin-tiny-cityscapes-semantic"
        num_queries = config['model'].get('num_queries', 100) if config else 100

        # Load model and processor
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_repo, 
            num_queries=num_queries, 
            num_labels=num_classes, 
            ignore_mismatched_sizes=True
        )

        # Inject custom loss weights from config if available
        if config and 'loss' in config:
            # Mask2Former uses specific weight names in its configuration
            # Map our config names to HF names
            # Default HF: class_weight=2.0, mask_weight=5.0, dice_weight=5.0
            self.model.config.class_weight = float(config['loss'].get('class_weight', 2.0))
            self.model.config.mask_weight = float(config['loss'].get('mask_weight', 20.0))
            self.model.config.dice_weight = float(config['loss'].get('dice_weight', 1.0))
            
            print(f"🎯 Mask2Former Loss Weights Injected: Class={self.model.config.class_weight}, Mask={self.model.config.mask_weight}, Dice={self.model.config.dice_weight}")

        self.processor = Mask2FormerImageProcessor.from_pretrained(pretrained_repo)
        
    def forward(self, x, labels=None):
        """
        Forward pass for Mask2Former.
        Args:
            x (torch.Tensor): Input images (B, 3, H, W)
            labels (torch.Tensor, optional): Ground truth masks (B, H, W). 
                                            If provided, model computes loss.
        """
        if labels is not None:
            
            # For semantic segmentation, we convert (B, H, W) to the format expected by HF
            # Each unique class in the mask (except ignore_index) becomes a binary mask.
            batch_mask_labels = []
            batch_class_labels = []
            
            for i in range(x.shape[0]):
                mask = labels[i]
                unique_classes = torch.unique(mask)
                mask_labels = []
                class_labels = []
                
                for cls in unique_classes:
                    if cls == 255: # ignore_index
                        continue
                    mask_labels.append((mask == cls).float())
                    class_labels.append(cls.long())
                
                if len(mask_labels) > 0:
                    batch_mask_labels.append(torch.stack(mask_labels))
                    batch_class_labels.append(torch.stack(class_labels))
                else:
                    # Handle empty masks (e.g. all background or all ignored)
                    batch_mask_labels.append(torch.zeros((0, mask.shape[0], mask.shape[1]), device=x.device))
                    batch_class_labels.append(torch.zeros((0,), dtype=torch.long, device=x.device))

            outputs = self.model(
                pixel_values=x,
                mask_labels=batch_mask_labels,
                class_labels=batch_class_labels,
                return_dict=True
            )
        else:
            outputs = self.model(pixel_values=x, return_dict=True)
        
        return outputs

    def post_process(self, outputs, target_sizes):
        """
        Convert raw model outputs to semantic segmentation maps.
        """
        # target_sizes is a list of (H, W) or a tensor (B, 2)
        return self.processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
