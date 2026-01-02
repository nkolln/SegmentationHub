import torch
import torch.nn as nn
import torch.nn.functional as F

class DinoV2Seg(nn.Module):
    """
    Segmentation model using DINOv2 backbone.
    DINOv2 features are extremely robust for small datasets.
    """
    def __init__(self, num_classes, model_type='dinov2_vits14'):
        super().__init__()
        self.num_classes = num_classes
        
        # Load DINOv2 from Torch Hub
        # Options: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
        print(f"Loading DINOv2 backbone: {model_type}...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_type)
        
        # Determine embedding dimension
        if 'vits' in model_type:
            embed_dim = 384
        elif 'vitb' in model_type:
            embed_dim = 768
        elif 'vitl' in model_type:
            embed_dim = 1024
        else: # vitg
            embed_dim = 1536
            
        # Segmentation Head (Simple Linear Head as a strong SOTA baseline for DINOv2)
        # For more complex tasks, we could use an UperNet or Mask2Former head.
        self.classifier = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        
        # DINOv2 ViT expects multiples of patch_size (14)
        # We handle this by resizing internally or relying on external resize.
        
        # Get intermediate features or patches
        # Here we use the patch tokens
        features = self.backbone.get_intermediate_layers(x, n=1)[0] # (B, NumPatches, EmbedDim)
        
        # Reshape patches to spatial grid
        ph, pw = h // 14, w // 14
        features = features.permute(0, 2, 1).reshape(-1, features.shape[-1], ph, pw) # (B, C, PH, PW)
        
        # Upsample and classify
        logits = self.classifier(features)
        
        # Final upsampling to original size
        return F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
