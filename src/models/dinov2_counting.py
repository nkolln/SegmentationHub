import torch
import torch.nn as nn
import torch.nn.functional as F

class DinoV2SegCounting(nn.Module):
    """
    Segmentation model using DINOv2 backbone with an additional auxiliary counting head.
    Returns both segmentation logits and a scalar for the window count.
    """
    def __init__(self, num_classes, model_type='dinov2_vits14'):
        super().__init__()
        self.num_classes = num_classes
        
        # Load DINOv2 from Torch Hub
        print(f"Loading DINOv2 backbone for counting: {model_type}...")
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
            
        # Segmentation Head
        self.classifier = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # Counting Head (Regressor)
        # Takes the [CLS] token (global context) and predicts a scalar count
        self.counting_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1) # Outputs a single scalar value
        )

    def forward(self, x):
        h, w = x.shape[2:]
        
        # DINOv2 forward pass with 'forward_features' to get both CLS and Patch tokens
        # features_dict = self.backbone.forward_features(x)
        # patch_tokens = features_dict["x_norm_patchtokens"]
        # cls_token = features_dict["x_norm_clstoken"]
        
        # NOTE: standard hub models might vary slightly in API, checking implementation.
        # Alternatively we can use get_intermediate_layers for patches, but handling CLS is cleaner with forward_features.
        # Let's try the standard approach for DINOv2:
        
        ret = self.backbone.forward_features(x)
        patch_tokens = ret["x_norm_patchtokens"]
        cls_token = ret["x_norm_clstoken"] # (B, EmbedDim)
        
        # 1. Segmentation Branch
        # Reshape patches to spatial grid
        ph, pw = h // 14, w // 14
        # patch_tokens is (B, N_Patches, C)
        features = patch_tokens.permute(0, 2, 1).reshape(x.shape[0], -1, ph, pw) # (B, C, PH, PW)
        
        seg_logits = self.classifier(features)
        seg_logits = F.interpolate(seg_logits, size=(h, w), mode='bilinear', align_corners=False)
        
        # 2. Counting Branch
        count_pred = self.counting_head(cls_token) # (B, 1)
        count_pred = count_pred.squeeze(1) # (B,)
        
        return seg_logits, count_pred
