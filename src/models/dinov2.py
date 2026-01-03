import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            
        # Image pooling
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        self.convs = nn.ModuleList(modules)
        
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            out = conv(x)
            # Handle image pooling case (needs upsampling)
            if out.shape[2:] != x.shape[2:]:
                out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
            res.append(out)
        res = torch.cat(res, dim=1)
        return self.project(res)

class MultiScaleHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        # 4 layers * embed_dim per layer
        input_dim = embed_dim * 4
        
        self.decode = nn.Sequential(
            nn.Conv2d(input_dim, embed_dim, 1, bias=False), # Projection to reduce dim
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(embed_dim, 256, 3, padding=1, bias=False), # Spatial mixing
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, num_classes, 1) # Final classification
        )
        
    def forward(self, features_list, h, w):
        # features_list: list of 4 tensors (B, NumPatches, EmbedDim)
        # We need to reshape and upsample all to the same grid size if needed (usually they are same size for ViT)
        
        projs = []
        ph, pw = h // 14, w // 14
        
        for feat in features_list:
            # Reshape: (B, N, C) -> (B, C, H, W)
            # N = PH * PW
            feat = feat.permute(0, 2, 1).reshape(-1, feat.shape[-1], ph, pw)
            projs.append(feat)
            
        # Concatenate along channel dimension
        x = torch.cat(projs, dim=1)
        return self.decode(x)

class DinoV2Seg(nn.Module):
    """
    Segmentation model using DINOv2 backbone with configurable heads.
    """
    def __init__(self, num_classes, model_type='dinov2_vits14', head_type='simple'):
        super().__init__()
        self.num_classes = num_classes
        self.head_type = head_type
        
        # Load DINOv2 from Torch Hub
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
            
        self.embed_dim = embed_dim
            
        # Segmentation Head
        print(f"Initializing Segmentation Head: {head_type}")
        if head_type == 'aspp':
            self.head = nn.Sequential(
                ASPP(embed_dim, 256),
                nn.Conv2d(256, num_classes, kernel_size=1)
            )
        elif head_type == 'multiscale':
             self.head = MultiScaleHead(embed_dim, num_classes)
        else: # simple/baseline
            self.head = nn.Sequential(
                nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, kernel_size=1)
            )

    def forward(self, x):
        h, w = x.shape[2:]
        
        if self.head_type == 'multiscale':
             # Get last 4 layers
             features = self.backbone.get_intermediate_layers(x, n=4)
             logits = self.head(features, h, w)
        else:
            # Get intermediate features (patch tokens) - single layer
            features = self.backbone.get_intermediate_layers(x, n=1)[0] # (B, NumPatches, EmbedDim)
            
            # Reshape patches to spatial grid
            ph, pw = h // 14, w // 14
            features = features.permute(0, 2, 1).reshape(-1, features.shape[-1], ph, pw) # (B, C, PH, PW)
            
            # Apply head
            logits = self.head(features)
        
        # Final upsampling to original size
        return F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
