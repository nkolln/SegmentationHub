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
    def __init__(self, num_classes, model_type='dinov2_vitb14', 
                 head_type='multiscale', freeze_backbone=True):
        super().__init__()
        self.num_classes = num_classes
        self.head_type = head_type
        
        # Load backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_type)
        self.patch_size = 14  # DINOv2 uses 14x14 patches
        
        # Embedding dimensions
        embed_dims = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
            'dinov2_vitg14': 1536
        }
        self.embed_dim = embed_dims.get(model_type, 768)
        
        # Segmentation head
        if head_type == 'multiscale':
            self.head = MultiScaleHead(self.embed_dim, num_classes)
        elif head_type == 'aspp':
            self.head = nn.Sequential(
                ASPP(self.embed_dim, 256),
                nn.Conv2d(256, num_classes, 1)
            )
        else:
            self.head = nn.Sequential(
                nn.Conv2d(self.embed_dim, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, 1)
            )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"✓ DINOv2 backbone frozen")
    
    def forward(self, x, labels=None):
        h, w = x.shape[2:]
        
        # Extract features
        if self.head_type == 'multiscale':
            features = self.backbone.get_intermediate_layers(x, n=4)
            logits = self.head(features, h, w)
        else:
            features = self.backbone.get_intermediate_layers(x, n=1)[0]
            ph, pw = h // self.patch_size, w // self.patch_size
            features = features.permute(0, 2, 1).reshape(-1, self.embed_dim, ph, pw)
            logits = self.head(features)
        
        # Upsample to input size
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        
        # # Return format matching Mask2Former interface
        # output = {"logits": logits}
        
        # if labels is not None:
        #     loss = F.cross_entropy(logits, labels, ignore_index=-1)
        #     output["loss"] = loss
            
        return logits
    
    def unfreeze_backbone(self):
        """Call this after initial training to fine-tune end-to-end"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✓ DINOv2 backbone unfrozen for fine-tuning")