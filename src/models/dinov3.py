import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
# ... [Keep your ASPP class exactly as it was] ...
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
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
            if out.shape[2:] != x.shape[2:]:
                out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
            res.append(out)
        res = torch.cat(res, dim=1)
        return self.project(res)

class MultiScaleHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        input_dim = embed_dim * 4
        
        self.decode = nn.Sequential(
            nn.Conv2d(input_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(embed_dim, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, num_classes, 1)
        )
        
    def forward(self, features_list, h, w):
        # The backbone wrapper (DinoV3Seg) has already reshaped features 
        # to (B, C, H, W). We just need to concatenate them.
        
        # Resize all to the same size if necessary (usually they are same in ViT)
        # but just in case of slight variations or padding differences:
        resized_feats = []
        target_h, target_w = features_list[0].shape[2:]
        
        for feat in features_list:
            if feat.shape[2:] != (target_h, target_w):
                feat = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
            resized_feats.append(feat)
            
        x = torch.cat(resized_feats, dim=1)
        return self.decode(x)


class DinoV3Seg(nn.Module):
    def __init__(self, num_classes, model_type='facebook/dinov3-vitl16-pretrain-lvd1689m', 
                 head_type='multiscale', freeze_backbone=True, hf_token=None):
        super().__init__()
        self.num_classes = num_classes
        self.head_type = head_type
        
        print(f"Loading {model_type} via Hugging Face Transformers...")
        
        # 1. Load Config
        self.config = AutoConfig.from_pretrained(model_type, token=hf_token)
        self.patch_size = self.config.patch_size # Auto-detects 16
        self.embed_dim = self.config.hidden_size # Auto-detects 1024
        
        # 2. Load Backbone
        self.backbone = AutoModel.from_pretrained(model_type, token=hf_token)
        
        print(f"✓ Model Loaded: Patch Size={self.patch_size}, Dim={self.embed_dim}")
        
        # 3. Setup Head
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
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ Backbone frozen")
    
    def forward(self, x, labels=None):
        # 1. Handle Padding
        h, w = x.shape[2:]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        h_padded, w_padded = x.shape[2:]
        ph, pw = h_padded // self.patch_size, w_padded // self.patch_size
        
        # 2. Forward Pass
        outputs = self.backbone(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # 3. Extract Features
        features_to_process = []
        
        if self.head_type == 'multiscale':
            layers_to_use = hidden_states[-4:] 
        else:
            layers_to_use = [hidden_states[-1]]

        for feat in layers_to_use:
            # Feat: (B, SeqLen, Dim)
            # DINOv3: 1 CLS + 4 Registers + Patches
            
            # Remove first 5 tokens
            spatial_feat = feat[:, 5:, :] 
            
            # Reshape to (B, C, H, W)
            spatial_feat = spatial_feat.permute(0, 2, 1).reshape(-1, self.embed_dim, ph, pw)
            features_to_process.append(spatial_feat)
            
        # 4. Pass to Head (features_to_process is already a list of 4D tensors)
        if self.head_type == 'multiscale':
            logits = self.head(features_to_process, h_padded, w_padded)
        else:
            logits = self.head(features_to_process[0])
        
        # 5. Crop and Upsample
        if pad_h > 0 or pad_w > 0:
            logits = logits[:, :, :h, :w]
            
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
            
        return logits