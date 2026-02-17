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
        
        # Determine architecture type (ViT vs ConvNeXt) based on config attributes
        if hasattr(self.config, 'patch_size'):
            self.patch_size = self.config.patch_size
            self.embed_dim = self.config.hidden_size
            self.is_convnext = False
        else:
            self.patch_size = None # ConvNeXt doesn't utilize patches in the same way
            self.is_convnext = True
            # ConvNeXt often has varying channel depths (e.g., [128, 256, 512, 1024])
            # We'll project them all to the final embedding dimension for the head
            self.feature_channels = getattr(self.config, 'hidden_sizes', [128, 256, 512, 1024]) # Default for base
            self.embed_dim = self.feature_channels[-1] 
            
            # Adapters to project all levels to the same embedding dimension
            self.adapter_convs = nn.ModuleList([
                nn.Conv2d(c, self.embed_dim, 1, bias=False) 
                for c in self.feature_channels
            ])
            
        
        # 2. Load Backbone
        self.backbone = AutoModel.from_pretrained(model_type, token=hf_token)
        
        print(f"✓ Model Loaded: Type={'ConvNeXt' if self.is_convnext else 'ViT'}, Dim={self.embed_dim}")
        
        # 3. Setup Head
        if head_type == 'multiscale':
            self.head = MultiScaleHead(self.embed_dim, num_classes)
        elif head_type == 'aspp':
            self.projection = nn.Sequential(
                nn.Conv2d(self.embed_dim * 4, 512, 3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
            self.aspp = ASPP(512,256,atrous_rates = [6,12,18])
            self.classifier = nn.Conv2d(256, num_classes, 1)
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
        features_to_process = []
        h, w = x.shape[2:]
        h_padded, w_padded = h, w

        if not self.is_convnext:
            # --- ViT Path ---
            # 1. Handle Padding
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
            if self.head_type in ['multiscale', 'aspp']:
                layers_to_use = hidden_states[-4:]
            else:
                layers_to_use = [hidden_states[-1]]

            for feat in layers_to_use:
                spatial_feat = feat[:, 5:, :]
                spatial_feat = spatial_feat.permute(0, 2, 1).reshape(-1, self.embed_dim, ph, pw)
                features_to_process.append(spatial_feat)
        
        else:
            # --- ConvNeXt Path ---
            # 1. Forward Pass (No manual padding usually needed for ConvNeXt)
            outputs = self.backbone(x, output_hidden_states=True)
            hidden_states = outputs.hidden_states 
            # hidden_states often includes the embedding layer output, so we take the last 4 actual stages
            # The indices might vary, but usually they are the last 4. 
            
            # 2. Extract and Adapt Features
            # We assume the backbone outputs multiple levels. We need the last 4 for multiscale/ASPP.
            # Adapter convs are aligned with the `hidden_sizes`.
            
            layers_to_use = hidden_states[-4:]
            
            if len(layers_to_use) != len(self.adapter_convs):
                 # Fallback/Safety: Try to match by channel count if possible, or just take last few
                 # This handles cases where hidden_states might include stem outputs
                 pass

            for i, feat in enumerate(layers_to_use):
                # Project to common embed_dim
                # Note: ConvNeXt outputs are already (B, C, H, W)
                adapted_feat = self.adapter_convs[i](feat)
                features_to_process.append(adapted_feat)
        
           
        # 4. Pass to Head
        if self.head_type == 'multiscale':
            logits = self.head(features_to_process, h_padded, w_padded)
        elif self.head_type == 'aspp':
            # Concatenate 4 layers → Project → ASPP → Classify
            # Resize all to largest spatial dim (usually the first of the 4, P2) for concatenation
            target_h_feat, target_w_feat = features_to_process[0].shape[2:]
            
            resized_feats = []
            for feat in features_to_process:
                if feat.shape[2:] != (target_h_feat, target_w_feat):
                    feat = F.interpolate(feat, size=(target_h_feat, target_w_feat), mode='bilinear', align_corners=False)
                resized_feats.append(feat)
                
            x_concat = torch.cat(resized_feats, dim=1)  # (B, 4096, H, W)
            x_proj = self.projection(x_concat)  # (B, 512, H, W)
            x_aspp = self.aspp(x_proj)  # (B, 256, H, W)
            logits = self.classifier(x_aspp)  # (B, num_classes, H, W)
        
        # 5. Crop and Upsample (if needed, mostly for ViT Padding or ConvNeXt downsampling)
        if logits.shape[2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
            
        return logits
