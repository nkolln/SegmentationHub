import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Module):
    """Atrous Spatial Pyramid Pooling convolution block."""
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ASPPPooling(nn.Module):
    """Global average pooling branch of ASPP."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.pool(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.
    
    Captures multi-scale context using parallel dilated convolutions.
    Essential for segmenting objects at different scales (windows, doors, walls).
    """
    def __init__(self, in_channels, out_channels=256, dilations=(6, 12, 18)):
        super().__init__()
        
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        ]
        
        for dilation in dilations:
            modules.append(ASPPConv(in_channels, out_channels, dilation))
        
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class AttentionRefinement(nn.Module):
    """
    Lightweight attention module for feature refinement.
    Uses channel attention (SE-style) + spatial attention.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_att(x)
        x = x * ca
        # Spatial attention
        sa = self.spatial_att(x)
        x = x * sa
        return x


class EnhancedDecoder(nn.Module):
    """
    Enhanced segmentation decoder with:
    1. ASPP for multi-scale context
    2. Skip connections from encoder (UNet-style)
    3. Attention refinement
    4. Gradual upsampling (avoids checkerboard artifacts)
    
    Input: Dictionary with encoder features at different scales
           {'stage1': (B,C1,H/4,W/4), 'stage2': (B,C2,H/8,W/8), ...}
    """
    def __init__(self, encoder_channels, num_classes, decoder_channels=256):
        """
        Args:
            encoder_channels: List of channel dims from encoder stages [C1, C2, C3, C4]
            num_classes: Number of output classes
            decoder_channels: Internal decoder channel dimension
        """
        super().__init__()
        
        # ASPP on deepest features
        self.aspp = ASPP(encoder_channels[-1], decoder_channels)
        
        # Lateral connections (1x1 conv to match channels)
        self.lateral3 = nn.Conv2d(encoder_channels[-2], decoder_channels, 1)
        self.lateral2 = nn.Conv2d(encoder_channels[-3], decoder_channels, 1)
        self.lateral1 = nn.Conv2d(encoder_channels[-4], decoder_channels, 1)
        
        # Fusion blocks (after skip connection merge)
        self.fuse3 = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Attention refinement
        self.attention = AttentionRefinement(decoder_channels // 2)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(decoder_channels // 2, decoder_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels // 4, num_classes, 1)
        )

    def forward(self, features):
        """
        Args:
            features: List of encoder features [stage1, stage2, stage3, stage4]
                      from shallow to deep
        """
        c1, c2, c3, c4 = features
        
        # ASPP on deepest features
        x = self.aspp(c4)
        
        # Upsample and fuse with c3
        x = F.interpolate(x, size=c3.shape[2:], mode='bilinear', align_corners=False)
        x = x + self.lateral3(c3)
        x = self.fuse3(x)
        
        # Upsample and fuse with c2
        x = F.interpolate(x, size=c2.shape[2:], mode='bilinear', align_corners=False)
        x = x + self.lateral2(c2)
        x = self.fuse2(x)
        
        # Upsample and fuse with c1
        x = F.interpolate(x, size=c1.shape[2:], mode='bilinear', align_corners=False)
        x = x + self.lateral1(c1)
        x = self.fuse1(x)
        
        # Attention refinement
        x = self.attention(x)
        
        # Final classification
        x = self.classifier(x)
        
        return x
