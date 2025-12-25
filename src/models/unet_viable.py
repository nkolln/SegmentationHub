import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, skip=None):
        # Upsample x to match skip resolution logic or just x2
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        if skip is not None:
            # Handle potential padding issues if dimensions don't match exactly (e.g. odd sizes)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
            
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class ResNetUNet(nn.Module):
    """
    U-Net with ResNet34 Encoder.
    """
    def __init__(self, n_classes=12, pretrained=True):
        super().__init__()
        
        # Encoder
        self.base_model = models.resnet34(weights='DEFAULT' if pretrained else None)
        self.encoder_layers = list(self.base_model.children())
        
        # Extract layers for skips
        # input -> (conv1, bn1, relu) -> (maxpool) -> layer1 -> layer2 -> layer3 -> layer4
        self.layer0 = nn.Sequential(*self.encoder_layers[:3]) # H/2, 64
        self.layer0_pool = self.encoder_layers[3]             # H/4
        self.layer1 = self.encoder_layers[4]                  # H/4, 64
        self.layer2 = self.encoder_layers[5]                  # H/8, 128
        self.layer3 = self.encoder_layers[6]                  # H/16, 256
        self.layer4 = self.encoder_layers[7]                  # H/32, 512
        
        # Decoder
        # Layer 4 (512) -> Layer 3 (256) -> Out: 256
        self.dec4 = DecoderBlock(512, 256, 256)
        
        # Layer 4_out (256) -> Layer 2 (128) -> Out: 128
        self.dec3 = DecoderBlock(256, 128, 128)
        
        # Layer 3_out (128) -> Layer 1 (64) -> Out: 64
        self.dec2 = DecoderBlock(128, 64, 64)
        
        # Layer 2_out (64) -> Layer 0 (64) -> Out: 64
        self.dec1 = DecoderBlock(64, 64, 64)
        
        # Final Upsample: Layer 1_out (64) -> Original Size
        # No skip here from input usually in basic ResUNet, or just raw input?
        # Typically we just upsample and conv.
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # Encoder
        x0 = self.layer0(x)     # H/2
        x_pool = self.layer0_pool(x0) 
        x1 = self.layer1(x_pool)# H/4
        x2 = self.layer2(x1)    # H/8
        x3 = self.layer3(x2)    # H/16
        x4 = self.layer4(x3)    # H/32
        
        # Decoder
        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)
        
        out = self.final_conv(d1)
        
        return out
