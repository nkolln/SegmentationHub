import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ model using segmentation_models_pytorch.
    
    This is often better for building/facade segmentation due to:
    - Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context
    - Strong encoder-decoder with skip connections
    """
    def __init__(self, num_classes, encoder_name="resnet101", pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
        )
        
    def forward(self, x):
        return self.model(x)


class UNetPlusPlus(nn.Module):
    """
    UNet++ (Nested UNet) - another strong alternative.
    Often better for fine-grained segmentation than regular UNet.
    """
    def __init__(self, num_classes, encoder_name="efficientnet-b4", pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
        )
        
    def forward(self, x):
        return self.model(x)
