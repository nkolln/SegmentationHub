import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    Basic U-Net implementation.
    """
    def __init__(self, n_channels=3, n_classes=5):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Placeholder layers
        self.inc = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.inc(x)
        # ... full implementation later
        return self.outc(x)
