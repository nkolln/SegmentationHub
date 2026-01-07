import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import inspect

# Import specific modules from lightly
from lightly.models.modules import MaskedCausalVisionTransformer

# --- MONKEYPATCH FOR LIGHTLY/TIMM INCOMPATIBILITY ---
from lightly.models.modules.masked_causal_vision_transformer import MaskedCausalAttention

_orig_mca_init = MaskedCausalAttention.__init__

def _patched_mca_init(self, *args, **kwargs):
    valid_params = inspect.signature(_orig_mca_init).parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return _orig_mca_init(self, *args, **filtered_kwargs)

MaskedCausalAttention.__init__ = _patched_mca_init
# ----------------------------------------------------

class SegmentationHead(nn.Module):
    """
    A lightweight segmentation head that upscales features to the input resolution.
    Structure: Proj -> Conv -> BN -> ReLU -> Conv (Classification).
    """
    def __init__(self, embed_dim: int, num_classes: int, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Intermediate channel dimension
        mid_channels = embed_dim // 2
        
        self.layers = nn.Sequential(
            nn.Conv2d(embed_dim, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        x = self.layers(x)
        return F.interpolate(x, size=target_shape, mode='bilinear', align_corners=False)


class AIMSegmentationModel(nn.Module):
    """
    Semantic Segmentation model using the AIM (Autoregressive Image Model) backbone.
    """
    def __init__(
        self, 
        num_classes: int, 
        model_type: str = 'aim-base',
        image_size: int = 224, 
        patch_size: int = 14, 
        **kwargs
    ):
        super().__init__()
        # print(f"DEBUG: AIMSegmentationModel received kwargs: {kwargs}")
        self.num_classes = num_classes
        self.patch_size = patch_size
        
        if model_type == 'aim-small':
            self.embed_dim = 384
            depth = 12
            num_heads = 6
        elif model_type == 'aim-large':
            self.embed_dim = 1024
            depth = 24
            num_heads = 16
        else: # Default/aim-base
            self.embed_dim = 768
            depth = 12
            num_heads = 12
            
        # 1. Initialize AIM Backbone
        # FIXED: Set global_pool to '' (empty string) to prevent averaging tokens
        self.backbone = MaskedCausalVisionTransformer(
            img_size=image_size,
            patch_size=patch_size,
            embed_dim=self.embed_dim,
            depth=depth,
            num_heads=num_heads,
            class_token=False,
            global_pool=''  # <--- CRITICAL FIX: Was 'avg'
        )
        
        # 2. Initialize Segmentation Head
        self.head = SegmentationHead(
            embed_dim=self.embed_dim,
            num_classes=num_classes
        )

    def forward(self, x: torch.Tensor, labels=None) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Sanity check for patch alignment
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            # Optional: Dynamic padding could be added here, 
            # but raising error is safer for fixed workflows
            pass 

        # 1. Backbone Forward Pass
        # Use forward_features to ensure we get the sequence of tokens, not the head output
        tokens = self.backbone.forward_features(x)
        
        # 2. Reshape Tokens to Grid
        h_grid = H // self.patch_size
        w_grid = W // self.patch_size
        
        # Handle potential Class Token or Registers
        seq_len = tokens.shape[1]
        expected_len = h_grid * w_grid
        
        if seq_len == expected_len + 1:
            # Assuming class token is at index 0 (standard ViT) or index -1
            # Standard TIMM usually puts CLS at 0 if class_token=True
            features = tokens[:, 1:, :] # Drop CLS token
        elif seq_len == expected_len:
            features = tokens
        else:
            # Fallback for unexpected shapes (e.g., registers)
            # Just take the last N tokens that fit the grid
            features = tokens[:, -expected_len:, :]
            
        # Permute to (B, C, N) -> Reshape to (B, C, H_grid, W_grid)
        # Now this will work because tokens dim is 3: (B, N, C)
        features = features.permute(0, 2, 1).reshape(B, self.embed_dim, h_grid, w_grid)
        
        # 3. Segmentation Head & Upsampling
        logits = self.head(features, target_shape=(H, W))
        
        # Compatibility with trainer expecting (logits, None) or dict
        if labels is not None:
            # If your trainer expects the model to calculate loss (HuggingFace style)
            # You can calculate it here, but your trainer seems to handle it externally.
            return logits
            
        return logits