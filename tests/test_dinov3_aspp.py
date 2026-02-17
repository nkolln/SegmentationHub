
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.dinov3 import DinoV3Seg

def test_dinov3_aspp_forward():
    print("Initializing DinoV3Seg with head_type='aspp'...")
    # Using a smaller model or mock if possible, but DinoV3 requires loading from HF
    # utilizing the model_type default in the class or a known one
    # Note: This might download the model if not cached. 
    # The user was using 'facebook/dinov3-vitl16-pretrain-lvd1689m'.
    # I'll use the default which seems to be that one.
    
    num_classes = 12
    model = DinoV3Seg(num_classes=num_classes, head_type='aspp')
    model.eval()
    
    # Input size: (B, 3, H, W). DINOv3 usually expects multiple of 14.
    # 770 is used in config which is 55*14
    h, w = 770, 770
    x = torch.randn(1, 3, h, w)
    
    print(f"Running forward pass with input shape {x.shape}...")
    try:
        with torch.no_grad():
            output = model(x)
        print("Forward pass successful!")
        print(f"Output shape: {output.shape}")
        
        expected_shape = (1, num_classes, h, w)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print("Shape verification successful.")
        
    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_dinov3_aspp_forward()
