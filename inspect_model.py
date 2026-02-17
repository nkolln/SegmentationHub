
import torch
from transformers import AutoConfig, AutoModel

model_name = "facebook/dinov3-convnext-base-pretrain-lvd1689m"
# Fallback to a known public convnext if the specific one is private/doesn't exist for testing logic, 
# but user implies this one exists. We will try to load it. 
# If it fails, we might need to assume it behaves like standard convnext.

try:
    print(f"Loading config for {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    print("\nConfig attributes:")
    for key, val in config.__dict__.items():
        if not key.startswith('_'):
            print(f"{key}: {val}")

    print(f"\nHas patch_size? {'patch_size' in dir(config)}")
    print(f"Has hidden_size? {'hidden_size' in dir(config)}")
    print(f"Hidden sizes: {getattr(config, 'hidden_sizes', 'N/A')}")
    print(f"Depths: {getattr(config, 'depths', 'N/A')}")
    
    print("\nLoading model...")
    model = AutoModel.from_pretrained(model_name)
    
    dummy_input = torch.randn(1, 3, 770, 770) # size from user config
    print(f"\nRunning forward pass with input {dummy_input.shape}...")
    
    outputs = model(dummy_input, output_hidden_states=True)
    
    print("\nOutput keys:", outputs.keys())
    if hasattr(outputs, 'hidden_states'):
        print(f"Number of hidden states: {len(outputs.hidden_states)}")
        for i, hs in enumerate(outputs.hidden_states):
            print(f"Hidden state {i} shape: {hs.shape}")
            
except Exception as e:
    print(f"\nError: {e}")
