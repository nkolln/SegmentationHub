import torch
import sys

def check_gpu():
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("✅ CUDA is available")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
        # Test allocation
        try:
            x = torch.rand(1000, 1000).cuda()
            print("✅ Tensor allocation on GPU successful")
            print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        except Exception as e:
            print(f"❌ Tensor allocation failed: {e}")
    else:
        print("❌ CUDA is NOT available. Training will run on CPU.")

if __name__ == "__main__":
    check_gpu()
