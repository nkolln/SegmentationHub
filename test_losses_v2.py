import torch
import torch.nn as nn
from src.training.losses import CombinedLoss

def test_combined_loss():
    num_classes = 4
    batch_size = 2
    h, w = 32, 32
    
    logits = torch.randn(batch_size, num_classes, h, w)
    targets = torch.randint(0, num_classes, (batch_size, h, w))
    
    # Test 1: CE + Dice
    config1 = {
        'cross_entropy': 1.0,
        'dice': 0.5
    }
    loss_fn1 = CombinedLoss(config1, num_classes=num_classes)
    loss1 = loss_fn1(logits, targets)
    print(f"Test 1 (CE+Dice) Loss: {loss1.item():.4f}")
    assert loss1 > 0
    
    # Test 2: Focal + Boundary
    config2 = {
        'focal': {'weight': 1.0, 'alpha': 0.25, 'gamma': 2.0},
        'boundary': 0.1
    }
    loss_fn2 = CombinedLoss(config2, num_classes=num_classes)
    loss2 = loss_fn2(logits, targets)
    print(f"Test 2 (Focal+Boundary) Loss: {loss2.item():.4f}")
    assert loss2 > 0
    
    # Test 3: All active
    config3 = {
        'cross_entropy': 1.0,
        'dice': 1.0,
        'focal': {'weight': 1.0},
        'boundary': 1.0
    }
    loss_fn3 = CombinedLoss(config3, num_classes=num_classes)
    loss3 = loss_fn3(logits, targets)
    print(f"Test 3 (All) Loss: {loss3.item():.4f}")
    assert loss3 > 0
    
    # Test 4: Default fallback
    config4 = {}
    loss_fn4 = CombinedLoss(config4, num_classes=num_classes)
    loss4 = loss_fn4(logits, targets)
    print(f"Test 4 (Default) Loss: {loss4.item():.4f}")
    assert loss4 > 0
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_combined_loss()
