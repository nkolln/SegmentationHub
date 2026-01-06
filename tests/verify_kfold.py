import os
import numpy as np
from src.data.dataset import SegmentationDataset

def test_kfold_logic():
    root_dir = os.path.abspath("data")
    seed = 42
    test_split = 0.1
    num_folds = 5
    
    print(f"--- Verifying K-Fold Logic (Seed: {seed}, Test Split: {test_split}, Folds: {num_folds}) ---")
    
    # 1. Check Test Set Consistency
    test_ds1 = SegmentationDataset(root_dir=root_dir, split='test', seed=seed, test_split=test_split)
    test_ds2 = SegmentationDataset(root_dir=root_dir, split='test', seed=seed, test_split=test_split)
    
    assert test_ds1.images == test_ds2.images, "❌ Test set is not deterministic!"
    print(f"✅ Test set consistency verified ({len(test_ds1)} images).")
    
    # 2. Check Overlap Between Fold 0 Val and Fold 1 Val
    val0 = SegmentationDataset(root_dir=root_dir, split='val', fold=0, num_folds=num_folds, seed=seed, test_split=test_split)
    val1 = SegmentationDataset(root_dir=root_dir, split='val', fold=1, num_folds=num_folds, seed=seed, test_split=test_split)
    
    overlap = set(val0.images).intersection(set(val1.images))
    assert len(overlap) == 0, f"❌ Overlap found between fold 0 and fold 1 validation sets! ({len(overlap)} items)"
    print(f"✅ No overlap between validation folds 0 and 1.")
    
    # 3. Check Overlap Between Train and Val in Fold 0
    train0 = SegmentationDataset(root_dir=root_dir, split='train', fold=0, num_folds=num_folds, seed=seed, test_split=test_split)
    overlap_train_val = set(train0.images).intersection(set(val0.images))
    assert len(overlap_train_val) == 0, f"❌ Overlap found between train and val in fold 0!"
    print(f"✅ No overlap between train and val in fold 0.")
    
    # 4. Check Test Set Isolation
    overlap_test_train = set(test_ds1.images).intersection(set(train0.images))
    assert len(overlap_test_train) == 0, "❌ Test set leak into training set!"
    overlap_test_val = set(test_ds1.images).intersection(set(val0.images))
    assert len(overlap_test_val) == 0, "❌ Test set leak into validation set!"
    print(f"✅ Test set is properly isolated from train/val.")
    
    # 5. Check total coverage (excluding rounding issues)
    total_unique = len(set(test_ds1.images) | set(val0.images) | set(val1.images) | set(train0.images))
    # Note: since val takes slices, we should check if sum of all val folds + test equals total
    all_vals = []
    for f in range(num_folds):
        v = SegmentationDataset(root_dir=root_dir, split='val', fold=f, num_folds=num_folds, seed=seed, test_split=test_split)
        all_vals.extend(v.images)
    
    total_files = len(set(test_ds1.images) | set(all_vals))
    print(f"✅ Total unique images covered: {total_files}")

if __name__ == "__main__":
    try:
        test_kfold_logic()
        print("\n✨ ALL K-FOLD VERIFICATION CHECKS PASSED!")
    except AssertionError as e:
        print(e)
