
import unittest
import os
import shutil
import numpy as np
from src.data.dataset import SegmentationDataset

class TestDatasetLoading(unittest.TestCase):
    def test_loading(self):
        root_dir = os.path.abspath("data")
        dataset = SegmentationDataset(root_dir=root_dir, split='train')
        print(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            img, mask = dataset[0]
            print(f"Image shape: {img.shape}")
            print(f"Mask shape: {mask.shape}")
            print(f"Mask unique values: {np.unique(mask)}")
            
            # Basic checks
            self.assertEqual(len(img.shape), 3) # H, W, 3 (since we load RGB)
            # self.assertEqual(len(mask.shape), 2) # H, W (Mask is usually single channel 2D array from PIL)
            # Actually, if albumentations is not passed, it returns numpy array.
            
            # Check min/max values of mask (should be 0-11)
            self.assertTrue(np.min(mask) >= 0)
            self.assertTrue(np.max(mask) <= 11)

if __name__ == '__main__':
    unittest.main()
