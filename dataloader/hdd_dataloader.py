import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
from pathlib import Path
import glob
from typing import Dict, Tuple, List, Optional
import torchvision.transforms as transforms

class HDDDepthFusionDataset(Dataset):
    """
    Dataset class for HDD Depth Completion Multi-Modal Fusion
    Loads sparse depth, grayscale, depth_anything_v2, ground truth, and various masks
    """
    def __init__(self, 
                 dataset_root: str,
                 split: str = 'train',
                 normalize_depth: bool = True,
                 depth_min: float = None,
                 depth_max: float = None,
                 train_on_shadow_regions: bool = False):
        """
        Args:
            dataset_root: Path to dataset root directory
            split: Dataset split ('train', 'val', 'test') - for future use
            normalize_depth: Whether to normalize depth values to [0,1]
            depth_min: Minimum depth value for normalization (if None, computed from data)
            depth_max: Maximum depth value for normalization (if None, computed from data)
            train_on_shadow_regions: Whether to train on shadow regions (if False, applies shadow mask to sparse_mask)
        """
        self.split = split
        if split in ['valid', 'test']:
            self.dataset_root = Path(dataset_root) / split
        else:
            self.dataset_root = Path(dataset_root)  # 'train' uses parent directory
        self.normalize_depth = normalize_depth
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.train_on_shadow_regions = train_on_shadow_regions
        
        # Define subfolder paths
        self.sparse_mask_dir = self.dataset_root / 'sparse_mask'
        self.sparse_depth_dir = self.dataset_root / 'sparse_depth_z'
        self.grayscale_dir = self.dataset_root / 'grayscale'
        self.depth_anything_dir = self.dataset_root / 'depth_anything_v2_map_1512'
        self.gt_depth_dir = self.dataset_root / 'gt_depth'
        self.shadow_mask_dir = self.dataset_root / 'shadow_mask_gt'
        self.background_mask_dir = self.dataset_root / 'background_mask_gt'
        
        # Verify all directories exist
        self._verify_directories()
        
        # Get list of sample names (assuming consistent naming across all modalities)
        self.sample_names = self._get_sample_names()
        
        # Center crop transform (514x544 -> 512x512)
        self.center_crop = transforms.CenterCrop((512, 512))
        
        print(f"Loaded {len(self.sample_names)} samples from {dataset_root}")
        print(f"First few samples: {self.sample_names[:3]}")
    
    def _verify_directories(self):
        """Verify all required directories exist"""
        dirs_to_check = [
            self.sparse_mask_dir, self.sparse_depth_dir, self.grayscale_dir,
            self.depth_anything_dir, self.gt_depth_dir, self.shadow_mask_dir,
            self.background_mask_dir
        ]
        
        for dir_path in dirs_to_check:
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        print("All required directories verified.")
    
    def _get_sample_names(self) -> List[str]:
        """Get list of sample names from one of the directories (using sparse_mask as reference)"""
        # Get all PNG files from sparse_mask directory
        mask_files = list(self.sparse_mask_dir.glob('*.png'))
        sample_names = [f.stem for f in mask_files]  # Remove .png extension
        sample_names.sort()  # Ensure consistent ordering
        
        if len(sample_names) == 0:
            raise ValueError(f"No PNG files found in {self.sparse_mask_dir}")
        
        return sample_names
    
    def _load_png_image(self, file_path: Path, normalize: bool = True) -> torch.Tensor:
        """Load PNG image and convert to tensor"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        image = Image.open(file_path)
        # Convert to numpy array
        image_array = np.array(image)
        
        # Convert to tensor and add channel dimension if needed
        if len(image_array.shape) == 2:  # Grayscale
            tensor = torch.from_numpy(image_array).unsqueeze(0).float()
        else:  # RGB
            tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        
        # Normalize to [0, 1] if requested
        if normalize:
            tensor = tensor / 255.0
        
        # Center crop to 512x512
        tensor = self.center_crop(tensor)
        
        return tensor
    
    def _load_csv_depth(self, file_path: Path) -> torch.Tensor:
        """Load depth data from CSV file"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load CSV without headers
        depth_data = pd.read_csv(file_path, header=None).values
        
        # Convert to tensor and add channel dimension
        tensor = torch.from_numpy(depth_data).unsqueeze(0).float()
        
        # Center crop to 512x512
        tensor = self.center_crop(tensor)
        
        return tensor
    
    def _normalize_depth_tensor(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize depth tensor to [0, 1] range"""
        if self.depth_min is None or self.depth_max is None:
            # If normalization parameters not provided, use tensor min/max
            min_val = depth_tensor.min()
            max_val = depth_tensor.max()
        else:
            min_val = self.depth_min
            max_val = self.depth_max
        
        # Avoid division by zero
        if max_val == min_val:
            return torch.zeros_like(depth_tensor)
        
        normalized = (depth_tensor - min_val) / (max_val - min_val)
        return normalized
    
    def __len__(self) -> int:
        return len(self.sample_names)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset
        
        Returns:
            Dictionary containing all modalities and masks as tensors
        """
        sample_name = self.sample_names[idx]
        
        try:
            # 1. Load sparse mask (PNG -> divide by 255 -> center crop -> 512x512x1)
            sparse_mask_path = self.sparse_mask_dir / f"{sample_name}.png"
            sparse_mask = self._load_png_image(sparse_mask_path, normalize=True)
            sparse_mask = 1.0 - sparse_mask
            
            # 2. Load sparse depth (CSV -> center crop -> 512x512x1)
            sparse_depth_path = self.sparse_depth_dir / f"{sample_name}.csv"
            sparse_depth = self._load_csv_depth(sparse_depth_path)
            if self.normalize_depth:
                sparse_depth = self._normalize_depth_tensor(sparse_depth)
            
            # 3. Load grayscale image (PNG -> divide by 255 -> center crop -> 512x512x1)
            grayscale_path = self.grayscale_dir / f"{sample_name}.png"
            grayscale = self._load_png_image(grayscale_path, normalize=True)
            
            # 4. Load depth_anything_v2 (PNG -> divide by 255 -> first channel -> invert -> center crop -> 512x512x1)
            depth_anything_path = self.depth_anything_dir / f"{sample_name}.png"
            depth_anything = self._load_png_image(depth_anything_path, normalize=True)
            depth_anything = depth_anything[0:1, :, :]  # Take first channel only
            depth_anything = 1.0 - depth_anything  # Invert: tensor = 1 - tensor
            
            # 5. Load ground truth dense depth (CSV -> center crop -> 512x512x1)
            gt_depth_path = self.gt_depth_dir / f"{sample_name}.csv"
            gt_depth = self._load_csv_depth(gt_depth_path)
            if self.normalize_depth:
                gt_depth = self._normalize_depth_tensor(gt_depth)
            
            # 6. Load shadow mask (PNG -> divide by 255 -> center crop -> 512x512x1)
            shadow_mask_path = self.shadow_mask_dir / f"{sample_name}.png"
            shadow_mask = self._load_png_image(shadow_mask_path, normalize=True)
            
            # 7. Load background mask (PNG -> divide by 255 -> center crop -> 512x512x1)
            background_mask_path = self.background_mask_dir / f"{sample_name}.png"
            background_mask = self._load_png_image(background_mask_path, normalize=True)
            
            # 8. Create shadow-masked ground truth: gt_depth_shadowmasked = gt_depth * shadow_mask
            gt_depth_shadowmasked = gt_depth * shadow_mask

            #Newly added on 08/142025 - only apply if not training on shadow regions
            if not self.train_on_shadow_regions:
                sparse_mask = sparse_mask * shadow_mask
            
            # Return dictionary with all modalities
            sample = {
                'sample_name': sample_name,
                'sparse_mask': sparse_mask,
                'sparse_depth': sparse_depth,
                'grayscale': grayscale,
                'depth_anything_v2': depth_anything,
                'gt_depth': gt_depth,
                'gt_depth_shadowmasked': gt_depth_shadowmasked,
                'shadow_mask': shadow_mask,
                'background_mask': background_mask
            }
            
            return sample
            
        except Exception as e:
            print(f"Error loading sample {sample_name}: {str(e)}")
            raise e
    
    def get_depth_stats(self, num_samples: int = 40) -> Dict[str, float]:
        """
        Compute depth statistics from a subset of the dataset
        Useful for determining normalization parameters
        """
        sparse_depths = []
        gt_depths = []
        
        num_samples = min(num_samples, len(self.sample_names))
        
        print(f"Computing depth statistics from {num_samples} samples...")
        
        for i in range(num_samples):
            sample_name = self.sample_names[i]
            
            # Load sparse depth
            sparse_depth_path = self.sparse_depth_dir / f"{sample_name}.csv"
            sparse_depth = self._load_csv_depth(sparse_depth_path)
            # Only consider non-zero values (valid depth measurements)
            valid_sparse = sparse_depth[sparse_depth > 0]
            if len(valid_sparse) > 0:
                sparse_depths.extend(valid_sparse.flatten().tolist())
            
            # Load gt depth
            gt_depth_path = self.gt_depth_dir / f"{sample_name}.csv"
            gt_depth = self._load_csv_depth(gt_depth_path)
            # Only consider non-zero values (valid depth measurements)
            valid_gt = gt_depth[gt_depth > 0]
            if len(valid_gt) > 0:
                gt_depths.extend(valid_gt.flatten().tolist())

            # gt_depths.extend(gt_depth.flatten().tolist())
        
        sparse_depths = np.array(sparse_depths)
        gt_depths = np.array(gt_depths)
        
        stats = {
            'sparse_min': float(sparse_depths.min()) if len(sparse_depths) > 0 else 0.0,
            'sparse_max': float(sparse_depths.max()) if len(sparse_depths) > 0 else 1.0,
            'sparse_mean': float(sparse_depths.mean()) if len(sparse_depths) > 0 else 0.5,
            'sparse_std': float(sparse_depths.std()) if len(sparse_depths) > 0 else 0.1,
            'gt_min': float(gt_depths.min()),
            'gt_max': float(gt_depths.max()),
            'gt_mean': float(gt_depths.mean()),
            'gt_std': float(gt_depths.std()),
        }
        
        print("Depth Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.6f}")
        
        return stats

def create_dataloader(dataset_root: str,
                     split: str = 'train',
                     batch_size: int = 4,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     normalize_depth: bool = True,
                     depth_min: float = None,
                     depth_max: float = None,
                     train_on_shadow_regions: bool = False) -> DataLoader:
    """
    Create DataLoader for HDD Depth Fusion Dataset
    
    Args:
        dataset_root: Path to dataset root directory
        split: Dataset split ('train', 'val', 'test') - for future use
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
        normalize_depth: Whether to normalize depth values
        depth_min: Minimum depth value for normalization
        depth_max: Maximum depth value for normalization
        train_on_shadow_regions: Whether to train on shadow regions
    
    Returns:
        DataLoader instance
    """
    dataset = HDDDepthFusionDataset(
        dataset_root=dataset_root,
        split=split,
        normalize_depth=normalize_depth,
        depth_min=depth_min,
        depth_max=depth_max,
        train_on_shadow_regions=train_on_shadow_regions
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        drop_last=False   # Keep all samples
    )
    
    return dataloader

# Test and demonstration code
if __name__ == "__main__":
    # Test the dataloader
    dataset_path = '/home/badri/Hard Disk Project/neuralnets/depth_fusion/virtual_dataset_complete/'
    #dataset_path = '/home/badri/Hard Disk Project/neuralnets/depth_fusion/real_dataset_complete/'
    print("Testing HDD Depth Fusion DataLoader...")
    
    try:
        # Create dataset instance
        dataset = HDDDepthFusionDataset(dataset_path, normalize_depth=False)
        
        # Get depth statistics for normalization
        stats = dataset.get_depth_stats(num_samples=40)
        
        # Create dataset with normalization using computed stats
        # Using combined min/max from both sparse and gt depths
        combined_min_stats = min(stats['sparse_min'], stats['gt_min'])
        combined_max_stats = max(stats['sparse_max'], stats['gt_max'])

        combined_min=combined_min_stats
        combined_max=combined_max_stats
        #set combined_min and combined_max manually
        combined_min_manual_real = 0.420008
        combined_max_manual_real = 0.468000

        combined_min_manual_virtual = 0.410288
        combined_max_manual_virtual = 0.478991

        combined_min=0.410288
        combined_max=0.478991


        dataset_normalized = HDDDepthFusionDataset(
            dataset_path, 
            normalize_depth=False,
            depth_min=combined_min,
            depth_max=combined_max,
            train_on_shadow_regions=False
        )
        
        # Create dataloader
        dataloader = create_dataloader(
            dataset_path,
            batch_size=2,
            shuffle=False,
            num_workers=2,
            normalize_depth=False,
            depth_min=combined_min,
            depth_max=combined_max, 
            train_on_shadow_regions=False
        )
        
        # Test loading a batch
        print("\nTesting batch loading...")
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Batch size: {len(batch['sample_name'])}")
            print(f"  Sample names: {batch['sample_name']}")
            
            # Print tensor shapes and ranges
            for key, tensor in batch.items():
                if isinstance(tensor, torch.Tensor):
                    print(f"  {key}: {tensor.shape}, range: [{tensor.min():.3f}, {tensor.max():.3f}]")
            
            # Only test first batch
            break
        
        print("\nDataLoader test completed successfully!")
        
        # Test individual sample access
        print(f"\nTesting individual sample access...")
        sample = dataset_normalized[0]
        print(f"Sample name: {sample['sample_name']}")
        for key, tensor in sample.items():
            if isinstance(tensor, torch.Tensor):
                print(f"  {key}: {tensor.shape}, range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()