import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Dict, Tuple, Optional
from .hdd_dataloader import HDDDepthFusionDataset, create_dataloader
import random

class HDDDepthFusionAugmentedDataset(Dataset):
    """
    Augmented Dataset class for HDD Depth Completion Multi-Modal Fusion
    Applies albumentations-based augmentation to grayscale images and geometric transforms to all modalities
    """
    def __init__(self, 
                 dataset_root: str,
                 base_dataset: HDDDepthFusionDataset = None,
                 augment_probability: float = 0.8,
                 geometric_augment_probability: float = 0.7,
                 custom_probabilities: Dict[str, float] = None,
                 train_on_shadow_regions: bool = False):
        """
        Args:
            dataset_root: Path to dataset root directory
            base_dataset: Pre-initialized base dataset (optional)
            augment_probability: Overall probability of applying any augmentation
            geometric_augment_probability: Probability of applying geometric transforms
            custom_probabilities: Custom probabilities for individual transforms
            train_on_shadow_regions: Whether to train on shadow regions
        """
        self.dataset_root = dataset_root
        self.augment_probability = augment_probability
        self.geometric_augment_probability = geometric_augment_probability
        
        # Initialize base dataset if not provided
        if base_dataset is None:
            self.base_dataset = HDDDepthFusionDataset(dataset_root, train_on_shadow_regions=train_on_shadow_regions)
        else:
            self.base_dataset = base_dataset
        
        # Set up augmentation probabilities based on typical GitHub CV codebases
        self.probabilities = {
            # Image-only transforms (grayscale)
            "clahe": 0.1,
            "color_jitter": 0.1,
            "random_gamma": 0.1,
            "gaussian_blur": 0.1,
            "defocus": 0.05,
            "horizontal_flip": 0.5,
            "vertical_flip": 0.5    
        }
        
        # Override with custom probabilities if provided
        if custom_probabilities:
            self.probabilities.update(custom_probabilities)
        
        # Initialize augmentation pipelines
        self._setup_augmentation_pipelines()
        
        print(f"Augmented dataset initialized with {len(self.base_dataset)} samples")
        print(f"Augmentation probabilities: {self.probabilities}")
    
    def _setup_augmentation_pipelines(self):
        """Setup albumentations pipelines for different augmentation types"""
        
        # Image-only transforms for grayscale images
        # Note: For grayscale images, we only use brightness and contrast from ColorJitter
        self.image_only_transforms = A.Compose([
            A.CLAHE(
                clip_limit=3.0,
                tile_grid_size=(8, 8),
                p=self.probabilities['clahe']
            ),
            A.ColorJitter(
                brightness=(0.8, 1.2),
                contrast=(0.8, 1.2),
                saturation=0,  # No saturation for grayscale
                hue=0,         # No hue for grayscale
                p=self.probabilities['color_jitter']
            ),
            A.RandomGamma(
                gamma_limit=(80, 125),
                p=self.probabilities['random_gamma']
            ),
            A.GaussianBlur(
                blur_limit=0,
                sigma_limit=(0.5, 1.0),
                p=self.probabilities['gaussian_blur']
            ),
            A.Defocus(
                radius=(1, 1.5),
                alias_blur=(0.1, 0.5),
                p=self.probabilities['defocus']
            ),
        ], p=1.0)  # Always apply the compose, individual transforms have their own probabilities
        
        # Geometric transforms for all modalities (applied consistently)
        self.geometric_transforms = A.ReplayCompose([
            A.HorizontalFlip(p=self.probabilities['horizontal_flip']),
            A.VerticalFlip(p=self.probabilities['vertical_flip']),
        ], p=1.0)
    
    def _apply_geometric_transforms_to_all(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply geometric transforms consistently to all spatial modalities
        
        Args:
            sample: Dictionary containing all modalities as tensors
            
        Returns:
            Augmented sample with consistent geometric transforms
        """
        # Extract spatial tensors (excluding sample_name)
        spatial_keys = [
            'sparse_mask', 'sparse_depth', 'grayscale', 'depth_anything_v2',
            'gt_depth', 'gt_depth_shadowmasked', 'shadow_mask', 'background_mask'
        ]
        
        # Convert tensors to numpy arrays for albumentations (HWC format)
        numpy_arrays = {}
        for key in spatial_keys:
            if key in sample:
                # Convert from CHW to HWC
                tensor = sample[key]
                if tensor.dim() == 3 and tensor.shape[0] == 1:
                    numpy_array = tensor.squeeze(0).numpy()  # Remove channel dimension for single channel
                else:
                    numpy_array = tensor.permute(1, 2, 0).numpy()  # CHW -> HWC
                numpy_arrays[key] = numpy_array
        
        # Apply geometric transforms consistently
        if len(numpy_arrays) > 0:
            first_key = list(numpy_arrays.keys())[0]
            first_array = numpy_arrays[first_key]
            
            # Apply transform to get parameters and result
            transformed = self.geometric_transforms(image=first_array)
            
            # Get the replay parameters to apply same transform to all arrays
            replay_params = transformed.get('replay', None)
            
            # Apply same transform to all arrays using replay if available
            augmented_arrays = {}
            augmented_arrays[first_key] = transformed['image']
            
            for key, array in numpy_arrays.items():
                if key != first_key:
                    if replay_params is not None:
                        # Use replay to apply same transform
                        augmented = A.ReplayCompose.replay(replay_params, image=array)
                        augmented_arrays[key] = augmented['image']
                    #else:
                        # Fallback: apply transform directly (may not be identical)
                        #augmented = self.geometric_transforms(image=array)
                        #augmented_arrays[key] = augmented['image']
        
        # Convert back to tensors
        augmented_sample = sample.copy()
        for key, array in augmented_arrays.items():
            if len(array.shape) == 2:  # Single channel
                tensor = torch.from_numpy(array).unsqueeze(0).float()  # Add channel dimension
            else:  # Multi-channel
                tensor = torch.from_numpy(array).permute(2, 0, 1).float()  # HWC -> CHW
            augmented_sample[key] = tensor
        
        return augmented_sample
    
    def _apply_image_transforms_to_grayscale(self, grayscale_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply image-only transforms specifically to grayscale images
        
        Args:
            grayscale_tensor: Grayscale image tensor (1, H, W)
            
        Returns:
            Augmented grayscale tensor
        """
        # Convert to numpy array (HW format for single channel)
        grayscale_array = grayscale_tensor.squeeze(0).numpy()
        
        # Ensure values are in [0, 255] range for albumentations
        if grayscale_array.max() <= 1.0:
            grayscale_array = (grayscale_array * 255).astype(np.uint8)
        else:
            grayscale_array = grayscale_array.astype(np.uint8)
        
        # Apply image transforms
        augmented = self.image_only_transforms(image=grayscale_array)
        augmented_array = augmented['image']
        
        # Convert back to tensor and normalize to [0, 1]
        augmented_tensor = torch.from_numpy(augmented_array).unsqueeze(0).float() / 255.0
        
        return augmented_tensor
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get augmented sample from the dataset
        
        Returns:
            Dictionary containing all modalities with applied augmentations
        """
        # Get base sample
        sample = self.base_dataset[idx]
        
        # Decide whether to apply augmentation
        if random.random() > self.augment_probability:
            return sample  # Return original sample without augmentation
        
        # Apply image-only transforms to grayscale
        if 'grayscale' in sample:
            sample['grayscale'] = self._apply_image_transforms_to_grayscale(sample['grayscale'])
        
        # Apply geometric transforms to all spatial modalities
        if random.random() < self.geometric_augment_probability:
            sample = self._apply_geometric_transforms_to_all(sample)
        
        return sample
    
    def set_augmentation_probabilities(self, new_probabilities: Dict[str, float]):
        """Update augmentation probabilities and reinitialize pipelines"""
        self.probabilities.update(new_probabilities)
        self._setup_augmentation_pipelines()
        print(f"Updated augmentation probabilities: {self.probabilities}")
    
    def disable_augmentation(self):
        """Disable all augmentations (useful for validation/testing)"""
        self.augment_probability = 0.0
        print("Augmentation disabled")
    
    def enable_augmentation(self, probability: float = 0.8):
        """Re-enable augmentations with specified probability"""
        self.augment_probability = probability
        print(f"Augmentation enabled with probability {probability}")

def create_augmented_dataloader(dataset_root: str,
                               batch_size: int = 4,
                               split: str = 'train',
                               shuffle: bool = True,
                               num_workers: int = 4,
                               normalize_depth: bool = True,
                               depth_min: float = None,
                               depth_max: float = None,
                               augment_probability: float = 0.8,
                               geometric_augment_probability: float = 0.7,
                               custom_probabilities: Dict[str, float] = None,
                               train_on_shadow_regions: bool = False) -> DataLoader:
    """
    Create augmented DataLoader for HDD Depth Fusion Dataset
    
    Args:
        dataset_root: Path to dataset root directory
        batch_size: Batch size for DataLoader
        split: Dataset split ('train', 'val', 'test') - for future use
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
        normalize_depth: Whether to normalize depth values
        depth_min: Minimum depth value for normalization
        depth_max: Maximum depth value for normalization
        augment_probability: Overall probability of applying any augmentation
        geometric_augment_probability: Probability of applying geometric transforms
        custom_probabilities: Custom probabilities for individual transforms
        train_on_shadow_regions: Whether to train on shadow regions
    
    Returns:
        DataLoader instance with augmentation
    """
    # Create base training dataset
    base_dataset = HDDDepthFusionDataset(
        dataset_root=dataset_root,
        split=split,
        normalize_depth=normalize_depth,
        depth_min=depth_min,
        depth_max=depth_max,
        train_on_shadow_regions=train_on_shadow_regions
    )
    
    # Create augmented training dataset
    augmented_dataset = HDDDepthFusionAugmentedDataset(
        dataset_root=dataset_root,
        base_dataset=base_dataset,
        augment_probability=augment_probability,
        geometric_augment_probability=geometric_augment_probability,
        custom_probabilities=custom_probabilities,
        train_on_shadow_regions=train_on_shadow_regions
    )
    
    dataloader = DataLoader(
        augmented_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader

# Test and demonstration code
if __name__ == "__main__":
    print("Testing HDD Depth Fusion Data Augmentation...")
    
    dataset_path = '/home/badri/Hard Disk Project/neuralnets/depth_fusion/virtual_dataset_complete/'
    
    try:
        # Create base dataset to get depth statistics
        base_dataset = HDDDepthFusionDataset(dataset_path, normalize_depth=False)
        stats = base_dataset.get_depth_stats(num_samples=40)
        
        # Compute combined normalization parameters
        combined_min = min(stats['sparse_min'], stats['gt_min'])
        combined_max = max(stats['sparse_max'], stats['gt_max'])
        
        print(f"\nUsing depth normalization: [{combined_min:.6f}, {combined_max:.6f}]")
        
        # Create augmented dataloader
        augmented_dataloader = create_augmented_dataloader(
            dataset_path,
            batch_size=2,
            shuffle=False,
            num_workers=1,  # Use 1 worker for testing
            normalize_depth=True,
            depth_min=combined_min,
            depth_max=combined_max,
            augment_probability=1.0,  # Always augment for testing
            geometric_augment_probability=1.0,  # Always apply geometric transforms for testing
            train_on_shadow_regions=False # Pass the parameter
        )
        
        print("\nTesting augmented batch loading...")
        for batch_idx, batch in enumerate(augmented_dataloader):
            print(f"\nAugmented Batch {batch_idx + 1}:")
            print(f"  Batch size: {len(batch['sample_name'])}")
            print(f"  Sample names: {batch['sample_name']}")
            
            # Print tensor shapes and ranges
            for key, tensor in batch.items():
                if isinstance(tensor, torch.Tensor):
                    print(f"  {key}: {tensor.shape}, range: [{tensor.min():.3f}, {tensor.max():.3f}]")
            
            # Only test first batch
            break
        
        print("\nAugmented DataLoader test completed successfully!")
        
        # Test custom probabilities
        print("\nTesting custom augmentation probabilities...")
        custom_probs = {
            'clahe': 0.8,
            'horizontal_flip': 0.9,
            'vertical_flip': 0.1
        }
        
        custom_augmented_dataloader = create_augmented_dataloader(
            dataset_path,
            batch_size=1,
            custom_probabilities=custom_probs,
            normalize_depth=True,
            depth_min=combined_min,
            depth_max=combined_max,
            train_on_shadow_regions=False # Pass the parameter
        )
        
        print("Custom probabilities applied successfully!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()




CONSERVATIVE_CONFIG = {
    'clahe': 0.1,
    'color_jitter': 0.1,
    'random_gamma': 0.1,
    'gaussian_blur': 0.1,
    'defocus': 0.05,
    'horizontal_flip': 0.5,
    'vertical_flip': 0.5,
}


STANDARD_CONFIG = {
    'clahe': 0.25,
    'color_jitter': 0.30,
    'random_gamma': 0.20,
    'gaussian_blur': 0.15,
    'defocus': 0.10,
    'horizontal_flip': 0.30,
    'vertical_flip': 0.20,
}


AGGRESSIVE_CONFIG = {
    'clahe': 0.40,
    'color_jitter': 0.45,
    'random_gamma': 0.35,
    'gaussian_blur': 0.25,
    'defocus': 0.15,
    'horizontal_flip': 0.40,
    'vertical_flip': 0.30,
}