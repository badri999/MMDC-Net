import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import time
import argparse
import json
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
from typing import Dict, Tuple, Optional, List
import cv2
import matplotlib
matplotlib.use('Agg')  # Force non-Qt backend
import matplotlib.pyplot as plt
import pandas as pd
import random

# Add parent directory to Python path to import modules from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your custom modules
from circularkernel_2_5 import (
    SparseDepthBranch, GrayscaleBranch, RelativeDepthBranch, FusionBranch
)
from emergency_weight_fix import emergency_fix_all_circular_weights
from dataloader.hdd_dataloader import HDDDepthFusionDataset, create_dataloader
from dataloader.hdd_data_augmenter import create_augmented_dataloader, CONSERVATIVE_CONFIG
from compute_loss import (
    l1_loss_with_mask, l2_loss_with_mask, 
    ssim_loss_with_mask, laplacian_loss_with_mask
)

class HDDDepthFusionModel(nn.Module):
    """Complete Multi-Modal Fusion Model for HDD Depth Completion"""
    def __init__(self, base_channels=64, dropout_prob=0.0):
        super(HDDDepthFusionModel, self).__init__()
        
        # Initialize all four branches
        self.sparse_branch = SparseDepthBranch(base_channels=base_channels)
        self.grayscale_branch = GrayscaleBranch(base_channels=base_channels)
        self.relative_depth_branch = RelativeDepthBranch(base_channels=base_channels)
        self.fusion_branch = FusionBranch(base_channels=base_channels, dropout_prob=dropout_prob)
    
    def forward(self, sparse_input, grayscale_input, relative_depth_input):
        """
        Forward pass through the complete multi-modal fusion network
        
        Args:
            sparse_input: Sparse depth input (B, 1, 512, 512)
            grayscale_input: Grayscale image input (B, 1, 512, 512)
            relative_depth_input: Relative depth input (B, 1, 512, 512)
        
        Returns:
            Dense depth prediction (B, 1, 512, 512)
        """
        # Forward pass through parallel branches
        sparse_output, sparse_features = self.sparse_branch(sparse_input)
        grayscale_output, grayscale_features = self.grayscale_branch(grayscale_input)
        relative_output, relative_features = self.relative_depth_branch(relative_depth_input)
        
        # Forward pass through fusion branch
        final_output = self.fusion_branch(
            sparse_input, sparse_output, grayscale_output, relative_output,
            sparse_features, grayscale_features, relative_features
        )
        
        return final_output
    
    def set_dropout_prob(self, prob):
        """Update dropout probability for fine-tuning"""
        self.fusion_branch.set_dropout_prob(prob)
    
    def freeze_branches(self, freeze_sparse=False, freeze_grayscale=False, freeze_relative=False):
        """Freeze specified parallel branches"""
        if freeze_sparse:
            for param in self.sparse_branch.parameters():
                param.requires_grad = False
        
        if freeze_grayscale:
            for param in self.grayscale_branch.parameters():
                param.requires_grad = False
        
        if freeze_relative:
            for param in self.relative_depth_branch.parameters():
                param.requires_grad = False

class CombinedDepthLoss(nn.Module):
    """Combined loss function using multiple depth estimation losses"""
    def __init__(self, 
                 l1_active=True, 
                 l1_shadow_weight=1.0,
                 l1_sparse_weight=1.0,
                 l2_active=False, 
                 l2_shadow_weight=1.0,
                 l2_sparse_weight=1.0,
                 ssim_active=False, 
                 ssim_shadow_weight=1.0,
                 ssim_sparse_weight=1.0,
                 laplacian_active=False,
                 laplacian_shadow_weight=1.0,
                 laplacian_sparse_weight=1.0):
        super(CombinedDepthLoss, self).__init__()
        self.l1_active = l1_active
        self.l2_active = l2_active
        self.ssim_active = ssim_active
        self.laplacian_active = laplacian_active
        
        # Store weight parameters as instance attributes
        self.l1_shadow_weight = l1_shadow_weight
        self.l1_sparse_weight = l1_sparse_weight
        self.l2_shadow_weight = l2_shadow_weight
        self.l2_sparse_weight = l2_sparse_weight
        self.ssim_shadow_weight = ssim_shadow_weight
        self.ssim_sparse_weight = ssim_sparse_weight
        self.laplacian_shadow_weight = laplacian_shadow_weight
        self.laplacian_sparse_weight = laplacian_sparse_weight
        
        # Store which losses are active (non-zero weight)
        self.active_losses = []
        # Store the weights for each active loss
        self.active_loss_weights = []
        if l1_active == True:
            self.active_losses.append('l1')
            self.active_loss_weights.append(l1_shadow_weight)
            self.active_loss_weights.append(l1_sparse_weight)
        if l2_active == True:
            self.active_losses.append('l2')
            self.active_loss_weights.append(l2_shadow_weight)
            self.active_loss_weights.append(l2_sparse_weight)
        if ssim_active == True:
            self.active_losses.append('ssim')
            self.active_loss_weights.append(ssim_shadow_weight)
            self.active_loss_weights.append(ssim_sparse_weight)
        if laplacian_active == True:
            self.active_losses.append('laplacian')
            self.active_loss_weights.append(laplacian_shadow_weight)
            self.active_loss_weights.append(laplacian_sparse_weight)
        
        print(f"Active losses: {self.active_losses}")
        print(f"Loss weights: {self.active_loss_weights}")
    
    def forward(self, pred, target, shadow_mask=None,sparse_mask=None):
        """
        Compute combined loss
        
        Args:
            pred: Predicted depth (B, 1, H, W) - values in [0, 1] range
            target: Ground truth depth (B, 1, H, W) - values in [0, 1] range
            mask: Optional mask for valid pixels (B, 1, H, W) - binary mask
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        total_loss = 0.0
        
        # L1 Loss
        if self.l1_active == True:
            l1_loss_shadow = l1_loss_with_mask(pred, target, shadow_mask)
            l1_loss_sparse = l1_loss_with_mask(pred, target, sparse_mask)
            #l1_loss_background= l1_loss_with_mask(pred, target, background_mask)
            loss_dict['l1_loss_excluding_shadow'] = l1_loss_shadow.item()
            loss_dict['l1_loss_for_sparse'] = l1_loss_sparse.item()
            loss_dict['l1_loss'] = self.l1_shadow_weight * l1_loss_shadow.item() + self.l1_sparse_weight * l1_loss_sparse.item()
            #total_loss += self.l1_weight * l1_loss
            total_loss += self.l1_shadow_weight * l1_loss_shadow + self.l1_sparse_weight * l1_loss_sparse
        else:
            loss_dict['l1_loss_excluding_shadow'] = 0.0
            loss_dict['l1_loss_for_sparse'] = 0.0
            loss_dict['l1_loss'] = 0.0
        
        # L2 Loss  
        if self.l2_active == True:
            l2_loss_shadow  = l2_loss_with_mask(pred, target, shadow_mask)
            l2_loss_sparse = l2_loss_with_mask(pred, target, sparse_mask)
            #l2_loss_background= l2_loss_with_mask(pred, target, background_mask)
            loss_dict['l2_loss_excluding_shadow'] = l2_loss_shadow.item()
            loss_dict['l2_loss_for_sparse'] = l2_loss_sparse.item()
            loss_dict['l2_loss'] = self.l2_shadow_weight * l2_loss_shadow.item() + self.l2_sparse_weight * l2_loss_sparse.item()
            #total_loss += self.l2_weight * l2_loss
            total_loss += self.l2_shadow_weight * l2_loss_shadow + self.l2_sparse_weight * l2_loss_sparse
        else:
            loss_dict['l2_loss_excluding_shadow'] = 0.0
            loss_dict['l2_loss_for_sparse'] = 0.0
            loss_dict['l2_loss'] = 0.0
        
        # SSIM Loss
        if self.ssim_active == True:
            ssim_loss_shadow = ssim_loss_with_mask(pred, target, shadow_mask)
            ssim_loss_sparse = ssim_loss_with_mask(pred, target, sparse_mask)
            #ssim_loss_background= ssim_loss_with_mask(pred, target, background_mask)
            loss_dict['ssim_loss_excluding_shadow'] = ssim_loss_shadow.item()
            loss_dict['ssim_loss_for_sparse'] = ssim_loss_sparse.item()
            loss_dict['ssim_loss'] = self.ssim_shadow_weight * ssim_loss_shadow.item() + self.ssim_sparse_weight * ssim_loss_sparse.item()
            #total_loss += self.ssim_weight * ssim_loss
            total_loss += self.ssim_shadow_weight * ssim_loss_shadow + self.ssim_sparse_weight * ssim_loss_sparse
        else:
            loss_dict['ssim_loss_excluding_shadow'] = 0.0
            loss_dict['ssim_loss_for_sparse'] = 0.0
            loss_dict['ssim_loss'] = 0.0
        
        # Laplacian Pyramid Loss
        if self.laplacian_active == True:
            lap_loss_shadow = laplacian_loss_with_mask(pred, target, shadow_mask)
            lap_loss_sparse = laplacian_loss_with_mask(pred, target, sparse_mask)
            #lap_loss_background= laplacian_loss_with_mask(pred, target, background_mask)
            loss_dict['laplacian_loss_excluding_shadow'] = lap_loss_shadow.item()
            loss_dict['laplacian_loss_for_sparse'] = lap_loss_sparse.item()
            loss_dict['laplacian_loss'] = self.laplacian_shadow_weight * lap_loss_shadow.item() + self.laplacian_sparse_weight * lap_loss_sparse.item()
            #total_loss += self.laplacian_weight * lap_loss
            total_loss += self.laplacian_shadow_weight * lap_loss_shadow + self.laplacian_sparse_weight * lap_loss_sparse
        else:
            loss_dict['laplacian_loss_excluding_shadow'] = 0.0
            loss_dict['laplacian_loss_for_sparse'] = 0.0
            loss_dict['laplacian_loss'] = 0.0

        
        # Store total loss
        loss_dict['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, loss_dict

class HDDTrainer:
    """Trainer class for HDD Depth Fusion Model"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Multi-GPU setup
        self.num_gpus = torch.cuda.device_count()
        self.use_multi_gpu = self.num_gpus > 1 and config.get('use_multi_gpu', True)
        
        # Memory optimization settings
        if self.use_multi_gpu:
            # DataParallel requires more memory on GPU 0, so we need conservative settings
            torch.backends.cudnn.benchmark = False  # Disable for consistent memory usage
            torch.backends.cudnn.deterministic = True
        
        # Define model-specific name for directory creation
        self.model_name = "circularkernel_2_5_real_and_synthetic"
        
        # Settings for saving predictions (initialize before logging setup)
        self.save_predictions = config.get('save_predictions', True)
        self.max_batches_to_save = config.get('max_batches_to_save', 5)
        self.prediction_save_threshold = config.get('prediction_save_threshold', float('inf'))  # Only save predictions if val_loss < threshold
        self.selected_batch_indices = {}
        
        # Specific batch indices to save for each phase
        self.train_batch_indices_to_save = set(config.get('train_batch_indices_to_save', []))
        self.val_batch_indices_to_save = set(config.get('val_batch_indices_to_save', []))
        self.test_batch_indices_to_save = set(config.get('test_batch_indices_to_save', []))
        
        # Setup logging
        self.setup_logging()
        
        # Log prediction saving configuration
        self.logger.info(f"Prediction saving configuration:")
        self.logger.info(f"  Save predictions: {self.save_predictions}")
        if self.save_predictions:
            self.logger.info(f"  Prediction save threshold: {self.prediction_save_threshold}")
            self.logger.info(f"  Train batch indices: {sorted(self.train_batch_indices_to_save) if self.train_batch_indices_to_save else 'Random selection'}")
            self.logger.info(f"  Val batch indices: {sorted(self.val_batch_indices_to_save) if self.val_batch_indices_to_save else 'Random selection'}")
            self.logger.info(f"  Test batch indices: {sorted(self.test_batch_indices_to_save) if self.test_batch_indices_to_save else 'Random selection'}")
        
        # Log GPU information
        self.logger.info(f"Available GPUs: {self.num_gpus}")
        if self.use_multi_gpu:
            self.logger.info(f"Using multi-GPU training with DataParallel across {self.num_gpus} GPUs")
            for i in range(self.num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            self.logger.info(f"Using single GPU training on {torch.cuda.get_device_name(0)}")
        
        # Initialize model
        self.model = HDDDepthFusionModel(
            base_channels=config.get('base_channels', 64),
            dropout_prob=config.get('dropout_prob', 0.0)
        )

        # Apply emergency weight fix
        emergency_fix_all_circular_weights(self.model)
        
        # Freeze branches if specified in config
        freeze_sparse = config.get('freeze_sparse_branch', False)
        freeze_grayscale = config.get('freeze_grayscale_branch', False)
        freeze_relative = config.get('freeze_relative_branch', False)
        
        if any([freeze_sparse, freeze_grayscale, freeze_relative]):
            self.model.freeze_branches(
                freeze_sparse=freeze_sparse,
                freeze_grayscale=freeze_grayscale,
                freeze_relative=freeze_relative
            )
            frozen_branches = []
            if freeze_sparse:
                frozen_branches.append('sparse_branch')
            if freeze_grayscale:
                frozen_branches.append('grayscale_branch')
            if freeze_relative:
                frozen_branches.append('relative_depth_branch')
            self.logger.info(f"Frozen branches: {', '.join(frozen_branches)}")
        
        # Wrap model with DataParallel for multi-GPU training
        if self.use_multi_gpu:
            self.model = nn.DataParallel(self.model)
            self.logger.info("Model wrapped with DataParallel")
            self.logger.info(f"DataParallel will use GPUs: {list(range(self.num_gpus))}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Clear any cached memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize loss function
        self.criterion = CombinedDepthLoss(
            l1_active=config.get('l1_active', True),
            l1_shadow_weight=config.get('l1_shadow_weight', 1.0),
            l1_sparse_weight=config.get('l1_sparse_weight', 1.0),
            l2_active=config.get('l2_active', False),
            l2_shadow_weight=config.get('l2_shadow_weight', 1.0),
            l2_sparse_weight=config.get('l2_sparse_weight', 1.0),
            ssim_active=config.get('ssim_active', False),
            ssim_shadow_weight=config.get('ssim_shadow_weight', 1.0),
            ssim_sparse_weight=config.get('ssim_sparse_weight', 1.0),
            laplacian_active=config.get('laplacian_active', False),
            laplacian_shadow_weight=config.get('laplacian_shadow_weight', 1.0),
            laplacian_sparse_weight=config.get('laplacian_sparse_weight', 1.0)
        )
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.get('lr_factor', 0.5),
            patience=config.get('lr_patience', 10)
        )
        
        # Training state
        self.epoch = config.get('start_epoch', 0)
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.is_best_epoch = False  # Add this line
        
        # Setup directories
        self.setup_directories()
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False):
            wandb_project = config.get('wandb_project', 'hdd-depth-fusion')
            wandb_project_name = f"{wandb_project}-{self.model_name}"
            wandb.init(project=wandb_project_name)
            
            # Log the entire config to wandb
            wandb.config.update(config)
            
            # Log loss weights and training parameters specifically for easy access
            training_params = {
                'weight_decay': config.get('weight_decay', 1e-5),
                'l1_active': config.get('l1_active', True),
                'l1_shadow_weight': config.get('l1_shadow_weight', 1.0),
                'l1_sparse_weight': config.get('l1_sparse_weight', 1.0),
                'l2_active': config.get('l2_active', False),
                'l2_shadow_weight': config.get('l2_shadow_weight', 1.0),
                'l2_sparse_weight': config.get('l2_sparse_weight', 1.0),
                'ssim_active': config.get('ssim_active', False),
                'ssim_shadow_weight': config.get('ssim_shadow_weight', 1.0),
                'ssim_sparse_weight': config.get('ssim_sparse_weight', 1.0),
                'laplacian_active': config.get('laplacian_active', False),
                'laplacian_shadow_weight': config.get('laplacian_shadow_weight', 1.0),
                'laplacian_sparse_weight': config.get('laplacian_sparse_weight', 1.0)
            }
            wandb.config.update({"training_params": training_params})
            
            # Upload config file as artifact (if running from main, not resuming)
            if hasattr(self, '_config_file_path'):
                artifact = wandb.Artifact('config', type='config')
                artifact.add_file(self._config_file_path)
                wandb.log_artifact(artifact)
                self.logger.info(f"Config file uploaded to wandb as artifact: {self._config_file_path}")
            
            # Log model info
            wandb.config.update({
                "model_name": self.model_name,
                "num_gpus": self.num_gpus,
                "use_multi_gpu": self.use_multi_gpu,
                "model_parameters": sum(p.numel() for p in self.model.parameters())
            })
            
            wandb.watch(self.model)
            self.logger.info("Config, training parameters (including weight decay and loss weights) logged to wandb")
    
    def setup_logging(self):
        """Setup logging configuration with model-specific log file"""
        # Create model-specific logs directory
        log_dir = Path('logs') / self.model_name
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """Create necessary model-specific directories"""
        base_checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        base_results_dir = Path(self.config.get('results_dir', 'results'))
        
        # Create model-specific directories
        self.checkpoint_dir = base_checkpoint_dir / self.model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = base_results_dir / self.model_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for saving predictions
        if self.config.get('save_predictions', True):
            self.predictions_dir = self.results_dir / 'predictions'
            self.predictions_dir.mkdir(parents=True, exist_ok=True)
            
            for phase in ['train', 'val', 'test']:
                phase_dir = self.predictions_dir / phase
                phase_dir.mkdir(parents=True, exist_ok=True)
                
                # Create subdirectories for different types of outputs
                for output_type in ['predicted_depth', 'ground_truth', 'sparse_input', 'grayscale_input', 'relative_input', 'sparse_mask', 'shadow_mask', 'background_mask', 'comparison']:
                    (phase_dir / output_type).mkdir(parents=True, exist_ok=True)
    
    def save_prediction_batch(self, batch_data, predictions, batch_idx, phase='train'):
        """Save predictions from a single batch"""
        if not self.save_predictions:
            return
            
        # Check if this batch should be saved (randomized selection)
        if batch_idx not in self.selected_batch_indices.get(phase, set()):
            return
            
        phase_dir = self.predictions_dir / phase
        
        # Convert tensors to numpy arrays
        pred_np = predictions.detach().cpu().numpy()
        target_np = batch_data['gt_depth'].detach().cpu().numpy()
        sparse_np = batch_data['sparse_depth'].detach().cpu().numpy()
        grayscale_np = batch_data['grayscale'].detach().cpu().numpy()
        relative_np = batch_data['depth_anything_v2'].detach().cpu().numpy()
        
        # Convert mask tensors to numpy arrays
        sparse_mask_np = batch_data['sparse_mask'].detach().cpu().numpy()
        shadow_mask_np = batch_data['shadow_mask'].detach().cpu().numpy()
        background_mask_np = batch_data['background_mask'].detach().cpu().numpy()
        
        batch_size = pred_np.shape[0]
        
        for sample_idx in range(batch_size):
            sample_name = f"epoch_{self.epoch+1}_batch_{batch_idx+1}_sample_{sample_idx+1}"
            
            # Save individual components as images and CSV files
            self._save_depth_image(pred_np[sample_idx, 0], phase_dir / 'predicted_depth' / f"{sample_name}.png")
            self._save_depth_image(target_np[sample_idx, 0], phase_dir / 'ground_truth' / f"{sample_name}.png")
            self._save_depth_image(sparse_np[sample_idx, 0], phase_dir / 'sparse_input' / f"{sample_name}.png")
            self._save_depth_image(grayscale_np[sample_idx, 0], phase_dir / 'grayscale_input' / f"{sample_name}.png")
            self._save_depth_image(relative_np[sample_idx, 0], phase_dir / 'relative_input' / f"{sample_name}.png")
            
            # Save mask components as images and CSV files
            self._save_mask_image(sparse_mask_np[sample_idx, 0], phase_dir / 'sparse_mask' / f"{sample_name}.png")
            self._save_mask_image(shadow_mask_np[sample_idx, 0], phase_dir / 'shadow_mask' / f"{sample_name}.png")
            self._save_mask_image(background_mask_np[sample_idx, 0], phase_dir / 'background_mask' / f"{sample_name}.png")
            
            # Save CSV files for further analysis
            self._save_depth_csv(pred_np[sample_idx, 0], phase_dir / 'predicted_depth' / f"{sample_name}.csv")
            self._save_depth_csv(target_np[sample_idx, 0], phase_dir / 'ground_truth' / f"{sample_name}.csv")
            self._save_depth_csv(sparse_np[sample_idx, 0], phase_dir / 'sparse_input' / f"{sample_name}.csv")
            self._save_depth_csv(relative_np[sample_idx, 0], phase_dir / 'relative_input' / f"{sample_name}.csv")
            
            # Save mask CSV files
            self._save_depth_csv(sparse_mask_np[sample_idx, 0], phase_dir / 'sparse_mask' / f"{sample_name}.csv")
            self._save_depth_csv(shadow_mask_np[sample_idx, 0], phase_dir / 'shadow_mask' / f"{sample_name}.csv")
            self._save_depth_csv(background_mask_np[sample_idx, 0], phase_dir / 'background_mask' / f"{sample_name}.csv")
            
            # Create comparison visualization
            self._save_comparison_image(
                sparse_np[sample_idx, 0], grayscale_np[sample_idx, 0], relative_np[sample_idx, 0], 
                pred_np[sample_idx, 0], target_np[sample_idx, 0],
                sparse_mask_np[sample_idx, 0], shadow_mask_np[sample_idx, 0], background_mask_np[sample_idx, 0],
                phase_dir / 'comparison' / f"{sample_name}_comparison.png"
            )
    
    def _save_depth_image(self, depth_array, filepath):
        """Save a depth array as a normalized image"""
        # Normalize to 0-255 range
        depth_normalized = ((depth_array - depth_array.min()) / (depth_array.max() - depth_array.min() + 1e-8) * 255).astype(np.uint8)
        
        # Apply colormap for better visualization
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        # Save the image
        cv2.imwrite(str(filepath), depth_colored)
    
    def _save_mask_image(self, mask_array, filepath):
        """Save a mask array as a binary image"""
        # Convert mask to 0-255 range (assuming mask values are 0-1)
        mask_normalized = (mask_array * 255).astype(np.uint8)
        
        # Save as grayscale image
        cv2.imwrite(str(filepath), mask_normalized)
    
    def _save_depth_csv(self, depth_array, filepath):
        """Save a depth array as a CSV file"""
        df = pd.DataFrame(depth_array)
        df.to_csv(filepath, index=False)
    
    def _save_comparison_image(self, sparse_input, grayscale_input, relative_input, prediction, ground_truth, sparse_mask, shadow_mask, background_mask, filepath):
        """Create and save a comparison visualization"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        # Top row: inputs
        axes[0, 0].imshow(sparse_input, cmap='viridis')
        axes[0, 0].set_title('Sparse Depth Input')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(grayscale_input, cmap='gray')
        axes[0, 1].set_title('Grayscale Input')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(relative_input, cmap='viridis')
        axes[0, 2].set_title('Relative Depth Input')
        axes[0, 2].axis('off')
        
        # Middle row: outputs
        axes[1, 0].imshow(prediction, cmap='viridis')
        axes[1, 0].set_title('Predicted Depth')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(ground_truth, cmap='viridis')
        axes[1, 1].set_title('Ground Truth')
        axes[1, 1].axis('off')
        
        # Error map
        error = np.abs(prediction - ground_truth)
        im = axes[1, 2].imshow(error, cmap='hot')
        axes[1, 2].set_title('Absolute Error')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2])
        
        # Bottom row: masks
        axes[2, 0].imshow(sparse_mask, cmap='gray')
        axes[2, 0].set_title('Sparse Mask')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(shadow_mask, cmap='gray')
        axes[2, 1].set_title('Shadow Mask')
        axes[2, 1].axis('off')
        
        axes[2, 2].imshow(background_mask, cmap='gray')
        axes[2, 2].set_title('Background Mask')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _select_batch_indices_to_save(self, dataloader_len, phase):
        """Select batch indices to save predictions for - use specified indices or fallback to random"""
        if phase == 'train' and self.train_batch_indices_to_save:
            # Filter indices that exist in the dataloader
            valid_indices = {idx for idx in self.train_batch_indices_to_save if idx < dataloader_len}
            self.selected_batch_indices[phase] = valid_indices
            self.logger.info(f"Using specified batch indices for {phase}: {sorted(valid_indices)}")
        elif phase == 'val' and self.val_batch_indices_to_save:
            valid_indices = {idx for idx in self.val_batch_indices_to_save if idx < dataloader_len}
            self.selected_batch_indices[phase] = valid_indices
            self.logger.info(f"Using specified batch indices for {phase}: {sorted(valid_indices)}")
        elif phase == 'test' and self.test_batch_indices_to_save:
            valid_indices = {idx for idx in self.test_batch_indices_to_save if idx < dataloader_len}
            self.selected_batch_indices[phase] = valid_indices
            self.logger.info(f"Using specified batch indices for {phase}: {sorted(valid_indices)}")
        else:
            # Fallback to random selection
            max_batches = min(self.max_batches_to_save, dataloader_len)
            selected_indices = set(random.sample(range(dataloader_len), max_batches))
            self.selected_batch_indices[phase] = selected_indices
            self.logger.info(f"Using random batch indices for {phase}: {sorted(selected_indices)}")
    
    def _should_save_predictions_this_epoch(self, current_val_loss=None):
        """Check if predictions should be saved this epoch (best model + threshold or checkpoint frequency)"""
        save_freq = self.config.get('save_freq', 10)
        
        # Save every save_freq epochs regardless of loss
        if (self.epoch + 1) % save_freq == 0:
            return True
            
        # Save on best model only if validation loss is below threshold
        if self.is_best_epoch and current_val_loss is not None:
            if current_val_loss < self.prediction_save_threshold:
                self.logger.info(f"Saving predictions: Best epoch with val_loss {current_val_loss:.6f} < threshold {self.prediction_save_threshold:.6f}")
                return True
            else:
                self.logger.info(f"Not saving predictions: Best epoch but val_loss {current_val_loss:.6f} >= threshold {self.prediction_save_threshold:.6f}")
                return False
        
        return False
    
    def save_predictions_for_epoch(self, train_dataloader, val_dataloader=None, test_dataloader=None):
        """Save predictions for selected batches after epoch completion"""
        if not self.save_predictions:
            return
            
        self.logger.info(f"Saving prediction batches for epoch {self.epoch+1}")
        
        # Save training predictions
        self._select_batch_indices_to_save(len(train_dataloader), 'train')
        self._save_selected_predictions(train_dataloader, 'train')
        
        # Save validation predictions
        if val_dataloader is not None:
            self._select_batch_indices_to_save(len(val_dataloader), 'val')
            self._save_selected_predictions(val_dataloader, 'val')
        
        # Save test predictions
        if test_dataloader is not None:
            self._select_batch_indices_to_save(len(test_dataloader), 'test')
            self._save_selected_predictions(test_dataloader, 'test')
    
    def _save_selected_predictions(self, dataloader, phase):
        """Save predictions for selected batch indices"""
        self.model.eval()
        
        if phase == 'train':
            desc = f'Saving {phase} predictions'
        else:
            desc = f'Saving {phase} predictions'
            
        pbar = tqdm(dataloader, desc=desc)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Only process selected batches
                if batch_idx not in self.selected_batch_indices.get(phase, set()):
                    continue
                    
                # Move data to device
                sparse_input = batch['sparse_depth'].to(self.device)
                grayscale_input = batch['grayscale'].to(self.device)
                relative_input = batch['depth_anything_v2'].to(self.device)
                
                # Forward pass
                pred = self.model(sparse_input, grayscale_input, relative_input)
                
                # Save predictions
                self.save_prediction_batch(batch, pred, batch_idx, phase=phase)

    def save_checkpoint(self, filename='checkpoint.pth'):
        """Save model checkpoint"""
        # Handle DataParallel wrapped models
        model_state_dict = self.model.module.state_dict() if self.use_multi_gpu else self.model.state_dict()
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'test_losses': self.test_losses,
            'use_multi_gpu': self.use_multi_gpu,
            'num_gpus': self.num_gpus
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Handle DataParallel wrapped models
        if self.use_multi_gpu:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Handle case where checkpoint was saved with DataParallel but loading without
            state_dict = checkpoint['model_state_dict']
            if any(key.startswith('module.') for key in state_dict.keys()):
                # Remove 'module.' prefix from keys
                state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            self.model.load_state_dict(state_dict)
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        # Log checkpoint info
        checkpoint_multi_gpu = checkpoint.get('use_multi_gpu', False)
        checkpoint_num_gpus = checkpoint.get('num_gpus', 1)
        self.logger.info(f"Checkpoint loaded: {filepath}")
        self.logger.info(f"Checkpoint was saved with: {checkpoint_num_gpus} GPU(s), DataParallel: {checkpoint_multi_gpu}")
        self.logger.info(f"Current setup: {self.num_gpus} GPU(s), DataParallel: {self.use_multi_gpu}")
    
    def compute_metrics(self, pred, target, shadow_mask=None, sparse_mask=None):
        """Compute evaluation metrics"""
        # Convert all tensors to numpy
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Convert masks to numpy as well
        if shadow_mask is not None:
            shadow_mask_np = shadow_mask.detach().cpu().numpy().astype(bool)
        else:
            shadow_mask_np = np.ones_like(pred_np, dtype=bool)
            
        if sparse_mask is not None:
            sparse_mask_np = sparse_mask.detach().cpu().numpy().astype(bool)
        else:
            sparse_mask_np = np.ones_like(pred_np, dtype=bool)
        
        # Compute metrics using numpy masks
        mae = np.mean(np.abs(pred_np[shadow_mask_np] - target_np[shadow_mask_np]))
        mae_sparse = np.mean(np.abs(pred_np[sparse_mask_np] - target_np[sparse_mask_np]))
        mse = np.mean((pred_np[shadow_mask_np] - target_np[shadow_mask_np]) ** 2)
        mse_sparse = np.mean((pred_np[sparse_mask_np] - target_np[sparse_mask_np]) ** 2)
        rmse = np.sqrt(mse)
        rmse_sparse = np.sqrt(mse_sparse)
        
        # Compute relative metrics
        rel_error = np.mean(np.abs(pred_np[shadow_mask_np] - target_np[shadow_mask_np]) / (target_np[shadow_mask_np] + 1e-8))
        rel_error_sparse = np.mean(np.abs(pred_np[sparse_mask_np] - target_np[sparse_mask_np]) / (target_np[sparse_mask_np] + 1e-8))
        
        return {
            'mae': mae,
            'mae_sparse': mae_sparse,
            'mse': mse,
            'mse_sparse': mse_sparse,
            'rmse': rmse,
            'rmse_sparse': rmse_sparse,
            'rel_error': rel_error,
            'rel_error_sparse': rel_error_sparse
        }
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_loss_components = {'l1_loss':0.0,'l1_loss_excluding_shadow': 0.0, 'l1_loss_for_sparse': 0.0, 'l2_loss':0.0,'l2_loss_excluding_shadow': 0.0, 'l2_loss_for_sparse': 0.0, 'ssim_loss':0.0,'ssim_loss_excluding_shadow': 0.0, 'ssim_loss_for_sparse': 0.0, 'laplacian_loss':0.0,'laplacian_loss_excluding_shadow': 0.0, 'laplacian_loss_for_sparse': 0.0}
        total_metrics = {'mae': 0.0, 'mae_sparse': 0.0, 'mse': 0.0, 'mse_sparse': 0.0, 'rmse': 0.0, 'rmse_sparse': 0.0, 'rel_error': 0.0, 'rel_error_sparse': 0.0}
        
        # Prediction saving will be handled after epoch completion in main train loop
        
        pbar = tqdm(dataloader, desc=f'Epoch {self.epoch+1} Training')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            sparse_input = batch['sparse_depth'].to(self.device)
            grayscale_input = batch['grayscale'].to(self.device)
            relative_input = batch['depth_anything_v2'].to(self.device)
            target = batch['gt_depth'].to(self.device)
            
            # Optional: use shadow mask for loss computation
            mask_shadow = batch.get('shadow_mask', None)
            mask_sparse = batch.get('sparse_mask', None)
            if mask_shadow is not None:
                mask_shadow = mask_shadow.to(self.device)
            if mask_sparse is not None:
                mask_sparse = mask_sparse.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(sparse_input, grayscale_input, relative_input)
            
            # Compute loss
            loss, loss_dict = self.criterion(pred, target, mask_shadow, mask_sparse)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            # Compute metrics
            metrics = self.compute_metrics(pred, target, mask_shadow, mask_sparse)
            
            # Update running averages
            total_loss += loss.item()
            for key in total_loss_components:
                total_loss_components[key] += loss_dict[key]
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            # Update progress bar
            pbar.set_postfix({
                'total': f'{loss.item():.6f}',
                'mae': f'{metrics["mae"]:.6f}',
                'rmse': f'{metrics["rmse"]:.6f}'
            })
            
            # Memory cleanup for multi-GPU training
            #if self.use_multi_gpu and (batch_idx + 1) % 10 == 0:
            #    torch.cuda.empty_cache()
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb_dict = {
                    'train_batch_total_loss': loss.item(),
                    'train_batch_mae': metrics['mae'],
                    'train_batch_rmse': metrics['rmse'],
                    'train_batch_mae_sparse': metrics['mae_sparse'],
                    'train_batch_mse': metrics['mse'],
                    'train_batch_mse_sparse': metrics['mse_sparse'],
                    'train_batch_rmse_sparse': metrics['rmse_sparse'],
                    'train_batch_rel_error': metrics['rel_error'],
                    'train_batch_rel_error_sparse': metrics['rel_error_sparse']
                }
                # Add individual loss components
                for loss_name, loss_value in loss_dict.items():
                    wandb_dict[f'train_batch_{loss_name}'] = loss_value
                wandb.log(wandb_dict)
        
        # Compute epoch averages
        avg_loss = total_loss / len(dataloader)
        avg_loss_components = {key: val / len(dataloader) for key, val in total_loss_components.items()}
        avg_metrics = {key: val / len(dataloader) for key, val in total_metrics.items()}
        
        return avg_loss, avg_loss_components, avg_metrics
    
    def validate_epoch(self, dataloader, name='val'):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_loss_components = {'l1_loss':0.0,'l1_loss_excluding_shadow': 0.0, 'l1_loss_for_sparse': 0.0, 'l2_loss':0.0,'l2_loss_excluding_shadow': 0.0, 'l2_loss_for_sparse': 0.0, 'ssim_loss':0.0,'ssim_loss_excluding_shadow': 0.0, 'ssim_loss_for_sparse': 0.0, 'laplacian_loss':0.0,'laplacian_loss_excluding_shadow': 0.0, 'laplacian_loss_for_sparse': 0.0}
        total_metrics = {'mae': 0.0, 'mae_sparse': 0.0, 'mse': 0.0, 'mse_sparse': 0.0, 'rmse': 0.0, 'rmse_sparse': 0.0, 'rel_error': 0.0, 'rel_error_sparse': 0.0}
        
        # Prediction saving will be handled after epoch completion in main train loop
        
        if name == 'val':
            pbar = tqdm(dataloader, desc=f'Epoch {self.epoch+1} Validation')
        else:
            pbar = tqdm(dataloader, desc=f'Epoch {self.epoch+1} Test')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                sparse_input = batch['sparse_depth'].to(self.device)
                grayscale_input = batch['grayscale'].to(self.device)
                relative_input = batch['depth_anything_v2'].to(self.device)
                target = batch['gt_depth'].to(self.device)
                
                # Optional: use shadow mask for loss computation
                mask_shadow = batch.get('shadow_mask', None)
                mask_sparse = batch.get('sparse_mask', None)
                if mask_shadow is not None:
                    mask_shadow = mask_shadow.to(self.device)
                if mask_sparse is not None:
                    mask_sparse = mask_sparse.to(self.device)
                
                # Forward pass
                pred = self.model(sparse_input, grayscale_input, relative_input)
                
                # Compute loss
                loss, loss_dict = self.criterion(pred, target, mask_shadow, mask_sparse)
                
                # Compute metrics
                metrics = self.compute_metrics(pred, target, mask_shadow, mask_sparse)
                
                # Update running averages
                total_loss += loss.item()
                for key in total_loss_components:
                    total_loss_components[key] += loss_dict[key]
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
                
                # Update progress bar
                pbar.set_postfix({
                    'total': f'{loss.item():.6f}',
                    'mae': f'{metrics["mae"]:.6f}',
                    'rmse': f'{metrics["rmse"]:.6f}'
                })
        
        # Compute epoch averages
        avg_loss = total_loss / len(dataloader)
        avg_loss_components = {key: val / len(dataloader) for key, val in total_loss_components.items()}
        avg_metrics = {key: val / len(dataloader) for key, val in total_metrics.items()}
        
        return avg_loss, avg_loss_components, avg_metrics
    
    def train(self, train_dataloader, val_dataloader=None, test_dataloader=None):
        """Main training loop"""
        self.logger.info(f"Starting training on {self.device}")
        if self.use_multi_gpu:
            self.logger.info(f"Multi-GPU training enabled - using {self.num_gpus} GPUs")
            self.logger.info(f"Effective batch size per GPU: {self.config['batch_size'] // self.num_gpus}")
            if self.config['batch_size'] % self.num_gpus != 0:
                self.logger.warning(f"Batch size {self.config['batch_size']} is not evenly divisible by {self.num_gpus} GPUs. "
                                  f"Consider using a batch size that's a multiple of {self.num_gpus} for optimal performance.")
        else:
            self.logger.info(f"Single GPU training - batch size: {self.config['batch_size']}")
        
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config['start_epoch'], self.config['start_epoch']+self.config['num_epochs']):
            self.epoch = epoch
            
            # Train epoch
            train_loss, train_loss_components, train_metrics = self.train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            
            # Validate epoch
            if val_dataloader is not None:
                val_loss, val_loss_components, val_metrics = self.validate_epoch(val_dataloader,name='val')
                self.val_losses.append(val_loss)
            else:
                val_loss, val_loss_components, val_metrics = train_loss, train_loss_components, train_metrics
            
            # Test epoch (if test dataloader provided)
            if test_dataloader is not None:
                test_loss, test_loss_components, test_metrics = self.validate_epoch(test_dataloader,name='test')
                self.test_losses.append(test_loss)
            else:
                test_loss, test_loss_components, test_metrics = val_loss, val_loss_components, val_metrics
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Log epoch results
            self.logger.info(f"Epoch {epoch+1}/{self.config['start_epoch']+self.config['num_epochs']}")
            self.logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Test Loss: {test_loss:.6f}")
            self.logger.info(f"Train MAE: {train_metrics['mae']:.6f}, Val MAE: {val_metrics['mae']:.6f}, Test MAE: {test_metrics['mae']:.6f}")
            self.logger.info(f"Train RMSE: {train_metrics['rmse']:.6f}, Val RMSE: {val_metrics['rmse']:.6f}, Test RMSE: {test_metrics['rmse']:.6f}")
            self.logger.info(f"Train MAE Sparse: {train_metrics['mae_sparse']:.6f}, Val MAE Sparse: {val_metrics['mae_sparse']:.6f}, Test MAE Sparse: {test_metrics['mae_sparse']:.6f}")
            self.logger.info(f"Train MSE: {train_metrics['mse']:.6f}, Val MSE: {val_metrics['mse']:.6f}, Test MSE: {test_metrics['mse']:.6f}")
            self.logger.info(f"Train MSE Sparse: {train_metrics['mse_sparse']:.6f}, Val MSE Sparse: {val_metrics['mse_sparse']:.6f}, Test MSE Sparse: {test_metrics['mse_sparse']:.6f}")
            self.logger.info(f"Train RMSE Sparse: {train_metrics['rmse_sparse']:.6f}, Val RMSE Sparse: {val_metrics['rmse_sparse']:.6f}, Test RMSE Sparse: {test_metrics['rmse_sparse']:.6f}")
            self.logger.info(f"Train Rel Error: {train_metrics['rel_error']:.6f}, Val Rel Error: {val_metrics['rel_error']:.6f}, Test Rel Error: {test_metrics['rel_error']:.6f}")
            self.logger.info(f"Train Rel Error Sparse: {train_metrics['rel_error_sparse']:.6f}, Val Rel Error Sparse: {val_metrics['rel_error_sparse']:.6f}, Test Rel Error Sparse: {test_metrics['rel_error_sparse']:.6f}")

            # Log individual loss components
            self.logger.info(f"Train Loss Components - L1: {train_loss_components['l1_loss']:.6f}, "
                           f"L1 Excluding Shadow: {train_loss_components['l1_loss_excluding_shadow']:.6f}, "
                           f"L1 For Sparse: {train_loss_components['l1_loss_for_sparse']:.6f}, "
                           f"L2: {train_loss_components['l2_loss']:.6f}, "
                           f"L2 Excluding Shadow: {train_loss_components['l2_loss_excluding_shadow']:.6f}, "
                           f"L2 For Sparse: {train_loss_components['l2_loss_for_sparse']:.6f}, "
                           f"SSIM: {train_loss_components['ssim_loss']:.6f}, "
                           f"SSIM Excluding Shadow: {train_loss_components['ssim_loss_excluding_shadow']:.6f}, "
                           f"SSIM For Sparse: {train_loss_components['ssim_loss_for_sparse']:.6f}, "
                           f"Laplacian: {train_loss_components['laplacian_loss']:.6f}, "
                           f"Laplacian Excluding Shadow: {train_loss_components['laplacian_loss_excluding_shadow']:.6f}, "
                           f"Laplacian For Sparse: {train_loss_components['laplacian_loss_for_sparse']:.6f}")
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb_dict = {
                    'epoch': epoch + 1,
                    'train_total_loss': train_loss,
                    'val_total_loss': val_loss,
                    'test_total_loss': test_loss,
                    'train_mae': train_metrics['mae'],
                    'val_mae': val_metrics['mae'],
                    'test_mae': test_metrics['mae'],
                    'train_rmse': train_metrics['rmse'],
                    'val_rmse': val_metrics['rmse'],
                    'test_rmse': test_metrics['rmse'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                
                # Add individual loss components
                for loss_name, loss_value in train_loss_components.items():
                    wandb_dict[f'train_{loss_name}'] = loss_value
                for loss_name, loss_value in val_loss_components.items():
                    wandb_dict[f'val_{loss_name}'] = loss_value
                for loss_name, loss_value in test_loss_components.items():
                    wandb_dict[f'test_{loss_name}'] = loss_value
                    
                wandb.log(wandb_dict)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.is_best_epoch = True  # Add this line
                self.save_checkpoint(f'best-epoch{epoch+1}.pth')  # Change filename format
                self.logger.info(f"New best model saved with validation loss: {val_loss:.6f}")
            else:
                self.is_best_epoch = False  # Add this line
            
            # Save predictions if conditions are met
            should_save_predictions = self._should_save_predictions_this_epoch(val_loss)
            if should_save_predictions:
                self.save_predictions_for_epoch(train_dataloader, val_dataloader, test_dataloader)
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get('save_freq', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        self.logger.info("Training completed!")

    def reset_best_validation_loss(self):
        """Reset the best validation loss to start fresh tracking"""
        old_best_loss = self.best_val_loss
        self.best_val_loss = float('inf')
        self.logger.info(f"Best validation loss reset from {old_best_loss:.6f} to inf - will start tracking new best from current point")

    def set_learning_rate(self, lr):
        """Update the learning rate for the optimizer"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.logger.info(f"Learning rate updated to: {lr}")

def create_dataloaders(config):
    """Create training and validation dataloaders"""
    dataset_path = config['dataset_path']
    
    # Get dataset statistics for normalization
    base_dataset = HDDDepthFusionDataset(dataset_path, normalize_depth=False)
    stats = base_dataset.get_depth_stats(num_samples=40)
    
    # Compute combined normalization parameters
    depth_min = min(stats['sparse_min'], stats['gt_min'])
    depth_max = max(stats['sparse_max'], stats['gt_max'])
    
    # Create augmented training dataloader
    train_dataloader = create_augmented_dataloader(
        dataset_path,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        normalize_depth=False,
        depth_min=depth_min,
        depth_max=depth_max,
        augment_probability=config.get('augment_prob', 0.8),
        custom_probabilities=config.get('augment_config', CONSERVATIVE_CONFIG),
        train_on_shadow_regions=config['train_on_shadow_regions']
    )
    
    # Create validation dataloader (no augmentation)
    val_dataloader = create_dataloader(
        dataset_path,
        split='valid',
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        normalize_depth=False,
        depth_min=depth_min,
        depth_max=depth_max,
        train_on_shadow_regions=config['train_on_shadow_regions']
    )
    
    # Create test dataloader (no augmentation)
    test_dataloader = create_dataloader(
        dataset_path,
        split='test',
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        normalize_depth=False,
        depth_min=depth_min,
        depth_max=depth_max,
        train_on_shadow_regions=config['train_on_shadow_regions']
    )
    
    return train_dataloader, val_dataloader, test_dataloader, (depth_min, depth_max)

def main():
    parser = argparse.ArgumentParser(description='Train HDD Depth Fusion Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--resume-lr', type=float, help='Custom learning rate when resuming training')
    parser.add_argument('--reset-best-loss', action='store_true', help='Reset best validation loss when resuming (useful when changing loss coefficients)')
    parser.add_argument('--train-on-shadow-regions', action='store_true', help='Include-or-exclude-shadow-regions-while-training')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    if args.train_on_shadow_regions:
        config['train_on_shadow_regions'] = True
    else:
        config['train_on_shadow_regions'] = False
    
    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader, norm_params = create_dataloaders(config)
    
    # Initialize trainer
    trainer = HDDTrainer(config)
    
    # Store config file path for wandb artifact upload
    trainer._config_file_path = args.config
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
        # Reset best validation loss if requested
        if args.reset_best_loss:
            trainer.reset_best_validation_loss()
    
    # Override learning rate if custom resume LR is specified
    if args.resume_lr:
        trainer.set_learning_rate(args.resume_lr)
    
    # Start training
    trainer.train(train_dataloader, val_dataloader, test_dataloader)

if __name__ == "__main__":
    main()
