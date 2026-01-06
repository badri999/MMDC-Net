#!/usr/bin/env python3
"""
Inference script for HDD Depth Fusion Model (circularkernel_2_5)
Loads a trained checkpoint and runs inference on test data, saving predictions
in the exact same format as the training script.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
import json
from pathlib import Path
import logging
from tqdm import tqdm
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
import importlib
from dataloader.hdd_dataloader import HDDDepthFusionDataset, create_dataloader
from dataloader.hdd_data_augmenter import create_augmented_dataloader, CONSERVATIVE_CONFIG


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


class HDDInferenceEngine:
    """Inference engine for HDD Depth Fusion Model"""
    def __init__(self, checkpoint_path, model_file='circularkernel_2_5_real_and_synthetic', use_emergency_fix=True, device=None):
        self.checkpoint_path = checkpoint_path
        self.model_file = model_file
        self.use_emergency_fix = use_emergency_fix
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint and extract config
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = self.checkpoint.get('config', {})
        
        # Extract model configuration from checkpoint
        self.use_multi_gpu = self.checkpoint.get('use_multi_gpu', False)
        self.num_gpus = self.checkpoint.get('num_gpus', 1)
        
        # Setup logging
        self.setup_logging()
        
        # Dynamically import model components
        self.import_model_components()
        
        # Initialize model
        self.model = HDDDepthFusionModel(
            base_channels=self.config.get('base_channels', 64),
            dropout_prob=0.0  # Set to 0 for inference
        )
        
        # Apply emergency weight fix if specified
        if self.use_emergency_fix:
            self.emergency_fix_all_circular_weights(self.model)
            self.logger.info("Applied emergency weight fix")
        
        # Load model weights
        self.load_model_weights()
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Log model info
        self.logger.info(f"Loaded checkpoint from: {checkpoint_path}")
        self.logger.info(f"Model file: {model_file}")
        self.logger.info(f"Emergency fix applied: {use_emergency_fix}")
        self.logger.info(f"Checkpoint epoch: {self.checkpoint.get('epoch', 'unknown')}")
        self.logger.info(f"Checkpoint val loss: {self.checkpoint.get('best_val_loss', 'unknown')}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Using device: {self.device}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
    
    def import_model_components(self):
        """Dynamically import model components and emergency fix function"""
        try:
            # Import model branches from specified model file
            model_module = importlib.import_module(self.model_file)
            global SparseDepthBranch, GrayscaleBranch, RelativeDepthBranch, FusionBranch
            SparseDepthBranch = model_module.SparseDepthBranch
            GrayscaleBranch = model_module.GrayscaleBranch
            RelativeDepthBranch = model_module.RelativeDepthBranch
            FusionBranch = model_module.FusionBranch
            
            # Import emergency fix function if needed
            if self.use_emergency_fix:
                emergency_module = importlib.import_module('emergency_weight_fix')
                self.emergency_fix_all_circular_weights = emergency_module.emergency_fix_all_circular_weights
            
            self.logger.info(f"Successfully imported components from {self.model_file}")
        except ImportError as e:
            self.logger.error(f"Failed to import model components from {self.model_file}: {e}")
            raise
    
    def load_model_weights(self):
        """Load model weights from checkpoint"""
        state_dict = self.checkpoint['model_state_dict']
        
        # Handle DataParallel saved models
        if any(key.startswith('module.') for key in state_dict.keys()):
            # Remove 'module.' prefix from keys
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.logger.info("Model weights loaded successfully")
    
    def create_output_directories(self, output_dir):
        """Create necessary directories for saving predictions"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different types of outputs
        subdirs = ['predicted_depth', 'ground_truth', 'sparse_input', 'grayscale_input', 
                  'relative_input', 'sparse_mask', 'shadow_mask', 'background_mask', 'comparison']
        
        for subdir in subdirs:
            (output_path / subdir).mkdir(parents=True, exist_ok=True)
        
        return output_path
    
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
    
    def _save_comparison_image(self, sparse_input, grayscale_input, relative_input, prediction, ground_truth, 
                              sparse_mask, shadow_mask, background_mask, filepath):
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
    
    def save_prediction_batch(self, batch_data, predictions, batch_idx, output_path, prefix="inference"):
        """Save predictions from a single batch - exact same format as training script"""
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
            sample_name = f"{prefix}_batch_{batch_idx+1}_sample_{sample_idx+1}"
            
            # Save individual components as images and CSV files
            self._save_depth_image(pred_np[sample_idx, 0], output_path / 'predicted_depth' / f"{sample_name}.png")
            self._save_depth_image(target_np[sample_idx, 0], output_path / 'ground_truth' / f"{sample_name}.png")
            self._save_depth_image(sparse_np[sample_idx, 0], output_path / 'sparse_input' / f"{sample_name}.png")
            self._save_depth_image(grayscale_np[sample_idx, 0], output_path / 'grayscale_input' / f"{sample_name}.png")
            self._save_depth_image(relative_np[sample_idx, 0], output_path / 'relative_input' / f"{sample_name}.png")
            
            # Save mask components as images
            self._save_mask_image(sparse_mask_np[sample_idx, 0], output_path / 'sparse_mask' / f"{sample_name}.png")
            self._save_mask_image(shadow_mask_np[sample_idx, 0], output_path / 'shadow_mask' / f"{sample_name}.png")
            self._save_mask_image(background_mask_np[sample_idx, 0], output_path / 'background_mask' / f"{sample_name}.png")
            
            # Save CSV files for further analysis
            self._save_depth_csv(pred_np[sample_idx, 0], output_path / 'predicted_depth' / f"{sample_name}.csv")
            self._save_depth_csv(target_np[sample_idx, 0], output_path / 'ground_truth' / f"{sample_name}.csv")
            self._save_depth_csv(sparse_np[sample_idx, 0], output_path / 'sparse_input' / f"{sample_name}.csv")
            self._save_depth_csv(relative_np[sample_idx, 0], output_path / 'relative_input' / f"{sample_name}.csv")
            
            # Save mask CSV files
            self._save_depth_csv(sparse_mask_np[sample_idx, 0], output_path / 'sparse_mask' / f"{sample_name}.csv")
            self._save_depth_csv(shadow_mask_np[sample_idx, 0], output_path / 'shadow_mask' / f"{sample_name}.csv")
            self._save_depth_csv(background_mask_np[sample_idx, 0], output_path / 'background_mask' / f"{sample_name}.csv")
            
            # Create comparison visualization
            self._save_comparison_image(
                sparse_np[sample_idx, 0], grayscale_np[sample_idx, 0], relative_np[sample_idx, 0], 
                pred_np[sample_idx, 0], target_np[sample_idx, 0],
                sparse_mask_np[sample_idx, 0], shadow_mask_np[sample_idx, 0], background_mask_np[sample_idx, 0],
                output_path / 'comparison' / f"{sample_name}_comparison.png"
            )
    
    def compute_metrics(self, pred, target, shadow_mask=None, sparse_mask=None):
        """Compute evaluation metrics - same as training script"""
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
    
    def run_inference(self, dataloader, output_dir, save_all_batches=False, max_batches=None):
        """Run inference on the provided dataloader"""
        output_path = self.create_output_directories(output_dir)
        
        total_metrics = {
            'mae': 0.0, 'mae_sparse': 0.0, 'mse': 0.0, 'mse_sparse': 0.0, 
            'rmse': 0.0, 'rmse_sparse': 0.0, 'rel_error': 0.0, 'rel_error_sparse': 0.0
        }
        
        self.logger.info(f"Starting inference on {len(dataloader)} batches")
        self.logger.info(f"Output directory: {output_path}")
        self.logger.info(f"Save all batches: {save_all_batches}")
        if max_batches:
            self.logger.info(f"Maximum batches to process: {max_batches}")
        
        pbar = tqdm(dataloader, desc='Running Inference')
        processed_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Check if we've reached the maximum number of batches
                if max_batches and processed_batches >= max_batches:
                    break
                
                # Move data to device
                sparse_input = batch['sparse_depth'].to(self.device)
                grayscale_input = batch['grayscale'].to(self.device)
                relative_input = batch['depth_anything_v2'].to(self.device)
                target = batch['gt_depth'].to(self.device)
                
                # Get masks
                mask_shadow = batch.get('shadow_mask', None)
                mask_sparse = batch.get('sparse_mask', None)
                if mask_shadow is not None:
                    mask_shadow = mask_shadow.to(self.device)
                if mask_sparse is not None:
                    mask_sparse = mask_sparse.to(self.device)
                
                # Forward pass
                pred = self.model(sparse_input, grayscale_input, relative_input)
                
                # Compute metrics
                metrics = self.compute_metrics(pred, target, mask_shadow, mask_sparse)
                
                # Update running averages
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
                
                # Save predictions based on configuration
                should_save = False
                if save_all_batches:
                    # Save all batches
                    should_save = True
                elif max_batches is not None:
                    # If max_batches is set, save first 10 batches for sample outputs
                    should_save = processed_batches < 10
                else:
                    # If max_batches is None (process all), save first 10 batches by default for sample outputs
                    should_save = processed_batches < 10
                
                if should_save:
                    self.save_prediction_batch(batch, pred, batch_idx, output_path)
                
                # Update progress bar
                pbar.set_postfix({
                    'mae': f'{metrics["mae"]:.6f}',
                    'rmse': f'{metrics["rmse"]:.6f}',
                    'rel_err': f'{metrics["rel_error"]:.6f}'
                })
                
                processed_batches += 1
        
        # Compute final averages
        num_batches = min(processed_batches, len(dataloader))
        avg_metrics = {key: val / num_batches for key, val in total_metrics.items()}
        
        # Log final results
        self.logger.info(f"\nInference completed on {num_batches} batches")
        self.logger.info("Final Metrics:")
        self.logger.info(f"  MAE: {avg_metrics['mae']:.6f}")
        self.logger.info(f"  MAE Sparse: {avg_metrics['mae_sparse']:.6f}")
        self.logger.info(f"  RMSE: {avg_metrics['rmse']:.6f}")
        self.logger.info(f"  RMSE Sparse: {avg_metrics['rmse_sparse']:.6f}")
        self.logger.info(f"  MSE: {avg_metrics['mse']:.6f}")
        self.logger.info(f"  MSE Sparse: {avg_metrics['mse_sparse']:.6f}")
        self.logger.info(f"  Relative Error: {avg_metrics['rel_error']:.6f}")
        self.logger.info(f"  Relative Error Sparse: {avg_metrics['rel_error_sparse']:.6f}")
        
        # Save metrics to file
        metrics_file = output_path / 'inference_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(avg_metrics, f, indent=2)
        self.logger.info(f"Metrics saved to: {metrics_file}")
        
        return avg_metrics


def create_inference_dataloader(dataset_path, config):
    """Create dataloader for inference using the same configuration as training"""
    # Get dataset statistics for normalization
    base_dataset = HDDDepthFusionDataset(dataset_path, normalize_depth=False)
    stats = base_dataset.get_depth_stats(num_samples=40)
    
    # Compute combined normalization parameters
    depth_min = min(stats['sparse_min'], stats['gt_min'])
    depth_max = max(stats['sparse_max'], stats['gt_max'])
    
    # Create dataloader (no augmentation for inference)
    dataloader = create_dataloader(
        dataset_path,
        split=config.get('inference_split', 'test'),  # Default to test split
        batch_size=config.get('inference_batch_size', config.get('batch_size', 4)),
        shuffle=False,  # No shuffling for inference
        num_workers=config.get('num_workers', 4),
        normalize_depth=False,
        depth_min=depth_min,
        depth_max=depth_max,
        train_on_shadow_regions=config.get('train_on_shadow_regions', False)
    )
    
    return dataloader, (depth_min, depth_max)


def load_config(config_path):
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = ['dataset_path', 'checkpoint_path', 'output_dir']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required field '{field}' not found in configuration file")
    
    # Set default values for optional fields
    config.setdefault('inference_split', 'test')
    config.setdefault('batch_size', 4)
    config.setdefault('num_workers', 4)
    config.setdefault('train_on_shadow_regions', False)
    config.setdefault('save_all_batches', False)
    config.setdefault('max_batches', None)
    config.setdefault('device', 'auto')
    config.setdefault('model_file', 'circularkernel_2_5')
    config.setdefault('use_emergency_fix', True)
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Run inference with HDD Depth Fusion Model using JSON config')
    parser.add_argument('--config', type=str, default='example_inference_config.json',
                       help='Path to JSON configuration file (default: example_inference_config.json)')
    parser.add_argument('--override-device', type=str, default=None,
                       help='Override device setting from config (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load configuration from JSON file
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Validate inputs
    if not os.path.exists(config['checkpoint_path']):
        raise FileNotFoundError(f"Checkpoint file not found: {config['checkpoint_path']}")
    if not os.path.exists(config['dataset_path']):
        raise FileNotFoundError(f"Dataset path not found: {config['dataset_path']}")
    
    # Set device (with optional override)
    if args.override_device:
        device = torch.device(args.override_device)
        print(f"Device overridden via command line: {device}")
    elif config['device'] != 'auto':
        device = torch.device(config['device'])
        print(f"Using device from config: {device}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Auto-detected device: {device}")
    
    # Initialize inference engine
    inference_engine = HDDInferenceEngine(
        checkpoint_path=config['checkpoint_path'], 
        model_file=config['model_file'],
        use_emergency_fix=config['use_emergency_fix'],
        device=device
    )
    
    # Update config with inference parameters
    inference_config = inference_engine.config.copy()
    inference_config['inference_split'] = config['inference_split']
    inference_config['inference_batch_size'] = config['batch_size']
    inference_config['num_workers'] = config['num_workers']
    inference_config['train_on_shadow_regions'] = config['train_on_shadow_regions']
    
    # Create dataloader
    print(f"Creating dataloader for split: {config['inference_split']}")
    dataloader, norm_params = create_inference_dataloader(config['dataset_path'], inference_config)
    print(f"Dataloader created with {len(dataloader)} batches")
    print(f"Normalization parameters: min={norm_params[0]:.6f}, max={norm_params[1]:.6f}")
    
    # Run inference
    metrics = inference_engine.run_inference(
        dataloader=dataloader,
        output_dir=config['output_dir'],
        save_all_batches=config['save_all_batches'],
        max_batches=config['max_batches']
    )
    
    print(f"\nInference completed successfully!")
    print(f"Results saved to: {config['output_dir']}")
    print(f"Final MAE: {metrics['mae']:.6f}")
    print(f"Final RMSE: {metrics['rmse']:.6f}")
    
    # Save the used configuration for reproducibility
    used_config_path = os.path.join(config['output_dir'], 'used_config.json')
    os.makedirs(config['output_dir'], exist_ok=True)
    with open(used_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Used configuration saved to: {used_config_path}")


if __name__ == "__main__":
    main()
