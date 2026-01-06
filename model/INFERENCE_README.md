# HDD Depth Fusion Model - Inference Script

This directory contains the inference script for the HDD Depth Fusion Model using circular convolutions (circularkernel_2_5).

## Overview

The `inference_script_2_5_ck.py` script loads a trained checkpoint and runs inference on test data, saving predictions in exactly the same format as the training script's `_save_selected_predictions` function.

## Features

- Loads any checkpoint saved by the training script
- Supports both single-GPU and multi-GPU trained models
- Saves predictions in identical format to training script:
  - PNG images with colormaps for visualization
  - CSV files for numerical analysis
  - Comparison visualizations
  - All masks and inputs
- Computes comprehensive evaluation metrics
- Configurable batch processing and output saving

## Usage

### Basic Usage

```bash
python model/inference_script_2_5_ck.py \
    --checkpoint /path/to/your/checkpoint.pth \
    --dataset-path /path/to/test/dataset \
    --output-dir /path/to/output/results
```

### Advanced Usage

```bash
python model/inference_script_2_5_ck.py \
    --checkpoint /path/to/your/best-epoch50.pth \
    --dataset-path /path/to/real_dataset_complete \
    --output-dir ./inference_results \
    --save-all \
    --split test \
    --batch-size 8 \
    --max-batches 100 \
    --device cuda
```

## Arguments

### Required Arguments
- `--checkpoint`: Path to the checkpoint file (.pth)
- `--dataset-path`: Path to the dataset directory
- `--output-dir`: Directory where results will be saved

### Optional Arguments
- `--save-all`: Save predictions for all batches (default: save first 10 only)
- `--max-batches INT`: Maximum number of batches to process (default: all)
- `--batch-size INT`: Override batch size for inference (default: use checkpoint config)
- `--split {train,valid,test}`: Dataset split to use (default: test)
- `--device {cuda,cpu}`: Device to use (default: auto-detect)

## Dataset Structure

The script expects the same dataset structure as the training script:

```
dataset_path/
├── test/  (or train/, valid/ depending on --split)
│   ├── sparse_mask/
│   ├── sparse_depth_z/
│   ├── grayscale/
│   ├── depth_anything_v2_map_1512/
│   ├── gt_depth/
│   ├── shadow_mask_gt/
│   └── background_mask_gt/
```

## Output Structure

The script creates the following output structure (identical to training script):

```
output_dir/
├── predicted_depth/
│   ├── *.png  (visualizations)
│   └── *.csv  (raw data)
├── ground_truth/
│   ├── *.png
│   └── *.csv
├── sparse_input/
│   ├── *.png
│   └── *.csv
├── grayscale_input/
│   ├── *.png
│   └── *.csv
├── relative_input/
│   ├── *.png
│   └── *.csv
├── sparse_mask/
│   ├── *.png
│   └── *.csv
├── shadow_mask/
│   ├── *.png
│   └── *.csv
├── background_mask/
│   ├── *.png
│   └── *.csv
├── comparison/
│   └── *_comparison.png  (3x3 grid visualizations)
└── inference_metrics.json  (evaluation metrics)
```

## Key Features

### 1. Exact Format Compatibility
The inference script uses the exact same saving functions as the training script:
- `_save_depth_image()` - saves depth maps with COLORMAP_JET
- `_save_mask_image()` - saves binary masks
- `_save_depth_csv()` - saves raw numerical data
- `_save_comparison_image()` - creates 3x3 comparison grids

### 2. Comprehensive Metrics
Calculates the same metrics as training:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error) 
- RMSE (Root Mean Squared Error)
- Relative Error
- All metrics computed both for shadow-excluded and sparse regions

### 3. Checkpoint Compatibility
- Handles both single-GPU and multi-GPU trained models
- Automatically removes `module.` prefixes from DataParallel models
- Extracts training configuration from checkpoint
- Applies the same weight fixes as training

### 4. Memory Efficient
- Processes data in batches
- Configurable batch size
- Optional early stopping with `--max-batches`
- GPU memory cleanup

## Examples

### Example 1: Quick Test Run
```bash
python model/inference_script_2_5_ck.py \
    --checkpoint checkpoints/circularkernel_2_5_realdata/best-epoch25.pth \
    --dataset-path /data/real_dataset_complete \
    --output-dir ./quick_test \
    --max-batches 5
```

### Example 2: Full Evaluation
```bash
python model/inference_script_2_5_ck.py \
    --checkpoint checkpoints/circularkernel_2_5_realdata/best-epoch50.pth \
    --dataset-path /data/real_dataset_complete \
    --output-dir ./full_evaluation \
    --save-all \
    --split test
```

### Example 3: Validation Set Analysis
```bash
python model/inference_script_2_5_ck.py \
    --checkpoint checkpoints/circularkernel_2_5_realdata/checkpoint_epoch_30.pth \
    --dataset-path /data/real_dataset_complete \
    --output-dir ./validation_analysis \
    --split valid \
    --batch-size 2
```

## Output Files

### Metrics File (inference_metrics.json)
```json
{
  "mae": 0.012345,
  "mae_sparse": 0.008901,
  "mse": 0.000234,
  "mse_sparse": 0.000156,
  "rmse": 0.015321,
  "rmse_sparse": 0.012487,
  "rel_error": 0.028456,
  "rel_error_sparse": 0.019234
}
```

### Sample Naming Convention
Files are named as: `inference_batch_{batch_num}_sample_{sample_num}.{ext}`

Example: `inference_batch_1_sample_1.png`, `inference_batch_1_sample_1.csv`

## Notes

1. The script automatically detects and uses the same normalization parameters as training
2. Dropout is set to 0.0 for inference (even if checkpoint was trained with dropout)
3. Model is automatically set to evaluation mode
4. Supports both CPU and GPU inference
5. Progress bars show real-time metrics during inference
6. All tensor operations are performed with `torch.no_grad()` for memory efficiency

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `--batch-size` or use `--max-batches`
2. **Checkpoint Loading Error**: Ensure checkpoint was saved by the same model architecture
3. **Dataset Path Error**: Verify the dataset follows the expected directory structure
4. **Permission Error**: Ensure write permissions for `--output-dir`

### Performance Tips

1. Use `--max-batches` for quick testing
2. Increase `--batch-size` for faster processing (if GPU memory allows)
3. Use `--save-all` only when you need all predictions saved
4. For large datasets, process in chunks using `--max-batches`
