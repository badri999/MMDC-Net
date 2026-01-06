# Multi-Modal Depth Completion with Circular Convolutions

Official implementation of the depth completion framework published in Optics and Lasers in Engineering.

**Paper:** [https://doi.org/10.1016/j.optlaseng.2025.109587](https://doi.org/10.1016/j.optlaseng.2025.109587)

## Overview

This repository implements a multi-modal depth completion network that fuses sparse depth measurements, grayscale imagery, and monocular depth estimates to generate dense depth maps. The model employs circular convolutions and a multi-branch architecture optimized for efficient inference (~2.5M parameters).

## Key Features

- **Circular Convolution Kernels**: Novel convolution operation for improved spatial feature extraction
- **Multi-Modal Fusion**: Combines three complementary modalities (sparse depth, grayscale, relative depth)
- **Lightweight Architecture**: SqueezeNet-inspired design with Fire modules for efficiency
- **Gated Convolutions**: Adaptive feature gating for improved fusion quality
- **Squeeze-and-Excitation**: Channel attention mechanism for feature refinement
- **Flexible Loss Functions**: Supports L1, L2, SSIM, and Laplacian pyramid losses with custom weighting

## Architecture

The model consists of four main branches:

1. **Sparse Depth Branch**: Processes sparse structured light measurements
2. **Grayscale Branch**: Extracts texture and edge information from intensity images
3. **Relative Depth Branch**: Incorporates monocular depth priors (Depth-Anything-v2)
4. **Fusion Branch**: Combines multi-scale features with gated convolutions and SE blocks

Each parallel branch uses an encoder-decoder architecture with skip connections. The fusion branch integrates outputs and encoder features from all three modalities to predict dense depth maps.

## Installation

```bash
# Clone the repository


# Create virtual environment


# Install dependencies in environment.yml

```

## Dataset Structure

Organize your dataset as follows:

```
dataset_root/
├── 
│   ├── sparse_depth_z/       # Sparse depth measurements (CSV)
│   ├── sparse_mask/           # Validity masks (PNG)
│   ├── grayscale/             # Grayscale images (PNG)
│   ├── depth_anything_v2_map_1512/  # Monocular depth (PNG)
│   ├── gt_depth/              # Ground truth depth (CSV)
│   ├── shadow_mask_gt/        # Shadow region masks (PNG)
│   └── background_mask_gt/    # Background masks (PNG)
├── valid/
│   └── [same structure as train]
└── test/
    └── [same structure as train]
```

## Usage

### Training

```bash
python hdd_training_script_2_5_ck.py \
    --config config.json \
    --train-on-shadow-regions
```

### Resume Training

```bash
python hdd_training_script_2_5_ck.py \
    --config config.json \
    --resume checkpoints/best-epoch50.pth \
    --resume-lr 0.0001 \
    --reset-best-loss
```

### Configuration

Use existing json files in repo


### Inference

Please run inference_script.py
```

## Model Components

### Fire Module
Efficient squeeze-expand module with 1×1 squeeze convolution followed by parallel 1×1 and 3×3 expand convolutions (with circular kernels).

### Gated Circular Convolution
Feature gating mechanism: `output = φ(features) ⊙ σ(gating)` where both feature and gating branches use circular convolutions.

### Circular Convolution (CircleConv3x3)
Custom convolution operation with circular transformation matrix for enhanced spatial feature learning.

## Training Features

- **Multi-Loss Training**: Combine L1, L2, SSIM, and Laplacian losses with custom weights
- **Shadow Region Handling**: Optional masking of shadow regions during training
- **Data Augmentation**: Built-in augmentation pipeline (see `hdd_data_augmenter.py`)
- **WandB Integration**: Automatic experiment tracking and visualization
- **Checkpoint Management**: Automatic best model saving and periodic checkpoints
- **Learning Rate Scheduling**: Support for step, cosine, and plateau schedulers

## Loss Functions

The framework supports multiple loss functions defined in `compute_loss.py`:

- **L1 Loss**: Mean Absolute Error with optional masking
- **L2 Loss**: Mean Squared Error with optional masking
- **SSIM Loss**: Structural similarity with cropping and masking
- **Laplacian Loss**: Multi-scale edge-aware loss using Laplacian pyramids

Each loss can be independently weighted for shadow and sparse regions.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{balasubramaniam2026application,
    title={Application-driven multi-modal depth completion in fringe projection profilometry},
    author={Balasubramaniam, Badrinath and Suresh, Vignesh and Cheng, Yang and Li, Jiaqiong and Li, Beiwen},
    journal={Optics and Lasers in Engineering},
    year={2026},
    doi={10.1016/j.optlaseng.2025.109587},
    url={https://doi.org/10.1016/j.optlaseng.2025.109587}
}
```

## License
GPL-3.0 license


## Acknowledgments

- Depth-Anything-v2 for monocular depth estimation
- PyTorch-MSSSIM for SSIM loss implementation
- Circular Convolution Implementation by https://github.com/JHL-HUST/CircularKernel  

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
