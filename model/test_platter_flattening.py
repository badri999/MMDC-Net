"""
Test script for automatic platter detection and RANSAC flattening
Run this on your existing model predictions to test the post-processing
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
from platter_detection import detect_platter_auto, PlatterDetector
from ransac_postprocessing import flatten_platter_ransac, process_hdd_sample

def test_single_sample(grayscale_path, predicted_depth_path, sparse_depth_path=None, 
                      sparse_mask_path=None):
    """
    Test the complete pipeline on a single sample
    
    Args:
        grayscale_path: Path to grayscale image (.png)
        predicted_depth_path: Path to your model's predicted depth (.csv)
        sparse_depth_path: Optional path to sparse depth (.csv)
        sparse_mask_path: Optional path to sparse mask (.png)
    """
    print(f"Testing sample: {Path(grayscale_path).stem}")
    print("=" * 50)
    
    # Load data
    grayscale = cv2.imread(grayscale_path, cv2.IMREAD_GRAYSCALE)
    predicted_depth = np.loadtxt(predicted_depth_path, delimiter=',')
    
    sparse_depth = None
    sparse_mask = None
    if sparse_depth_path and os.path.exists(sparse_depth_path):
        sparse_depth = np.loadtxt(sparse_depth_path, delimiter=',')
    
    if sparse_mask_path and os.path.exists(sparse_mask_path):
        sparse_mask = cv2.imread(sparse_mask_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"Loaded data shapes:")
    print(f"  Grayscale: {grayscale.shape}")
    print(f"  Predicted depth: {predicted_depth.shape}")
    if sparse_depth is not None:
        print(f"  Sparse depth: {sparse_depth.shape}")
    if sparse_mask is not None:
        print(f"  Sparse mask: {sparse_mask.shape}")
    
    # Step 1: Detect platter
    print(f"\n=== Step 1: Platter Detection ===")
    platter_mask = detect_platter_auto(
        grayscale, 
        sparse_depth=sparse_depth, 
        sparse_mask=sparse_mask,
        min_radius=60,  # Adjust based on your HDD size
        max_radius=250,
        visualize=True
    )
    
    platter_area = np.sum(platter_mask > 0)
    total_area = grayscale.shape[0] * grayscale.shape[1]
    print(f"Detected platter area: {platter_area} pixels ({platter_area/total_area:.1%} of image)")
    
    if platter_area == 0:
        print("Warning: No platter detected! Try adjusting detection parameters.")
        return None, None, None
    
    # Step 2: Analyze depth variation in detected platter
    print(f"\n=== Step 2: Platter Depth Analysis ===")
    platter_depths = predicted_depth[platter_mask > 0]
    print(f"Platter depth statistics:")
    print(f"  Mean: {np.mean(platter_depths):.6f}")
    print(f"  Std: {np.std(platter_depths):.6f}")
    print(f"  Min: {np.min(platter_depths):.6f}")
    print(f"  Max: {np.max(platter_depths):.6f}")
    print(f"  Range: {np.max(platter_depths) - np.min(platter_depths):.6f}")
    
    # Step 3: RANSAC flattening
    print(f"\n=== Step 3: RANSAC Flattening ===")
    
    # Try different thresholds based on current variation
    current_std = np.std(platter_depths)
    suggested_threshold = min(0.001, current_std * 0.5)  # Start with half the current std
    
    print(f"Current platter std: {current_std:.6f}")
    print(f"Suggested RANSAC threshold: {suggested_threshold:.6f}")
    
    flattened_depth, diagnostics = flatten_platter_ransac(
        predicted_depth, 
        platter_mask,
        residual_threshold=suggested_threshold,
        max_trials=200,
        visualize=True,
        compare_methods=True
    )
    
    # Step 4: Results analysis
    print(f"\n=== Step 4: Results Analysis ===")
    if diagnostics.get('success'):
        flattened_platter_depths = flattened_depth[platter_mask > 0]
        improvement = np.std(platter_depths) - np.std(flattened_platter_depths)
        
        print(f"Flattening results:")
        print(f"  Original platter std: {np.std(platter_depths):.6f}")
        print(f"  Flattened platter std: {np.std(flattened_platter_depths):.6f}")
        print(f"  Improvement: {improvement:.6f} ({improvement/np.std(platter_depths)*100:.1f}% reduction)")
        print(f"  Mean change: {np.mean(flattened_platter_depths - platter_depths):.6f}")
    
    return flattened_depth, platter_mask, diagnostics

def test_multiple_samples(dataset_root, sample_prefixes=['sample_001', 'sample_002'], 
                         results_dir='test_flattening_results'):
    """
    Test on multiple samples from your dataset
    
    Args:
        dataset_root: Path to your dataset
        sample_prefixes: List of sample names to test (without extensions)
        results_dir: Where to save results
    """
    os.makedirs(results_dir, exist_ok=True)
    
    results_summary = []
    
    for sample_name in sample_prefixes:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {sample_name}")
        print(f"{'='*60}")
        
        # Construct file paths based on your dataset structure
        grayscale_path = f"{dataset_root}/grayscale/{sample_name}.png"
        predicted_depth_path = f"{dataset_root}/predicted_depth/{sample_name}.csv"  # Your model outputs
        sparse_depth_path = f"{dataset_root}/sparse_depth_z/{sample_name}.csv"
        sparse_mask_path = f"{dataset_root}/sparse_mask/{sample_name}.png"
        
        # Check if files exist
        if not os.path.exists(grayscale_path):
            print(f"Skipping {sample_name}: grayscale not found at {grayscale_path}")
            continue
        
        if not os.path.exists(predicted_depth_path):
            print(f"Skipping {sample_name}: predicted depth not found at {predicted_depth_path}")
            continue
        
        try:
            flattened_depth, platter_mask, diagnostics = test_single_sample(
                grayscale_path, predicted_depth_path, 
                sparse_depth_path if os.path.exists(sparse_depth_path) else None,
                sparse_mask_path if os.path.exists(sparse_mask_path) else None
            )
            
            if flattened_depth is not None:
                # Save results
                output_path = f"{results_dir}/{sample_name}_flattened.csv"
                np.savetxt(output_path, flattened_depth, delimiter=',', fmt='%.6f')
                
                mask_path = f"{results_dir}/{sample_name}_platter_mask.png"
                cv2.imwrite(mask_path, platter_mask)
                
                # Record summary
                if diagnostics.get('success'):
                    results_summary.append({
                        'sample': sample_name,
                        'success': True,
                        'inlier_ratio': diagnostics.get('inlier_ratio', 0),
                        'mean_residual': diagnostics.get('mean_residual', 0),
                        'std_residual': diagnostics.get('std_residual', 0)
                    })
                else:
                    results_summary.append({
                        'sample': sample_name,
                        'success': False,
                        'error': diagnostics.get('error', 'Unknown')
                    })
        
        except Exception as e:
            print(f"Error processing {sample_name}: {e}")
            results_summary.append({
                'sample': sample_name,
                'success': False,
                'error': str(e)
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY RESULTS")
    print(f"{'='*60}")
    
    successful = [r for r in results_summary if r['success']]
    failed = [r for r in results_summary if not r['success']]
    
    print(f"Successful: {len(successful)}/{len(results_summary)}")
    
    if successful:
        avg_inlier_ratio = np.mean([r['inlier_ratio'] for r in successful])
        avg_residual = np.mean([r['mean_residual'] for r in successful])
        print(f"Average inlier ratio: {avg_inlier_ratio:.1%}")
        print(f"Average residual: {avg_residual:.6f}")
    
    if failed:
        print(f"\nFailed samples:")
        for r in failed:
            print(f"  {r['sample']}: {r['error']}")
    
    return results_summary

def quick_test_with_synthetic_data():
    """
    Quick test with synthetic data if you don't have real data ready
    """
    print("Creating synthetic test data...")
    
    # Create synthetic HDD-like image
    img_size = (512, 512)
    grayscale = np.zeros(img_size, dtype=np.uint8)
    
    # Add platter (large circle)
    center = (256, 256)
    radius = 150
    cv2.circle(grayscale, center, radius, 128, -1)  # Filled circle
    
    # Add some other components
    cv2.circle(grayscale, (256, 256), 20, 64, -1)  # Spindle
    cv2.rectangle(grayscale, (100, 100), (150, 400), 192, -1)  # Actuator arm
    
    # Add noise
    noise = np.random.normal(0, 10, img_size).astype(np.int16)
    grayscale = np.clip(grayscale.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Create synthetic depth with platter artifacts
    base_depth = 10.0  # mm
    depth_map = np.full(img_size, base_depth, dtype=np.float32)
    
    # Add platter region with slight artifacts
    y, x = np.ogrid[:img_size[0], :img_size[1]]
    platter_region = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    
    # Add artificial artifacts to platter (wavy pattern)
    artifact_magnitude = 0.002  # 2 micrometers
    wave_x = np.sin(2 * np.pi * x / 50) * artifact_magnitude
    wave_y = np.cos(2 * np.pi * y / 40) * artifact_magnitude
    depth_map[platter_region] += (wave_x + wave_y)[platter_region]
    
    print("Testing on synthetic data...")
    flattened_depth, platter_mask, diagnostics = test_single_sample(
        grayscale, depth_map, None, None
    )
    
    return grayscale, depth_map, flattened_depth, platter_mask

if __name__ == "__main__":
    # Example usage - modify paths for your data
    
    # Option 1: Test with synthetic data
    print("=== Testing with synthetic data ===")
    quick_test_with_synthetic_data()
    
    # Option 2: Test single real sample (uncomment and modify paths)
    # test_single_sample(
    #     grayscale_path="path/to/your/grayscale/sample_001.png",
    #     predicted_depth_path="path/to/your/predictions/sample_001.csv"
    # )
    
    # Option 3: Test multiple samples (uncomment and modify paths)
    # test_multiple_samples(
    #     dataset_root="path/to/your/dataset",
    #     sample_prefixes=['sample_001', 'sample_002', 'sample_003'],
    #     results_dir="flattening_test_results"
    # )