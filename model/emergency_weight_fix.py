import torch
import torch.nn as nn
import numpy as np

def emergency_fix_all_circular_weights(model):
    """Emergency fix for all circular convolution weight initialization"""
    
    print("=== Applying Emergency Weight Fix ===")
    
    fixed_count = 0
    for name, module in model.named_modules():
        # Check if this is a circular convolution
        if hasattr(module, 'w_transform_matrix') and hasattr(module, 'weight'):
            print(f"Fixing circular conv: {name}")
            
            try:
                # Get transformation matrix
                transform_matrix = module.w_transform_matrix
                
                # Method 1: Compute safe initialization scale
                # Find the maximum amplification factor
                eigenvals = torch.linalg.eigvals(transform_matrix)
                max_eigenval_mag = torch.abs(eigenvals).max().item()
                
                # Scale down initialization by this factor
                safe_scale = 0.01 / np.sqrt(max_eigenval_mag)  # Very conservative
                
                print(f"  Max eigenvalue magnitude: {max_eigenval_mag:.3f}")
                print(f"  Safe scale factor: {safe_scale:.6f}")
                
                with torch.no_grad():
                    # Apply safe initialization
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    std = safe_scale * np.sqrt(2.0 / fan_in)  # He initialization with scaling
                    
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                    
                    if hasattr(module, 'bias') and module.bias is not None:
                        bound = safe_scale / np.sqrt(fan_in)
                        nn.init.uniform_(module.bias, -bound, bound)
                    
                    print(f"  ✓ Applied safe initialization (std={std:.6f})")
                
                fixed_count += 1
                
            except Exception as e:
                print(f"  ⚠️  Could not analyze {name}: {e}")
                print(f"  Applying ultra-conservative fallback...")
                
                # Ultra-conservative fallback
                with torch.no_grad():
                    nn.init.normal_(module.weight, mean=0.0, std=0.0001)  # Extremely small
                    if hasattr(module, 'bias') and module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                
                print(f"  ✓ Applied ultra-conservative fallback")
                fixed_count += 1
    
    print(f"\n✓ Fixed {fixed_count} circular convolution layers")
    return model

def test_emergency_fix():
    """Test the emergency fix with multiple seeds"""
    
    print("=== Testing Emergency Fix ===\n")
    
    from circularkernel_7_6 import SparseDepthBranch, GrayscaleBranch, RelativeDepthBranch
    
    # Test the problematic seeds from your results
    problematic_seeds = [123, 456, 789, 999]
    
    for seed in problematic_seeds:
        print(f"Testing seed {seed} (was problematic):")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create branches
        sparse_branch = SparseDepthBranch()
        grayscale_branch = GrayscaleBranch() 
        relative_branch = RelativeDepthBranch()
        
        # Apply emergency fix
        sparse_branch = emergency_fix_all_circular_weights(sparse_branch)
        grayscale_branch = emergency_fix_all_circular_weights(grayscale_branch)
        relative_branch = emergency_fix_all_circular_weights(relative_branch)
        
        # Test with same input
        test_input = torch.randn(1, 1, 64, 64)  # Smaller for speed
        
        branches = {
            'sparse': sparse_branch,
            'grayscale': grayscale_branch,
            'relative': relative_branch
        }
        
        all_stable = True
        for name, branch in branches.items():
            try:
                output, _ = branch(test_input)
                loss = output.mean()
                
                if torch.isnan(loss):
                    print(f"  ❌ {name}: Still NaN after fix")
                    all_stable = False
                elif abs(loss.item()) > 1000:
                    print(f"  ⚠️  {name}: Large value {loss.item():.2f}")
                    all_stable = False
                else:
                    print(f"  ✓ {name}: {loss.item():.6f}")
                    
            except Exception as e:
                print(f"  ❌ {name}: Error - {e}")
                all_stable = False
        
        if all_stable:
            print(f"  🎉 Seed {seed} is now STABLE!")
        else:
            print(f"  😞 Seed {seed} still has issues")
        print()

def create_fully_fixed_model():
    """Create your full model with emergency fix applied"""
    
    print("=== Creating Fully Fixed Model ===\n")
    
    from circularkernel_7_6 import SqueezeUNetFusionCircular
    
    # Use a known good seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create model
    model = SqueezeUNetFusionCircular(base_channels=64, dropout_prob=0.0)
    
    # Apply emergency fix
    model = emergency_fix_all_circular_weights(model)
    
    # Test full model
    print("Testing full fixed model:")
    test_inputs = [torch.randn(1, 1, 64, 64) for _ in range(3)]  # Smaller for speed
    
    try:
        output = model(*test_inputs)
        loss = output.mean()
        
        if torch.isnan(loss):
            print("❌ Full model still produces NaN")
        else:
            print(f"✓ Full model works: {loss.item():.6f}")
            print("🎉 Model is ready for training!")
            
    except Exception as e:
        print(f"❌ Full model error: {e}")
    
    return model

if __name__ == "__main__":
    test_emergency_fix()
    create_fully_fixed_model()
