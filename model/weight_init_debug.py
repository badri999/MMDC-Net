import torch
import torch.nn as nn
import numpy as np

def debug_weight_initialization():
    """Debug weight initialization differences between branches"""
    
    print("=== Weight Initialization Debug ===\n")
    
    from conv.CircleConv3x3 import CircleConv3x3
    from circularkernel_7_6 import SparseDepthBranch, GrayscaleBranch, RelativeDepthBranch
    
    # Set different seeds to see the effect
    seeds = [42, 123, 456, 789, 999]
    
    for seed in seeds:
        print(f"Testing with seed {seed}:")
        torch.manual_seed(seed)
        np.random.seed(seed)
        # Create branches
        sparse_branch = SparseDepthBranch()
        torch.manual_seed(seed)  # Reset to same seed
        np.random.seed(seed)
        grayscale_branch = GrayscaleBranch()
        torch.manual_seed(seed)  # Reset to same seed
        np.random.seed(seed)
        relative_branch = RelativeDepthBranch()
        
        branches = {
            'sparse': sparse_branch,
            'grayscale': grayscale_branch, 
            'relative': relative_branch
        }
        
        # Test each branch
        test_input = torch.randn(1, 1, 64, 64)  # Smaller for speed
        
        for name, branch in branches.items():
            try:
                output, _ = branch(test_input)
                loss = output.mean()
                
                if torch.isnan(loss):
                    print(f"  ❌ {name}: NaN (seed {seed})")
                else:
                    print(f"  ✓ {name}: {loss.item():.6f}")
                    
            except Exception as e:
                print(f"  ❌ {name}: Error - {e}")
        print()

def analyze_circular_conv_weights():
    """Analyze how circular transformation affects different weight initializations"""
    
    print("=== Circular Conv Weight Analysis ===\n")
    
    from conv.CircleConv3x3 import CircleConv3x3
    
    # Test multiple random initializations
    for trial in range(10):
        print(f"Trial {trial + 1}:")
        
        # Create circular conv with random weights
        conv = CircleConv3x3(1, 64, kernel_size=3, padding=1)
        
        # Get original and transformed weights
        original_weights = conv.weight.data.clone()
        
        # Simulate the transformation that happens in forward pass
        w_flat = original_weights.view(-1, 9)
        transformed_flat = w_flat @ conv.w_transform_matrix
        transformed_weights = transformed_flat.view_as(original_weights)
        
        # Analyze statistics
        orig_min, orig_max = original_weights.min().item(), original_weights.max().item()
        orig_std = original_weights.std().item()
        
        trans_min, trans_max = transformed_weights.min().item(), transformed_weights.max().item()
        trans_std = transformed_weights.std().item()
        
        amplification = trans_std / orig_std
        
        print(f"  Original: min={orig_min:.4f}, max={orig_max:.4f}, std={orig_std:.4f}")
        print(f"  Transformed: min={trans_min:.4f}, max={trans_max:.4f}, std={trans_std:.4f}")
        print(f"  Amplification: {amplification:.2f}x")
        
        # Check for extreme values
        if abs(trans_max) > 10 or abs(trans_min) > 10:
            print(f"  ⚠️  EXTREME VALUES detected!")
        if amplification > 5:
            print(f"  ⚠️  HIGH AMPLIFICATION detected!")
        
        # Test forward pass
        test_input = torch.randn(1, 1, 32, 32) * 0.1
        try:
            output = conv(test_input)
            if torch.isnan(output).any():
                print(f"  ❌ NaN in output!")
            elif output.abs().max() > 100:
                print(f"  ⚠️  Very large output: {output.abs().max().item():.2f}")
            else:
                print(f"  ✓ Output OK: range [{output.min().item():.3f}, {output.max().item():.3f}]")
        except:
            print(f"  ❌ Forward pass failed!")
        
        print()

def find_stable_initialization():
    """Find initialization that works consistently"""
    
    print("=== Finding Stable Initialization ===\n")
    
    from conv.CircleConv3x3 import CircleConv3x3
    
    # Test different initialization strategies
    strategies = [
        ("Default Kaiming", lambda conv: None),  # Do nothing, use default
        ("Scaled Kaiming", lambda conv: conv.weight.data.mul_(0.1)),
        ("Xavier Normal", lambda conv: nn.init.xavier_normal_(conv.weight)),
        ("Xavier Uniform", lambda conv: nn.init.xavier_uniform_(conv.weight)),
        ("Small Normal", lambda conv: nn.init.normal_(conv.weight, 0, 0.01)),
        ("Orthogonal", lambda conv: nn.init.orthogonal_(conv.weight)),
    ]
    
    test_input = torch.randn(1, 1, 32, 32)
    
    for name, init_fn in strategies:
        print(f"Testing {name}:")
        
        success_count = 0
        total_trials = 20
        
        for trial in range(total_trials):
            try:
                # Create conv and apply initialization
                conv = CircleConv3x3(1, 32, kernel_size=3, padding=1)
                if init_fn:
                    init_fn(conv)
                
                # Test forward pass
                output = conv(test_input)
                
                if not torch.isnan(output).any() and output.abs().max() < 100:
                    success_count += 1
                    
            except:
                pass  # Count as failure
        
        success_rate = success_count / total_trials * 100
        print(f"  Success rate: {success_rate:.1f}% ({success_count}/{total_trials})")
        
        if success_rate >= 90:
            print(f"  ✓ {name} is stable!")
        elif success_rate >= 70:
            print(f"  ⚠️  {name} is mostly stable")
        else:
            print(f"  ❌ {name} is unstable")
        print()

if __name__ == "__main__":
    debug_weight_initialization()
    analyze_circular_conv_weights()
    find_stable_initialization()
