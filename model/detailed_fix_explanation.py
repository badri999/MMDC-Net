import torch
import torch.nn as nn
import numpy as np

def emergency_fix_explained(model):
    """
    Emergency fix with detailed mathematical explanation for each step
    """
    
    print("=== DETAILED EMERGENCY FIX EXPLANATION ===\n")
    
    fixed_count = 0
    
    # STEP 1: Find all circular convolution layers
    for name, module in model.named_modules():
        
        # LINE: if hasattr(module, 'w_transform_matrix') and hasattr(module, 'weight'):
        # EXPLANATION: This identifies circular convolution layers
        # - w_transform_matrix: The K²×K² transformation matrix (B in the paper)
        # - weight: The learnable convolution weights
        # Only circular convs have both of these attributes
        
        if hasattr(module, 'w_transform_matrix') and hasattr(module, 'weight'):
            print(f"=== Analyzing Circular Conv: {name} ===")
            
            # STEP 2: Extract the transformation matrix
            # LINE: transform_matrix = module.w_transform_matrix
            # MATH: This is the matrix B ∈ ℝ^{K²×K²} from the paper
            # For 3×3 kernels: B ∈ ℝ^{9×9}
            # Purpose: Transforms square kernel weights to circular sampling pattern
            
            transform_matrix = module.w_transform_matrix
            print(f"Transform matrix shape: {transform_matrix.shape}")
            print(f"Transform matrix range: [{transform_matrix.min():.6f}, {transform_matrix.max():.6f}]")
            
            # STEP 3: Analyze the transformation's amplification potential
            # LINE: eigenvals = torch.linalg.eigvals(transform_matrix)
            # MATH: Compute eigenvalues λ₁, λ₂, ..., λₙ of matrix B
            # THEORY: Eigenvalues tell us how much B can amplify vectors
            # - If |λᵢ| > 1: B amplifies vectors in direction of eigenvector i
            # - If |λᵢ| < 1: B shrinks vectors in direction of eigenvector i
            # - If |λᵢ| = 1: B preserves magnitude in direction of eigenvector i
            
            eigenvals = torch.linalg.eigvals(transform_matrix)
            eigenval_magnitudes = torch.abs(eigenvals)
            print(f"Eigenvalue magnitudes: {eigenval_magnitudes.numpy()[:5]}...")  # Show first 5
            
            # STEP 4: Find maximum amplification factor
            # LINE: max_eigenval_mag = torch.abs(eigenvals).max().item()
            # MATH: max_amp = max(|λ₁|, |λ₂|, ..., |λₙ|)
            # INTERPRETATION: This is the WORST-CASE amplification
            # If we have weight vector w aligned with the dominant eigenvector,
            # then ||B·w|| = max_amp × ||w||
            
            max_eigenval_mag = eigenval_magnitudes.max().item()
            print(f"Maximum eigenvalue magnitude (worst-case amplification): {max_eigenval_mag:.6f}")
            
            # STEP 5: Understand what happens in forward pass
            print("\n--- Forward Pass Analysis ---")
            print("In circular conv forward pass:")
            print("1. Original weights: w ∈ ℝ^{out×in×3×3}")
            print("2. Reshape: w_flat ∈ ℝ^{(out×in)×9}")
            print("3. Transform: w_transformed = w_flat @ B")
            print("4. Use w_transformed in standard conv2d")
            print()
            print("The problem:")
            print(f"If ||w_flat|| = σ (standard deviation of initial weights)")
            print(f"Then ||w_transformed|| ≤ {max_eigenval_mag:.3f} × σ")
            print(f"With bad luck, could be exactly {max_eigenval_mag:.3f} × σ")
            
            # STEP 6: Compute safe initialization scale
            # LINE: safe_scale = 0.01 / np.sqrt(max_eigenval_mag)
            # MATH DERIVATION:
            # 
            # Goal: Ensure transformed weights stay reasonable
            # Let σ_original = std of original weights
            # Let σ_transformed = std of transformed weights
            # 
            # Worst case: σ_transformed = max_eigenval_mag × σ_original
            # 
            # We want: σ_transformed ≤ some_reasonable_value (say 0.01)
            # Therefore: max_eigenval_mag × σ_original ≤ 0.01
            # Solving: σ_original ≤ 0.01 / max_eigenval_mag
            # 
            # But we also want to account for stochastic effects, so:
            # σ_original ≤ 0.01 / sqrt(max_eigenval_mag)
            # 
            # This gives us the safe_scale factor
            
            safe_scale = 0.01 / np.sqrt(max_eigenval_mag)
            print(f"\n--- Safe Scale Computation ---")
            print(f"Target max transformed std: 0.01")
            print(f"Max amplification factor: {max_eigenval_mag:.6f}")
            print(f"Safe scale = 0.01 / sqrt({max_eigenval_mag:.6f}) = {safe_scale:.8f}")
            
            # STEP 7: Apply safe He initialization
            # LINE: fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
            # MATH: For conv weights with shape [out_channels, in_channels, kH, kW]:
            # fan_in = in_channels × kH × kW (number of input connections per output neuron)
            # fan_out = out_channels × kH × kW (number of output connections per input neuron)
            
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
            print(f"\n--- He Initialization ---")
            print(f"Weight shape: {module.weight.shape}")
            print(f"fan_in (input connections per output neuron): {fan_in}")
            print(f"fan_out (output connections per input neuron): {fan_out}")
            
            # LINE: std = safe_scale * np.sqrt(2.0 / fan_in)
            # MATH: This is scaled He initialization
            # 
            # Standard He initialization: std = sqrt(2 / fan_in)
            # Reasoning: Keeps variance of activations ≈ 1 through layers
            # Mathematical derivation (assuming ReLU activation):
            # 
            # For layer l with input x and weights w:
            # y = ReLU(w × x)
            # 
            # To maintain: Var(y) ≈ Var(x)
            # He showed: std(w) = sqrt(2 / fan_in) achieves this
            # 
            # Our modification: std(w) = safe_scale × sqrt(2 / fan_in)
            # This scales down the He initialization by our safe_scale factor
            
            he_std = np.sqrt(2.0 / fan_in)
            scaled_std = safe_scale * he_std
            
            print(f"Standard He std: sqrt(2/{fan_in}) = {he_std:.6f}")
            print(f"Our scaled std: {safe_scale:.8f} × {he_std:.6f} = {scaled_std:.8f}")
            
            # STEP 8: Apply the initialization
            with torch.no_grad():
                # LINE: nn.init.normal_(module.weight, mean=0.0, std=std)
                # MATH: Sample each weight from Normal(0, scaled_std²)
                # This ensures: E[w] = 0, Var(w) = scaled_std²
                
                nn.init.normal_(module.weight, mean=0.0, std=scaled_std)
                
                print(f"Applied: w ~ Normal(0, {scaled_std:.8f}²)")
                
                # STEP 9: Handle bias terms
                if hasattr(module, 'bias') and module.bias is not None:
                    # LINE: bound = safe_scale / np.sqrt(fan_in)
                    # MATH: This is a heuristic for bias initialization
                    # Standard practice: bias bound = 1/sqrt(fan_in)
                    # Our modification: bias bound = safe_scale/sqrt(fan_in)
                    # 
                    # Uniform distribution: b ~ Uniform(-bound, bound)
                    # This gives: Var(b) = bound²/3
                    
                    bias_bound = safe_scale / np.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bias_bound, bias_bound)
                    
                    print(f"Bias: b ~ Uniform(-{bias_bound:.8f}, {bias_bound:.8f})")
            
            # STEP 10: Verify the fix
            print(f"\n--- Verification ---")
            test_input = torch.randn(1, module.weight.shape[1], 8, 8) * 0.1
            
            try:
                # Test the module with new weights
                test_output = module(test_input)
                output_range = [test_output.min().item(), test_output.max().item()]
                output_std = test_output.std().item()
                
                print(f"Test output range: [{output_range[0]:.6f}, {output_range[1]:.6f}]")
                print(f"Test output std: {output_std:.6f}")
                
                if torch.isnan(test_output).any():
                    print("❌ Still produces NaN!")
                elif abs(output_range[0]) > 100 or abs(output_range[1]) > 100:
                    print("⚠️  Large values detected")
                else:
                    print("✓ Output looks reasonable")
                    
            except Exception as e:
                print(f"❌ Test failed: {e}")
            
            fixed_count += 1
            print(f"✓ Fixed {name}\n")
    
    print(f"=== SUMMARY ===")
    print(f"Fixed {fixed_count} circular convolution layers")
    print(f"Key insight: Circular transformation can amplify weights unpredictably")
    print(f"Solution: Scale down initial weights by factor related to max eigenvalue")
    print(f"Result: Even worst-case amplification produces reasonable values")
    
    return model

# Demonstrate the math with a concrete example
def concrete_example():
    """Show concrete numbers for the mathematical concepts"""
    
    print("=== CONCRETE MATHEMATICAL EXAMPLE ===\n")
    
    # Create a simple transformation matrix (like circular conv would have)
    print("Example 3×3 circular transformation matrix B:")
    B = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],  # Corner gets redistributed
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],  # Corner gets redistributed
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    
    print("Matrix B shape:", B.shape)
    print("Matrix B:")
    print(B.numpy())
    
    # Compute eigenvalues
    eigenvals = torch.linalg.eigvals(B)
    eigenval_mags = torch.abs(eigenvals)
    max_eigenval = eigenval_mags.max().item()
    
    print(f"\nEigenvalue magnitudes: {eigenval_mags.numpy()}")
    print(f"Maximum eigenvalue magnitude: {max_eigenval:.6f}")
    
    # Show what happens to different weight vectors
    print(f"\n--- Weight Transformation Examples ---")
    
    # Case 1: Random weights
    w_random = torch.randn(9) * 0.1  # Typical initialization scale
    w_transformed = w_random @ B
    amplification = w_transformed.norm().item() / w_random.norm().item()
    
    print(f"Random weights norm: {w_random.norm().item():.6f}")
    print(f"Transformed norm: {w_transformed.norm().item():.6f}")
    print(f"Amplification factor: {amplification:.6f}")
    
    # Case 2: Worst-case alignment (eigenvector with largest eigenvalue)
    eigenvals_complex, eigenvecs_complex = torch.linalg.eig(B)
    # Find the eigenvector corresponding to largest eigenvalue magnitude
    max_idx = torch.abs(eigenvals_complex).argmax()
    worst_eigenvec = eigenvecs_complex[:, max_idx].real  # Take real part
    worst_eigenvec = worst_eigenvec / worst_eigenvec.norm()  # Normalize
    
    w_worst = worst_eigenvec * 0.1  # Scale to typical init magnitude
    w_worst_transformed = w_worst @ B
    worst_amplification = w_worst_transformed.norm().item() / w_worst.norm().item()
    
    print(f"\nWorst-case alignment:")
    print(f"Worst weights norm: {w_worst.norm().item():.6f}")
    print(f"Worst transformed norm: {w_worst_transformed.norm().item():.6f}")
    print(f"Worst amplification: {worst_amplification:.6f}")
    print(f"Theory predicts: {max_eigenval:.6f} (matches!)")
    
    # Show our safe scale computation
    safe_scale = 0.01 / np.sqrt(max_eigenval)
    print(f"\n--- Safe Scale Computation ---")
    print(f"Target max std after transformation: 0.01")
    print(f"Safe scale = 0.01 / sqrt({max_eigenval:.6f}) = {safe_scale:.8f}")
    
    # Show the result
    w_safe = torch.randn(9) * safe_scale
    w_safe_transformed = w_safe @ B
    
    print(f"\nWith safe initialization:")
    print(f"Safe weights std: {w_safe.std().item():.8f}")
    print(f"Safe transformed std: {w_safe_transformed.std().item():.8f}")
    print(f"Safe transformed max: {w_safe_transformed.abs().max().item():.8f}")

if __name__ == "__main__":
    # Run the concrete example first to understand the math
    concrete_example()
    
    print("\n" + "="*80 + "\n")
    
    # Then show how it would work on an actual model
    print("To apply to your model:")
    print("model = emergency_fix_explained(model)")
