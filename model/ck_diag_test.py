# Quick test to isolate the problem
import torch
torch.cuda.manual_seed(42)
from circularkernel_7_6 import (
    SparseDepthBranch, GrayscaleBranch, RelativeDepthBranch, FusionBranch
)
def test_single_circular_conv():
    from conv.CircleConv3x3 import CircleConv3x3
    
    conv = CircleConv3x3(1, 1, kernel_size=3, padding=1)
    x = torch.randn(1, 1, 8, 8) * 0.01  # Very small input
    
    print("Transform matrix stats:")
    matrix = conv.w_transform_matrix
    print(f"Min/max: [{matrix.min():.6f}, {matrix.max():.6f}]")
    print(f"Condition number: {torch.linalg.cond(matrix):.2e}")
    
    output = conv(x)
    print(f"Output contains NaN: {torch.isnan(output).any()}")
    
def find_problematic_layer():
    # Test each component separately
    
    # 1. Test just parallel branches
    sparse_branch = SparseDepthBranch()
    x1 = torch.randn(1, 1, 512, 512, requires_grad=True)
    print("Testing Sparse Branch")
    try:
        output, _ = sparse_branch(x1)
        loss = output.mean()
        if torch.isnan(loss):
            print("Sparse Branch Loss Before Backprop is NaN")
        loss.backward()
        if torch.isnan(loss):
            print("Sparse Branch Loss after Backprop is NaN")
        
        #print("Sparse branch OK")
        print(loss)
    except:
        print("Sparse branch causes NaN")
        
    grayscale_branch = GrayscaleBranch()
    x2 = torch.randn(1, 1, 512, 512, requires_grad=True)
    print("Testing Grayscale Branch")
    try:
        output, _ = grayscale_branch(x2)
        loss = output.mean()
        if torch.isnan(loss):
            print("Grayscale Branch Loss Before Backprop is NaN")
        loss.backward()
        if torch.isnan(loss):
            print("Grayscale Branch Loss after Backprop is NaN")
        #loss.backward()
        #print("Grayscale branch OK")
        print(loss)
    except:
        print("Grayscale branch causes NaN")
        
        
    relativedepth_branch = RelativeDepthBranch()
    x3 = torch.randn(1, 1, 512, 512, requires_grad=True)
    print("Testing Relative Depth Branch")
    try:
        output, _ = relativedepth_branch(x3)
        loss = output.mean()
        if torch.isnan(loss):
            print("Relative Depth Branch Loss Before Backprop is NaN")
        loss.backward()
        if torch.isnan(loss):
            print("Relative Depth Branch Loss after Backprop is NaN")
        #loss.backward()
        #print("Relative Depth branch OK")
        print(loss)
    except:
        print("Relative Depth causes NaN")
    
    # 2. Test just fusion branch with regular inputs
    fusion_branch = FusionBranch()
    inputs = [torch.randn(1, 1, 512, 512) for _ in range(4)]
    features = {'enc1': torch.randn(1, 64, 512, 512),
                'enc2': torch.randn(1, 128, 256, 256), 
                'enc3': torch.randn(1, 256, 128, 128)}
    print("Testing Fusion Branch")
    try:
        output = fusion_branch(*inputs, features, features, features)
        loss = output.mean()
        if torch.isnan(loss):
            print("Fusion Branch Loss Before Backprop is NaN")
        loss.backward()
        if torch.isnan(loss):
            print("Fusion Branch Loss after Backprop is NaN")
        #loss.backward()
        #print("Fusion branch OK")
        print(loss)
    except:
        print("Fusion branch causes NaN")
    
if __name__ == "__main__":
    test_single_circular_conv()
    find_problematic_layer()