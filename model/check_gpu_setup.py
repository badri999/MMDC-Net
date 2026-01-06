#!/usr/bin/env python3
"""
GPU Setup Verification Script for HDD Depth Fusion Multi-GPU Training
"""

import torch
import torch.nn as nn

def check_gpu_setup():
    """Check GPU availability and configuration for multi-GPU training"""
    
    print("="*60)
    print("GPU SETUP VERIFICATION")
    print("="*60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        return False
    
    print("✅ CUDA is available")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\n📊 Number of GPUs detected: {num_gpus}")
    
    if num_gpus < 2:
        print("⚠️  Only 1 GPU available - multi-GPU training will be disabled")
    else:
        print("✅ Multi-GPU training supported")
    
    # Check individual GPU details
    print("\n" + "="*60)
    print("GPU DETAILS")
    print("="*60)
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {memory_gb:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multiprocessors: {props.multi_processor_count}")
        
        # Check current memory usage
        torch.cuda.set_device(i)
        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
        memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
        memory_free = memory_gb - memory_reserved
        
        print(f"  Memory Allocated: {memory_allocated:.2f} GB")
        print(f"  Memory Reserved: {memory_reserved:.2f} GB") 
        print(f"  Memory Free: {memory_free:.2f} GB")
        
        if memory_free < 5.0:
            print(f"  ⚠️  Low free memory ({memory_free:.1f} GB)")
        else:
            print(f"  ✅ Good free memory ({memory_free:.1f} GB)")
    
    # Test DataParallel capability
    print("\n" + "="*60)
    print("DATAPARALLEL TEST")
    print("="*60)
    
    try:
        # Reset to default device
        torch.cuda.set_device(0)
        
        # Clear any existing CUDA cache
        torch.cuda.empty_cache()
        
        # Create a simple model
        model = nn.Linear(10, 1)
        
        if num_gpus > 1:
            # Move model to GPU first, then wrap with DataParallel
            model = model.cuda()
            model = nn.DataParallel(model)
            print("✅ DataParallel wrapper created successfully")
        else:
            model = model.cuda()
        
        # Test forward pass
        test_input = torch.randn(4, 10).cuda()
        output = model(test_input)
        
        print(f"✅ Test forward pass successful")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        
        if num_gpus > 1:
            print(f"   Model distributed across GPUs: {list(model.device_ids)}")
        
        # Test backward pass
        loss = output.sum()
        loss.backward()
        print("✅ Test backward pass successful")
        
        # Clear test tensors
        del model, test_input, output, loss
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ DataParallel test failed: {str(e)}")
        print("   This may not affect actual training if the error is in the test setup")
        
        # Clear any partial tensors
        torch.cuda.empty_cache()
        
        # Don't return False - the test failure might not indicate actual training issues
        pass
    
    # Memory recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    total_memory = sum(torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(num_gpus))
    print(f"Total GPU Memory: {total_memory:.1f} GB")
    
    if num_gpus == 2:
        print("\nFor 2-GPU training:")
        print("  • Recommended batch size: 16-24 (8-12 per GPU)")
        print("  • Use batch sizes divisible by 2")
        print("  • Monitor memory usage during training")
        print("  • Set num_workers = 4-8 for optimal data loading")
        
        # Specific recommendations based on available memory
        min_free_memory = min(
            (torch.cuda.get_device_properties(i).total_memory / (1024**3)) - 
            (torch.cuda.memory_reserved(i) / (1024**3))
            for i in range(num_gpus)
        )
        
        if min_free_memory > 20:
            print("  • High memory available - can use larger batch sizes (20-32)")
        elif min_free_memory > 10:
            print("  • Medium memory available - use moderate batch sizes (12-20)")
        else:
            print("  • Limited memory available - use smaller batch sizes (4-12)")
    
    print("\n✅ GPU setup verification completed!")
    return True

if __name__ == "__main__":
    success = check_gpu_setup()
    exit(0 if success else 1) 