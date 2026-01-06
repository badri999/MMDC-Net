import torch
from pytorch_msssim import ssim

def gauss_kernel(size=5, channels=1):
    kernel = torch.zeros((size, size, channels), dtype=torch.float32)
    kernel[:, :, 0] = torch.tensor([[1., 4., 6., 4., 1.],
                                    [4., 16., 24., 16., 4.],
                                    [6., 24., 36., 24., 6.],
                                    [4., 16., 24., 16., 4.],
                                    [1., 4., 6., 4., 1.]])
    kernel /= 256.0
    kernel = kernel.unsqueeze(-1)  # Add a fourth dimension for depthwise convolution
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    batch_size, channels, height, width = x.shape
    cc = torch.cat([x, torch.zeros_like(x)], dim=3)
    cc = cc.reshape(batch_size, channels, height, width * 2)
    cc = cc.transpose(2, 3)
    cc = torch.cat([cc, torch.zeros_like(cc)], dim=3)
    cc = cc.reshape(batch_size, channels, width * 2, height * 2)
    x_up = cc.transpose(2, 3)
    return conv_gauss(x_up, gauss_kernel(size=5, channels=x.size(1)))

def conv_gauss(img, kernel):
    pad_width = (2, 2, 2, 2)  # PyTorch padding order: (left, right, top, bottom)
    img = torch.nn.functional.pad(img, pad_width, mode='reflect')
    kernel = kernel.permute(3, 2, 0, 1)  # Rearrange kernel dimensions for depthwise convolution
    kernel = kernel.to(img.device)
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])  # Depthwise convolution in PyTorch
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current - up
        pyr.append(diff)
        current = down
    return pyr

def laplacian_loss(pred, target):
    kernel = gauss_kernel(size=5, channels=1)
    pyr_input = laplacian_pyramid(img=target, kernel=kernel, max_levels=3)
    pyr_target = laplacian_pyramid(img=pred, kernel=kernel, max_levels=3)
    laploss = sum(torch.mean(torch.abs(a - b)) for a, b in zip(pyr_input, pyr_target))
    return laploss

def laplacian_loss_with_mask(pred, target, mask=None):
    if mask is not None:
        # Apply mask to both inputs before computing pyramids
        masked_pred = pred * mask      # ❌ Creates artificial boundaries
        masked_target = target * mask  # ❌ Creates artificial boundaries
        
        # Compute Laplacian pyramids on masked inputs - includes boundary artifacts!
        kernel = gauss_kernel(size=5, channels=1)
        pyr_input = laplacian_pyramid(img=masked_target, kernel=kernel, max_levels=3)
        pyr_target = laplacian_pyramid(img=masked_pred, kernel=kernel, max_levels=3)
        loss = sum(torch.mean(torch.abs(a - b)) for a, b in zip(pyr_input, pyr_target))

    else:
        # Standard Laplacian loss
        kernel = gauss_kernel(size=5, channels=1)
        pyr_input = laplacian_pyramid(img=target, kernel=kernel, max_levels=3)
        pyr_target = laplacian_pyramid(img=pred, kernel=kernel, max_levels=3)
        loss = sum(torch.mean(torch.abs(a - b)) for a, b in zip(pyr_input, pyr_target))
    
    return loss

def ssim_loss(pred, target):
    return 1 - ssim(pred, target, data_range=1.0, size_average=True)

def ssim_loss_with_mask(pred, target, mask=None):
    """
    Compute SSIM loss between two images with optional masking.
    
    Args:
        pred (torch.Tensor): Predicted image/depth map with values between 0 and 1
        target (torch.Tensor): Target/ground truth image/depth map with values between 0 and 1
        mask (torch.Tensor, optional): Binary mask where 1 indicates valid pixels, 0 for invalid
        
    Returns:
        torch.Tensor: SSIM loss value (scalar)
    """
    # Ensure both tensors have the same shape
    assert pred.shape == target.shape, "Predicted and target must have the same shape"
    
    if mask is not None:
        assert mask.shape == pred.shape, "Mask must have the same shape as input tensors"
        
        #Normalize the input tensors with min value of 0.4 and max value of 0.5
        pred_norm=(pred-0.4)/(0.5-0.4)
        target_norm=(target-0.4)/(0.5-0.4)
        
        # Apply mask to entire batch first - efficient!
        #masked_pred = pred * mask
        #masked_target = target * mask

        masked_pred = pred_norm * mask
        masked_target = target_norm * mask
        
        batch_size = pred.shape[0]
        total_loss = 0.0
        valid_samples = 0
        
        # Process each sample in the batch separately for cropping
        for b in range(batch_size):
            sample_mask = mask[b, 0]  # Shape: (H, W)
            
            valid_indices = torch.nonzero(sample_mask, as_tuple=False)
            if len(valid_indices) == 0:
                continue
            
            min_h, min_w = valid_indices.min(dim=0)[0]
            max_h, max_w = valid_indices.max(dim=0)[0]
            
            # Crop the already-masked tensors
            pred_crop = masked_pred[b:b+1, :, min_h:max_h+1, min_w:max_w+1]
            target_crop = masked_target[b:b+1, :, min_h:max_h+1, min_w:max_w+1]
            
            if pred_crop.shape[-1] < 11 or pred_crop.shape[-2] < 11:
                continue
            
            

            ssim_value = ssim(pred_crop, target_crop, data_range=1.0, size_average=True)
            ssim_value = torch.clamp(ssim_value, 0.0, 1.0)
            
            total_loss += (1 - ssim_value)
            valid_samples += 1
        
        if valid_samples == 0:
            return torch.tensor(0.0, device=pred.device)
        
        loss = total_loss / valid_samples
    else:
        loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
    
    return loss

def l2_loss(pred_depth, target_depth):
    """
    Compute L2 (Mean Squared Error) loss between two depth maps.
    
    Args:
        pred_depth (torch.Tensor): Predicted depth map with values between 0 and 1
        target_depth (torch.Tensor): Target/ground truth depth map with values between 0 and 1
        
    Returns:
        torch.Tensor: L2 loss value (scalar)
    """
    # Ensure both tensors have the same shape
    assert pred_depth.shape == target_depth.shape, "Predicted and target depth maps must have the same shape"
    
    # Compute L2 loss (Mean Squared Error)
    loss = torch.mean((pred_depth - target_depth) ** 2)
    
    return loss

def l2_loss_with_mask(pred_depth, target_depth, mask=None):
    """
    Compute L2 loss between two depth maps with optional masking.
    Useful when you want to ignore certain pixels (e.g., invalid depth values).
    
    Args:
        pred_depth (torch.Tensor): Predicted depth map with values between 0 and 1
        target_depth (torch.Tensor): Target/ground truth depth map with values between 0 and 1
        mask (torch.Tensor, optional): Binary mask where 1 indicates valid pixels, 0 for invalid
        
    Returns:
        torch.Tensor: L2 loss value (scalar)
    """
    # Ensure both tensors have the same shape
    assert pred_depth.shape == target_depth.shape, "Predicted and target depth maps must have the same shape"
    
    # Compute squared differences
    squared_diff = (pred_depth - target_depth) ** 2
    
    if mask is not None:
        assert mask.shape == pred_depth.shape, "Mask must have the same shape as depth maps"
        # Apply mask and compute mean only over valid pixels
        masked_diff = squared_diff * mask
        loss = torch.sum(masked_diff) / torch.sum(mask)
    else:
        # Standard L2 loss
        loss = torch.mean(squared_diff)
    
    return loss

def l1_loss(pred_depth, target_depth):
    """
    Compute L1 (Mean Absolute Error) loss between two depth maps.
    
    Args:
        pred_depth (torch.Tensor): Predicted depth map with values between 0 and 1
        target_depth (torch.Tensor): Target/ground truth depth map with values between 0 and 1
        
    Returns:
        torch.Tensor: L1 loss value (scalar)
    """
    # Ensure both tensors have the same shape
    assert pred_depth.shape == target_depth.shape, "Predicted and target depth maps must have the same shape"
    
    # Compute L1 loss (Mean Absolute Error)
    loss = torch.mean(torch.abs(pred_depth - target_depth))
    
    return loss

def l1_loss_with_mask(pred_depth, target_depth, mask=None):
    """
    Compute L1 loss between two depth maps with optional masking.
    Useful when you want to ignore certain pixels (e.g., invalid depth values).
    
    Args:
        pred_depth (torch.Tensor): Predicted depth map with values between 0 and 1
        target_depth (torch.Tensor): Target/ground truth depth map with values between 0 and 1
        mask (torch.Tensor, optional): Binary mask where 1 indicates valid pixels, 0 for invalid
        
    Returns:
        torch.Tensor: L1 loss value (scalar)
    """
    # Ensure both tensors have the same shape
    assert pred_depth.shape == target_depth.shape, "Predicted and target depth maps must have the same shape"
    
    # Compute absolute differences
    abs_diff = torch.abs(pred_depth - target_depth)
    
    if mask is not None:
        assert mask.shape == pred_depth.shape, "Mask must have the same shape as depth maps"
        # Apply mask and compute mean only over valid pixels
        masked_diff = abs_diff * mask
        loss = torch.sum(masked_diff) / torch.sum(mask)
    else:
        # Standard L1 loss
        loss = torch.mean(abs_diff)
    
    return loss


    