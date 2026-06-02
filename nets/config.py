"""
Configuration module for the parametrized medical segmentation framework.

This file provides factory functions that build a complete configuration dictionary
from high-level parameters. The Trainer calls get_network_config() and passes
the resulting dictionary to the network constructor.

No module-level state is stored here; all configuration flows explicitly
from the Trainer through the dictionary.
"""

import torch.nn as nn

def get_network_config(spatial_dims: int,
                       norm_type: str,
                       act_type: str,
                       upsample_type: str,
                       **kwargs) -> dict:
    """
    Dynamically generates a configuration dictionary containing the appropriate 
    network operations and their corresponding keyword arguments based 
    on high-level parameters.

    Args:
        spatial_dims (int): The spatial dimensionality of the network (1, 2, or 3).
        norm_type (str): The type of normalization layer ('gn' for GroupNorm, 
                         'ln' for LayerNorm, 'in' for InstanceNorm, 'bn' for BatchNorm).
        act_type (str): The type of activation function ('relu', 'leakyrelu', or 'gelu').
        upsample_type (str): The upsampling strategy ('transpose' or 'upsample').
        **kwargs: Additional overrides (kernel_size, bias, num_groups, norm_eps,
                  negative_slope, bn_momentum, id, preact, etc.).
        
    Returns:
        dict: A configuration dictionary mapping standard architectural components 
              (e.g., 'conv_op', 'norm_op') to their respective classes 
              and default keyword arguments.
    """
    config = {}

    # Convolution
    if spatial_dims == 1:
        conv_op = nn.Conv1d
    elif spatial_dims == 2:
        conv_op = nn.Conv2d
    elif spatial_dims == 3:
        conv_op = nn.Conv3d
    else:
        raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")
    config['conv_op'] = conv_op
    config['conv_kwargs'] = {
        "kernel_size": kwargs.get("kernel_size", 3),
        "bias": kwargs.get("bias", True)
    }

    # Normalization    
    if norm_type == "gn":
        config['norm_op'] = nn.GroupNorm
        config['norm_kwargs'] = {
            "num_groups": kwargs.get("num_groups", 8),
            "eps": kwargs.get("norm_eps", 1e-5)
        }
    elif norm_type == "ln":
        config['norm_op'] = nn.LayerNorm
        config['norm_kwargs'] = {
            "eps": kwargs.get("norm_eps", 1e-5),
            "bias": True
        }
    elif norm_type == "in":
        if spatial_dims == 1:
            norm_op = nn.InstanceNorm1d
        elif spatial_dims == 2:
            norm_op = nn.InstanceNorm2d
        elif spatial_dims == 3:
            norm_op = nn.InstanceNorm3d
        else:
            raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")            
        config['norm_op'] = norm_op
        config['norm_kwargs'] = {
            "eps": kwargs.get("norm_eps", 1e-5)
        }
    elif norm_type == "bn":
        if spatial_dims == 1:
            norm_op = nn.BatchNorm1d
        elif spatial_dims == 2:
            norm_op = nn.BatchNorm2d
        elif spatial_dims == 3:
            norm_op = nn.BatchNorm3d
        else:
            raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")
        config['norm_op'] = norm_op
        config['norm_kwargs'] = {
            "eps": kwargs.get("norm_eps", 1e-5),
            "momentum": kwargs.get("bn_momentum", 0.1)
        }
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")

    # Adaptive pooling and interpolation routing
    if spatial_dims == 1:
        adaptive_avg_pool_op = nn.AdaptiveAvgPool1d
        adaptive_max_pool_op = nn.AdaptiveMaxPool1d
        interp_mode = 'linear'
    elif spatial_dims == 2:
        adaptive_avg_pool_op = nn.AdaptiveAvgPool2d
        adaptive_max_pool_op = nn.AdaptiveMaxPool2d
        interp_mode = 'bilinear'
    elif spatial_dims == 3:
        adaptive_avg_pool_op = nn.AdaptiveAvgPool3d
        adaptive_max_pool_op = nn.AdaptiveMaxPool3d
        interp_mode = 'trilinear'
    else:
        raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")
    config['adaptive_avg_pool_op'] = adaptive_avg_pool_op
    config['adaptive_max_pool_op'] = adaptive_max_pool_op
    config['interp_mode'] = interp_mode

    # Activation
    if act_type == "relu":
        config['act_op'] = nn.ReLU
        config['act_kwargs'] = {"inplace": True}
    elif act_type == "leakyrelu":
        config['act_op'] = nn.LeakyReLU
        config['act_kwargs'] = {
            "negative_slope": kwargs.get("negative_slope", 0.01),
            "inplace": True
        }
    elif act_type == "gelu":
        config['act_op'] = nn.GELU
        config['act_kwargs'] = {}
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")

    # Upsampling
    config['pool_op'], config['transpconv_op'] = get_pooling_operations(spatial_dims)
    if upsample_type not in ("transpose", "upsample"):
        raise ValueError(f"Unsupported upsample type: {upsample_type}")
    config['upsample_type'] = upsample_type    
    
    # Network
    config['network_kwargs'] = {
        "id": kwargs.get("id", True),
        "preact": kwargs.get("preact", True),
    }
    
    return config

def get_pooling_operations(spatial_dims: int):
    """
    Retrieve dimension-specific pooling and transposed convolution operations.
    The upsample_type routing is handled by UpConv via the 'upsample_type' config key.
    Prevents global state leakage when training multiple models simultaneously.
    """
    if spatial_dims == 1:
        pool_op = nn.MaxPool1d
        transpconv_op = nn.ConvTranspose1d
    elif spatial_dims == 2:
        pool_op = nn.MaxPool2d
        transpconv_op = nn.ConvTranspose2d
    elif spatial_dims == 3:
        pool_op = nn.MaxPool3d
        transpconv_op = nn.ConvTranspose3d
    else:
        raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")

    return pool_op, transpconv_op
