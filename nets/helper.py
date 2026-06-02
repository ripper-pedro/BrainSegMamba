"""
Helper module for the parametrized medical segmentation framework.

This file contains utility functions designed to support the dynamic nature
of the architecture. By isolating mathematical and formatting operations here,
we keep the core neural network blocks clean and strictly focused on deep learning logic.
"""

import inspect
import numpy as np
import torch.nn as nn

def maybe_convert_scalar_to_list(conv_op, scalar):
    """
    Converts a scalar value into a list based on the spatial dimensions of the given convolutional operator.
    This is essential for making the architecture dimension-agnostic (1D, 2D, or 3D).

    Args:
        conv_op (Type[nn.Module]): The convolutional class (e.g., nn.Conv2d, nn.Conv3d).
        scalar (Union[int, tuple, list, np.ndarray]): The value to be converted (e.g., kernel_size=3).

    Returns:
        List/Tuple of scalars (e.g., [3, 3, 3] for 3D or [3, 3] for 2D).
    """
    if not isinstance(scalar, (tuple, list, np.ndarray)):
        # If scalar is not a list, tuple or np.ndarray, convert it to a list
        if issubclass(conv_op , nn.Conv2d):
            return [scalar] * 2
        elif issubclass(conv_op , nn.Conv3d):
            return [scalar] * 3
        elif issubclass(conv_op , nn.Conv1d):
            return [scalar] * 1
        else:
            raise RuntimeError("Invalid conv op: %s" % str(conv_op))
    else:
        return scalar

# Note: only works for normalization functions that use num_channels/num_features
def build_norm(norm_op, num_channels, norm_kwargs):
    """
    Dynamically instantiates a normalization layer based on the provided operator class.

    This function handles the specific instantiation logic for different normalization layers.
    It explicitly supports `nn.GroupNorm` by mapping `num_channels` and default arguments. 
    For other normalization layers (e.g., `nn.BatchNorm2d`, `nn.InstanceNorm3d`), it relies 
    on `inspect.signature` to safely filter `norm_kwargs` and inject only the valid arguments 
    expected by the layer's constructor alongside the channel dimension.

    Args:
        norm_op (Type[nn.Module]): The uninstantiated normalization class.
        num_channels (int): The number of channels/features the normalization layer expects.
        norm_kwargs (dict): A dictionary of keyword arguments for the normalization layer.

    Returns:
        nn.Module: The instantiated normalization layer.
    """
    if norm_op is nn.GroupNorm:
        # Standard scenario
        return norm_op(
            num_groups=norm_kwargs.get("num_groups", 8),
            num_channels=num_channels,
            eps=norm_kwargs.get("eps", 1e-5)
        )
    else:
        # For other normalization layers, dynamically inspect the constructor signature to prevent passing unexpected keyword arguments
        sig = inspect.signature(norm_op.__init__)
        valid_kwargs = {}
        for k, v in norm_kwargs.items():
            if k in sig.parameters:
                valid_kwargs[k] = v

        return norm_op(num_channels, **valid_kwargs)
