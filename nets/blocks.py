"""
Neural network building blocks for the parametrized medical segmentation framework.

This module provides the core components (ConvBlock, ResidualBlock) and
routing mechanisms (SpatialReducer, UpConv) used to construct the U-Net architecture.
All blocks are designed to be independent of spatial dimensions (1D, 2D, or 3D),
dynamically adapting based on the routing parameters defined by the network orchestrator.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from typing import Union, List, Tuple, Type, Optional, Dict, Any

from nnunetv2.nets.helper import maybe_convert_scalar_to_list, build_norm

class ConvBlock(nn.Module):
    """
    Fundamental building block consisting of Convolution, Normalization, and Activation.
    
    Automatically handles spatial padding to maintain dimensions (same padding) and adapts
    normalization channels based on its relative position to the convolution layer.

    Args:
        input_channels: Number of input feature channels.
        output_channels: Number of output feature channels.
        stride: Spatial resolution reduction factor (broadcastable to 1D/2D/3D).
        conv_op: Convolutional operator type (e.g., nn.Conv2d, nn.Conv3d).
        conv_kwargs: Dictionary of parameters for the convolutional layer.
        norm_op: Normalization operator type.
        norm_kwargs: Dictionary of parameters for the normalization layer.
        act_op: Activation operator type.
        act_kwargs: Dictionary of parameters for the activation layer.
        layers_order: Sequence of operations (e.g., ("conv", "norm", "act")).
        **kwargs: Additional parameters passed by the orchestrator.

    Returns:
        torch.Tensor: The processed feature map.
    """
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd] = None,
                 conv_kwargs: dict = None,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_kwargs: dict = None,
                 act_op: Union[None, Type[nn.Module]] = None,
                 act_kwargs: dict = None,
                 layers_order: Tuple["str", ...] = ("conv", "norm", "act"),
                 **kwargs
                 ):
        super().__init__()

        # Ensure dictionaries are initialized to prevent unpacking errors
        conv_kwargs = conv_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}

        # Adapt parameters to the specific dimensionality (1D, 2D, or 3D)
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        kernel_size = maybe_convert_scalar_to_list(conv_op, conv_kwargs["kernel_size"])

        # Calculate 'same' padding automatically based on kernel size
        padding = [(i - 1) // 2 for i in kernel_size]

        layers = {}

        # Convolutional layer
        layers["conv"] = conv_op(
            input_channels,
            output_channels,
            stride=stride,
            padding=padding,
            kernel_size=kernel_size,
            bias=conv_kwargs["bias"]
        )

        # Normalization layer (if requested)
        if "norm" in layers_order:
            # Logic for Channel Adaptation:
            # If Norm comes before Conv, it uses input_channels.
            # If it comes after, it uses output_channels.
            if "conv" in layers_order and layers_order.index("norm") < layers_order.index("conv"):
                norm_channels = input_channels
            else:
                norm_channels = output_channels

            # Normalization function agnostic
            layers["norm"] = build_norm(norm_op, norm_channels, norm_kwargs)
            if norm_op is nn.LayerNorm:
                layers["norm"] = ChannelLastWrapper(layers["norm"])

        # Activation layer (if requested)
        if "act" in layers_order:
            layers["act"] = act_op(**act_kwargs)

        # Assemble the final block according to the specified sequence
        module_list = []
        for layer_name in layers_order:
            if layer_name not in layers:
                continue
            module_list.append(layers[layer_name])

        self.block = nn.Sequential(*module_list)

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    """
    Residual block featuring two convolutional stages and an identity skip connection.
    
    Supports both standard (Conv-Norm-Act) and pre-activation (Norm-Act-Conv) regimes
    based on the injected network_kwargs.

    Args:
        input_channels: Number of input feature channels.
        output_channels: Target number of output feature channels.
        stride: Downsampling factor for the primary convolutional stage.
        conv_op: Convolutional operator type.
        conv_kwargs: Parameters for the convolutional layer.
        norm_op: Normalization operator type.
        norm_kwargs: Parameters for the normalization layer.
        act_op: Activation operator type.
        act_kwargs: Parameters for the activation layer.
        network_kwargs: Dictionary containing flags 'preact' and 'id'.
        **kwargs: Additional parameters passed by the orchestrator.

    Returns:
        torch.Tensor: The residually fused feature map.
    """
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 conv_op: Type[_ConvNd] = None,
                 conv_kwargs: dict = None,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_kwargs: dict = None,
                 act_op: Union[None, Type[nn.Module]] = None,
                 act_kwargs: dict = None,
                 network_kwargs: dict = None,
                 **kwargs
                 ):
        super().__init__()

        # Ensure dictionaries are initialized to prevent unpacking errors
        conv_kwargs = conv_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}
        network_kwargs = network_kwargs or {}

        # Determine layer sequence based on the pre-activation strategy
        if network_kwargs["preact"]:
            order = ("norm", "act", "conv")
        else:
            order = ("conv", "norm", "act")

        # Defining common parameters for both internal convolutional components.
        # All ops (conv_op, norm_op, act_op) are explicitly propagated to ConvBlock
        # so that injected ops from the top-level model flow through correctly.
        common_params = {
            "conv_op" : conv_op,
            "conv_kwargs" : conv_kwargs,
            "norm_op" : norm_op,
            "norm_kwargs" : norm_kwargs,
            "act_op" : act_op,
            "act_kwargs" : act_kwargs,
            "layers_order" : order,
            "output_channels" : output_channels,
        }

        # First block handles potential spatial downsampling via stride
        self.block1 = ConvBlock(
            input_channels=input_channels,
            stride=stride,
            **common_params
        )

        # Second block performs feature refinement at the target resolution
        self.block2 = ConvBlock(
            input_channels=output_channels,
            stride=1,
            **common_params
        )

        # Logic for either using identity mapping or a 1x1 convolution to ensure compatibility between input and output number of channels
        stride_list = maybe_convert_scalar_to_list(conv_op, stride)
        is_identity = (input_channels == output_channels) and all(s == 1 for s in stride_list) and network_kwargs["id"]
        if is_identity:
            self.id = nn.Identity()
        else:
            self.id = conv_op(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False
            )

    def forward(self, x):
        residual = self.id(x)
        out = self.block1(x)
        out = self.block2(out)
        return out + residual

class SpatialReducer(nn.Module):
    """
    Downsampling router for spatial dimensionality reduction.

    Args:
        input_channels: Number of input channels.
        output_channels: Target number of channels after reduction.
        down_type: Strategy for resolution reduction: 
            - 'residual': Learnable bottleneck blocks.
            - 'conv': Strided convolutions.
            - 'maxpool': Fixed pooling followed by convolution.
            - 'mamba': Patch merging approach.
        stride: Scaling factor for spatial reduction.
        pool_op: Pooling operator type (required if down_type='maxpool').
        **kwargs: Downstream operations injected by the model.

    Returns:
        torch.Tensor: The downsampled feature map.
    """
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 down_type: str,
                 stride: Union[int, List[int], Tuple[int, ...]] = 2,
                 pool_op: Type[nn.Module] = None,
                 **kwargs):
        super().__init__()

        if down_type == "residual":
            # Residual-based downsampling with learnable parameters
            self.block = ResidualBlock(
                input_channels=input_channels,
                output_channels=output_channels,
                stride=stride,
                **kwargs
            )

        elif down_type == "conv":
            # Strided convolution for feature extraction and resolution reduction
            self.block = ConvBlock(
                input_channels=input_channels,
                output_channels=output_channels,
                stride=stride,
                **kwargs
            )

        elif down_type == "maxpool":
            # Non-learnable max pooling followed by a feature projection layer
            pool_layer = pool_op(kernel_size=stride, stride=stride)

            conv_block = ConvBlock(
                input_channels=input_channels,
                output_channels=output_channels,
                stride=1,
                **kwargs
            )
            self.block = nn.Sequential(pool_layer, conv_block)
        
        elif down_type == "mamba":
            # Non-convolutional block for mamba stages
            self.block = PatchMerging(
                input_channels=input_channels,
                output_channels=output_channels,
                conv_op=kwargs['conv_op']
            )

        else:
            raise ValueError(f"Downsample type '{down_type}' not supported. Use 'residual', 'conv', 'maxpool' or 'mamba'.")

    def forward(self, x):
        return self.block(x)

class UpConv(nn.Module):
    """
    Upsampling router for spatial dimensionality expansion.
    
    Automatically infers the optimal interpolation mode ('trilinear', 'bilinear', 
    or 'linear') based on the dimension of the injected transpconv_op.

    Args:
        input_channels: Number of input feature channels.
        output_channels: Target number of output feature channels.
        upsample_type: Expansion strategy:
            - 'transpose': Learnable transposed convolutions.
            - 'upsample': Fixed parameter-free interpolation followed by convolution.
        stride: Scaling factor for spatial expansion.
        transpconv_op: Transposed convolutional operator type.
        **kwargs: Additional parameters (conv_kwargs, norm_op, etc.).

    Returns:
        torch.Tensor: The upsampled feature map.
    """
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 upsample_type: str,
                 stride: Union[int, List[int], Tuple[int, ...]] = 2,
                 transpconv_op: Type[_ConvNd] = None,
                 **kwargs):
        super().__init__()

        # Automatic inference of interpolation mode based on operator dimensionality
        if issubclass(transpconv_op, nn.ConvTranspose3d):
            upsample_mode = "trilinear"
        elif issubclass(transpconv_op, nn.ConvTranspose2d):
            upsample_mode = "bilinear"
        elif issubclass(transpconv_op, nn.ConvTranspose1d):
            upsample_mode = "linear"
        else:
            raise ValueError(f"Transposition operation '{transpconv_op}' not supported.")

        if upsample_type == "transpose":
            # Learnable spatial expansion via transposed convolution
            bias = kwargs["conv_kwargs"]["bias"]

            self.block = transpconv_op(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=stride,
                stride=stride,
                bias=bias
            )

        elif upsample_type == "upsample":
            # Non-learnable interpolation followed by a convolutional refinement block
            upsample_op = nn.Upsample(scale_factor=stride, mode=upsample_mode, align_corners=False)

            conv_block = ConvBlock(
                input_channels=input_channels,
                output_channels=output_channels,
                stride=1,
                **kwargs
            )
            self.block = nn.Sequential(upsample_op, conv_block)

        else:
            raise ValueError(f"Upsample type '{upsample_type}' not supported. Use 'transpose' or 'upsample'.")

    def forward(self, x):
        return self.block(x)

# All blocks below are related to VMUnetV2 implementation

class PatchMerging(nn.Module):
    """
    Dimensionality-agnostic Patch Merging block.

    Halves the spatial resolution by concatenating adjacent spatial elements
    into the channel dimension, followed by a linear projection. Designed 
    to facilitate hierarchical feature extraction in state-space models (SSMs) 
    without relying on strided convolutions.

    Args:
        input_channels: Number of input channels.
        output_channels: Target number of output channels after projection.
        conv_op: Convolutional operator type used to deduce spatial dimension.
        **kwargs: Additional parameters (ignored/passed through).

    Returns:
        torch.Tensor: The merged and projected feature map.
    """
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 conv_op: Type[_ConvNd],
                 **kwargs):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.norm_layer = nn.LayerNorm
        
        # Deduce spatial dimensions from conv_op
        if issubclass(conv_op, nn.Conv1d):
            self.num_spatial = 1
        elif issubclass(conv_op, nn.Conv2d):
            self.num_spatial = 2
        elif issubclass(conv_op, nn.Conv3d):
            self.num_spatial = 3
        else:
            raise ValueError(f"Unsupported conv_op: {conv_op}")

        group_size = 2 ** self.num_spatial
        expanded_channels = input_channels * group_size

        # Create parameters in __init__ so Optimizer can track them
        self.reduction_proj = conv_op(expanded_channels, self.output_channels, kernel_size=1, bias=False)
        self.norm = self.norm_layer(expanded_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        spatial_dims = x.shape[2:]
        new_spatial_dims = [max(1, d // 2) for d in spatial_dims]
        expanded_channels = C * (2 ** self.num_spatial)

        if self.num_spatial == 3:
            D, H, W = spatial_dims
            pad_d, pad_h, pad_w = (2 - D % 2) % 2, (2 - H % 2) % 2, (2 - W % 2) % 2
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
                D, H, W = D + pad_d, H + pad_h, W + pad_w
                new_spatial_dims = [D//2, H//2, W//2]

            x = x.view(B, C, new_spatial_dims[0], 2, new_spatial_dims[1], 2, new_spatial_dims[2], 2)
            x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
            x = x.view(B, expanded_channels, *new_spatial_dims)
            
        elif self.num_spatial == 2:
            H, W = spatial_dims
            pad_h, pad_w = (2 - H % 2) % 2, (2 - W % 2) % 2
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h))
                H, W = H + pad_h, W + pad_w
                new_spatial_dims = [H//2, W//2]

            x = x.view(B, C, new_spatial_dims[0], 2, new_spatial_dims[1], 2)
            x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
            x = x.view(B, expanded_channels, *new_spatial_dims)
            
        elif self.num_spatial == 1:
            L = spatial_dims[0]
            pad_l = (2 - L % 2) % 2
            if pad_l > 0:
                x = F.pad(x, (0, pad_l))
                L = L + pad_l
                new_spatial_dims = [L//2]

            x = x.view(B, C, new_spatial_dims[0], 2)
            x = x.permute(0, 1, 3, 2).contiguous()
            x = x.view(B, expanded_channels, *new_spatial_dims)

        x = x.permute(0, *range(2, self.num_spatial + 2), 1)
        x = self.norm(x)
        x = x.permute(0, self.num_spatial + 1, *range(1, self.num_spatial + 1))
        
        x = self.reduction_proj(x)
        return x

class PatchEmbed(nn.Module):
    """
    Dimensionality-agnostic Patch Embedding block.
    
    Splits the input volume into non-overlapping patches using a strided convolution 
    and projects them to an embedding space. Handles 1D, 2D, and 3D data dynamically.

    Args:
        input_channels: Number of input channels.
        embed_dim: Dimensionality of the resulting embedding.
        conv_op: Convolutional operator type used to deduce spatial dimension.
        patch_size: Size of the projection patches.
        **kwargs: Additional parameters (ignored/passed through).

    Returns:
        torch.Tensor: The embedded feature map.
    """
    def __init__(self,
                 input_channels: int,
                 embed_dim: int,
                 conv_op: Type[_ConvNd],
                 patch_size: Union[int, Tuple[int, ...]] = 4,
                 **kwargs):
        super().__init__()
        
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.norm_layer = nn.LayerNorm
        
        if issubclass(conv_op, nn.Conv1d):
            self.num_spatial = 1
        elif issubclass(conv_op, nn.Conv2d):
            self.num_spatial = 2
        elif issubclass(conv_op, nn.Conv3d):
            self.num_spatial = 3
        else:
            raise ValueError(f"Unsupported conv_op: {conv_op}")

        if isinstance(self.patch_size, int):
            p_size = (self.patch_size,) * self.num_spatial
        else:
            p_size = self.patch_size

        # Create parameters in __init__
        self.proj = conv_op(self.input_channels, self.embed_dim, kernel_size=p_size, stride=p_size, bias=False)
        self.norm = self.norm_layer(self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_dims = x.shape[2:]
        p_size = (self.patch_size,) * self.num_spatial if isinstance(self.patch_size, int) else self.patch_size

        pad_sizes = []
        pad_needed = False
        for i in reversed(range(self.num_spatial)):
            rem = spatial_dims[i] % p_size[i]
            pad = (p_size[i] - rem) % p_size[i]
            pad_sizes.extend([0, pad])
            if pad > 0: pad_needed = True

        if pad_needed:
            x = F.pad(x, tuple(pad_sizes))

        x = self.proj(x)
        
        x = x.permute(0, *range(2, self.num_spatial + 2), 1)
        x = self.norm(x)
        x = x.permute(0, self.num_spatial + 1, *range(1, self.num_spatial + 1)).contiguous()
            
        return x

class ChannelLastWrapper(nn.Module):
    """
    Generic wrapper to handle Channel-First (B, C, ...) to Channel-Last (B, ..., C) conversion.
    
    Args:
        module: The PyTorch module to wrap.

    Returns:
        torch.Tensor: The processed tensor reverted to Channel-First format.
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_spatial = x.dim() - 2
        x = x.permute(0, *range(2, num_spatial + 2), 1)
        x = self.module(x)
        x = x.permute(0, num_spatial + 1, *range(1, num_spatial + 1)).contiguous()
        return x

class MambaWrapper(nn.Module):
    """
    Wrapper for VSSLayer3D combining Channel-Last conversion and cubic padding.
    
    Args:
        mamba_layer: The instantiated VSSLayer3D module.

    Returns:
        torch.Tensor: The processed tensor, restored to its original spatial shape.
    """
    def __init__(self, mamba_layer: nn.Module):
        super().__init__()
        self.mamba = ChannelLastWrapper(mamba_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_dims = x.shape[2:]
        max_dim = max(spatial_dims)
        
        pad_sizes = []
        pad_needed = False
        for d in reversed(spatial_dims):
            pad = max_dim - d
            pad_sizes.extend([0, pad])
            if pad > 0: pad_needed = True
            
        if pad_needed:
            x = F.pad(x, tuple(pad_sizes))
            
        x = self.mamba(x)
        
        if pad_needed:
            slices = [slice(None), slice(None)] + [slice(0, d) for d in spatial_dims]
            x = x[tuple(slices)]
            
        return x

class ChannelAttention(nn.Module):
    """
    Channel Attention Module.
    
    Extracts channel-wise attention weights by applying average and max pooling
    spatial operations, followed by a shared Multi-Layer Perceptron (MLP).

    Args:
        channels: Number of input channels.
        conv_op: Convolutional operator type (used to infer spatial dimensions).
        reduction_ratio: Reduction factor for the hidden MLP layer size.
        act_op: Activation operator type for the MLP.
        act_kwargs: Parameters for the activation operator.
        **kwargs: Required to contain 'adaptive_avg_pool_op' and 'adaptive_max_pool_op'.

    Returns:
        torch.Tensor: The channel-refined feature map.
    """
    def __init__(
        self,
        channels: int,
        conv_op: Type[_ConvNd],
        act_op: Type[nn.Module],
        act_kwargs: dict,
        reduction_ratio: int = 16,
        **kwargs):
        super().__init__()
          
        hidden_channels = max(channels // reduction_ratio, 1)

        self.avg_pool = kwargs['adaptive_avg_pool_op'](1)
        self.max_pool = kwargs['adaptive_max_pool_op'](1)

        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=True),
            act_op(**act_kwargs),
            nn.Linear(hidden_channels, channels, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels = x.size()[:2]

        out1 = self.avg_pool(x).view(batch_size, channels)
        out1 = self.mlp(out1)

        out2 = self.max_pool(x).view(batch_size, channels)
        out2 = self.mlp(out2)

        out = out1 + out2
        scale = torch.sigmoid(out).view(batch_size, channels, *([1] * (x.dim() - 2)))
        return x * scale

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Calculates spatial attention masks by concatenating average-pooled and
    max-pooled features across the channel dimension, followed by a spatial convolution.

    Args:
        conv_op: Convolutional operator type.
        spatial_kernel_size: Spatial size of the convolutional kernel. Must be an odd integer.
        **kwargs: Additional parameters.

    Returns:
        torch.Tensor: The spatially-refined feature map.
    """
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        spatial_kernel_size: int = 7,
        **kwargs
    ) -> None:
        super().__init__()
        if spatial_kernel_size % 2 == 0:
            raise ValueError(f"Spatial attention kernel size must be odd, got {spatial_kernel_size}.")

        self.conv = conv_op(
            in_channels=2,
            out_channels=1,
            kernel_size=spatial_kernel_size,
            padding=spatial_kernel_size // 2,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg_out, max_out], dim=1)
        scale = torch.sigmoid(self.conv(out))

        return x * scale

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Sequentially applies channel attention followed by spatial attention to
    refine feature maps.

    Args:
        channels: Number of input channels.
        conv_op: Convolutional operator type.
        reduction_ratio: MLP reduction ratio for channel attention.
        spatial_kernel_size: Kernel size for spatial attention. Must be odd.
        act_op: Activation operator type.
        act_kwargs: Activation arguments.
        no_spatial: If True, bypasses the spatial attention branch.
        no_channel: If True, bypasses the channel attention branch.
        **kwargs: Additional arguments injected by the model orchestrator.

    Returns:
        torch.Tensor: The feature map processed by CBAM.
    """
    def __init__(self,
                 channels: int,
                 conv_op: Type[_ConvNd],
                 act_op: Type[nn.Module],
                 act_kwargs: dict,
                 reduction_ratio: int = 16,
                 spatial_kernel_size: int = 7,
                 no_spatial: bool = False,
                 no_channel: bool = False,
                 **kwargs):
        super().__init__()
        if no_spatial and no_channel:
            raise ValueError("CBAM module instantiated with both channel and spatial attention disabled.")

        self.no_spatial = no_spatial
        self.no_channel = no_channel

        if not no_channel:
            self.channel_attention = ChannelAttention(
                channels=channels,
                conv_op=conv_op,
                reduction_ratio=reduction_ratio,
                act_op=act_op,
                act_kwargs=act_kwargs,
                **kwargs
            )

        if not no_spatial:
            self.spatial_attention = SpatialAttention(
                conv_op=conv_op,
                spatial_kernel_size=spatial_kernel_size,
                **kwargs
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.no_channel:
            x = self.channel_attention(x)
        if not self.no_spatial:
            x = self.spatial_attention(x)
        return x