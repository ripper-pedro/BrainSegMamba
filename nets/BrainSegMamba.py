"""
Main architectural implementation of the parametrized medical segmentation framework.

This module orchestrates the high-level components of the network,
which are composed by the simpler building blocks defined in 'blocks' file,
integrating the contractive path (Encoder) and the expansive path (Decoder) into a cohesive
U-Net structure. It features a parametrized design that uses State-Space Models (VSSLayer3D)
and can dynamically adjust its architecture based on the configuration profile
injected by the Trainer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.modules.conv import _ConvNd
from typing import List, Type
import warnings

from nnunetv2.nets.VSS3D import VSSLayer3D
from nnunetv2.nets.blocks import ConvBlock, UpConv, SpatialReducer, ResidualBlock
from nnunetv2.nets.blocks import ChannelLastWrapper, MambaWrapper, PatchEmbed, CBAM

class HybridConvMambaEncoder(nn.Module):
    """
    Hybrid Encoder combining Conv3D for high-resolution early stages
    and VSSLayer3D for deep, low-resolution global context.

        
    Args:
        input_channels: Channel count from the input volume.
        encoder_channel_sizes: Target channel dimensions for each resolution stage.
        stage_sizes: Pre-calculated spatial dimensions for HSCANS parameterization.
        num_convs_per_stage: Number of processing blocks per stage.
        skip_mamba_depths: Number of VSSLayer3D blocks applied before skip output.
        down_type: Defines the reduction operation ("conv" or "mamba") per stage.
        mamba_kwargs: Configuration dictionary for VSSLayer3D blocks.
        **block_kwargs: nnU-Net dependency injections (conv_op, norm_op, etc.).
    """
    def __init__(self,
                 input_channels: int,
                 encoder_channel_sizes: List[int],
                 stage_sizes: List[int],
                 num_convs_per_stage: List[int],
                 skip_mamba_depths: List[int],
                 down_type: List[str],
                 mamba_kwargs: dict,
                 **block_kwargs):
        super().__init__()
        
        # ---- Stem ----
        if block_kwargs["network_kwargs"]["preact"]:
            self.stem = ConvBlock(
                input_channels=input_channels,
                output_channels=encoder_channel_sizes[0],
                stride=1,
                layers_order=("conv",),
                **block_kwargs
            )
        else:
            self.stem = nn.Identity()
            
        self.stages = nn.ModuleList()
        self.mamba_layers = nn.ModuleList()
        in_ch = encoder_channel_sizes[0]

        # ---- Loop ----
        for i, out_ch in enumerate(encoder_channel_sizes):
            stage_blocks = []
            current_size = stage_sizes[i]
            
            # Spatial Reduction (Only for i > 0)
            if i > 0:
                stage_blocks.append(SpatialReducer(
                    input_channels=in_ch,
                    output_channels=out_ch,
                    down_type=down_type[i],
                    stride=2,
                    **block_kwargs
                ))
            
            # Processing: Mamba or Convolution
            if down_type[i] == "mamba":
                body_depth = max(1, num_convs_per_stage[i]) 
                mamba_layer = VSSLayer3D(
                    dim=out_ch, 
                    depth=body_depth, 
                    size=current_size, 
                    **mamba_kwargs
                )
                stage_blocks.append(MambaWrapper(mamba_layer))
            else:
                num_blocks = num_convs_per_stage[i] if i == 0 else max(1, num_convs_per_stage[i] - 1)
                for _ in range(num_blocks):
                    stage_blocks.append(ResidualBlock(
                        input_channels=out_ch,
                        output_channels=out_ch,
                        **block_kwargs
                    ))

            self.stages.append(nn.Sequential(*stage_blocks))
            in_ch = out_ch

            # Mamba Skips Processing
            if skip_mamba_depths[i] > 0:
                mamba_stage = MambaWrapper(VSSLayer3D(
                    dim=out_ch, 
                    depth=skip_mamba_depths[i], 
                    size=current_size, 
                    **mamba_kwargs
                ))
            else:
                mamba_stage = nn.Identity()
                
            self.mamba_layers.append(mamba_stage)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for stage, mamba_skip in zip(self.stages, self.mamba_layers):
            x = stage(x)
            # Optional: Enable gradient checkpointing for VRAM optimization (training takes longer).
            # x = checkpoint(stage, x, use_reentrant=False)

            if not isinstance(mamba_skip, nn.Identity):
                feature_to_skip = mamba_skip(x)
                # feature_to_skip =  checkpoint(mamba_skip, x, use_reentrant=False)
            else:
                feature_to_skip = x
            
            skips.append(feature_to_skip)

        return skips

class PureMambaEncoder(nn.Module):
    """
        Encoder strictly based on State-Space Model (SSM) philosophy.
    Uses PatchEmbed for initial downsampling and VSSLayer3D for all subsequent stages.
    
    Args:
        input_channels: Channel count from the input volume.
        encoder_channel_sizes: Target channel dimensions for each resolution stage.
        stage_sizes: Pre-calculated spatial dimensions for HSCANS parameterization.
        num_convs_per_stage: Number of VSSLayer3D blocks per stage.
        skip_mamba_depths: Number of VSSLayer3D blocks applied before skip output.
        stem_patch_size: Initial spatial reduction factor applied by PatchEmbed.
        mamba_kwargs: Configuration dictionary for VSSLayer3D blocks.
        **block_kwargs: nnU-Net dependency injections (conv_op, norm_op, etc.).

    """
    def __init__(self,
                 input_channels: int,
                 encoder_channel_sizes: List[int],
                 stage_sizes: List[int],
                 num_convs_per_stage: List[int],
                 skip_mamba_depths: List[int],
                 stem_patch_size: int,
                 mamba_kwargs: dict,
                 **block_kwargs):
        super().__init__()
        
        # ---- Stem ----
        self.stem = PatchEmbed(
            input_channels=input_channels,
            embed_dim=encoder_channel_sizes[0],
            conv_op=block_kwargs['conv_op'],
            patch_size=stem_patch_size
        )
        
        self.stages = nn.ModuleList()
        self.mamba_layers = nn.ModuleList()
        in_ch = encoder_channel_sizes[0]

        # ---- Loop ----
        for i, out_ch in enumerate(encoder_channel_sizes):
            stage_blocks = []
            current_size = stage_sizes[i]
            
            # Spatial Reduction (Only for i > 0, as PatchEmbed handled i=0)
            if i > 0:
                stage_blocks.append(SpatialReducer(
                    input_channels=in_ch,
                    output_channels=out_ch,
                    down_type="mamba",
                    stride=2,
                    **block_kwargs
                ))
            
            # Pure Mamba Processing
            body_depth = max(1, num_convs_per_stage[i]) 
            mamba_layer = VSSLayer3D(
                dim=out_ch, 
                depth=body_depth, 
                size=current_size, 
                **mamba_kwargs
            )
            stage_blocks.append(MambaWrapper(mamba_layer))

            self.stages.append(nn.Sequential(*stage_blocks))
            in_ch = out_ch

            # Mamba Skips Processing
            if skip_mamba_depths[i] > 0:
                mamba_stage = MambaWrapper(VSSLayer3D(
                    dim=out_ch, 
                    depth=skip_mamba_depths[i], 
                    size=current_size, 
                    **mamba_kwargs
                ))
            else:
                mamba_stage = nn.Identity()
                
            self.mamba_layers.append(mamba_stage)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for stage, mamba_skip in zip(self.stages, self.mamba_layers):
            x = stage(x)
            # Optional: Enable gradient checkpointing for VRAM optimization (training takes longer).
            # x = checkpoint(stage, x, use_reentrant=False)

            if not isinstance(mamba_skip, nn.Identity):
                feature_to_skip = mamba_skip(x)
                # feature_to_skip =  checkpoint(mamba_skip, x, use_reentrant=False)
            else:
                feature_to_skip = x
            
            skips.append(feature_to_skip)

        return skips

class BrainSegMambaBottleneck(nn.Module):
    """
    Bridge between encoder and decoder.
    Expands channels, processes with VSSLayer3D, and compresses back.

    Args:
        input_channels: Feature dimension from the deepest encoder stage.
        embedding_dim: Expanded channel dimension for Mamba processing.
        output_channels: Target feature dimension for the decoder input.
        bottleneck_depth: Number of VSSLayer3D blocks in the core Mamba stage.
        bottleneck_size: Spatial dimension at the bottleneck level (for HSCANS).
        mamba_kwargs: Configuration dictionary for VSSLayer3D blocks.
        **kwargs: nnU-Net dependency injections.
    """
    def __init__(self,
                 input_channels: int,
                 embedding_dim: int,
                 output_channels: int,
                 bottleneck_depth: int,
                 bottleneck_size: int,
                 mamba_kwargs: dict,
                 **kwargs):
        super().__init__()

        self.expand = ConvBlock(input_channels=input_channels,
                                output_channels=embedding_dim,
                                stride=1,
                                layers_order=("norm", "act", "conv"),
                                **kwargs)

        self.mamba = MambaWrapper(VSSLayer3D(dim=embedding_dim,
                                depth=bottleneck_depth,
                                size=bottleneck_size,
                                **mamba_kwargs))

        self.pre_ln = kwargs['norm_op'](embedding_dim, **kwargs['norm_kwargs'])
        if kwargs['norm_op'] is nn.LayerNorm:
            self.pre_ln = ChannelLastWrapper(self.pre_ln)


        self.compress1 = ConvBlock(input_channels=embedding_dim,
                                   output_channels=output_channels,
                                   stride=1,
                                   layers_order=("conv", "norm", "act"), **kwargs)
        self.compress2 = ConvBlock(input_channels=output_channels,
                                   output_channels=output_channels,
                                   stride=1,
                                   layers_order=("conv", "norm", "act"), **kwargs)

    def forward(self, x):
        x = self.expand(x)
        x = self.mamba(x)
        # Optional: Enable gradient checkpointing for VRAM optimization (training takes longer).
        #x = checkpoint(self.mamba, x, use_reentrant=False)
        x = self.pre_ln(x)
        x = self.compress1(x)
        x = self.compress2(x)
        return x

class HeavyDecoder(nn.Module):
    """
    Standard nnU-Net style decoder.
    Uses Transposed Convolutions and multiple Residual Blocks.
    
    Args:
        skip_type: Determines fusion logic ("concat", "sum", "sdi", or "mamba").
        num_classes: Number of output segmentation classes.
        decoder_channel_sizes: Target channel dimensions for upsampling stages.
        encoder_channel_sizes: Channel dimensions expected from skip connections.
        num_convs_per_stage: Number of ResidualBlocks applied after fusion.
        decoder_mamba_depths: Number of VSSLayer3D blocks applied if the stage is Mamba-based.
            # NOTE: Accessed in reverse order [i - 1] during bottom-up construction.
        down_type: List indicating the architectural paradigm ("residual", "mamba", etc) per stage.
            # NOTE: Accessed in reverse order [i - 1] during bottom-up construction.
        deep_supervision: Enables multi-scale topological loss.
        mamba_kwargs: Dictionary containing state-space specific parameters.
        **kwargs: nnU-Net dependency injections.
    """
    def __init__(self,
                 skip_type: str,
                 num_classes: int,
                 decoder_channel_sizes: List[int],
                 encoder_channel_sizes: List[int],
                 num_convs_per_stage: List[int],
                 decoder_mamba_depths: List[int],
                 down_type: List[str],
                 stage_sizes: List[int],
                 stem_patch_size: int,
                 deep_supervision: bool,
                 mamba_kwargs: dict,
                 **kwargs):
        super().__init__()

        self.skip_type = skip_type
        self.deep_supervision = deep_supervision
        conv_op = kwargs['conv_op']
        transpconv_op = kwargs['transpconv_op']
        self.interp_mode = kwargs['interp_mode']

        self.upsamples = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.seg_layers = nn.ModuleList()

        # Build decoder stages from deepest to shallowest resolution
        for i in range(len(decoder_channel_sizes) - 1, 0, -1):
            dec_in_ch = decoder_channel_sizes[i]
            dec_out_ch = decoder_channel_sizes[i - 1]
            skip_ch = encoder_channel_sizes[i - 1]
            current_size = stage_sizes[i - 1]
            stage_type = down_type[i - 1]

            # UpConv naturally halves the channels: dec_in_ch -> dec_out_ch
            self.upsamples.append(UpConv(
                input_channels=dec_in_ch,
                output_channels=dec_out_ch,
                **kwargs
            ))

            # Channel geometry based on the skip connection type
            if self.skip_type in ["concat", "mamba"]:
                fusion_in_ch = dec_out_ch + skip_ch
            elif self.skip_type in ["sum", "sdi"]:
                fusion_in_ch = dec_out_ch
            else:
                raise ValueError(f"Unknown skip_type in Decoder: {self.skip_type}")

            stage_blocks = []

            # First block digests the fused features
            stage_blocks.append(ResidualBlock(
                input_channels=fusion_in_ch,
                output_channels=dec_out_ch,
                **kwargs
            ))

            if stage_type == "mamba": 
                # Dynamically instantiate Mamba layers based on the specified depths
                mamba_depth = decoder_mamba_depths[i - 1]
                if mamba_depth > 0:
                    mamba_kwargs = mamba_kwargs or {}
                    stage_blocks.append(MambaWrapper(VSSLayer3D(
                        dim=dec_out_ch,
                        depth=mamba_depth,
                        size=current_size,
                        **mamba_kwargs
                    )))
            else:
                for _ in range(max(0, num_convs_per_stage[i - 1] - 1)):
                    stage_blocks.append(ResidualBlock(
                        input_channels=dec_out_ch,
                        output_channels=dec_out_ch,
                        **kwargs
                    ))
            
            self.stages.append(nn.Sequential(*stage_blocks))

            # Unified geometric DS heads mapped to each decoder scale output
            self.seg_layers.append(conv_op(dec_out_ch, num_classes, kernel_size=1))

        # Learnable final cascade to reconstruct features compressed by the stem
        num_final = int(round(math.log2(stem_patch_size))) if stem_patch_size > 1 else 0
        
        self.final_ups = nn.ModuleList([
            transpconv_op(decoder_channel_sizes[0], decoder_channel_sizes[0], kernel_size=2, stride=2)
            for _ in range(num_final)
        ])
        
        if num_final > 1:
            self.ds_heads_cascade = nn.ModuleList([
                conv_op(decoder_channel_sizes[0], num_classes, kernel_size=1)
                for _ in range(num_final - 1)
            ])
        else:
            self.ds_heads_cascade = None

        self.final_conv = conv_op(decoder_channel_sizes[0], num_classes, kernel_size=1)


    def forward(self, x, skips):
        decoder_ds = []

        for i, (up, stage) in enumerate(zip(self.upsamples, self.stages)):
            x = up(x)
            skip = skips[-(i + 1)]
            
            # Spatial alignment safeguard
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode=self.interp_mode, align_corners=False)
            
            # Fusion operation
            if self.skip_type in ["concat", "mamba"]:
                x = torch.cat((skip, x), dim=1)
            elif self.skip_type in ["sum", "sdi"]:
                x = x + skip

            x = stage(x)

            # Collect spatial intermediate maps from the heavy decoder blocks
            if self.deep_supervision:
                if len(self.final_ups) > 0 or i < len(self.stages) - 1:
                    decoder_ds.append(self.seg_layers[i](x))

        # Process the learnable final cascade to full input size
        cascade_ds = []
        for k, up in enumerate(self.final_ups):
            x = up(x)
            if self.deep_supervision and (self.ds_heads_cascade is not None) and (k < len(self.final_ups) - 1):
                cascade_ds.append(self.ds_heads_cascade[k](x))
                
        final_seg = self.final_conv(x)

        if self.deep_supervision:
            return [final_seg] + cascade_ds[::-1] + decoder_ds[::-1]
        return final_seg

class LightDecoder(nn.Module):
    """
    Lightweight decoder optimized for minimal parameter count.
    Uses parameter-free interpolation and 1x1 convolutions for feature alignment.
    
    Args:
        skip_type: Determines fusion logic ("concat", "sum", "sdi", or "mamba").
        num_classes: Number of output segmentation classes.
        decoder_channel_sizes: Target channel dimensions for upsampling stages.
        encoder_channel_sizes: Channel dimensions expected from skip connections.
        decoder_mamba_depths: List indicating the VSSLayer3D depth per decoder stage.
            # NOTE: Accessed in reverse order [i - 1] during bottom-up construction.
        deep_supervision: If True, enables multi-scale loss extraction.
        **kwargs: nnU-Net dependency injections.
    """
    def __init__(self,
                 skip_type: str,
                 num_classes: int,
                 decoder_channel_sizes: List[int],
                 encoder_channel_sizes: List[int],
                 decoder_mamba_depths: List[int],
                 stage_sizes: List[int],
                 stem_patch_size: int,
                 deep_supervision: bool,
                 mamba_kwargs: dict = None,
                 **kwargs):
        super().__init__()

        self.skip_type = skip_type
        self.deep_supervision = deep_supervision
        self.upsample_type = kwargs['upsample_type']
        conv_op = kwargs['conv_op']
        transpconv_op = kwargs['transpconv_op']
        self.interp_mode = kwargs['interp_mode']
        
        self.align_convs = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.ds_heads = nn.ModuleList()

        if self.upsample_type == "transpose":
            self.upsamples = nn.ModuleList()
        
        # Build lightweight stages (bottom-up)
        for i in range(len(decoder_channel_sizes) - 1, 0, -1):
            dec_in_ch = decoder_channel_sizes[i]
            dec_out_ch = decoder_channel_sizes[i - 1]
            skip_ch = encoder_channel_sizes[i - 1]
            current_size = stage_sizes[i - 1]
            
            if self.upsample_type == "transpose":
                self.upsamples.append(UpConv(
                    input_channels=dec_in_ch,
                    output_channels=dec_in_ch,
                    **kwargs
                ))

            # Alignment logic for F.interpolate (which preserves dec_in_ch)
            if self.skip_type in ["sum", "sdi"]:
                # We must project the interpolated 'x' down to match the skip channels before summing
                self.align_convs.append(conv_op(dec_in_ch, skip_ch, kernel_size=1, bias=False))
                fusion_in_ch = skip_ch
            elif self.skip_type in ["concat", "mamba"]:
                # Concat mode: no need to align before concatenating
                self.align_convs.append(nn.Identity())
                fusion_in_ch = dec_in_ch + skip_ch
            else:
                raise ValueError(f"Unsupported skip type: {self.skip_type}")
                
            stage_blocks = []
            
            # 1x1 conv to fuse and project to the next stage's channel size
            stage_blocks.append(conv_op(
                in_channels=fusion_in_ch, 
                out_channels=dec_out_ch, 
                kernel_size=1, 
                bias=False
            ))

            # Dynamically instantiate Mamba layers based on the specified depths
            mamba_depth = decoder_mamba_depths[i - 1]
            if mamba_depth > 0:
                mamba_kwargs = mamba_kwargs or {}
                stage_blocks.append(MambaWrapper(VSSLayer3D(
                    dim=dec_out_ch,
                    depth=mamba_depth,
                    size=current_size,
                    **mamba_kwargs
                )))
            
            self.stages.append(nn.Sequential(*stage_blocks))

            # Geometric heads based on output channels for each state
            self.ds_heads.append(conv_op(dec_out_ch, num_classes, kernel_size=1))

        # Final upsampling cascade to mitigate resolution loss from the stem
        num_final = int(round(math.log2(stem_patch_size))) if stem_patch_size > 1 else 0
        
        self.final_ups = nn.ModuleList([
            transpconv_op(decoder_channel_sizes[0], decoder_channel_sizes[0], kernel_size=2, stride=2)
            for _ in range(num_final)
        ])
        
        if num_final > 1:
            self.ds_heads_cascade = nn.ModuleList([
                conv_op(decoder_channel_sizes[0], num_classes, kernel_size=1)
                for _ in range(num_final - 1)
            ])
        else:
            self.ds_heads_cascade = None

        self.final_conv = conv_op(decoder_channel_sizes[0], num_classes, kernel_size=1)

    def forward(self, x, skips):
        decoder_ds = []
        
        for i, (align, stage) in enumerate(zip(self.align_convs, self.stages)):
            skip = skips[-(i + 1)]
            
            if self.upsample_type == "transpose":
                x = self.upsamples[i](x)
            else:
                x = F.interpolate(x, size=skip.shape[2:], mode=self.interp_mode, align_corners=True)

            # Spatial mismatch protection
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode=self.interp_mode, align_corners=True)
            
            # Fusion operation
            if self.skip_type in ["sum", "sdi"]:
                x = align(x) # Project channels down to match skip
                x = x + skip
            else:
                x = torch.cat([skip, x], dim=1)
                
            x = stage(x)

            # Stores intermediate resolutions of the decoder (smaller to larger)
            if self.deep_supervision:
                if len(self.final_ups) > 0 or i < len(self.stages) - 1:
                    decoder_ds.append(self.ds_heads[i](x))

        # Execution of the final learnable cascade towards the ground truth
        cascade_ds = []
        for k, up in enumerate(self.final_ups):
            x = up(x)
            if self.deep_supervision and (self.ds_heads_cascade is not None) and (k < len(self.final_ups) - 1):
                cascade_ds.append(self.ds_heads_cascade[k](x))
                
        final_seg = self.final_conv(x)
        if self.deep_supervision:
            # Reverts order to guarantee the convention of nnU-Net: larger to smaller
            return [final_seg] + cascade_ds[::-1] + decoder_ds[::-1]
        return final_seg

class SDI(nn.Module):
    """
    Semantics and Detail Infusion (SDI) module.

    Fuses multi-scale features from an encoder architecture. It accepts a dynamic
    list of input scales and aligns them to a specified spatial anchor via 
    adaptive upsampling and downsampling. The fusion is enhanced using a 
    Convolutional Block Attention Module (CBAM) per scale and finalized via
    an element-wise Hadamard product.

    Args:
        in_channels_list: List of channel dimensions from all encoder stages.
        mid_channel: Target processing width (anchored to base_num_features).
        cbam_kwargs: Structural configurations for attention sub-modules.
        block_kwargs: Operational parameters including convolution operator types.
        expand_channels: If True, restores outputs to original encoder widths.
            If False, maintains mid_channel width across all scales.
    """
    def __init__(
        self,
        in_channels_list: List[int],
        mid_channel: int,
        cbam_kwargs: dict,
        block_kwargs: dict,
        expand_channels: bool = True,
    ) -> None:
        super().__init__()
        
        self.expand_channels = expand_channels

        conv_op = block_kwargs['conv_op']
        self.num_scales = len(in_channels_list)
        if self.num_scales < 2:
            raise ValueError(f"SDI module requires at least 2 input scales. Received {self.num_scales}.")

        self.cbams = nn.ModuleList([
            CBAM(
                channels=ch,
                **cbam_kwargs,
                **block_kwargs
            )
            for ch in in_channels_list
        ])

        self.reductions = nn.ModuleList([
            conv_op(ch, mid_channel, kernel_size=1, bias=False)
            for ch in in_channels_list
        ])

        self.smooth_convs = nn.ModuleList([
            conv_op(mid_channel, mid_channel, kernel_size=3, padding=1, bias=False)
            for _ in range(self.num_scales)
        ])

        self.expansions = nn.ModuleList([
            conv_op(mid_channel, ch, kernel_size=1, bias=False)
            for ch in in_channels_list
        ])

        self.interp_mode = block_kwargs['interp_mode']
        
        if issubclass(conv_op, nn.Conv1d):
            self.adaptive_pool_func = F.adaptive_avg_pool1d
        elif issubclass(conv_op, nn.Conv2d):
            self.adaptive_pool_func = F.adaptive_avg_pool2d
        else:
            self.adaptive_pool_func = F.adaptive_avg_pool3d

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        processed_xs = []
        for i, x in enumerate(xs):
            x = self.cbams[i](x)
            x = self.reductions[i](x)
            processed_xs.append(x)

        enriched_skips = []
        for target_idx, target_x in enumerate(xs):
            target_size = target_x.shape[2:]
            aligned_features = []
            
            for i, x in enumerate(processed_xs):
                if x.shape[2:] != target_size:
                    if x.shape[-1] > target_size[-1]:  
                        x = self.adaptive_pool_func(x, target_size)
                    else:                              
                        x = F.interpolate(x, size=target_size, mode=self.interp_mode, align_corners=True)
                x = self.smooth_convs[i](x)
                aligned_features.append(x)

            result = aligned_features[0]
            for x in aligned_features[1:]:
                result = result * x
            if self.expand_channels:
                result = self.expansions[target_idx](result)
            enriched_skips.append(result)

        return enriched_skips

class BrainSegMamba(nn.Module):
    """
    Parametrized Medical Segmentation Framework.
    
    Dynamically routes to the correct Encoder and Decoder based on Trainer profiles,
    supporting pure State-Space Models (Mamba), Hybrid Convolutional-Mamba architectures,
    and multiple skip connection paradigms.

    Args:
        input_channels: Number of input image channels (e.g., 4 for BraTS).
        num_classes: Number of output segmentation classes.
        encoder_type: Architectural strategy for feature extraction. Options:
            - "pure_mamba": Uses PatchEmbed and Mamba blocks for all stages.
            - "hybrid": Uses a convolutional stem and downsampling, mixed with Mamba.
        decoder_type: Architectural strategy for upsampling. Options:
            - "heavy": Standard nnU-Net style with Transposed Convs and ResidualBlocks.
            - "light": VM-UNet style with parameter-free interpolation and 1x1 Convs.
        deep_supervision: If True, outputs segmentation maps at all resolution stages.
        skip_type: Fusion strategy for encoder-decoder skip connections. Options:
            - "concat": Standard U-Net feature concatenation.
            - "sum": Element-wise addition (requires equal channel sizes).
            - "sdi": Routes all skips through a Semantics and Detail Infusion module.
            - "mamba": Processes skip features through VSSLayer3D before concatenation.
        conv_op, conv_kwargs, norm_op, norm_kwargs, act_op, act_kwargs: Base nnU-Net
            dependency injections for convolutional operations.
        encoder_channel_sizes: List of channel dimensions for each encoder stage.
        decoder_channel_sizes: List of channel dimensions for each decoder stage.
        down_type: List of strings dictating the operation per stage.
        num_convs_per_stage: List indicating the depth of processing blocks per stage.
        skip_mamba_depths: List indicating the VSSLayer3D depth for skip connections.
        bottleneck_depth: Number of VSSLayer3D blocks at the deepest resolution.
        embedding_dim: Channel dimension at the bottleneck.
        img_dim: Base spatial dimension of the input volume (used for HSCANS size).
        stem_patch_size: Downsampling factor for the initial PatchEmbed (if pure_mamba).
        mamba_kwargs: Dictionary of VSSLayer3D specific parameters (d_state, drop, etc.).
        cbam_kwargs: Dictionary of Attention module parameters (if SDI is used).
    """
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 encoder_type: str,
                 decoder_type: str,
                 deep_supervision: bool,
                 skip_type: str,
                 
                 # ---- Operations (from get_network_config) ----
                 conv_op: Type[_ConvNd],
                 conv_kwargs: dict,
                 norm_op: Type[nn.Module],
                 norm_kwargs: dict,
                 act_op: Type[nn.Module],
                 act_kwargs: dict,
                 pool_op: Type[nn.Module],
                 transpconv_op: Type[_ConvNd],
                 upsample_type: str,
                 adaptive_avg_pool_op: Type[nn.Module],
                 adaptive_max_pool_op: Type[nn.Module],
                 interp_mode: str,
                 network_kwargs: dict,
                 
                 # ---- Architecture sizes ----
                 encoder_channel_sizes: List[int],
                 decoder_channel_sizes: List[int],
                 down_type: List[str],
                 num_convs_per_stage: List[int],
                 skip_mamba_depths: List[int],
                 decoder_mamba_depths: List[int],
                 bottleneck_depth: int,
                 embedding_dim: int,
                 img_dim: int,
                 stem_patch_size: int,
                 
                 # ---- Configuration Dicts ----
                 mamba_kwargs: dict,
                 cbam_kwargs: dict):
        super().__init__()

        # INTEGRITY CHECKS & WARNINGS
        valid_skip_types = ["concat", "sum", "sdi", "mamba"]
        if skip_type not in valid_skip_types:
            raise ValueError(f"Invalid skip_type '{skip_type}'. Must be one of {valid_skip_types}.")
            
        if len(encoder_channel_sizes) != len(down_type):
            raise ValueError("down_type must have the same length as encoder_channel_sizes!")
        if len(encoder_channel_sizes) != len(num_convs_per_stage):
            raise ValueError("num_convs_per_stage must have the same length as encoder_channel_sizes!")
        if len(encoder_channel_sizes) != len(skip_mamba_depths):
            raise ValueError("skip_mamba_depths must have the same length as encoder_channel_sizes!")
        if len(encoder_channel_sizes) != len(decoder_mamba_depths):
            raise ValueError("decoder_mamba_depths must have the same length as encoder_channel_sizes!")

        if skip_type in ["sdi", "sum", "concat"] and any(d > 0 for d in skip_mamba_depths):
            warnings.warn(
                f"skip_type='{skip_type}' overrides skip_mamba_depths. Setting all to 0. "
                "If you wanted Mamba skips, set skip_type='mamba'."
            )
            skip_mamba_depths = [0] * len(skip_mamba_depths)
        if encoder_type == "pure_mamba" and any(d != "mamba" for d in down_type):
            warnings.warn(
                f"encoder_type='pure_mamba' overrides down_type. All stages will use 'mamba' downsampling. "
                f"Your input down_type={down_type} will be ignored."
            )
            down_type = ["mamba"] * len(down_type)
            
        self.skip_type = skip_type
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.sdi_expand_channels = network_kwargs["sdi_expand_channels"]
        self.bottleneck_depth = bottleneck_depth
        
        # Determine base spatial reduction from stem configuration.
        self.stem_reduces_by = stem_patch_size if encoder_type == "pure_mamba" else 1

        # Centralized kwargs for building blocks
        block_kwargs = {
            "conv_op": conv_op,
            "conv_kwargs": conv_kwargs,
            "pool_op": pool_op,
            "transpconv_op": transpconv_op,
            "upsample_type": upsample_type,
            "norm_op": norm_op,
            "adaptive_avg_pool_op": adaptive_avg_pool_op,
            "adaptive_max_pool_op": adaptive_max_pool_op,
            "interp_mode": interp_mode,
            "norm_kwargs": norm_kwargs,
            "act_op": act_op,
            "act_kwargs": act_kwargs,
            "network_kwargs": network_kwargs,
        }
        self.block_kwargs = block_kwargs

        # Pre-calculate spatial dimensions for Mamba's HSCANS
        stage_sizes = []
        curr = max(1, -(-img_dim // self.stem_reduces_by))
        for i in range(len(encoder_channel_sizes)):
            if i >= 1: curr = max(1, -(-curr // 2))
            stage_sizes.append(curr)

        # ENCODER INSTANTIATION
        if encoder_type == "pure_mamba":
            self.encoder = PureMambaEncoder(
                input_channels=input_channels,
                encoder_channel_sizes=encoder_channel_sizes,
                stage_sizes=stage_sizes,
                num_convs_per_stage=num_convs_per_stage,
                skip_mamba_depths=skip_mamba_depths,
                stem_patch_size=stem_patch_size,
                mamba_kwargs=mamba_kwargs,
                **block_kwargs
            )
        elif encoder_type == "hybrid":
            self.encoder = HybridConvMambaEncoder(
                input_channels=input_channels,
                encoder_channel_sizes=encoder_channel_sizes,
                stage_sizes=stage_sizes,
                num_convs_per_stage=num_convs_per_stage,
                skip_mamba_depths=skip_mamba_depths,
                down_type=down_type,
                mamba_kwargs=mamba_kwargs,
                **block_kwargs
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # BOTTLENECK & SDI INSTANTIATION
        if bottleneck_depth > 0:
            self.bottleneck = BrainSegMambaBottleneck(
                input_channels=encoder_channel_sizes[-1],
                embedding_dim=embedding_dim,
                output_channels=decoder_channel_sizes[-1],
                bottleneck_depth=bottleneck_depth,
                bottleneck_size=stage_sizes[-1],
                mamba_kwargs=mamba_kwargs,
                **block_kwargs
            )
        else:
            self.bottleneck = nn.Identity()

        if self.skip_type == "sdi":
            self.sdi = SDI(
                in_channels_list=encoder_channel_sizes,
                mid_channel=decoder_channel_sizes[0],
                cbam_kwargs=cbam_kwargs,
                block_kwargs=block_kwargs,
                expand_channels=self.sdi_expand_channels
            )
            
            if not self.sdi_expand_channels:
                decoder_channel_sizes = [decoder_channel_sizes[0]] * len(decoder_channel_sizes)
                effective_skip_sizes = [decoder_channel_sizes[0]] * len(encoder_channel_sizes)
                
                # Align the deepest compressed skip to match the expected decoder input channels
                self.deepest_skip_align = block_kwargs['conv_op'](
                    decoder_channel_sizes[0],
                    decoder_channel_sizes[-1],
                    kernel_size=1,
                    bias=False
                )
            else:
                effective_skip_sizes = encoder_channel_sizes
                self.deepest_skip_align = nn.Identity()
        else:
            effective_skip_sizes = encoder_channel_sizes
            self.deepest_skip_align = nn.Identity()

        # DECODER INSTANTIATION
        if decoder_type == "light":
            self.decoder = LightDecoder(
                skip_type,
                num_classes=num_classes,
                decoder_channel_sizes=decoder_channel_sizes,
                encoder_channel_sizes=effective_skip_sizes,
                decoder_mamba_depths=decoder_mamba_depths,
                stage_sizes=stage_sizes,
                deep_supervision=deep_supervision,
                stem_patch_size=self.stem_reduces_by,
                mamba_kwargs=mamba_kwargs,
                **block_kwargs
            )
        elif decoder_type == "heavy":
            self.decoder = HeavyDecoder( 
                skip_type,
                num_classes=num_classes,
                decoder_channel_sizes=decoder_channel_sizes,
                encoder_channel_sizes=effective_skip_sizes,
                num_convs_per_stage=num_convs_per_stage,
                decoder_mamba_depths=decoder_mamba_depths,
                down_type=down_type,
                stage_sizes=stage_sizes,
                stem_patch_size=self.stem_reduces_by,
                deep_supervision=deep_supervision,
                mamba_kwargs=mamba_kwargs,
                **block_kwargs
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

        self.interp_mode = interp_mode

    def forward(self, x):
        # Feature Extraction
        skips = self.encoder(x)
        
        # Sequential Bottleneck Processing
        bottleneck_out = self.bottleneck(skips[-1])
        
        # Global Semantic Infusion
        if self.skip_type == "sdi":
            enriched_skips = self.sdi(skips)
            
            # Project the deepest skip connection to match decoder feature dimension
            aligned_deepest_skip = self.deepest_skip_align(enriched_skips[-1])
            
            if self.bottleneck_depth > 0:
                fused_bottleneck = bottleneck_out + aligned_deepest_skip
            else:
                fused_bottleneck = aligned_deepest_skip
                
            decoder_skips = enriched_skips[:-1]
        else:
            fused_bottleneck = bottleneck_out
            decoder_skips = skips[:-1]
        
        # Interpolation Alignment
        if fused_bottleneck.shape[2:] != decoder_skips[-1].shape[2:]:
            fused_bottleneck = F.interpolate(
                fused_bottleneck, 
                size=decoder_skips[-1].shape[2:], 
                mode=self.interp_mode, 
                align_corners=True
            )
             
        # Expansive Decoding
        decoder_outputs = self.decoder(fused_bottleneck, decoder_skips)
        
        return decoder_outputs