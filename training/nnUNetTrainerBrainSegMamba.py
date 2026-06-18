"""
Trainer configuration for the Parametrized BrainSegMamba Network.

This trainer configures and initializes the BrainSegMamba architecture within the nnU-Net framework.
It provides explicit control over the network's structural choices (Encoder, Decoder, Skip connections)
and the specific parameters for State-Space Models (VSSLayer3D).

Recommended Configurations (Architecture Profiles):
-------------------------------------------------
1. VM-UNet V2 Emulation (Lightweight, Global Context Focus):
    - encoder_type: "pure_mamba"
    - decoder_type: "light"
    - bottleneck_depth: 0
    - skip_type: "sdi"
    - enable_deep_supervision: True
    - upsample_type: "transpose"
    - stem_patch_size: 2 (Crucial for VRAM stability in 3D pure mamba)
    - norm_type: "ln" (LayerNorm)
    - act_type: "gelu"
    - d_state: 16
    - batch_size: 1 (Crucial for VRAM stability in 3D pure mamba)
    - sdi_expand_channels: False

2. MedSegMamba / Robust Hybrid (Early CNN Spatial Extraction):
    - encoder_type: "hybrid"
    - decoder_type: "heavy"
    - skip_type: "concat"
    - enable_deep_supervision: True
    - upsample_type: "transpose"
    - stem_patch_size: 1 (CNN handles initial high-res)
    - norm_type: "gn" (GroupNorm)
    - act_type: "leakyrelu"
    - d_state: 64
    - batch_size: 2
    - sdi_expand_channels: True
"""

import inspect
import torch
from torch import nn
from torch._dynamo import OptimizedModule

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.nets.config import get_network_config
from nnunetv2.nets.BrainSegMamba import BrainSegMamba


class nnUNetTrainerBrainSegMamba(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-4
        self.weight_decay = 1e-2
        self.num_epochs = 1000
        self.batch_size = 1 # Must be 1 for "pure_mamba" encoder to prevent OOM
        self.enable_deep_supervision = True

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(),
                                      self.initial_lr,
                                      weight_decay=self.weight_decay,
                                      eps=1e-4)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: list,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        """
        Constructs the BrainSegMamba architecture.
        
        This method defines the specific instantiation parameters for the network.
        
        Options Mapping:
        ----------------
        [Topology]
        encoder_type:
            "hybrid": Combines CNN downsampling with SSM context processing.
            "pure_mamba": Strict SSM/Mamba architecture with PatchEmbed block.
        decoder_type: 
            "heavy": Symmetrical CNN decoder supporting deep supervision.
            "light": Rapid. parameter-free stage fusion
        skip_type:
            "concat": Standard concatenation.
            "sum": Element-wise addition.
            "mamba": Skip connection feature refinement via VSSLayer3D.
            "sdi": Multi-scale Semantics and Detail Infusion.
        skip_mamba_depths:
            List of VSSLayer3D block depths applied at each skip connection.
            Is set to [0, 0, ...] if skip_type is not "mamba".
        decoder_mamba_depths:
            List of VSSLayer3D block depths applied per stage in the expansive path (Decoder).
            Indexed from shallowest (high-res) to deepest (low-res) by reading the list in reverse order when applying.
            For the heavy_decoder, it follows the "down_type" list (also in reverse order) to apply either a mamba block or a residual block.
        num_stages:
            Number of stages in the network.
        down_type (per stage):
            "residual": Residual-based downsampling with learnable parameters.
            "conv": Strided convolution for feature extraction and resolution reduction.
            "maxpool": Non-learnable max pooling followed by a feature projection layer.
            "mamba": Patch merging dimensionality reduction.
        num_convs_per_stage:
            List of the number of convolutional blocks in each stage.
        bottleneck_depth:
            Depth of the bottleneck layer. Set to 0 to completely disable
            the dedicated bottleneck processing path (bypassing via nn.Identity).

        [Channels & Sizes]
        stem_patch_size: 
            Must be >= 2 for "pure_mamba" encoder to prevent Out-Of-Memory errors in 3D.
            Use 1 for "hybrid" encoder.
        img_patch_size:
            Spatial dimensions of the input image patch.
        base_num_features:
            Base number of features in the network.
        max_features:
            Maximum cap for the number of features in the network.
        embedding_dim:
            Dimensionality of the embedding layer in the bottleneck.

        [Components & Operations]
        spatial_dims: 
            Number of spatial dimensions (e.g., 3 for 3D).
        norm_type: 
            "ln" (LayerNorm), "in" (InstanceNorm), "bn" (BatchNorm), "gn" (GroupNorm).
        act_type: 
            "gelu", "relu", "leakyrelu".
        upsample_type: 
            "transpose": Learnable spatial expansion via transposed convolution.
            "upsample": Non-learnable interpolation. Followed by convolutional refinement in heavy decoder.
        preact:
            True: Pre-activation layout (norm, act, conv).
            False: Standard layout (conv, norm, act).
        id:
            True: Identity mapping in residual blocks.
            False: 1x1 Conv projection in residual blocks.
        fusion_mode:
            Mathematical strategy used by the SDI module to fuse multi-scale features:
            - "sigmoid": Normalizes inputs and applies a Sigmoid attention mask before Hadamard product. Prevents gradient explosion while preserving semantic gating.
            - "sum": Simple element-wise addition (most stable).
            - "concat": Concatenates all scales across the channel dimension and compresses them back using a 1x1 convolution.
        sdi_expand_channels:
            - True: After fusion, the SDI module projects the features back to their original encoder channel dimensions using 1x1 convolutions.
            - False: The SDI module maintains the 'mid_channel' width across all output scales, significantly reducing parameters in the decoder.

        [State-Space Models (Mamba)]
        d_state:
            The hidden state dimension of the SSM. Typically 16 for deep architectures (like VM-UNet V2) or 64 for lighter hybrid architectures.
        version:
            The underlying VSS algorithm version (e.g., "v5").
        scan_type:
            The spatial scanning heuristic used to flatten 3D features into 1D sequences (e.g., "scan").
            
        [Attention (CBAM)]
        reduction_ratio:
            The channel reduction factor used in the Channel Attention MLP. Higher values save memory.
        spatial_kernel_size:
            Kernel size for spatial attention. Must be an odd number (e.g., 3, 5, 7).
        no_spatial / no_channel:
            If True, disables the respective attention branch. Cannot both be True simultaneously.
        """
        
        # =====================================================================
        # DYNAMIC NNUNET CONFIGURATION EXTRACTION (Modif by Pedro Ripper)
        # =====================================================================
        # Uncomment the block below if the architecture must dynamically adapt 
        # to nnU-Net's heuristic plans.json (patch_sizes, pooling layers, etc.).
        # 
        # caller_frame = inspect.currentframe().f_back
        # config_manager = caller_frame.f_locals.get('configuration_manager')
        # if config_manager is None:
        #     caller_self = caller_frame.f_locals.get('self')
        #     if caller_self is not None and hasattr(caller_self, 'configuration_manager'):
        #         config_manager = caller_self.configuration_manager
        # if config_manager is None:
        #     raise RuntimeError("Failed to trace configuration_manager in nnU-Net memory.")
        #
        # img_patch_size = max(config_manager.patch_size)
        # if img_patch_size > 128:
        #     raise RuntimeError(f"VRAM ALERT: Giant patch_size: {config_manager.patch_size}.")
        # 
        # num_stages = len(config_manager.pool_op_kernel_sizes)
        # if num_stages > 8:
        #     raise RuntimeError(f"VRAM ALERT: Deep network ({num_stages} stages).")
        #
        # base_num_features = min(getattr(config_manager, 'UNet_base_num_features', 32), 48)
        # 
        # Generalization loop for array parameters:
        # num_convs_per_stage = (num_convs_per_stage_design + [num_convs_per_stage_design[-1]] * 4)[:num_stages]
        # down_type = (down_type_design + [down_type_design[-1]] * 4)[:num_stages]
        # skip_mamba_depths = (skip_mamba_depths_design + [skip_mamba_depths_design[-1]] * 4)[:num_stages]
        # =====================================================================

        # ARCHITECTURAL TOPOLOGY
        encoder_type = "pure_mamba" 
        decoder_type = "light" 
        
        skip_type = "sdi"
        skip_mamba_depths = [0, 0, 1, 1, 1] # Doesn't change anything if "skip_type" is not "mamba"
        
        num_convs_per_stage = [2, 3, 4, 4, 8]
        num_stages = len(num_convs_per_stage)
        down_type = ["residual", "residual", "residual", "mamba", "mamba"] # Doesn't change anything if "encoder_type" is pure_mamba

        bottleneck_depth = 0
        decoder_mamba_depths = [2, 3, 4, 4, 8]
        
        # CHANNEL AND SIZES PARAMETERIZATION
        stem_patch_size = 2 # Must be >=2 for "pure_mamba" encoder to prevent OOM
        img_patch_size = 96
        
        base_num_features = 48
        max_features = 768
        encoder_channel_sizes = [min(base_num_features * 2 ** i, max_features) for i in range(num_stages)]
        decoder_channel_sizes = list(encoder_channel_sizes)
        embedding_dim = 768

        # COMPONENT CONFIGURATIONS
        spatial_dims = 3
        norm_type = "ln" 
        act_type = "gelu" 
        upsample_type = "transpose" 
        
        config = get_network_config(spatial_dims, norm_type, act_type, upsample_type)

        # Extra topological parameters
        network_kwargs = config.get('network_kwargs', {})
        network_kwargs['preact'] = True # Used in the ResidualBlock
        network_kwargs['id'] = True # Used in the ResidualBlock
        network_kwargs['sdi_expand_channels'] = False
        network_kwargs['fusion_mode'] = "sigmoid"
        config['network_kwargs'] = network_kwargs

        # Mamba / VSSLayer3D Specific Parameters
        mamba_kwargs = {
            "d_state": 16,
            "version": "v5",
            "scan_type": "scan",
            "expansion_factor": 1,
            "drop_path": 0.2,
            "attn_drop": 0.1,
            "mlp_drop": 0.1,
        }

        # SDI / Convolutional Block Attention Module Parameters
        cbam_kwargs = {
            "reduction_ratio": 16,
            "spatial_kernel_size": 3,
            "no_spatial": False,
            "no_channel": False,
        }

        # MODEL INSTANTIATION
        model = BrainSegMamba(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            deep_supervision=enable_deep_supervision,
            skip_type=skip_type,

            # Sub-module operations
            conv_op=config['conv_op'],
            conv_kwargs=config['conv_kwargs'],
            norm_op=config['norm_op'],
            norm_kwargs=config['norm_kwargs'],
            act_op=config['act_op'],
            act_kwargs=config['act_kwargs'],
            pool_op=config['pool_op'],
            transpconv_op=config['transpconv_op'],
            upsample_type=config['upsample_type'],
            adaptive_avg_pool_op=config['adaptive_avg_pool_op'],
            adaptive_max_pool_op=config['adaptive_max_pool_op'],
            interp_mode=config['interp_mode'],
            network_kwargs=config['network_kwargs'],
            
            # Architecture dimensions
            down_type=down_type,
            encoder_channel_sizes=encoder_channel_sizes,
            decoder_channel_sizes=decoder_channel_sizes,
            num_convs_per_stage=num_convs_per_stage,
            skip_mamba_depths=skip_mamba_depths,
            decoder_mamba_depths=decoder_mamba_depths,
            bottleneck_depth=bottleneck_depth,
            embedding_dim=embedding_dim,
            img_dim=img_patch_size,
            stem_patch_size=stem_patch_size,
            
            # Dictionaries
            mamba_kwargs=mamba_kwargs,
            cbam_kwargs=cbam_kwargs
        )

        return model