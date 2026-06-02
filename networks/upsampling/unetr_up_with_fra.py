import torch
import torch.nn as nn
from ..blocks.unetr_block import UnetrBasicBlock
from .fra import FRAUpsample2D, FRAUpsample25D
import torch.nn.functional as F

class UnetrUpBlockWithFRA(nn.Module):
    """
    FRA-Decoder UpBlock: Feature ReAssembly Decoder with FRA-Upsample.
    Replaces standard upsampling with content-adaptive feature reassembly.

    Architecture:
    - FRA-Upsample (2D or 2.5D) for content-aware spatial reconstruction
    - Channel alignment projection
    - Concatenation with encoder skip features
    - UNETR BasicBlock for feature fusion

    - spatial_dims=2 -> FRAUpsample2D
    - spatial_dims=3 -> FRAUpsample25D (H,W x2; D unchanged)
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        upsample_kernel_size: int = 2,
        norm_name: str = "instance",
        res_block: bool = True,
        fra_k_up: int = 5,
        fra_k_enc: int = 3,
        fra_compress: int = 4,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        r = upsample_kernel_size
        if spatial_dims == 2:
            self.fra_up = FRAUpsample2D(
                in_channels, scale_factor=r, k_up=fra_k_up, k_enc=fra_k_enc, compress_ratio=fra_compress
            )
        elif spatial_dims == 3:
            self.fra_up = FRAUpsample25D(
                in_channels, scale_factor=r, k_up=fra_k_up, k_enc=fra_k_enc, compress_ratio=fra_compress
            )
        else:
            raise ValueError("spatial_dims must be 2 or 3")

        self.proj = nn.ConvNd(spatial_dims, in_channels, out_channels, kernel_size=1) if hasattr(nn, 'ConvNd') else (
            nn.Conv2d(in_channels, out_channels, 1) if spatial_dims == 2 else nn.Conv3d(in_channels, out_channels, 1)
        )

        Conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        Norm = (nn.InstanceNorm2d if spatial_dims == 2 else nn.InstanceNorm3d) if norm_name == "instance" else None
        Act  = nn.GELU

        self.block = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )


    def forward(self, D_in, skip, modalities=None):
        D_up = self.fra_up(D_in)
        D_proj = self.proj(D_up)
        if D_proj.shape[2:] != skip.shape[2:]:
            D_proj = F.interpolate(D_proj, size=skip.shape[2:], mode='trilinear', align_corners=False)
        D_cat = torch.cat([D_proj, skip], dim=1)
        D_out = self.block(D_cat, modalities)
        return D_out