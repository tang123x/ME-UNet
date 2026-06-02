from __future__ import annotations
from collections.abc import Sequence

import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks.dynunet_block import UnetOutBlock
from ..blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from ..blocks.patch_merging import PatchMerging, PatchMergingV2
from monai.utils.deprecate_utils import deprecated_arg

from ..upsampling.unetr_up_with_fra import UnetrUpBlockWithFRA


rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "MEUNETR",
    "MERGING_MODE",
]

MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


class MambaSSM(nn.Module):
    """
    Mamba State Space Model (SSM) for sequence modeling.
    Implements selective state-space modeling with linear-time complexity.
    Operates on [B, N, C] sequences and returns [B, N, C].
    """
    def __init__(self, channels, state_dim=64, kernel=9, dropout=0.0, residual_scale=0.1):
        super().__init__()
        self.C = channels
        self.state_dim = state_dim
        self.kernel = kernel
        self.residual_scale = residual_scale

        self.norm = nn.LayerNorm(channels)

        self.to_state = nn.Linear(channels, state_dim)
        padding = (kernel - 1) // 2
        self.dw = nn.Conv1d(state_dim, state_dim, kernel_size=kernel, padding=padding, groups=state_dim)
        self.to_chan = nn.Linear(state_dim, channels)

        self.gate = nn.Linear(channels, channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.zeros_(self.to_chan.weight)
        nn.init.zeros_(self.to_chan.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 0.0)

    def forward(self, seq):
        B, N, C = seq.shape
        x = self.norm(seq)
        s = self.to_state(x)
        s_t = s.permute(0, 2, 1)
        s_t = self.dw(s_t)
        s = s_t.permute(0, 2, 1)
        s = self.to_chan(s)
        s = self.act(s)
        s = self.dropout(s)
        g = torch.sigmoid(self.gate(x))
        out = g * s + (1 - g) * x
        return seq + self.residual_scale * out


class MambaCell3D(nn.Module):
    """
    Apply MambaSSM to flattened 3D tokens with optional adaptive pooling.
    Input: x [B, C, D, H, W] -> project -> pool -> flatten -> MambaSSM -> reshape -> project -> upsample back
    """
    def __init__(self, in_ch, hidden_ch=None, state_dim=64, kernel=9, downsample=2):
        super().__init__()
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch if hidden_ch is not None else in_ch
        self.state_dim = state_dim
        self.downsample = downsample

        self.proj_in = nn.Conv3d(in_ch, self.hidden_ch, kernel_size=1)
        self.proj_out = nn.Conv3d(self.hidden_ch, in_ch, kernel_size=1)
        self.pos = nn.Parameter(torch.zeros(1, self.hidden_ch, 1, 1, 1))
        self.ssm = MambaSSM(self.hidden_ch, state_dim=state_dim, kernel=kernel)
        self.gate1 = nn.Conv1d(self.hidden_ch, self.hidden_ch, kernel_size=3, padding=1)

    def _compute_target(self, D, H, W):
        if isinstance(self.downsample, int) and self.downsample > 1:
            Dt = max(1, math.ceil(D / self.downsample))
            Ht = max(1, math.ceil(H / self.downsample))
            Wt = max(1, math.ceil(W / self.downsample))
            return (Dt, Ht, Wt)
        return (D, H, W)

    def forward(self, x, target_size=None):
        B, C, D, H, W = x.shape
        if target_size is None:
            Dt, Ht, Wt = self._compute_target(D, H, W)
        else:
            Dt, Ht, Wt = target_size

        if (Dt, Ht, Wt) != (D, H, W):
            x_ds = F.adaptive_avg_pool3d(x, output_size=(Dt, Ht, Wt))
        else:
            x_ds = x

        z = self.proj_in(x_ds)
        z = z + self.pos
        B2, Hc, Dp, Hp, Wp = z.shape
        seq = z.reshape(B2, Hc, Dp * Hp * Wp).permute(0, 2, 1)
        s_out = self.ssm(seq)
        g = torch.sigmoid(self.gate1(seq.permute(0, 2, 1))).permute(0, 2, 1)
        s_out = s_out * g
        s_out = s_out.permute(0, 2, 1).reshape(B2, Hc, Dp, Hp, Wp)
        out = self.proj_out(s_out)
        if (Dt, Ht, Wt) != (D, H, W):
            out = F.interpolate(out, size=(D, H, W), mode='trilinear', align_corners=False)
        return out


class BiDirectionalMamba(nn.Module):
    """
    Bidirectional scanning strategy to eliminate directional bias of causal SSMs in 3D imaging.
    Processes sequence in both forward and reverse directions, fusing outputs for isotropic receptive fields.
    O_bi = 0.5 * (O_uni^fwd + Flip(O_uni^bwd))
    """
    def __init__(self, in_ch, hidden_ch=None, state_dim=64, kernel=9, downsample=2):
        super().__init__()
        if hidden_ch is None:
            hidden_ch = in_ch
        self.fwd = MambaCell3D(in_ch, hidden_ch, state_dim=state_dim, kernel=kernel, downsample=downsample)
        self.rev = MambaCell3D(in_ch, hidden_ch, state_dim=state_dim, kernel=kernel, downsample=downsample)
        self.proj = nn.Conv3d(in_ch, in_ch, kernel_size=1)
        self.norm = nn.InstanceNorm3d(in_ch, affine=True)

    def forward(self, x):
        out_f = self.fwd(x)
        x_rev = torch.flip(x, dims=[2, 3, 4])
        out_r = self.rev(x_rev)
        out_r = torch.flip(out_r, dims=[2, 3, 4])
        out = 0.5 * (out_f + out_r)
        out = self.proj(out)
        out = self.norm(out + x)
        return out


class LocalFeatureExtraction(nn.Module):
    """
    Local Feature Extraction module in CMHB.
    Employs two 3D residual blocks to capture short-range anatomical context and texture continuity.
    Y = IN(Conv3D(GELU(IN(Conv3D(X))))) + X
    """
    def __init__(self, channels, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=kernel, padding=pad),
            nn.InstanceNorm3d(channels, affine=True),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=kernel, padding=pad),
        )
        self.norm_out = nn.InstanceNorm3d(channels, affine=True)

    def forward(self, x):
        return self.norm_out(self.conv(x) + x)


class CMHB(nn.Module):
    """
    Conv-Mamba Hybrid Block (CMHB): The fundamental building block of the encoder.
    Synergizes local textural extraction with global dependency modeling.
    
    Processing pipeline:
    1. Local Feature Extraction: Two 3D residual blocks capture short-range context
    2. Global Dependency Modeling: Bidirectional Mamba for long-range dependencies
    3. Feature fusion via learnable scaling parameter gamma
    """
    def __init__(self, channels, state_dim=64, kernel=9, downsample=2):
        super().__init__()
        self.local_extract = LocalFeatureExtraction(channels)
        self.mamba = BiDirectionalMamba(channels, hidden_ch=channels, state_dim=state_dim, kernel=kernel, downsample=downsample)
        self.gamma = nn.Parameter(torch.tensor(1e-2).repeat(1, channels, 1, 1, 1))
        self.norm_out = nn.InstanceNorm3d(channels, affine=True)

    def forward(self, x):
        y = self.local_extract(x)
        m = self.mamba(y)
        return self.norm_out(x + self.gamma * m)


class PatchEmbed3D(nn.Module):
    """3D patch embedding via stride convolution."""
    def __init__(self, in_ch, embed_dim, patch_size=2):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.InstanceNorm3d(embed_dim, affine=True)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return F.gelu(x)


class CMHBEncoder(nn.Module):
    """
    Encoder composed of CMHB blocks.
    Produces feature list: [feat1, feat2, feat3, feat4, feat5]
    Channels: embed, 2*embed, 4*embed, 8*embed, 16*embed
    """
    def __init__(self, in_chans=1, embed_dim=24, depths=(2,2,2,2), patch_size=2, state_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depths = depths

        self.patch1 = PatchEmbed3D(in_chans, embed_dim, patch_size=patch_size)
        self.stage1 = nn.ModuleList([CMHB(embed_dim, state_dim=state_dim) for _ in range(depths[0])])

        self.down2 = nn.Conv3d(embed_dim, embed_dim*2, kernel_size=2, stride=2)
        self.norm2 = nn.InstanceNorm3d(embed_dim*2, affine=True)
        self.stage2 = nn.ModuleList([CMHB(embed_dim*2, state_dim=state_dim) for _ in range(depths[1])])

        self.down3 = nn.Conv3d(embed_dim*2, embed_dim*4, kernel_size=2, stride=2)
        self.norm3 = nn.InstanceNorm3d(embed_dim*4, affine=True)
        self.stage3 = nn.ModuleList([CMHB(embed_dim*4, state_dim=state_dim) for _ in range(depths[2])])

        self.down4 = nn.Conv3d(embed_dim*4, embed_dim*8, kernel_size=2, stride=2)
        self.norm4 = nn.InstanceNorm3d(embed_dim*8, affine=True)
        self.stage4 = nn.ModuleList([CMHB(embed_dim*8, state_dim=state_dim) for _ in range(depths[3])])

        self.down5 = nn.Conv3d(embed_dim*8, embed_dim*16, kernel_size=2, stride=2)
        self.norm5 = nn.InstanceNorm3d(embed_dim*16, affine=True)
        self.stage5 = CMHB(embed_dim*16, state_dim=state_dim)

    def forward(self, x, normalize=True, modalities=None):
        feat1 = self.patch1(x)
        for blk in self.stage1:
            feat1 = blk(feat1)
        feat2 = self.down2(feat1); feat2 = self.norm2(feat2)
        for blk in self.stage2:
            feat2 = blk(feat2)
        feat3 = self.down3(feat2); feat3 = self.norm3(feat3)
        for blk in self.stage3:
            feat3 = blk(feat3)
        feat4 = self.down4(feat3); feat4 = self.norm4(feat4)
        for blk in self.stage4:
            feat4 = blk(feat4)
        feat5 = self.down5(feat4); feat5 = self.norm5(feat5)
        feat5 = self.stage5(feat5)
        return [feat1, feat2, feat3, feat4, feat5]


class CRA(nn.Module):
    """
    Complexity-Reduced Attention (CRA) module in AMSF.
    Decomposes 3D attention into two efficient pathways:
    - Spatial Reduction Branch: Compresses depth dimension, applies 2D conv, expands back
    - Channel Squeeze Branch: Global Average Pooling + MLP + sigmoid for channel-wise weights
    """
    def __init__(self, in_ch):
        super().__init__()
        self.in_ch = in_ch
        self.pool_depth = nn.AdaptiveAvgPool3d((1, None, None))
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.conv_pw = nn.Conv3d(in_ch, in_ch, kernel_size=1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_ch, max(in_ch // 8, 4), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(in_ch // 8, 4), in_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        dpool = self.pool_depth(x).squeeze(2)
        dmap = self.conv1(dpool)
        dmap = dmap.unsqueeze(2)
        out = self.conv_pw(dmap)
        out = F.interpolate(out, size=(D, H, W), mode='trilinear', align_corners=False)
        g = self.se(x)
        return out * g


class AMSF(nn.Module):
    """
    Adaptive Mamba Skip Fusion (AMSF).
    Bridges semantic gap between encoder and decoder representations.
    
    Process:
    1. Structural alignment: Project both features to common hidden space, resize encoder to decoder dimensions
    2. Encoder refinement: Mamba-based sequence modeling for global semantic stability
    3. Decoder refinement: CRA module for efficient attention
    4. Adaptive gating: Gated fusion of refined features
    """
    def __init__(self, dec_ch, skip_ch, mid_ch=None, state_dim=64, downsample=2):
        super().__init__()
        if mid_ch is None:
            mid_ch = skip_ch
        self.dec_proj = nn.Conv3d(dec_ch, mid_ch, kernel_size=1)
        self.skip_proj = nn.Conv3d(skip_ch, mid_ch, kernel_size=1)
        self.mamba = MambaCell3D(mid_ch, hidden_ch=mid_ch, state_dim=state_dim, downsample=downsample)
        self.cra = CRA(mid_ch)
        self.gate_mamba = nn.Conv3d(mid_ch, mid_ch, kernel_size=1)
        self.gate_cra = nn.Conv3d(mid_ch, mid_ch, kernel_size=1)
        self.out = nn.Conv3d(mid_ch, skip_ch, kernel_size=1)
        self.norm = nn.InstanceNorm3d(skip_ch, affine=True)

    def forward(self, dec_feat, enc_feat):
        B, _, Dd, Hd, Wd = dec_feat.shape
        _, _, De, He, We = enc_feat.shape
        dec_p = self.dec_proj(dec_feat)
        enc_p = self.skip_proj(enc_feat)
        mid_size = (Dd, Hd, Wd)
        if (De, He, We) != mid_size:
            if De >= Dd and He >= Hd and We >= Wd:
                enc_resized = F.adaptive_avg_pool3d(enc_p, mid_size)
            else:
                enc_resized = F.interpolate(enc_p, size=mid_size, mode='trilinear', align_corners=False)
        else:
            enc_resized = enc_p
        enc_m = self.mamba(enc_resized, target_size=mid_size)
        dec_h = self.cra(dec_p)
        Gm = torch.sigmoid(self.gate_mamba(enc_m))
        Gh = torch.sigmoid(self.gate_cra(dec_h))
        fused = Gm * enc_m + Gh * dec_h
        fused_proj = self.out(fused)
        if (Dd, Hd, Wd) != (De, He, We):
            if Dd >= De and Hd >= He and Wd >= We:
                fused_back = F.adaptive_avg_pool3d(fused_proj, output_size=(De, He, We))
            else:
                fused_back = F.interpolate(fused_proj, size=(De, He, We), mode='trilinear', align_corners=False)
        else:
            fused_back = fused_proj
        return self.norm(fused_back + enc_feat)


class MEUNETR(nn.Module):
    """
    ME-UNet architecture:
    - Encoder: CMHBEncoder composed of CMHB blocks
    - Bottleneck: BiDirectionalMamba for long-range modeling
    - Skip Fusion: AMSF for adaptive feature fusion
    - Decoder: FRA-based upsampling for content-aware reconstruction
    """

    @deprecated_arg(
        name="img_size",
        since="1.3",
        removed="1.5",
        msg_suffix="The img_size argument is not required anymore and "
                   "checks on the input size are run during forward().",
    )
    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        patch_size: int = 2,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Sequence[int] | int = 7,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        feature_size: int = 24,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample: str | nn.Module = "merging",
        vit_norm: tuple | str = "layer",
        dec_norm: tuple | str = "instance",
        enc_norm: tuple | str = "instance",
        freeze_enc: bool = False,
        use_v2: bool = False,
        mm_state_dim: int = 64,
        mm_kernel: int = 9,
    ) -> None:
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self.patch_size = patch_size

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(window_size, spatial_dims)

        self._check_input_size(img_size)

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.vit_norm = vit_norm[0] if isinstance(vit_norm, tuple) else vit_norm
        self.dec_norm = dec_norm[0] if isinstance(dec_norm, tuple) else dec_norm
        self.enc_norm = enc_norm[0] if isinstance(enc_norm, tuple) else enc_norm

        if self.dec_norm == "layer" or self.enc_norm == "layer":
            raise ValueError("Layer normalization not yet implemented for encoder and decoder blocks, please "
                             "select another normalization.")

        self.normalize = normalize

        self.encoder = CMHBEncoder(in_chans=in_channels, embed_dim=feature_size, depths=depths, patch_size=patch_size, state_dim=mm_state_dim)

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=enc_norm,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=enc_norm,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=enc_norm,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=enc_norm,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=enc_norm,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlockWithFRA(
            spatial_dims=spatial_dims, in_channels=16 * feature_size, out_channels=8 * feature_size,
            kernel_size=3, upsample_kernel_size=2, norm_name=dec_norm, res_block=True,
            fra_k_up=5, fra_k_enc=3, fra_compress=4
        )
        self.decoder4 = UnetrUpBlockWithFRA(
            spatial_dims=spatial_dims, in_channels=8 * feature_size, out_channels=4 * feature_size,
            kernel_size=3, upsample_kernel_size=2, norm_name=dec_norm, res_block=True
        )
        self.decoder3 = UnetrUpBlockWithFRA(
            spatial_dims=spatial_dims, in_channels=4 * feature_size, out_channels=2 * feature_size,
            kernel_size=3, upsample_kernel_size=2, norm_name=dec_norm, res_block=True
        )
        self.decoder2 = UnetrUpBlockWithFRA(
            spatial_dims=spatial_dims, in_channels=2 * feature_size, out_channels=feature_size,
            kernel_size=3, upsample_kernel_size=2, norm_name=dec_norm, res_block=True
        )
        self.decoder1 = UnetrUpBlockWithFRA(
            spatial_dims=spatial_dims, in_channels=feature_size, out_channels=feature_size,
            kernel_size=3, upsample_kernel_size=2, norm_name=dec_norm, res_block=True
        )

        fs = feature_size
        self.amsf_hs3 = AMSF(dec_ch=16*fs, skip_ch=8*fs, mid_ch=8*fs, state_dim=mm_state_dim, downsample=2)
        self.amsf_enc3 = AMSF(dec_ch=8*fs, skip_ch=4*fs, mid_ch=4*fs, state_dim=mm_state_dim, downsample=2)
        self.amsf_enc2 = AMSF(dec_ch=4*fs, skip_ch=2*fs, mid_ch=2*fs, state_dim=mm_state_dim, downsample=2)
        self.amsf_enc1 = AMSF(dec_ch=2*fs, skip_ch=1*fs, mid_ch=1*fs, state_dim=mm_state_dim, downsample=1)
        self.amsf_enc0 = AMSF(dec_ch=1*fs, skip_ch=1*fs, mid_ch=1*fs, state_dim=mm_state_dim, downsample=1)

        self.bottleneck = BiDirectionalMamba(in_ch=16 * feature_size, hidden_ch=16*feature_size, state_dim=mm_state_dim, downsample=2)

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

        if freeze_enc:
            self.encoder.requires_grad_(False)
            self.encoder1.requires_grad_(False)
            self.encoder2.requires_grad_(False)
            self.encoder3.requires_grad_(False)
            self.encoder4.requires_grad_(False)
            self.encoder10.requires_grad_(False)

    @classmethod
    def from_cfg(cls, cfg):
        mm_state_dim = getattr(cfg, "mm_state_dim", 64)
        mm_kernel = getattr(cfg, "mm_kernel", 9)
        return cls(
            img_size=(cfg.roi_x, cfg.roi_y, cfg.roi_z),
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            patch_size=cfg.patch_size,
            depths=tuple(cfg.depths) if hasattr(cfg, "depths") else (2,2,2,2),
            num_heads=getattr(cfg, "num_heads", (3,6,12,24)),
            window_size=getattr(cfg, "window_size", 7),
            qkv_bias=getattr(cfg, "qkv_bias", True),
            mlp_ratio=getattr(cfg, "mlp_ratio", 4.0),
            feature_size=cfg.feature_size,
            drop_rate=getattr(cfg, "drop_rate", 0.0),
            attn_drop_rate=getattr(cfg, "attn_drop_rate", 0.0),
            dropout_path_rate=getattr(cfg, "dropout_path_rate", 0.0),
            normalize=getattr(cfg, "normalize", True),
            patch_norm=getattr(cfg, "patch_norm", False),
            use_checkpoint=getattr(cfg, "use_checkpoint", False),
            spatial_dims=getattr(cfg, "spatial_dims", 3),
            downsample=getattr(cfg, "downsample", "merging"),
            vit_norm=tuple(cfg.vit_norm) if not isinstance(getattr(cfg, "vit_norm", "layer"), str) else getattr(cfg, "vit_norm", "layer"),
            dec_norm=tuple(cfg.dec_norm) if not isinstance(getattr(cfg, "dec_norm", "instance"), str) else getattr(cfg, "dec_norm", "instance"),
            enc_norm=tuple(cfg.enc_norm) if not isinstance(getattr(cfg, "enc_norm", "instance"), str) else getattr(cfg, "enc_norm", "instance"),
            freeze_enc=getattr(cfg, "freeze_enc", False),
            use_v2=getattr(cfg, "use_v2", False),
            mm_state_dim=mm_state_dim,
            mm_kernel=mm_kernel,
        )

    def load_from(self, weights):
        if not isinstance(weights, dict):
            warnings.warn("load_from expects a dict (state_dict wrapper). Skipping.")
            return
        wstate = weights.get("state_dict", weights)
        has_patch = any(k.endswith("patch_embed.proj.weight") for k in wstate.keys())
        if has_patch and hasattr(self.encoder, "patch1"):
            try:
                k_w = [k for k in wstate.keys() if k.endswith("patch_embed.proj.weight")][0]
                self.encoder.patch1.proj.weight.copy_(wstate[k_w])
                print("Partial weight mapping applied (patch). Other weights skipped.")
            except Exception as e:
                warnings.warn(f"Partial mapping failed: {e}. Skipping load_from mapping.")
        else:
            warnings.warn("Provided weights don't resemble Swin state dict or encoder not Swin; skipping mapping.")

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )

    def forward(self, x_in, modalities=None):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.encoder(x_in, self.normalize, modalities)
        enc0 = self.encoder1(x_in, modalities)
        enc1 = self.encoder2(hidden_states_out[0], modalities)
        enc2 = self.encoder3(hidden_states_out[1], modalities)
        enc3 = self.encoder4(hidden_states_out[2], modalities)

        dec4_feat = self.bottleneck(hidden_states_out[4])
        dec4 = self.encoder10(dec4_feat, modalities)

        hs3 = hidden_states_out[3]
        fused_hs3 = self.amsf_hs3(dec4, hs3)
        dec3 = self.decoder5(dec4, fused_hs3, modalities)

        fused_enc3 = self.amsf_enc3(dec3, enc3)
        dec2 = self.decoder4(dec3, fused_enc3, modalities)

        fused_enc2 = self.amsf_enc2(dec2, enc2)
        dec1 = self.decoder3(dec2, fused_enc2, modalities)

        fused_enc1 = self.amsf_enc1(dec1, enc1)
        dec0 = self.decoder2(dec1, fused_enc1, modalities)

        fused_enc0 = self.amsf_enc0(dec0, enc0)
        out = self.decoder1(dec0, fused_enc0, modalities)

        logits = self.out(out)
        return logits
