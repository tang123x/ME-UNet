import torch
import torch.nn as nn
import torch.nn.functional as F


class FRAUpsample2D(nn.Module):
    """
    Feature ReAssembly Upsample 2D (FRA-Upsample2D).
    Content-adaptive mechanism that reconstructs high-resolution features
    by dynamically predicting upsampling kernels conditioned on local context.

    Process:
    1. Channel compression: D_c = Conv_1x1(D_in)
    2. Kernel prediction: K = Conv_kenc(D_c)
    3. Patch extraction via unfold: P = Unfold_kup(D_in)
    4. Weighted reassembly: R = sum_l(w_l * P_l)
    5. Pixel shuffle: D_f = PixelShuffle_r(R)
    """
    def __init__(self, channels, scale_factor=2, k_up=5, k_enc=3, compress_ratio=4):
        super().__init__()
        assert k_up % 2 == 1, "k_up must be odd for center alignment."
        self.channels = channels
        self.r = scale_factor
        self.k_up = k_up
        self.k_enc = k_enc
        m = max(8, channels // compress_ratio)

        self.D_c = nn.Conv2d(channels, m, 1, bias=False)
        self.K = nn.Conv2d(m, (self.r ** 2) * (self.k_up ** 2), k_enc, padding=k_enc // 2, bias=True)

    def forward(self, D_in):  # D_in: [B, C, H, W]
        B, C, H, W = D_in.shape
        D_c = self.D_c(D_in)
        K = self.K(D_c)
        K = K.view(B, self.r ** 2, self.k_up ** 2, H, W)
        K = torch.softmax(K, dim=2)

        P = F.unfold(D_in, kernel_size=self.k_up, padding=self.k_up // 2)
        P = P.view(B, C, self.k_up ** 2, H, W)

        R = torch.einsum('bckhw,brkhw->bcrhw', P, K)
        D_f = R.view(B, C * self.r * self.r, H, W)
        D_f = F.pixel_shuffle(D_f, upscale_factor=self.r)
        return D_f


class FRAUpsample25D(nn.Module):
    """
    Feature ReAssembly Upsample 2.5D (FRA-Upsample25D).
    Applies FRA-Upsample2D slice-wise to 3D volume.
    PixelShuffle_r is applied only along H,W dimensions, preserving depth resolution.
    Output: [B, C, D, rH, rW]
    """
    def __init__(self, channels, scale_factor=2, k_up=5, k_enc=3, compress_ratio=4):
        super().__init__()
        self.scale = scale_factor
        self.fra_2d = FRAUpsample2D(
            channels=channels, scale_factor=scale_factor, k_up=k_up, k_enc=k_enc, compress_ratio=compress_ratio
        )

    def forward(self, D_in):  # D_in: [B, C, D, H, W]
        B, C, D, H, W = D_in.shape
        D_in = D_in.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
        D_f = self.fra_2d(D_in)
        _, _, rH, rW = D_f.shape
        D_f = D_f.view(B, D, C, rH, rW).permute(0, 2, 1, 3, 4).contiguous()
        return D_f