"""Fundamental building blocks for SwiftDet.

References:
    - CSPNet: Wang et al. 2020 (Cross Stage Partial Network)
    - CBAM: Woo et al. 2018 (Convolutional Block Attention Module)
    - SPP: He et al. 2015 (Spatial Pyramid Pooling)
    - DFL: Li et al. 2020 (Generalized Focal Loss)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(kernel_size, padding=None, dilation=1):
    """Compute 'same' padding for given kernel size and dilation."""
    if padding is not None:
        return padding
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1
    return kernel_size // 2


class ConvBnAct(nn.Module):
    """Conv2d + BatchNorm + Activation (SiLU by default)."""

    def __init__(self, c_in, c_out, kernel_size=1, stride=1, padding=None,
                 groups=1, dilation=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size, stride,
            autopad(kernel_size, padding, dilation),
            dilation=dilation, groups=groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Forward pass after fusing Conv+BN."""
        return self.act(self.conv(x))


class DWConv(ConvBnAct):
    """Depth-wise convolution."""

    def __init__(self, c_in, c_out, kernel_size=1, stride=1, act=True):
        super().__init__(c_in, c_out, kernel_size, stride,
                         groups=math.gcd(c_in, c_out), act=act)


class DWSepConv(nn.Module):
    """Depth-wise separable convolution: DWConv + Pointwise Conv."""

    def __init__(self, c_in, c_out, kernel_size=3, stride=1, act=True):
        super().__init__()
        self.dw = ConvBnAct(c_in, c_in, kernel_size, stride, groups=c_in, act=act)
        self.pw = ConvBnAct(c_in, c_out, 1, 1, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block (Hu et al. 2018)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, channels, 1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        w = x.mean(dim=(2, 3), keepdim=True)
        w = self.act(self.fc1(w))
        w = self.fc2(w).sigmoid()
        return x * w


class ChannelAttention(nn.Module):
    """Channel attention sub-module of CBAM."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.fc2 = nn.Conv2d(mid, channels, 1, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        avg_pool = x.mean(dim=(2, 3), keepdim=True)
        max_pool = x.amax(dim=(2, 3), keepdim=True)
        avg_out = self.fc2(self.act(self.fc1(avg_pool)))
        max_out = self.fc2(self.act(self.fc1(max_pool)))
        return (avg_out + max_out).sigmoid()


class SpatialAttention(nn.Module):
    """Spatial attention sub-module of CBAM."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return self.bn(self.conv(combined)).sigmoid()


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al. 2018).

    Applies channel attention followed by spatial attention.
    """

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x


class BottleneckBlock(nn.Module):
    """Standard bottleneck block with optional SE attention and residual."""

    def __init__(self, c_in, c_out, shortcut=True, groups=1, expansion=0.5, se=False):
        super().__init__()
        c_mid = int(c_out * expansion)
        self.cv1 = ConvBnAct(c_in, c_mid, 1)
        self.cv2 = ConvBnAct(c_mid, c_out, 3, groups=groups)
        self.se = SEBlock(c_out) if se else nn.Identity()
        self.add = shortcut and c_in == c_out

    def forward(self, x):
        out = self.se(self.cv2(self.cv1(x)))
        return out + x if self.add else out


class CSPBlock(nn.Module):
    """Cross Stage Partial block (Wang et al. 2020).

    Input → split via 1x1 convs
      ├── cv1 → [Bottleneck × n] → features
      └── cv2 → skip
    concat(features, skip) → cv3 → Output
    """

    def __init__(self, c_in, c_out, n=1, shortcut=True, expansion=0.5, se=False):
        super().__init__()
        c_mid = int(c_out * expansion)
        self.cv1 = ConvBnAct(c_in, c_mid, 1)
        self.cv2 = ConvBnAct(c_in, c_mid, 1)
        self.cv3 = ConvBnAct(2 * c_mid, c_out, 1)
        self.blocks = nn.Sequential(
            *[BottleneckBlock(c_mid, c_mid, shortcut=shortcut, se=se) for _ in range(n)]
        )

    def forward(self, x):
        y1 = self.blocks(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))


class SPP(nn.Module):
    """Spatial Pyramid Pooling (He et al. 2015, SPPNet).

    Applies parallel MaxPool2d at multiple kernel sizes to capture
    multi-scale spatial features, then concatenates with the input.
    """

    def __init__(self, c_in, c_out, kernel_sizes=(5, 9, 13)):
        super().__init__()
        c_mid = c_in // 2
        self.cv1 = ConvBnAct(c_in, c_mid, 1)
        self.pools = nn.ModuleList(
            nn.MaxPool2d(k, stride=1, padding=k // 2) for k in kernel_sizes
        )
        self.cv2 = ConvBnAct(c_mid * (1 + len(kernel_sizes)), c_out, 1)

    def forward(self, x):
        x = self.cv1(x)
        pooled = [pool(x) for pool in self.pools]
        return self.cv2(torch.cat([x] + pooled, dim=1))


class DFL(nn.Module):
    """Distribution Focal Loss layer (Li et al. 2020, Generalized Focal Loss).

    Converts discrete probability distributions to continuous box coordinates.
    """

    def __init__(self, num_bins=16):
        super().__init__()
        self.num_bins = num_bins
        self.conv = nn.Conv2d(num_bins, 1, 1, bias=False)
        # Initialize with fixed weights [0, 1, ..., num_bins-1]
        x = torch.arange(num_bins, dtype=torch.float)
        self.conv.weight = nn.Parameter(x.view(1, num_bins, 1, 1))
        self.conv.weight.requires_grad = False

    def forward(self, x):
        """x: (B, 4*num_bins, H, W) or (B, N, 4*num_bins)."""
        if x.dim() == 3:
            # (B, N, 4*num_bins) → (B, N, 4)
            b, n, _ = x.shape
            x = x.view(b, n, 4, self.num_bins)
            x = F.softmax(x, dim=-1)
            # Use matmul instead of conv for 3D input
            weight = torch.arange(self.num_bins, dtype=x.dtype, device=x.device)
            return (x * weight).sum(dim=-1)  # (B, N, 4)
        else:
            # (B, 4*num_bins, H, W) → (B, 4, H, W)
            b, _, h, w = x.shape
            x = x.view(b, 4, self.num_bins, h, w)
            x = F.softmax(x, dim=2)
            x = x.view(b * 4, self.num_bins, h, w)
            return self.conv(x).view(b, 4, h, w)
