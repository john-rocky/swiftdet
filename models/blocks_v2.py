"""Modern building blocks for SwiftDet2.

References:
    - RepVGG: Ding et al. 2021 (Reparameterizable convolution)
    - ConvNeXt: Liu et al. 2022 (Large kernel convolution blocks)
    - YOLOv12: Zhao et al. 2025 (Area-based attention)
    - FasterNet: Chen et al. 2023 (Partial convolution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBnAct, DFL, DWSepConv, SPP, autopad

__all__ = [
    "ConvBnAct",
    "DFL",
    "DWSepConv",
    "SPP",
    "RepConv",
    "LargeKernelBlock",
    "AreaAttention",
    "PConv",
    "PConvCSPBlock",
    "RepBottleneck",
    "RepCSPBlock",
]


# ---------------------------------------------------------------------------
# RepConv (Ding et al. 2021, "RepVGG: Making VGG-style ConvNets Great Again")
# ---------------------------------------------------------------------------

class RepConv(nn.Module):
    """Reparameterizable convolution block.

    Training mode uses three parallel branches whose outputs are summed:
        - 3x3 convolution + BatchNorm
        - 1x1 convolution + BatchNorm
        - Identity + BatchNorm  (only when c_in == c_out and stride == 1)

    Inference mode fuses all branches into a single 3x3 conv + bias via the
    ``fuse`` method, eliminating the multi-branch overhead.

    Args:
        c_in: Input channels.
        c_out: Output channels.
        kernel_size: Kernel size of the main convolution branch (default 3).
        stride: Convolution stride (default 1).
        act: Activation function. ``True`` for SiLU, ``nn.Module`` instance
            for a custom activation, or ``False`` / ``None`` for identity.
    """

    def __init__(self, c_in, c_out, kernel_size=3, stride=1, act=True):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.stride = stride
        padding = autopad(kernel_size)

        # Main 3x3 branch
        self.conv3x3 = nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=False)
        self.bn3x3 = nn.BatchNorm2d(c_out)

        # Auxiliary 1x1 branch
        self.conv1x1 = nn.Conv2d(c_in, c_out, 1, stride, 0, bias=False)
        self.bn1x1 = nn.BatchNorm2d(c_out)

        # Identity branch (only when dimensions match)
        self.has_identity = (c_in == c_out and stride == 1)
        if self.has_identity:
            self.bn_id = nn.BatchNorm2d(c_out)

        # Activation
        self.act = nn.SiLU(inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity()
        )

        # Flag set after fusing
        self._fused = False

    def forward(self, x):
        if self._fused:
            return self.act(self.fused_conv(x))

        out = self.bn3x3(self.conv3x3(x)) + self.bn1x1(self.conv1x1(x))
        if self.has_identity:
            out = out + self.bn_id(x)
        return self.act(out)

    # -- Fusion utilities ---------------------------------------------------

    @staticmethod
    def _fuse_bn(conv, bn):
        """Fuse a Conv2d and BatchNorm2d into equivalent (weight, bias) pair."""
        w = conv.weight  # (c_out, c_in, kH, kW)
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        std = (var + eps).sqrt()
        scale = gamma / std  # (c_out,)

        fused_w = w * scale.reshape(-1, 1, 1, 1)
        fused_b = beta - mean * scale
        return fused_w, fused_b

    @staticmethod
    def _pad_1x1_to_kxk(weight_1x1, kernel_size):
        """Zero-pad a 1x1 weight tensor to (kernel_size x kernel_size)."""
        if kernel_size == 1:
            return weight_1x1
        pad = kernel_size // 2
        return F.pad(weight_1x1, [pad, pad, pad, pad])

    def _identity_weight(self):
        """Build an identity-equivalent (kernel_size x kernel_size) conv weight."""
        channels = self.c_in  # c_in == c_out when identity exists
        w = torch.zeros(channels, channels, self.kernel_size, self.kernel_size,
                        device=self.conv3x3.weight.device,
                        dtype=self.conv3x3.weight.dtype)
        center = self.kernel_size // 2
        for i in range(channels):
            w[i, i, center, center] = 1.0
        return w

    def fuse(self):
        """Fuse all branches into a single Conv2d and register as ``fused_conv``."""
        if self._fused:
            return

        # Branch 1: 3x3
        w3, b3 = self._fuse_bn(self.conv3x3, self.bn3x3)

        # Branch 2: 1x1 -> pad to 3x3
        w1, b1 = self._fuse_bn(self.conv1x1, self.bn1x1)
        w1 = self._pad_1x1_to_kxk(w1, self.kernel_size)

        # Branch 3: identity
        if self.has_identity:
            id_conv = nn.Conv2d(
                self.c_in, self.c_out, self.kernel_size,
                stride=1, padding=self.kernel_size // 2, bias=False,
            ).to(self.conv3x3.weight.device)
            id_conv.weight.data = self._identity_weight()
            w_id, b_id = self._fuse_bn(id_conv, self.bn_id)
        else:
            w_id = torch.zeros_like(w3)
            b_id = torch.zeros_like(b3)

        # Sum all branches
        fused_w = w3 + w1 + w_id
        fused_b = b3 + b1 + b_id

        # Create the fused conv
        fused = nn.Conv2d(
            self.c_in, self.c_out, self.kernel_size,
            stride=self.stride,
            padding=autopad(self.kernel_size),
            bias=True,
        )
        fused.weight.data = fused_w
        fused.bias.data = fused_b

        self.fused_conv = fused
        self._fused = True

        # Remove training-only parameters to save memory
        if hasattr(self, "conv3x3"):
            del self.conv3x3, self.bn3x3
        if hasattr(self, "conv1x1"):
            del self.conv1x1, self.bn1x1
        if hasattr(self, "bn_id"):
            del self.bn_id


# ---------------------------------------------------------------------------
# LargeKernelBlock (Liu et al. 2022, "A ConvNet for the 2020s" / ConvNeXt)
# ---------------------------------------------------------------------------

class LargeKernelBlock(nn.Module):
    """Block using a large depthwise kernel for an extended receptive field.

    Structure:
        DWConv (7x7 default) -> LayerNorm -> Conv1x1 (expand) -> GELU
        -> Conv1x1 (project back) -> residual add

    Args:
        channels: Input and output channels (residual block).
        expansion: Hidden dimension expansion ratio (default 4).
        kernel_size: Depthwise convolution kernel size (default 7).
    """

    def __init__(self, channels, expansion=4, kernel_size=7):
        super().__init__()
        hidden = channels * expansion

        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size,
            padding=kernel_size // 2, groups=channels, bias=True,
        )
        self.norm = LayerNorm2d(channels)
        self.pw_expand = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act = nn.GELU()
        self.pw_project = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x):
        shortcut = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw_expand(x)
        x = self.act(x)
        x = self.pw_project(x)
        return x + shortcut


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension for (B, C, H, W) tensors."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        # Normalize over the channel dimension (dim=1)
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / (s + self.eps).sqrt()
        x = self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return x


# ---------------------------------------------------------------------------
# AreaAttention (Zhao et al. 2025, "YOLOv12: Attention-Centric Real-Time
#   Object Detectors")
# ---------------------------------------------------------------------------

class AreaAttention(nn.Module):
    """Area-based multi-head self-attention.

    Partitions the feature map into non-overlapping square areas and applies
    multi-head self-attention within each area. Complexity is
    O(N * area_size^2) instead of O(N^2), making it viable for high-resolution
    feature maps.

    The block includes a pre-norm, the attention itself, a post-attention
    residual, an MLP with expansion, and another residual -- following a
    standard pre-norm transformer block structure.

    Args:
        channels: Number of input (and output) channels.
        num_heads: Number of attention heads (default 4).
        area_size: Spatial side length of each attention area (default 4,
            giving 4x4 = 16 tokens per area).
    """

    def __init__(self, channels, num_heads=4, area_size=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.area_size = area_size

        # Pre-attention norm
        self.norm1 = nn.LayerNorm(channels)

        # QKV projection
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        # Post-attention MLP with residual
        self.norm2 = nn.LayerNorm(channels)
        mlp_hidden = channels * 4
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, channels),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        a = self.area_size

        # Pad spatial dims to be divisible by area_size
        pad_h = (a - H % a) % a
        pad_w = (a - W % a) % a
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape

        num_h = Hp // a
        num_w = Wp // a

        # Reshape: (B, C, H, W) -> (B * num_areas, area_size^2, C)
        # (B, C, num_h, a, num_w, a) -> (B, num_h, num_w, a, a, C)
        x = x.reshape(B, C, num_h, a, num_w, a)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, num_h, num_w, a, a, C)
        x = x.reshape(B * num_h * num_w, a * a, C)     # (B*N_areas, tokens, C)

        # --- Self-attention block with residual ---
        shortcut = x
        x = self.norm1(x)

        # QKV
        qkv = self.qkv(x).reshape(-1, a * a, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*N, heads, tokens, head_dim)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B*N, heads, tokens, head_dim)

        out = out.transpose(1, 2).reshape(-1, a * a, C)
        out = self.proj(out)
        x = shortcut + out

        # --- MLP block with residual ---
        x = x + self.mlp(self.norm2(x))

        # Reshape back: (B * num_areas, area_size^2, C) -> (B, C, H, W)
        x = x.reshape(B, num_h, num_w, a, a, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B, C, num_h, a, num_w, a)
        x = x.reshape(B, C, Hp, Wp)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        return x


# ---------------------------------------------------------------------------
# PConv (Chen et al. 2023, "Run, Don't Walk: Chasing Higher FLOPS for Faster
#   Neural Networks" / FasterNet)
# ---------------------------------------------------------------------------

class PConv(nn.Module):
    """Partial Convolution -- applies convolution to only a fraction of channels.

    The input is split along the channel axis: the first ``n_part`` channels
    are processed by a depthwise-like convolution while the remaining channels
    pass through unchanged. This drastically reduces FLOPs while maintaining
    representational capacity since the subsequent pointwise layers mix all
    channels.

    Args:
        channels: Total number of input/output channels.
        n_part: Number of channels to convolve (default ``channels // 4``).
        kernel_size: Convolution kernel size (default 3).
    """

    def __init__(self, channels, n_part=None, kernel_size=3):
        super().__init__()
        self.n_part = n_part or channels // 4
        self.conv = nn.Conv2d(
            self.n_part, self.n_part, kernel_size,
            padding=kernel_size // 2, bias=False,
        )
        self.bn = nn.BatchNorm2d(self.n_part)

    def forward(self, x):
        x1 = x[:, :self.n_part, :, :]
        x2 = x[:, self.n_part:, :, :]
        x1 = self.bn(self.conv(x1))
        return torch.cat([x1, x2], dim=1)


class PConvCSPBlock(nn.Module):
    """CSP block using partial convolution for efficiency.

    Uses the same split-concat CSP structure as the v1 ``CSPBlock``, but
    replaces standard bottlenecks with PConv-based bottlenecks and uses
    RepConv for the entry/exit convolutions.

    Args:
        c_in: Input channels.
        c_out: Output channels.
        n: Number of bottleneck repeats (default 1).
        shortcut: Whether to use residual connections (default True).
        expansion: Channel expansion ratio for the CSP split (default 0.5).
    """

    def __init__(self, c_in, c_out, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        c_mid = int(c_out * expansion)
        self.cv1 = RepConv(c_in, c_mid, 1, act=True)
        self.cv2 = RepConv(c_in, c_mid, 1, act=True)
        self.cv3 = ConvBnAct(2 * c_mid, c_out, 1)
        self.blocks = nn.Sequential(
            *[_PConvBottleneck(c_mid, shortcut=shortcut) for _ in range(n)]
        )

    def forward(self, x):
        y1 = self.blocks(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))


class _PConvBottleneck(nn.Module):
    """Internal bottleneck using PConv with an optional residual."""

    def __init__(self, channels, shortcut=True):
        super().__init__()
        self.pconv = PConv(channels)
        self.pw = ConvBnAct(channels, channels, 1)
        self.add = shortcut

    def forward(self, x):
        out = self.pw(self.pconv(x))
        return out + x if self.add else out


# ---------------------------------------------------------------------------
# RepBottleneck / RepCSPBlock
# ---------------------------------------------------------------------------

class RepBottleneck(nn.Module):
    """Bottleneck with RepConv for reparameterizable training.

    Uses RepConv instead of standard Conv in the bottleneck path, enabling
    structural reparameterization at inference time.

    Args:
        c_in: Input channels.
        c_out: Output channels.
        shortcut: Use residual connection when dims match (default True).
        expansion: Mid-channel expansion ratio (default 0.5).
    """

    def __init__(self, c_in, c_out, shortcut=True, expansion=0.5):
        super().__init__()
        c_mid = int(c_out * expansion)
        self.cv1 = RepConv(c_in, c_mid, 3, act=True)
        self.cv2 = RepConv(c_mid, c_out, 3, act=True)
        self.add = shortcut and c_in == c_out

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return out + x if self.add else out


class RepCSPBlock(nn.Module):
    """Cross Stage Partial block with RepConv bottlenecks.

    Same CSP structure as v1 but uses ``RepBottleneck`` internally and
    ``RepConv`` for the split/merge convolutions.

    Args:
        c_in: Input channels.
        c_out: Output channels.
        n: Number of RepBottleneck repeats (default 1).
        shortcut: Whether bottlenecks use residual connections (default True).
        expansion: CSP channel split ratio (default 0.5).
    """

    def __init__(self, c_in, c_out, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        c_mid = int(c_out * expansion)
        self.cv1 = ConvBnAct(c_in, c_mid, 1)
        self.cv2 = ConvBnAct(c_in, c_mid, 1)
        self.cv3 = ConvBnAct(2 * c_mid, c_out, 1)
        self.blocks = nn.Sequential(
            *[RepBottleneck(c_mid, c_mid, shortcut=shortcut) for _ in range(n)]
        )

    def forward(self, x):
        y1 = self.blocks(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))
