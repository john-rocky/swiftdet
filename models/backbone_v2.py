"""Modern backbone for SwiftDet2 using RepConv and AreaAttention.

Architecture (base channels before width scaling):
    Stem:    3 -> 48,  RepConv 6x6 s=2                              # P1/2
    Stage 1: 48 -> 96,  RepConv 3x3 s=2 + RepCSPBlock(n=1)          # P2/4
    Stage 2: 96 -> 192, RepConv 3x3 s=2 + RepCSPBlock(n=2) + AreaAttention  # P3/8  -> output
    Stage 3: 192 -> 384, RepConv 3x3 s=2 + RepCSPBlock(n=3) + AreaAttention  # P4/16 -> output
    Stage 4: 384 -> 768, RepConv 3x3 s=2 + RepCSPBlock(n=2) + AreaAttention  # P5/32
    SPP:     768 -> 768                                              # P5/32 -> output

References:
    - RepVGG: Ding et al. 2021
    - YOLOv12: Zhao et al. 2025
    - CSPNet: Wang et al. 2020
"""

import torch.nn as nn

from .blocks import ConvBnAct
from .blocks_v2 import AreaAttention, RepConv, RepCSPBlock, SPP


def _auto_heads(channels, min_head_dim=32):
    """Compute num_heads so that head_dim >= min_head_dim."""
    heads = 1
    while heads * 2 * min_head_dim <= channels:
        heads *= 2
    return heads


class ModernBackbone(nn.Module):
    """SwiftDet2 backbone combining RepConv, RepCSP, and AreaAttention.

    Uses reparameterizable convolutions throughout for training-time
    multi-branch regularisation that can be fused to a single-branch
    architecture at inference time. AreaAttention replaces the v1 CBAM
    module, providing stronger spatial modelling with local self-attention.

    Args:
        width_mult: Channel width multiplier (e.g., 0.30 for Nano).
        depth_mult: Depth multiplier for RepCSP block repeat counts.
        max_channels: Maximum channel count after width scaling.
        in_channels: Number of input image channels (default 3).
    """

    # Base channel widths and RepCSP depths (before scaling)
    BASE_CHANNELS = [48, 96, 192, 384, 768]
    BASE_DEPTHS = [1, 2, 3, 2]  # RepCSP repeats for stages 1-4

    def __init__(self, width_mult=1.0, depth_mult=1.0, max_channels=1024,
                 in_channels=3):
        super().__init__()

        def ch(c):
            """Scale channel count, round to multiple of 8, clamp."""
            return min(max(round(c * width_mult / 8) * 8, 8), max_channels)

        def dep(d):
            """Scale depth count, minimum 1."""
            return max(round(d * depth_mult), 1)

        c = [ch(x) for x in self.BASE_CHANNELS]
        d = [dep(x) for x in self.BASE_DEPTHS]

        # Stem: 6x6 conv, stride 2 -> P1/2 (standard conv, even kernel not suited for RepConv)
        self.stem = ConvBnAct(in_channels, c[0], 6, stride=2, padding=2)

        # Stage 1: P2/4 -- no attention
        self.stage1 = nn.Sequential(
            RepConv(c[0], c[1], 3, stride=2),
            RepCSPBlock(c[1], c[1], n=d[0]),
        )

        # Stage 2: P3/8 -- with AreaAttention
        self.stage2 = nn.Sequential(
            RepConv(c[1], c[2], 3, stride=2),
            RepCSPBlock(c[2], c[2], n=d[1]),
            AreaAttention(c[2], num_heads=_auto_heads(c[2])),
        )

        # Stage 3: P4/16 -- with AreaAttention
        self.stage3 = nn.Sequential(
            RepConv(c[2], c[3], 3, stride=2),
            RepCSPBlock(c[3], c[3], n=d[2]),
            AreaAttention(c[3], num_heads=_auto_heads(c[3])),
        )

        # Stage 4: P5/32 -- with AreaAttention
        self.stage4 = nn.Sequential(
            RepConv(c[3], c[4], 3, stride=2),
            RepCSPBlock(c[4], c[4], n=d[3]),
            AreaAttention(c[4], num_heads=_auto_heads(c[4])),
        )

        # SPP at the end of stage 4
        self.spp = SPP(c[4], c[4])

        # Store output channel counts for the neck / head
        self._out_channels = [c[2], c[3], c[4]]  # P3, P4, P5

    def forward(self, x):
        """Forward pass returning multi-scale features [P3, P4, P5]."""
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.spp(self.stage4(p4))
        return [p3, p4, p5]

    @property
    def out_channels(self):
        """Output channel sizes for P3, P4, P5."""
        return self._out_channels

    @property
    def channels(self):
        """Alias for ``out_channels`` (v1 compatibility)."""
        return self._out_channels

    def fuse(self):
        """Fuse all RepConv modules in the backbone for inference.

        Iterates over every sub-module and calls ``fuse()`` on any
        ``RepConv`` instance, converting multi-branch training blocks
        into single-conv inference blocks.
        """
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse()
        return self
