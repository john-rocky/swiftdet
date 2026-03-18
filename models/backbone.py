"""CSP Backbone with CBAM attention.

Architecture (base channels before width scaling):
    Stem:    3 → 48, Conv 6x6 s=2           # P1/2
    Stage 1: 48 → 96, Conv 3x3 s=2 + CSP    # P2/4
    Stage 2: 96 → 192, Conv 3x3 s=2 + CSP + CBAM   # P3/8  → output
    Stage 3: 192 → 384, Conv 3x3 s=2 + CSP + CBAM  # P4/16 → output
    Stage 4: 384 → 768, Conv 3x3 s=2 + CSP + CBAM  # P5/32
    SPP:    768 → 768                                 # P5/32 → output

References:
    - CSPNet: Wang et al. 2020
    - CBAM: Woo et al. 2018
"""

import torch.nn as nn

from .blocks import CBAM, SPP, ConvBnAct, CSPBlock


class CSPBackbone(nn.Module):
    """CSP backbone with CBAM attention at stages 2-4.

    Args:
        width_mult: Channel multiplier (e.g., 0.30 for Nano).
        depth_mult: Depth multiplier for CSP block repeats.
        max_channels: Maximum channel count after scaling.
        in_channels: Input image channels (default 3).
    """

    # Base channel widths and CSP depths (before scaling)
    BASE_CHANNELS = [48, 96, 192, 384, 768]
    BASE_DEPTHS = [1, 2, 3, 2]  # CSP block repeat counts for stages 1-4

    def __init__(self, width_mult=1.0, depth_mult=1.0, max_channels=1024,
                 in_channels=3):
        super().__init__()

        def ch(c):
            return min(max(round(c * width_mult / 8) * 8, 8), max_channels)

        def dep(d):
            return max(round(d * depth_mult), 1)

        c = [ch(x) for x in self.BASE_CHANNELS]
        d = [dep(x) for x in self.BASE_DEPTHS]

        # Stem: 6x6 conv, stride 2
        self.stem = ConvBnAct(in_channels, c[0], 6, stride=2, padding=2)

        # Stage 1: P2/4 — no CBAM
        self.stage1 = nn.Sequential(
            ConvBnAct(c[0], c[1], 3, stride=2),
            CSPBlock(c[1], c[1], n=d[0]),
        )

        # Stage 2: P3/8 — with CBAM
        self.stage2 = nn.Sequential(
            ConvBnAct(c[1], c[2], 3, stride=2),
            CSPBlock(c[2], c[2], n=d[1]),
            CBAM(c[2]),
        )

        # Stage 3: P4/16 — with CBAM
        self.stage3 = nn.Sequential(
            ConvBnAct(c[2], c[3], 3, stride=2),
            CSPBlock(c[3], c[3], n=d[2]),
            CBAM(c[3]),
        )

        # Stage 4: P5/32 — with CBAM + SPP
        self.stage4 = nn.Sequential(
            ConvBnAct(c[3], c[4], 3, stride=2),
            CSPBlock(c[4], c[4], n=d[3]),
            CBAM(c[4]),
        )
        self.spp = SPP(c[4], c[4])

        # Store output channel counts for neck
        self.out_channels = [c[2], c[3], c[4]]  # P3, P4, P5

    def forward(self, x):
        """Returns multi-scale features [P3, P4, P5]."""
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.spp(self.stage4(p4))
        return [p3, p4, p5]

    @property
    def channels(self):
        """Output channel sizes for P3, P4, P5."""
        return self.out_channels
