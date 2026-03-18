"""Bidirectional Feature Pyramid Network (BiFPN-Lite).

Top-down and bottom-up pathways with depth-wise separable convolutions
for efficient multi-scale feature fusion.

References:
    - PANet: Liu et al. 2018 (Path Aggregation Network)
    - BiFPN: Tan et al. 2020 (EfficientDet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBnAct, DWSepConv


class BiFPNLite(nn.Module):
    """Lightweight bidirectional FPN (PANet-style).

    Takes P3, P4, P5 from backbone and produces fused P3, P4, P5 features.

    Args:
        in_channels: List of [P3_ch, P4_ch, P5_ch] from backbone.
        out_channels: Unified output channel count for all levels.
    """

    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        c3_in, c4_in, c5_in = in_channels

        # Lateral 1x1 convs to unify channels
        self.lateral_p5 = ConvBnAct(c5_in, out_channels, 1)
        self.lateral_p4 = ConvBnAct(c4_in, out_channels, 1)
        self.lateral_p3 = ConvBnAct(c3_in, out_channels, 1)

        # Top-down pathway: P5 → P4 → P3
        self.td_p4 = DWSepConv(out_channels, out_channels, 3)
        self.td_p3 = DWSepConv(out_channels, out_channels, 3)

        # Bottom-up pathway: P3 → P4 → P5
        self.bu_p4 = DWSepConv(out_channels, out_channels, 3)
        self.bu_p5 = DWSepConv(out_channels, out_channels, 3)

        # Downsample convs for bottom-up
        self.down_p3_to_p4 = ConvBnAct(out_channels, out_channels, 3, stride=2)
        self.down_p4_to_p5 = ConvBnAct(out_channels, out_channels, 3, stride=2)

        self.out_channels = out_channels

    def forward(self, features):
        """
        Args:
            features: [P3, P4, P5] from backbone.

        Returns:
            [P3_out, P4_out, P5_out] fused features.
        """
        p3_in, p4_in, p5_in = features

        # Lateral projections
        p5 = self.lateral_p5(p5_in)
        p4 = self.lateral_p4(p4_in)
        p3 = self.lateral_p3(p3_in)

        # Top-down: upsample and add
        p4_td = self.td_p4(p4 + F.interpolate(p5, size=p4.shape[2:], mode="nearest"))
        p3_td = self.td_p3(p3 + F.interpolate(p4_td, size=p3.shape[2:], mode="nearest"))

        # Bottom-up: downsample and add
        p4_out = self.bu_p4(p4_td + self.down_p3_to_p4(p3_td))
        p5_out = self.bu_p5(p5 + self.down_p4_to_p5(p4_out))

        return [p3_td, p4_out, p5_out]
