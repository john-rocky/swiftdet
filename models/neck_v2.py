# MIT License
#
# Copyright (c) 2024 SwiftDet contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
"""GELAN-inspired FPN neck for SwiftDet2.

Uses GELAN aggregation blocks with multiple gradient paths instead of simple
DWSepConv fusion. The multiple-path design preserves gradient flow better
during training while maintaining efficient inference.

References:
    - GELAN: Wang et al. 2024 (YOLOv9: Learning What You Want to Learn
      Using Programmable Gradient Information)
    - PANet: Liu et al. 2018 (Path Aggregation Network)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBnAct, DWSepConv


class GELANBlock(nn.Module):
    """GELAN aggregation block with multiple gradient paths.

    Takes input, processes through parallel branches:
    1. conv1x1 -> conv3x3 -> conv3x3 (deep path)
    2. conv1x1 (shallow path)
    Then concatenates all intermediate features and projects.

    This creates multiple gradient paths of different lengths for better
    training dynamics, following the programmable gradient information
    principle from YOLOv9.

    Args:
        c_in: Input channels.
        c_out: Output channels.
        expansion: Ratio for internal branch channels relative to c_out.
    """

    def __init__(self, c_in, c_out, expansion=0.5):
        super().__init__()
        c_mid = int(c_out * expansion)

        # Shallow path: 1x1 projection
        self.shallow = ConvBnAct(c_in, c_mid, 1)

        # Deep path: 1x1 -> DWSepConv 3x3 -> DWSepConv 3x3
        self.deep_proj = ConvBnAct(c_in, c_mid, 1)
        self.deep_dw1 = DWSepConv(c_mid, c_mid, 3)
        self.deep_dw2 = DWSepConv(c_mid, c_mid, 3)

        # Output projection: concatenated features -> output
        # Concatenation of: shallow(c_mid) + deep_proj(c_mid) + dw1(c_mid) + dw2(c_mid)
        self.project = ConvBnAct(c_mid * 4, c_out, 1)

    def forward(self, x):
        """Forward pass with multi-path gradient flow.

        Collects intermediate features from both shallow and deep paths
        to create diverse gradient paths during backpropagation.
        """
        # Shallow path
        shallow_out = self.shallow(x)

        # Deep path with intermediate collection
        deep_proj = self.deep_proj(x)
        deep_mid = self.deep_dw1(deep_proj)
        deep_out = self.deep_dw2(deep_mid)

        # Concatenate all intermediate features
        combined = torch.cat([shallow_out, deep_proj, deep_mid, deep_out], dim=1)
        return self.project(combined)


class GELANNeck(nn.Module):
    """GELAN-inspired FPN neck for SwiftDet2.

    Top-down + Bottom-up pathways using GELANBlock for aggregation
    instead of simple DWSepConv. Maintains the PANet structure but
    replaces fusion convolutions with multi-gradient-path blocks.

    Args:
        in_channels: List of [P3_ch, P4_ch, P5_ch] from backbone.
        out_channels: Unified output channels for all levels.
    """

    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        c3_in, c4_in, c5_in = in_channels

        # Lateral 1x1 convs to unify channels
        self.lateral_p5 = ConvBnAct(c5_in, out_channels, 1)
        self.lateral_p4 = ConvBnAct(c4_in, out_channels, 1)
        self.lateral_p3 = ConvBnAct(c3_in, out_channels, 1)

        # Top-down pathway: P5 -> P4 -> P3 with GELAN aggregation
        self.td_p4 = GELANBlock(out_channels, out_channels)
        self.td_p3 = GELANBlock(out_channels, out_channels)

        # Bottom-up pathway: P3 -> P4 -> P5 with GELAN aggregation
        self.bu_p4 = GELANBlock(out_channels, out_channels)
        self.bu_p5 = GELANBlock(out_channels, out_channels)

        # Downsample convs for bottom-up
        self.down_p3_to_p4 = ConvBnAct(out_channels, out_channels, 3, stride=2)
        self.down_p4_to_p5 = ConvBnAct(out_channels, out_channels, 3, stride=2)

        self.out_channels = out_channels

    def forward(self, features):
        """Fuse multi-scale features through top-down and bottom-up pathways.

        Args:
            features: [P3, P4, P5] from backbone.

        Returns:
            [P3_out, P4_out, P5_out] fused features with unified channels.
        """
        p3_in, p4_in, p5_in = features

        # Lateral projections
        p5 = self.lateral_p5(p5_in)
        p4 = self.lateral_p4(p4_in)
        p3 = self.lateral_p3(p3_in)

        # Top-down: upsample P5 and add to P4, then upsample and add to P3
        p4_td = self.td_p4(p4 + F.interpolate(p5, size=p4.shape[2:], mode="nearest"))
        p3_td = self.td_p3(p3 + F.interpolate(p4_td, size=p3.shape[2:], mode="nearest"))

        # Bottom-up: downsample P3 and add to P4, then downsample and add to P5
        p4_out = self.bu_p4(p4_td + self.down_p3_to_p4(p3_td))
        p5_out = self.bu_p5(p5 + self.down_p4_to_p5(p4_out))

        return [p3_td, p4_out, p5_out]
