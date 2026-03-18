"""Anchor-free decoupled detection head with Distribution Focal Loss.

Decoupled classification and regression branches process features independently
at each FPN scale level, following the principle that classification and
localization tasks benefit from separate learned representations.

References:
    - Decoupled head: Ge et al. 2021 (YOLOX: Exceeding YOLO Series in 2021)
    - DFL: Li et al. 2020 (Generalized Focal Loss)
    - Anchor-free detection: Tian et al. 2019 (FCOS)
"""

import math

import torch
import torch.nn as nn

from .blocks import ConvBnAct, DFL


def make_anchors(feature_maps, strides, offset=0.5):
    """Generate anchor center points for all feature map levels.

    For each spatial location in each feature map, produces an (x, y) center
    coordinate in the input image space, along with the stride value.

    Args:
        feature_maps: List of tensors, one per FPN level. Only shape is used.
        strides: List of stride values corresponding to each feature level.
        offset: Sub-pixel offset for anchor center (default 0.5 = cell center).

    Returns:
        anchor_points: Tensor of shape (N_total, 2) with (x, y) center coords.
        stride_tensor: Tensor of shape (N_total, 1) with stride per anchor.
    """
    anchor_points = []
    stride_values = []
    dtype, device = feature_maps[0].dtype, feature_maps[0].device

    for feat, stride in zip(feature_maps, strides):
        h, w = feat.shape[2], feat.shape[3]
        # Grid of (x, y) coordinates at cell centers
        sy = torch.arange(h, device=device, dtype=dtype) + offset
        sx = torch.arange(w, device=device, dtype=dtype) + offset
        grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij")
        # Scale from feature-map coords to input-image coords
        points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1) * stride
        anchor_points.append(points)
        stride_values.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_points, dim=0), torch.cat(stride_values, dim=0)


def dist2bbox(distance, anchor_points):
    """Convert distance predictions to bounding box coordinates.

    Each row of distance is (left, top, right, bottom) relative to the
    corresponding anchor point. Positive distances extend outward from the
    anchor center in each direction.

    Args:
        distance: Tensor of shape (..., 4) with (left, top, right, bottom).
        anchor_points: Tensor of shape (..., 2) with (x, y) center coords.

    Returns:
        Bounding boxes in (x1, y1, x2, y2) format, same shape as distance.
    """
    lt = distance[..., :2]  # left, top
    rb = distance[..., 2:]  # right, bottom
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat([x1y1, x2y2], dim=-1)


def bbox2dist(bbox, anchor_points, reg_max):
    """Convert bounding box coordinates to distance representation.

    Inverse of dist2bbox. Clamps distances to [0, reg_max - 1].

    Args:
        bbox: Tensor of shape (..., 4) in (x1, y1, x2, y2) format.
        anchor_points: Tensor of shape (..., 2) with (x, y) center coords.
        reg_max: Maximum distance value (number of DFL bins).

    Returns:
        Distance tensor of shape (..., 4) with (left, top, right, bottom).
    """
    x1y1 = bbox[..., :2]
    x2y2 = bbox[..., 2:]
    lt = anchor_points - x1y1  # left, top
    rb = x2y2 - anchor_points  # right, bottom
    dist = torch.cat([lt, rb], dim=-1)
    return dist.clamp(0, reg_max - 1.0 - 0.01)


class DetectionHead(nn.Module):
    """Anchor-free decoupled detection head with DFL.

    For each scale level, applies:
    - Shared stem conv
    - Separate cls branch (2x Conv -> cls output with nc classes)
    - Separate reg branch (2x Conv -> reg output with 4*reg_max values)

    Args:
        nc: Number of classes.
        in_channels: List of input channel sizes from neck (one per scale).
        reg_max: Number of DFL bins (default 16).
    """

    # Default strides for P3, P4, P5
    DEFAULT_STRIDES = [8, 16, 32]

    def __init__(self, nc=80, in_channels=None, reg_max=16):
        super().__init__()
        if in_channels is None:
            in_channels = [256, 256, 256]

        self.nc = nc
        self.reg_max = reg_max
        self.n_levels = len(in_channels)
        self.strides = self.DEFAULT_STRIDES[: self.n_levels]

        # Build per-level branches
        self.stems = nn.ModuleList()
        self.cls_branches = nn.ModuleList()
        self.reg_branches = nn.ModuleList()
        self.cls_outputs = nn.ModuleList()
        self.reg_outputs = nn.ModuleList()

        for ch_in in in_channels:
            # Stem: project to a shared hidden dim
            hidden = max(ch_in, 64)
            self.stems.append(ConvBnAct(ch_in, hidden, 1))

            # Classification branch: 2 stacked 3x3 convs
            self.cls_branches.append(nn.Sequential(
                ConvBnAct(hidden, hidden, 3),
                ConvBnAct(hidden, hidden, 3),
            ))

            # Regression branch: 2 stacked 3x3 convs
            self.reg_branches.append(nn.Sequential(
                ConvBnAct(hidden, hidden, 3),
                ConvBnAct(hidden, hidden, 3),
            ))

            # Output projections (1x1 conv, no BN/activation)
            self.cls_outputs.append(nn.Conv2d(hidden, nc, 1))
            self.reg_outputs.append(nn.Conv2d(hidden, 4 * reg_max, 1))

        # DFL layer for converting distributions to coordinates
        self.dfl = DFL(reg_max)

        self._initialize_biases()

    def _initialize_biases(self):
        """Initialize output conv biases for stable early training.

        Classification bias is set so that the initial predicted probability
        is approximately 0.01 (following focal loss convention from
        Lin et al. 2017). Regression bias is zero-initialized.
        """
        prior_prob = 0.01
        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        for cls_out in self.cls_outputs:
            nn.init.constant_(cls_out.bias, bias_value)
        for reg_out in self.reg_outputs:
            nn.init.zeros_(reg_out.bias)

    def forward(self, features):
        """Process multi-scale features through decoupled branches.

        Args:
            features: List of feature maps from neck, one per scale level.
                Each has shape (B, C_i, H_i, W_i).

        Returns:
            Dictionary containing:
                cls: Raw classification logits (B, N_total, nc).
                box_dist: Raw box distributions (B, N_total, 4*reg_max).
                box_decoded: Decoded (x1, y1, x2, y2) boxes (B, N_total, 4).
                anchors: Anchor center points (N_total, 2).
                strides: Stride per anchor (N_total, 1).
        """
        cls_list = []
        reg_list = []

        for i, feat in enumerate(features):
            stem_out = self.stems[i](feat)

            # Classification path
            cls_feat = self.cls_branches[i](stem_out)
            cls_out = self.cls_outputs[i](cls_feat)  # (B, nc, H, W)

            # Regression path
            reg_feat = self.reg_branches[i](stem_out)
            reg_out = self.reg_outputs[i](reg_feat)  # (B, 4*reg_max, H, W)

            b, _, h, w = cls_out.shape
            # Reshape to (B, H*W, C)
            cls_list.append(cls_out.permute(0, 2, 3, 1).reshape(b, h * w, self.nc))
            reg_list.append(reg_out.permute(0, 2, 3, 1).reshape(b, h * w, 4 * self.reg_max))

        # Concatenate across all scale levels
        cls_raw = torch.cat(cls_list, dim=1)      # (B, N_total, nc)
        box_dist = torch.cat(reg_list, dim=1)      # (B, N_total, 4*reg_max)

        # Generate anchors from feature map shapes
        anchor_points, stride_tensor = make_anchors(features, self.strides)

        # Decode boxes: DFL on raw distributions, then convert distances to boxes
        box_decoded = self._decode_boxes(box_dist, anchor_points, stride_tensor)

        return {
            "cls": cls_raw,
            "box_dist": box_dist,
            "box_decoded": box_decoded,
            "anchors": anchor_points,
            "strides": stride_tensor,
        }

    def _decode_boxes(self, box_dist, anchor_points, stride_tensor):
        """Decode raw box distributions into (x1, y1, x2, y2) coordinates.

        Steps:
            1. Apply DFL to convert per-bin distributions to scalar distances.
            2. Scale distances by stride.
            3. Convert (l, t, r, b) distances to (x1, y1, x2, y2) boxes.

        Args:
            box_dist: Raw distributions (B, N, 4*reg_max).
            anchor_points: (N, 2) center coordinates in image space.
            stride_tensor: (N, 1) stride values.

        Returns:
            Decoded boxes (B, N, 4) in (x1, y1, x2, y2) format.
        """
        # DFL: (B, N, 4*reg_max) -> (B, N, 4) as expected distances
        distances = self.dfl(box_dist)  # (B, N, 4)
        # Scale by stride
        distances = distances * stride_tensor.unsqueeze(0)
        # Convert to box coords
        return dist2bbox(distances, anchor_points.unsqueeze(0))
