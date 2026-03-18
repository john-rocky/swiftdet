"""Full SwiftDet detector assembling backbone, neck, and detection head.

References:
    - CSPNet backbone: Wang et al. 2020
    - BiFPN neck: Tan et al. 2020 (EfficientDet)
    - Decoupled head: Ge et al. 2021 (YOLOX)
    - DFL: Li et al. 2020 (Generalized Focal Loss)
"""

import torch.nn as nn

from .backbone import CSPBackbone
from .blocks import ConvBnAct
from .head import DetectionHead
from .neck import BiFPNLite


# Variant configurations: (width_mult, depth_mult, max_channels, neck_channels)
VARIANT_CONFIGS = {
    "n": {"width_mult": 0.30, "depth_mult": 0.33, "max_ch": 512, "neck_ch": 128},
    "s": {"width_mult": 0.50, "depth_mult": 0.33, "max_ch": 512, "neck_ch": 192},
    "m": {"width_mult": 0.75, "depth_mult": 0.67, "max_ch": 768, "neck_ch": 256},
    "l": {"width_mult": 1.00, "depth_mult": 1.00, "max_ch": 1024, "neck_ch": 384},
}


class SwiftDetector(nn.Module):
    """Complete SwiftDet detection model.

    Assembles backbone (CSPBackbone), neck (BiFPNLite), and head (DetectionHead)
    into a single end-to-end detector.

    Args:
        config: Model configuration dictionary with keys:
            - width_mult (float): Backbone channel multiplier.
            - depth_mult (float): Backbone depth multiplier.
            - max_ch (int): Maximum backbone channel count.
            - neck_ch (int): Unified channel width in the neck.
            - nc (int): Number of object classes.
            - reg_max (int): Number of DFL bins.
    """

    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = {**VARIANT_CONFIGS["n"], "nc": 80, "reg_max": 16}

        nc = config.get("nc", 80)
        reg_max = config.get("reg_max", 16)
        neck_ch = config["neck_ch"]

        # Backbone: produces multi-scale features [P3, P4, P5]
        self.backbone = CSPBackbone(
            width_mult=config["width_mult"],
            depth_mult=config["depth_mult"],
            max_channels=config["max_ch"],
        )

        # Neck: fuse multi-scale features to a unified channel width
        self.neck = BiFPNLite(
            in_channels=self.backbone.out_channels,
            out_channels=neck_ch,
        )

        # Head: decoupled classification and regression branches
        head_in_channels = [neck_ch] * len(self.backbone.out_channels)
        self.head = DetectionHead(
            nc=nc,
            in_channels=head_in_channels,
            reg_max=reg_max,
        )

        self.nc = nc
        self.reg_max = reg_max

    def forward(self, x):
        """Run full detection pipeline.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Dictionary containing:
                cls: Raw classification logits (B, N_total, nc).
                box_dist: Raw box distributions (B, N_total, 4*reg_max).
                box_decoded: Decoded (x1, y1, x2, y2) boxes (B, N_total, 4).
                anchors: Anchor center points (N_total, 2).
                strides: Stride per anchor (N_total, 1).
        """
        features = self.backbone(x)
        fused = self.neck(features)
        return self.head(fused)

    def fuse(self):
        """Fuse Conv2d + BatchNorm2d layers for inference speedup.

        Iterates over all modules and replaces the forward method of ConvBnAct
        blocks with the fused variant that skips the BN computation. The BN
        parameters are folded into the conv weights in-place.

        Returns:
            self (for method chaining).
        """
        for module in self.modules():
            if isinstance(module, ConvBnAct) and hasattr(module, "bn"):
                # Fold BN into conv weights
                module.conv = _fuse_conv_bn(module.conv, module.bn)
                # Remove BN and switch to fused forward
                module.bn = nn.Identity()
                module.forward = module.forward_fuse
        return self


def _fuse_conv_bn(conv, bn):
    """Fuse a Conv2d and BatchNorm2d into a single Conv2d.

    The BN transform y = gamma * (x - mean) / sqrt(var + eps) + beta
    is absorbed into the conv weight and bias.

    Args:
        conv: nn.Conv2d module.
        bn: nn.BatchNorm2d module.

    Returns:
        Fused nn.Conv2d with bias.
    """
    import torch

    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    ).to(conv.weight.device)

    # BN parameters
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    # Fold: W_fused = gamma / sqrt(var) * W_conv
    scale = (gamma / var_sqrt).reshape(-1, 1, 1, 1)
    fused.weight.data = conv.weight.data * scale

    # Fold: b_fused = gamma * (b_conv - mean) / sqrt(var) + beta
    conv_bias = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=conv.weight.device)
    fused.bias.data = (conv_bias - mean) * (gamma / var_sqrt) + beta

    return fused


def build_swiftdet(variant="n", nc=80, reg_max=16):
    """Build a SwiftDet model from a variant name.

    Args:
        variant: Model size variant, one of "n" (nano), "s" (small),
            "m" (medium), or "l" (large).
        nc: Number of object classes (default 80 for COCO).
        reg_max: Number of DFL distribution bins (default 16).

    Returns:
        SwiftDetector model instance.

    Raises:
        ValueError: If variant is not recognized.
    """
    if variant not in VARIANT_CONFIGS:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: {list(VARIANT_CONFIGS.keys())}"
        )
    config = {**VARIANT_CONFIGS[variant], "nc": nc, "reg_max": reg_max}
    return SwiftDetector(config)
