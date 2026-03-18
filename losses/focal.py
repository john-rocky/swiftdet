# SwiftDet - Focal and Varifocal loss functions
# MIT License - Original implementation
#
# References:
#   - Focal Loss: Lin et al. 2017, "Focal Loss for Dense Object Detection"
#   - Varifocal Loss: Zhang et al. 2021, "VarifocalNet: An IoU-aware Dense Object Detector"

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in dense detection.

    From Lin et al. 2017 ("Focal Loss for Dense Object Detection"):
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    The modulating factor (1 - p_t)^gamma down-weights loss for well-classified
    examples, focusing training on hard negatives.

    Args:
        alpha: Balancing factor for positive/negative classes (default 0.25).
        gamma: Focusing parameter that controls rate of down-weighting (default 2.0).
        reduction: Reduction mode: 'none', 'mean', or 'sum' (default 'none').
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """Compute focal loss.

        Args:
            pred: (*, C) raw logits (pre-sigmoid).
            target: (*, C) binary targets, same shape as pred.

        Returns:
            Focal loss tensor with specified reduction.
        """
        # Binary cross-entropy (element-wise, no reduction)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # Predicted probability for modulating factor
        p = pred.sigmoid()
        # p_t: probability of correct class
        p_t = target * p + (1.0 - target) * (1.0 - p)

        # Modulating factor
        modulating = (1.0 - p_t) ** self.gamma

        # Alpha balancing factor
        alpha_t = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)

        loss = alpha_t * modulating * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class VarifocalLoss(nn.Module):
    """Varifocal Loss for IoU-aware classification in dense object detection.

    From Zhang et al. 2021 ("VarifocalNet: An IoU-aware Dense Object Detector"):
        VFL(p, q) = -q * [q * log(p) + (1-q) * alpha * p^gamma * log(1-p)]

    For positive samples (q > 0), q is the target IoU between the predicted box
    and the ground truth, producing a soft label that couples classification
    confidence with localization quality.

    For negative samples (q = 0), this reduces to a weighted focal-style loss
    that down-weights easy negatives via the p^gamma term.

    Args:
        alpha: Weighting factor for negative samples (default 0.75).
        gamma: Focusing parameter for negative sample modulation (default 2.0).
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """Compute varifocal loss.

        Args:
            pred: (B, N, C) raw logits (pre-sigmoid).
            target: (B, N, C) soft targets. Values > 0 for positives (IoU score),
                    0 for negatives.

        Returns:
            Scalar loss (sum over all elements).
        """
        p = pred.sigmoid()

        # Binary cross-entropy components (numerically stable via log-sum-exp)
        # log(p) = pred - softplus(pred), log(1-p) = -softplus(pred)
        log_p = F.logsigmoid(pred)
        log_one_minus_p = F.logsigmoid(-pred)

        # Positive mask (q > 0)
        pos_mask = (target > 0).float()

        # Positive loss: -q * log(p)  weighted by q (quality-aware)
        pos_loss = -target * log_p

        # Negative loss: -alpha * p^gamma * log(1 - p)
        neg_weight = self.alpha * p.detach().abs().pow(self.gamma)
        neg_loss = -neg_weight * log_one_minus_p

        # Combine: positives use pos_loss, negatives use neg_loss
        loss = pos_mask * pos_loss + (1.0 - pos_mask) * neg_loss

        return loss.sum()
