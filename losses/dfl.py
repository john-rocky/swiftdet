# SwiftDet - Distribution Focal Loss and combined box regression loss
# MIT License - Original implementation
#
# References:
#   - DFL: Li et al. 2020, "Generalized Focal Loss: Learning Qualified and
#     Distributed Bounding Boxes for Dense Object Detection"
#   - CIoU: Zheng et al. 2019, "Distance-IoU Loss: Faster and Better Learning
#     for Bounding Box Regression"

import torch
import torch.nn as nn
import torch.nn.functional as F

from .iou import bbox_iou


def bbox2dist(bbox, anchor_points, reg_max):
    """Convert bounding box coordinates to distance representation for DFL targets.

    This is the inverse of dist2bbox: given boxes in xyxy format and their
    corresponding anchor center points, compute (left, top, right, bottom) distances.
    Distances are clamped to [0, reg_max - 1] to lie within the DFL bin range.

    Args:
        bbox: (..., 4) tensor in (x1, y1, x2, y2) format.
        anchor_points: (..., 2) tensor with (x, y) center coordinates.
        reg_max: Maximum distance value (number of DFL bins).

    Returns:
        (..., 4) tensor with (left, top, right, bottom) distances clamped
        to [0, reg_max - 1 - epsilon].
    """
    x1y1 = bbox[..., :2]
    x2y2 = bbox[..., 2:]
    lt = anchor_points - x1y1  # left, top distances
    rb = x2y2 - anchor_points  # right, bottom distances
    dist = torch.cat([lt, rb], dim=-1)
    return dist.clamp(0, reg_max - 1.0 - 0.01)


class DFLoss(nn.Module):
    """Distribution Focal Loss for bounding box regression.

    From Li et al. 2020 ("Generalized Focal Loss"):
    Trains the model to predict a discrete probability distribution over
    a set of integer-valued bin positions for each box offset. The continuous
    regression target is decomposed into a two-point distribution over its
    two nearest integer neighbors, and the loss is the cross-entropy between
    the predicted distribution and this target distribution.

    For a continuous target y with floor(y)=y_l and ceil(y)=y_r:
        target_dist[y_l] = y_r - y   (weight on left neighbor)
        target_dist[y_r] = y - y_l   (weight on right neighbor)
        Loss = CE(predicted_dist, target_dist)

    Args:
        reg_max: Number of distribution bins (default 16).
    """

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, target):
        """Compute distribution focal loss.

        Args:
            pred_dist: (N, 4, reg_max) predicted logits for each bin.
            target: (N, 4) continuous regression targets in [0, reg_max-1].

        Returns:
            Scalar DFL loss averaged over all valid elements.
        """
        reg_max = self.reg_max

        # Decompose continuous target into two neighboring integers
        target = target.clamp(0, reg_max - 1.0 - 0.01)
        target_left = target.long()                # floor index
        target_right = target_left + 1             # ceil index

        # Weights for the two-point distribution
        weight_right = target - target_left.float()    # fractional part
        weight_left = 1.0 - weight_right

        # Cross-entropy with the two-point target distribution
        # pred_dist: (N, 4, reg_max) -> log_softmax over bin dimension
        log_probs = F.log_softmax(pred_dist, dim=-1)  # (N, 4, reg_max)

        # Gather log-probabilities at left and right indices
        # target_left/right: (N, 4) -> (N, 4, 1) for gather
        loss_left = -log_probs.gather(-1, target_left.unsqueeze(-1)).squeeze(-1)
        loss_right = -log_probs.gather(-1, target_right.unsqueeze(-1)).squeeze(-1)

        # Weighted combination
        loss = weight_left * loss_left + weight_right * loss_right

        return loss.mean()


class BboxLoss(nn.Module):
    """Combined bounding box regression loss: CIoU + Distribution Focal Loss.

    The regression loss for anchor-free detection consists of:
    1. CIoU loss: Measures overall box overlap quality including center distance
       and aspect ratio consistency (Zheng et al. 2019).
    2. DFL loss: Trains the discrete distribution over box offsets to have its
       mass concentrated around the true offset (Li et al. 2020).

    Both losses are weighted by the target classification score (from the
    task-aligned assigner) so that higher-quality assignments contribute
    proportionally more to the gradient.

    Args:
        reg_max: Number of DFL bins (default 16).
    """

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.dfl_loss = DFLoss(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride_tensor=None):
        """Compute combined box regression loss.

        Args:
            pred_dist: (B, N, 4*reg_max) raw distribution predictions.
            pred_bboxes: (B, N, 4) decoded bboxes in xyxy format (image coords).
            anchor_points: (N, 2) anchor center coordinates (image coords).
            target_bboxes: (B, N, 4) assigned ground-truth boxes in xyxy format.
            target_scores: (B, N, nc) soft classification targets from assigner.
            target_scores_sum: Scalar, sum of target scores for normalization.
            fg_mask: (B, N) boolean mask indicating foreground (positive) anchors.
            stride_tensor: (N, 1) stride per anchor, used to normalize for DFL.

        Returns:
            loss_iou: Scalar CIoU loss.
            loss_dfl: Scalar DFL loss.
        """
        # Per-anchor weight: sum of class scores across classes (IoU-aware quality)
        weight = target_scores.sum(dim=-1)[fg_mask].unsqueeze(-1)  # (n_pos, 1)

        # --- CIoU Loss ---
        iou = bbox_iou(
            pred_bboxes[fg_mask], target_bboxes[fg_mask], ciou=True,
        )  # (n_pos,)
        loss_iou = ((1.0 - iou) * weight.squeeze(-1)).sum() / target_scores_sum

        # --- DFL Loss ---
        batch_size = fg_mask.shape[0]

        # Expand anchor_points for batch dimension: (N, 2) -> (B, N, 2)
        anchors_expanded = anchor_points.unsqueeze(0).expand(batch_size, -1, -1)
        fg_anchors = anchors_expanded[fg_mask]  # (n_pos, 2)
        fg_targets = target_bboxes[fg_mask]     # (n_pos, 4)

        # Normalize by stride so DFL targets are in [0, reg_max-1] range
        if stride_tensor is not None:
            strides_expanded = stride_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            fg_strides = strides_expanded[fg_mask]  # (n_pos, 1)
            fg_anchors = fg_anchors / fg_strides
            fg_targets = fg_targets / fg_strides  # broadcasts (n_pos, 1) to (n_pos, 4)

        target_dist = bbox2dist(fg_targets, fg_anchors, self.reg_max)

        # Reshape predicted distributions for foreground
        fg_pred_dist = pred_dist[fg_mask]  # (n_pos, 4*reg_max)
        fg_pred_dist = fg_pred_dist.view(-1, 4, self.reg_max)  # (n_pos, 4, reg_max)

        loss_dfl = self.dfl_loss(fg_pred_dist, target_dist)
        # Weight DFL by target scores (same as IoU loss weighting)
        loss_dfl = loss_dfl * weight.sum() / target_scores_sum

        return loss_iou, loss_dfl
