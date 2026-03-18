# SwiftDet - Detection loss functions
# MIT License - Original implementation
#
# Provides the combined detection training loss and individual sub-losses.
#
# References:
#   - Focal Loss: Lin et al. 2017
#   - Varifocal Loss: Zhang et al. 2021
#   - CIoU: Zheng et al. 2019
#   - DFL: Li et al. 2020
#   - Knowledge Distillation: Hinton et al. 2015

import torch
import torch.nn as nn
import torch.nn.functional as F

from .focal import FocalLoss, VarifocalLoss
from .iou import bbox_iou
from .dfl import DFLoss, BboxLoss, bbox2dist
from .distill import FeatureDistillLoss, LogitDistillLoss

from swiftdet.utils.assigner import TaskAlignedAssigner


class DetectionLoss(nn.Module):
    """Combined detection loss: Varifocal Loss (cls) + CIoU (box) + DFL.

    Uses TaskAlignedAssigner for dynamic positive/negative sample assignment,
    then computes:
      - Varifocal Loss for classification with IoU-aware soft labels
      - CIoU loss for bounding box regression quality
      - Distribution Focal Loss for discrete box offset distributions

    References:
        - Assignment: Feng et al. 2021 (TOOD: Task-aligned One-stage Object Detection)
        - VFL: Zhang et al. 2021 (VarifocalNet)
        - CIoU: Zheng et al. 2019
        - DFL: Li et al. 2020 (Generalized Focal Loss)

    Args:
        nc: Number of object classes.
        reg_max: Number of DFL distribution bins (default 16).
        cls_gain: Classification loss weight (default 0.5).
        box_gain: Box regression (CIoU) loss weight (default 7.5).
        dfl_gain: Distribution focal loss weight (default 1.5).
        tal_topk: Number of top candidates per GT in assigner (default 10).
        tal_alpha: Classification alignment exponent in assigner (default 0.5).
        tal_beta: IoU alignment exponent in assigner (default 6.0).
    """

    def __init__(self, nc, reg_max=16, cls_gain=0.5, box_gain=7.5, dfl_gain=1.5,
                 tal_topk=10, tal_alpha=0.5, tal_beta=6.0):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.cls_gain = cls_gain
        self.box_gain = box_gain
        self.dfl_gain = dfl_gain

        # Sub-losses
        self.vfl = VarifocalLoss()
        self.bbox_loss = BboxLoss(reg_max)

        # Task-aligned assigner for dynamic label assignment
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk, alpha=tal_alpha, beta=tal_beta,
        )

    def forward(self, predictions, targets):
        """Compute combined detection training loss.

        Args:
            predictions: Dictionary from DetectionHead containing:
                'cls': (B, N, nc) raw classification logits (pre-sigmoid).
                'box_dist': (B, N, 4*reg_max) raw box distribution logits.
                'box_decoded': (B, N, 4) decoded boxes in xyxy format.
                'anchors': (N, 2) anchor center points in image coordinates.
                'strides': (N, 1) stride value per anchor.
            targets: (B, max_gt, 5) ground truth tensor where each row is
                [class_id, x1, y1, x2, y2]. Padded rows should have class_id < 0
                or all zeros.

        Returns:
            total_loss: Scalar weighted sum of all losses (scaled by batch size).
            loss_dict: Dictionary with individual losses {'cls': ..., 'box': ..., 'dfl': ...}.
        """
        cls_logits = predictions["cls"]          # (B, N, nc)
        box_dist = predictions["box_dist"]        # (B, N, 4*reg_max)
        box_decoded = predictions["box_decoded"]  # (B, N, 4)
        anchor_points = predictions["anchors"]    # (N, 2)
        stride_tensor = predictions["strides"]    # (N, 1)

        batch_size = cls_logits.shape[0]
        device = cls_logits.device

        # --- Prepare ground truth ---
        gt_labels, gt_bboxes, gt_mask = self._prepare_targets(targets, batch_size, device)
        # gt_labels: (B, max_gt) class indices
        # gt_bboxes: (B, max_gt, 4) xyxy boxes
        # gt_mask: (B, max_gt) boolean mask for valid GTs

        # --- Dynamic label assignment ---
        # Assigner needs predicted scores (sigmoid) and decoded boxes
        cls_scores = cls_logits.detach().sigmoid()  # (B, N, nc)
        pred_boxes = box_decoded.detach()            # (B, N, 4)

        assigned_labels, assigned_bboxes, assigned_scores, fg_mask = self.assigner.forward(
            cls_scores, pred_boxes, anchor_points,
            gt_labels.unsqueeze(-1), gt_bboxes, gt_mask.unsqueeze(-1),
        )
        # assigned_labels: (B, N) class index for each anchor (-1 or 0..nc-1)
        # assigned_bboxes: (B, N, 4) target boxes for each anchor
        # assigned_scores: (B, N, nc) soft target scores (IoU-weighted one-hot)
        # fg_mask: (B, N) boolean foreground mask

        target_scores_sum = max(assigned_scores.sum(), 1.0)

        # --- Classification Loss (Varifocal) ---
        loss_cls = self.vfl(cls_logits, assigned_scores)
        loss_cls = loss_cls / target_scores_sum

        # --- Box Regression Loss (CIoU + DFL) ---
        if fg_mask.any():
            loss_iou, loss_dfl = self.bbox_loss(
                box_dist,
                box_decoded,
                anchor_points,
                assigned_bboxes,
                assigned_scores,
                target_scores_sum,
                fg_mask,
                stride_tensor,
            )
        else:
            loss_iou = torch.tensor(0.0, device=device)
            loss_dfl = torch.tensor(0.0, device=device)

        # --- Weighted combination ---
        loss_cls_scaled = self.cls_gain * loss_cls
        loss_iou_scaled = self.box_gain * loss_iou
        loss_dfl_scaled = self.dfl_gain * loss_dfl

        total_loss = (loss_cls_scaled + loss_iou_scaled + loss_dfl_scaled) * batch_size

        loss_dict = {
            "cls": loss_cls_scaled.detach(),
            "box": loss_iou_scaled.detach(),
            "dfl": loss_dfl_scaled.detach(),
        }

        return total_loss, loss_dict

    @staticmethod
    def _prepare_targets(targets, batch_size, device):
        """Parse the targets tensor into labels, boxes, and a validity mask.

        Args:
            targets: (B, max_gt, 5) with [class_id, x1, y1, x2, y2] per row.
                Padding rows have class_id < 0 or box area == 0.
            batch_size: Number of images in the batch.
            device: Target device.

        Returns:
            gt_labels: (B, max_gt) long tensor of class indices.
            gt_bboxes: (B, max_gt, 4) float tensor of xyxy boxes.
            gt_mask: (B, max_gt) boolean tensor, True for valid GTs.
        """
        gt_labels = targets[..., 0].long()          # (B, max_gt)
        gt_bboxes = targets[..., 1:5].float()       # (B, max_gt, 4)

        # Valid GT: non-negative class id and positive box area
        box_area = (
            (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp(min=0)
            * (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp(min=0)
        )
        gt_mask = (gt_labels >= 0) & (box_area > 0)  # (B, max_gt)

        return gt_labels, gt_bboxes, gt_mask


__all__ = [
    "DetectionLoss",
    "FocalLoss",
    "VarifocalLoss",
    "bbox_iou",
    "DFLoss",
    "BboxLoss",
    "bbox2dist",
    "FeatureDistillLoss",
    "LogitDistillLoss",
]
