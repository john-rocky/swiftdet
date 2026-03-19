# SwiftDet - Task-Aligned Assigner
# MIT License - Original implementation
# Reference: Feng et al. 2021, "TOOD: Task-aligned One-stage Object Detection"
#            (NeurIPS 2021, arXiv:2108.07755)
#
# The task-aligned assigner uses a composite metric combining classification
# confidence and localization quality to assign ground truth boxes to anchors.
# alignment_metric = cls_score^alpha * iou^beta

import torch
import torch.nn.functional as F

from .boxes import box_iou


class TaskAlignedAssigner:
    """Assigns ground truth targets to anchor points using a task-aligned metric.

    The alignment metric jointly considers classification confidence and
    localization quality:

        alignment_metric = cls_score^alpha * iou^beta

    For each ground truth, the top-k anchors by alignment metric are selected
    as positive candidates.  When an anchor is assigned to multiple ground
    truths, the GT with the highest IoU wins.

    Args:
        topk: Number of candidate anchors per ground truth (default 13).
        alpha: Exponent for classification score in alignment metric (default 1.0).
        beta: Exponent for IoU in alignment metric (default 6.0).
    """

    def __init__(self, topk: int = 13, alpha: float = 1.0, beta: float = 6.0):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def forward(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ):
        """Assign ground truths to anchors via task-aligned metric.

        Args:
            cls_scores: (B, N, nc) predicted classification scores (after sigmoid).
            bbox_preds: (B, N, 4) predicted bounding boxes in xyxy format.
            anchor_points: (N, 2) anchor center points (x, y).
            gt_labels: (B, max_gt, 1) ground truth class labels.
            gt_bboxes: (B, max_gt, 4) ground truth bounding boxes in xyxy format.
            mask_gt: (B, max_gt, 1) binary mask indicating valid ground truths.

        Returns:
            target_labels: (B, N) assigned class labels (0 = background).
            target_bboxes: (B, N, 4) assigned target bounding boxes.
            target_scores: (B, N, nc) soft classification targets.
            fg_mask: (B, N) boolean foreground mask.
        """
        B, N, nc = cls_scores.shape
        max_gt = gt_bboxes.shape[1]
        device = cls_scores.device

        if max_gt == 0:
            return (
                torch.zeros(B, N, dtype=torch.long, device=device),
                torch.zeros(B, N, 4, device=device),
                torch.zeros(B, N, nc, device=device),
                torch.zeros(B, N, dtype=torch.bool, device=device),
            )

        # --- Step 1: Determine which anchors fall inside each GT box ---
        # anchor_points: (N, 2), gt_bboxes: (B, max_gt, 4)
        # Expand for broadcasting: anchor (1, N, 1, 2) vs gt (B, 1, max_gt, 4)
        ap = anchor_points[None, :, None, :]  # (1, N, 1, 2)
        gt = gt_bboxes[:, None, :, :]  # (B, 1, max_gt, 4)

        # Check if anchor center is inside the GT box
        inside_left = ap[..., 0] > gt[..., 0]  # (B, N, max_gt)
        inside_top = ap[..., 1] > gt[..., 1]
        inside_right = ap[..., 0] < gt[..., 2]
        inside_bottom = ap[..., 1] < gt[..., 3]
        inside_mask = inside_left & inside_top & inside_right & inside_bottom  # (B, N, max_gt)

        # --- Step 2: Compute IoU between predictions and GTs ---
        # bbox_preds: (B, N, 4), gt_bboxes: (B, max_gt, 4)
        # Force float32 to prevent iou^beta underflow in AMP (float16)
        ious = torch.zeros(B, N, max_gt, device=device, dtype=torch.float32)
        for b in range(B):
            if mask_gt[b].any():
                ious[b] = box_iou(bbox_preds[b].float(), gt_bboxes[b].float())  # (N, max_gt)

        # --- Step 3: Compute alignment metric ---
        # Gather classification scores for the GT class of each GT
        gt_labels_long = gt_labels.squeeze(-1).long()  # (B, max_gt)
        # For each (batch, anchor, gt), gather the cls score for that GT's class
        # cls_scores: (B, N, nc) -> index with gt_labels to get (B, N, max_gt)
        gt_cls_idx = gt_labels_long[:, None, :].expand(B, N, max_gt).clamp(0, nc - 1)
        batch_cls = cls_scores.float().gather(2, gt_cls_idx)  # (B, N, max_gt) float32

        # alignment_metric = cls_score^alpha * iou^beta (float32 to avoid underflow)
        alignment_metric = batch_cls.pow(self.alpha) * ious.pow(self.beta)  # (B, N, max_gt)

        # Mask out anchors not inside GT boxes and invalid GTs
        mask_gt_expanded = mask_gt.squeeze(-1)[:, None, :].expand_as(alignment_metric)  # (B, N, max_gt)
        alignment_metric = alignment_metric * inside_mask.float() * mask_gt_expanded.float()

        # --- Step 4: Select top-k anchors per GT ---
        # For each GT, find the top-k anchors with highest alignment metric
        # alignment_metric: (B, N, max_gt) -> transpose to (B, max_gt, N)
        am_transposed = alignment_metric.permute(0, 2, 1)  # (B, max_gt, N)

        topk_k = min(self.topk, N)
        _, topk_indices = am_transposed.topk(topk_k, dim=-1)  # (B, max_gt, topk)

        # Build assignment mask: (B, N, max_gt)
        candidate_mask = torch.zeros(B, max_gt, N, device=device)
        candidate_mask.scatter_(2, topk_indices, 1.0)
        candidate_mask = candidate_mask.permute(0, 2, 1)  # (B, N, max_gt)

        # Apply inside and validity masks
        candidate_mask = candidate_mask * inside_mask.float() * mask_gt_expanded.float()

        # --- Step 5: Resolve conflicts (anchor assigned to multiple GTs) ---
        # For each anchor assigned to multiple GTs, keep the one with highest IoU
        # candidate_mask: (B, N, max_gt)
        overlaps = ious * candidate_mask  # (B, N, max_gt)

        # Number of GT assignments per anchor
        n_assignments = candidate_mask.sum(dim=-1)  # (B, N)
        conflict_mask = n_assignments > 1  # (B, N)

        if conflict_mask.any():
            # For conflicted anchors, keep only the GT with highest IoU
            best_gt = overlaps.argmax(dim=-1)  # (B, N)
            new_mask = torch.zeros_like(candidate_mask)
            new_mask.scatter_(2, best_gt.unsqueeze(-1), 1.0)
            conflict_expanded = conflict_mask.unsqueeze(-1).expand_as(candidate_mask)
            candidate_mask = torch.where(conflict_expanded, new_mask, candidate_mask)
            candidate_mask = candidate_mask * inside_mask.float() * mask_gt_expanded.float()

        # --- Step 6: Build final targets ---
        fg_mask = candidate_mask.sum(dim=-1) > 0  # (B, N)
        assigned_gt_idx = candidate_mask.argmax(dim=-1)  # (B, N)

        # Target labels: class id for foreground, 0 for background
        target_labels = torch.zeros(B, N, dtype=torch.long, device=device)
        target_bboxes = torch.zeros(B, N, 4, device=device)
        target_scores = torch.zeros(B, N, nc, device=device)

        for b in range(B):
            fg = fg_mask[b]  # (N,)
            if not fg.any():
                continue

            gt_idx = assigned_gt_idx[b][fg]  # (N_fg,)
            target_labels[b][fg] = gt_labels_long[b][gt_idx]
            target_bboxes[b][fg] = gt_bboxes[b][gt_idx]

            # Soft targets: scale one-hot by normalized alignment metric * max IoU
            # YOLOv8-style: norm = (am / max_am_per_gt) * max_iou_per_gt
            fg_indices = fg.nonzero(as_tuple=False).squeeze(-1)
            am_values = alignment_metric[b][fg_indices, gt_idx]  # (N_fg,)
            iou_values = ious[b][fg_indices, gt_idx]  # (N_fg,)

            one_hot = F.one_hot(target_labels[b][fg], nc).float()  # (N_fg, nc)

            # Per-GT normalization: scatter_reduce to get max AM and max IoU per GT
            max_am_per_gt = torch.zeros(max_gt, device=device)
            max_iou_per_gt = torch.zeros(max_gt, device=device)
            max_am_per_gt.scatter_reduce_(0, gt_idx, am_values, reduce='amax')
            max_iou_per_gt.scatter_reduce_(0, gt_idx, iou_values, reduce='amax')

            am_max_gathered = max_am_per_gt[gt_idx].clamp(min=1e-7)
            iou_max_gathered = max_iou_per_gt[gt_idx].clamp(min=1e-7)

            norm_am = (am_values / am_max_gathered) * iou_max_gathered
            norm_am = norm_am.clamp(min=0)
            target_scores[b][fg] = one_hot * norm_am.unsqueeze(-1)

        return target_labels, target_bboxes, target_scores, fg_mask
