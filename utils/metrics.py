# SwiftDet - COCO-style mAP evaluation
# MIT License - Original implementation
# Reference: Lin et al. 2014, "Microsoft COCO: Common Objects in Context"
# Implements the standard COCO AP protocol: 101-point interpolated precision-recall
# across IoU thresholds from 0.50 to 0.95 in steps of 0.05.

from typing import Dict, List, Optional

import numpy as np
import torch

from .boxes import box_iou


class APMetrics:
    """Compute Average Precision (AP) and mean AP for object detection.

    Follows the COCO evaluation protocol (Lin et al. 2014) with IoU thresholds
    0.5:0.05:0.95 and 101-point precision-recall interpolation.

    Args:
        nc: Number of object classes.
        iou_thresholds: Optional 1-D array of IoU thresholds. Defaults to
            np.arange(0.5, 1.0, 0.05) (10 thresholds).
    """

    def __init__(self, nc: int, iou_thresholds: Optional[np.ndarray] = None):
        self.nc = nc
        self.iou_thresholds = iou_thresholds if iou_thresholds is not None else np.arange(0.5, 1.0, 0.05)
        self.reset()

    def reset(self):
        """Reset all accumulated statistics."""
        # Per-class lists of (score, is_tp_per_threshold) tuples
        self._detections: List[List] = [[] for _ in range(self.nc)]
        # Per-class ground truth counts
        self._n_gt: np.ndarray = np.zeros(self.nc, dtype=np.int64)

    @torch.no_grad()
    def update(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]):
        """Accumulate a batch of predictions and ground truths.

        Args:
            predictions: List of (N_det, 6) tensors per image,
                each row is [x1, y1, x2, y2, confidence, class_id].
            targets: List of (N_gt, 5) tensors per image,
                each row is [x1, y1, x2, y2, class_id].
        """
        n_thresholds = len(self.iou_thresholds)

        for pred, gt in zip(predictions, targets):
            if gt.numel() == 0 and pred.numel() == 0:
                continue

            gt = gt.detach().cpu()
            pred = pred.detach().cpu()

            # Count ground truths per class
            if gt.numel() > 0:
                gt_classes = gt[:, 4].long()
                for c in gt_classes.tolist():
                    if 0 <= c < self.nc:
                        self._n_gt[c] += 1

            if pred.numel() == 0:
                continue

            pred_boxes = pred[:, :4]
            pred_scores = pred[:, 4]
            pred_classes = pred[:, 5].long()

            if gt.numel() == 0:
                # All predictions are false positives
                for j in range(pred.shape[0]):
                    c = pred_classes[j].item()
                    if 0 <= c < self.nc:
                        self._detections[c].append(
                            (pred_scores[j].item(), np.zeros(n_thresholds, dtype=bool))
                        )
                continue

            gt_boxes = gt[:, :4]
            gt_classes = gt[:, 4].long()

            # Compute IoU matrix between predictions and ground truths
            iou_matrix = box_iou(pred_boxes, gt_boxes).numpy()  # (N_det, N_gt)

            # For each IoU threshold, track which GTs have been matched
            gt_matched = np.zeros((n_thresholds, gt.shape[0]), dtype=bool)

            # Sort predictions by descending confidence
            order = pred_scores.argsort(descending=True)

            for idx in order.tolist():
                c = pred_classes[idx].item()
                if c < 0 or c >= self.nc:
                    continue

                score = pred_scores[idx].item()
                tp_flags = np.zeros(n_thresholds, dtype=bool)

                # Find ground truths of the same class
                gt_mask = (gt_classes == c).numpy()
                if gt_mask.any():
                    ious_for_det = iou_matrix[idx] * gt_mask  # zero out other classes
                    best_gt = int(ious_for_det.argmax())
                    best_iou = ious_for_det[best_gt]

                    for t_idx, iou_thr in enumerate(self.iou_thresholds):
                        if best_iou >= iou_thr and not gt_matched[t_idx, best_gt]:
                            tp_flags[t_idx] = True
                            gt_matched[t_idx, best_gt] = True

                self._detections[c].append((score, tp_flags))

    def _compute_ap_single(self, scores_and_flags: List, n_gt: int) -> np.ndarray:
        """Compute AP at each IoU threshold for one class via 101-point interpolation.

        Args:
            scores_and_flags: List of (score, tp_flags) tuples.
            n_gt: Number of ground truth instances for this class.

        Returns:
            (n_thresholds,) array of AP values.
        """
        n_thresholds = len(self.iou_thresholds)

        if n_gt == 0 or len(scores_and_flags) == 0:
            return np.zeros(n_thresholds)

        # Sort by descending confidence
        scores_and_flags.sort(key=lambda x: -x[0])

        tp_matrix = np.array([f for _, f in scores_and_flags])  # (N_det, n_thresholds)

        tp_cumsum = np.cumsum(tp_matrix, axis=0)  # (N_det, n_thresholds)
        n_dets = np.arange(1, len(scores_and_flags) + 1)[:, None]  # (N_det, 1)

        precision = tp_cumsum / n_dets  # (N_det, n_thresholds)
        recall = tp_cumsum / max(n_gt, 1)  # (N_det, n_thresholds)

        # 101-point interpolation (COCO-style)
        recall_levels = np.linspace(0.0, 1.0, 101)
        ap = np.zeros(n_thresholds)

        for t in range(n_thresholds):
            prec_t = precision[:, t]
            rec_t = recall[:, t]

            # Make precision monotonically decreasing (right-to-left max)
            for k in range(len(prec_t) - 2, -1, -1):
                prec_t[k] = max(prec_t[k], prec_t[k + 1])

            # Interpolate at each recall level
            interp_prec = np.zeros(101)
            for r_idx, r_level in enumerate(recall_levels):
                # Precision at the smallest recall >= r_level
                candidates = prec_t[rec_t >= r_level]
                interp_prec[r_idx] = candidates.max() if candidates.size > 0 else 0.0

            ap[t] = interp_prec.mean()

        return ap

    def compute(self) -> Dict:
        """Compute AP per class and mAP across all classes.

        Returns:
            Dictionary with keys:
                'mAP50': float - mean AP at IoU=0.50
                'mAP50_95': float - mean AP averaged over IoU 0.50:0.95
                'ap_per_class': (nc, n_thresholds) ndarray - per-class per-threshold AP
        """
        n_thresholds = len(self.iou_thresholds)
        ap_per_class = np.zeros((self.nc, n_thresholds))

        for c in range(self.nc):
            ap_per_class[c] = self._compute_ap_single(self._detections[c], int(self._n_gt[c]))

        # Classes with at least one ground truth instance
        valid_mask = self._n_gt > 0
        if valid_mask.any():
            mean_ap_per_threshold = ap_per_class[valid_mask].mean(axis=0)
        else:
            mean_ap_per_threshold = np.zeros(n_thresholds)

        # mAP@0.50 is the first threshold entry
        map50 = float(mean_ap_per_threshold[0])
        # mAP@0.50:0.95 is the mean over all thresholds
        map50_95 = float(mean_ap_per_threshold.mean())

        return {
            "mAP50": map50,
            "mAP50_95": map50_95,
            "ap_per_class": ap_per_class,
        }
