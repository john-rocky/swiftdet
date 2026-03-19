# SwiftDet - Non-Maximum Suppression
# MIT License - Original implementation
# NMS filters overlapping detections, keeping only the highest-confidence box
# among groups of heavily overlapping predictions.

from typing import List

import torch

from .boxes import box_iou


def _nms_fallback(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Pure-PyTorch NMS fallback when torchvision is unavailable.

    Greedy NMS: iteratively select the highest-scoring box and remove all
    boxes that overlap with it above the IoU threshold.

    Args:
        boxes: (N, 4) tensor in (x1, y1, x2, y2) format.
        scores: (N,) confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        1-D tensor of indices to keep, sorted by descending score.
    """
    order = scores.argsort(descending=True)
    keep: List[int] = []

    while order.numel() > 0:
        idx = order[0].item()
        keep.append(idx)

        if order.numel() == 1:
            break

        remaining = order[1:]
        ious = box_iou(boxes[idx : idx + 1], boxes[remaining])[0]  # (R,)
        mask = ious <= iou_threshold
        order = remaining[mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _get_nms_fn():
    """Return the best available NMS function."""
    try:
        from torchvision.ops import nms as tv_nms

        return tv_nms
    except ImportError:
        return _nms_fallback


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> List[torch.Tensor]:
    """Perform NMS on detection predictions.

    Args:
        prediction: (B, N, nc+4) tensor where the first 4 values per anchor
            are box coordinates in (x1, y1, x2, y2) format and the remaining
            values are per-class confidence scores.
        conf_thres: Minimum confidence to retain a detection.
        iou_thres: IoU threshold for NMS suppression.
        max_det: Maximum number of detections to return per image.

    Returns:
        List of (N_det, 6) tensors per image: [x1, y1, x2, y2, conf, cls].
    """
    nms_fn = _get_nms_fn()

    batch_size = prediction.shape[0]
    nc = prediction.shape[2] - 4  # number of classes
    output: List[torch.Tensor] = []

    for i in range(batch_size):
        pred = prediction[i]  # (N, nc+4)

        boxes = pred[:, :4]  # (N, 4)
        class_scores = pred[:, 4:]  # (N, nc)

        # Per-anchor maximum class confidence and its index
        conf, cls_idx = class_scores.max(dim=1)  # (N,), (N,)

        # Filter by confidence threshold
        mask = conf > conf_thres
        boxes = boxes[mask]
        conf = conf[mask]
        cls_idx = cls_idx[mask]

        if boxes.numel() == 0:
            output.append(torch.zeros((0, 6), device=prediction.device, dtype=prediction.dtype))
            continue

        # Class-aware NMS: offset boxes by class id so that boxes from different
        # classes never overlap. Use dynamic offset based on actual coordinate range.
        class_offset = cls_idx.float() * (boxes.abs().max().item() + 1)
        shifted_boxes = boxes + class_offset[:, None]

        keep = nms_fn(shifted_boxes, conf, iou_thres)

        # Limit number of detections
        if keep.numel() > max_det:
            keep = keep[:max_det]

        det = torch.cat(
            [boxes[keep], conf[keep].unsqueeze(1), cls_idx[keep].float().unsqueeze(1)],
            dim=1,
        )  # (N_det, 6)

        output.append(det)

    return output
