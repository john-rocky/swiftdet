# SwiftDet - Box utility functions
# MIT License - Original implementation
# Standard coordinate conversion and IoU computation for axis-aligned bounding boxes.

import torch


def xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from center-format (cx, cy, w, h) to corner-format (x1, y1, x2, y2).

    Args:
        boxes: (..., 4) tensor in (cx, cy, w, h) format.

    Returns:
        (..., 4) tensor in (x1, y1, x2, y2) format.
    """
    cx, cy, w, h = boxes.unbind(-1)
    half_w = w * 0.5
    half_h = h * 0.5
    return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=-1)


def xyxy2xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from corner-format (x1, y1, x2, y2) to center-format (cx, cy, w, h).

    Args:
        boxes: (..., 4) tensor in (x1, y1, x2, y2) format.

    Returns:
        (..., 4) tensor in (cx, cy, w, h) format.
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Compute areas of bounding boxes in (x1, y1, x2, y2) format.

    Args:
        boxes: (..., 4) tensor in (x1, y1, x2, y2) format.

    Returns:
        (...,) tensor of areas.
    """
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute the Intersection-over-Union (IoU) matrix between two sets of boxes.

    Both sets must be in (x1, y1, x2, y2) corner format.

    Args:
        boxes_a: (M, 4) tensor.
        boxes_b: (N, 4) tensor.

    Returns:
        (M, N) IoU matrix.
    """
    area_a = box_area(boxes_a)  # (M,)
    area_b = box_area(boxes_b)  # (N,)

    # Intersection coordinates
    top_left = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])  # (M, N, 2)
    bottom_right = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # (M, N, 2)

    wh = (bottom_right - top_left).clamp(min=0)  # (M, N, 2)
    intersection = wh[..., 0] * wh[..., 1]  # (M, N)

    union = area_a[:, None] + area_b[None, :] - intersection  # (M, N)

    return intersection / union.clamp(min=1e-7)
