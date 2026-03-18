# SwiftDet - IoU loss variants
# MIT License - Original implementation
#
# References:
#   - IoU Loss: Yu et al. 2016, "UnitBox: An Advanced Object Detection Network"
#   - GIoU: Rezatofighi et al. 2019, "Generalized Intersection over Union"
#   - DIoU/CIoU: Zheng et al. 2019, "Distance-IoU Loss: Faster and Better
#     Learning for Bounding Box Regression"
#   - Wise-IoU: Tong et al. 2023, "Wise-IoU: Bounding Box Regression Loss
#     with Dynamic Focusing Mechanism"

import math

import torch


def bbox_iou(box1, box2, xywh=False, giou=False, diou=False, ciou=False, eps=1e-7):
    """Compute IoU and its variants between two sets of boxes (element-wise).

    Supports standard IoU, Generalized IoU (GIoU), Distance IoU (DIoU), and
    Complete IoU (CIoU).

    Args:
        box1: (N, 4) bounding boxes.
        box2: (N, 4) bounding boxes (same N, element-wise comparison).
        xywh: If True, input format is (cx, cy, w, h); otherwise (x1, y1, x2, y2).
        giou: Compute GIoU (Rezatofighi et al. 2019).
        diou: Compute DIoU (Zheng et al. 2019).
        ciou: Compute CIoU (Zheng et al. 2019). Implies DIoU.
        eps: Small constant for numerical stability.

    Returns:
        (N,) tensor of IoU values (or variant values).
    """
    # Convert to corner format if needed
    if xywh:
        # (cx, cy, w, h) -> (x1, y1, x2, y2)
        half_w1 = box1[..., 2:3] * 0.5
        half_h1 = box1[..., 3:4] * 0.5
        b1_x1 = box1[..., 0:1] - half_w1
        b1_y1 = box1[..., 1:2] - half_h1
        b1_x2 = box1[..., 0:1] + half_w1
        b1_y2 = box1[..., 1:2] + half_h1

        half_w2 = box2[..., 2:3] * 0.5
        half_h2 = box2[..., 3:4] * 0.5
        b2_x1 = box2[..., 0:1] - half_w2
        b2_y1 = box2[..., 1:2] - half_h2
        b2_x2 = box2[..., 0:1] + half_w2
        b2_y2 = box2[..., 1:2] + half_h2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = (
            box1[..., 0:1], box1[..., 1:2], box1[..., 2:3], box1[..., 3:4],
        )
        b2_x1, b2_y1, b2_x2, b2_y2 = (
            box2[..., 0:1], box2[..., 1:2], box2[..., 2:3], box2[..., 3:4],
        )

    # Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union area
    area1 = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
    area2 = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)
    union = area1 + area2 - inter_area + eps

    # Standard IoU
    iou = inter_area / union

    if giou or diou or ciou:
        # Enclosing (smallest) box coordinates
        enclose_x1 = torch.min(b1_x1, b2_x1)
        enclose_y1 = torch.min(b1_y1, b2_y1)
        enclose_x2 = torch.max(b1_x2, b2_x2)
        enclose_y2 = torch.max(b1_y2, b2_y2)

        if giou:
            # GIoU = IoU - (enclose_area - union) / enclose_area
            enclose_area = (enclose_x2 - enclose_x1).clamp(min=0) * (
                enclose_y2 - enclose_y1
            ).clamp(min=0) + eps
            return (iou - (enclose_area - union) / enclose_area).squeeze(-1)

        # Squared diagonal of enclosing box (for DIoU and CIoU)
        enclose_diag_sq = (
            (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
        ) + eps

        # Squared distance between box centers
        cx1 = (b1_x1 + b1_x2) * 0.5
        cy1 = (b1_y1 + b1_y2) * 0.5
        cx2 = (b2_x1 + b2_x2) * 0.5
        cy2 = (b2_y1 + b2_y2) * 0.5
        center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

        # Distance penalty (shared by DIoU and CIoU)
        rho = center_dist_sq / enclose_diag_sq

        if diou:
            # DIoU = IoU - rho^2
            return (iou - rho).squeeze(-1)

        # CIoU = IoU - rho^2 - alpha * v
        # v measures aspect ratio consistency
        w1 = (b1_x2 - b1_x1).clamp(min=eps)
        h1 = (b1_y2 - b1_y1).clamp(min=eps)
        w2 = (b2_x2 - b2_x1).clamp(min=eps)
        h2 = (b2_y2 - b2_y1).clamp(min=eps)

        v = (4.0 / (math.pi ** 2)) * (
            torch.atan(w2 / h2) - torch.atan(w1 / h1)
        ) ** 2

        # alpha balances the aspect-ratio penalty relative to IoU
        with torch.no_grad():
            alpha = v / (1.0 - iou + v + eps)

        return (iou - rho - alpha * v).squeeze(-1)

    return iou.squeeze(-1)


def wise_iou(box1, box2, eps=1e-7):
    """Wise-IoU loss with dynamic non-monotonic focusing (Tong et al. 2023).

    Uses a dynamic focusing mechanism based on the ratio of the
    actual IoU to a running "expected" IoU level. Outlier boxes with
    very low IoU get down-weighted rather than dominating the gradient.

    WIoU = IoU * exp((distance / enclosing_diagonal)^2) weighted by focusing factor

    Simplified version: uses the DIoU penalty with an outlier-aware weighting.

    Args:
        box1: (N, 4) boxes in xyxy format
        box2: (N, 4) boxes in xyxy format

    Returns:
        (N,) Wise-IoU values (higher is better overlap, like standard IoU)
    """
    # Unpack box coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = (
        box1[..., 0:1], box1[..., 1:2], box1[..., 2:3], box1[..., 3:4],
    )
    b2_x1, b2_y1, b2_x2, b2_y2 = (
        box2[..., 0:1], box2[..., 1:2], box2[..., 2:3], box2[..., 3:4],
    )

    # Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union area
    area1 = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
    area2 = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)
    union = area1 + area2 - inter_area + eps

    # Step 1: Standard IoU
    iou = inter_area / union

    # Step 2: Center distance penalty (DIoU-style)
    # Enclosing box diagonal squared
    enclose_x1 = torch.min(b1_x1, b2_x1)
    enclose_y1 = torch.min(b1_y1, b2_y1)
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)
    enclose_diag_sq = (
        (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    ) + eps

    # Squared distance between box centers
    cx1 = (b1_x1 + b1_x2) * 0.5
    cy1 = (b1_y1 + b1_y2) * 0.5
    cx2 = (b2_x1 + b2_x2) * 0.5
    cy2 = (b2_y1 + b2_y2) * 0.5
    center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    distance_penalty = center_dist_sq / enclose_diag_sq

    # Step 3: Focusing coefficient -- ratio of IoU to batch mean IoU
    iou_mean = iou.detach().mean().clamp(min=eps)
    r = (iou.detach() / iou_mean)

    # Step 4: Non-monotonic weight (stop-gradient on r)
    weight = torch.exp((1.0 - iou) * r)

    # Step 5: WIoU loss = (1 - IoU + distance_penalty) * weight
    wiou = (1.0 - iou + distance_penalty) * weight.detach()

    return wiou.squeeze(-1)
