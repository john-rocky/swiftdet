# SwiftDet - Batch visualization utilities
# MIT License - Original implementation

import math

import cv2
import numpy as np

# Reuse the same 16-color palette as core/results.py
_PALETTE = [
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (0, 190, 246),
    (0, 165, 255),
    (0, 138, 255),
    (0, 94, 255),
    (255, 144, 30),
    (255, 78, 0),
    (255, 0, 0),
    (200, 0, 128),
]


def plot_batch(images, boxes, classes, names, confs=None, fname="batch.jpg",
               max_images=16, max_size=1920, conf_thres=0.25):
    """Draw bounding boxes on a batch of images and save as a mosaic grid.

    Args:
        images: (B, 3, H, W) float tensor, BGR 0-1.
        boxes: List of (N, 4) numpy xyxy arrays per image.
        classes: List of (N,) int arrays per image.
        names: Dict mapping class index to class name.
        confs: List of (N,) float arrays per image, or None for GT.
        fname: Output file path.
        max_images: Maximum number of images in the grid.
        max_size: Maximum mosaic dimension in pixels.
        conf_thres: Filter predictions below this threshold when confs given.
    """
    import torch

    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().float()

    b = min(images.shape[0], max_images)
    ncols = math.ceil(math.sqrt(b))
    nrows = math.ceil(b / ncols)

    _, _, h, w = images.shape
    # Scale cells so the mosaic fits within max_size
    scale = min(1.0, max_size / (ncols * w), max_size / (nrows * h))
    cell_w = int(w * scale)
    cell_h = int(h * scale)

    mosaic = np.full((nrows * cell_h, ncols * cell_w, 3), 114, dtype=np.uint8)

    # Auto-scale line width and font based on cell size
    lw = max(1, int(min(cell_h, cell_w) / 200))
    fs = max(0.3, lw / 3.0)
    ft = max(1, lw)

    for i in range(b):
        # Convert tensor to BGR uint8
        img = images[i].numpy()  # (3, H, W)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = img.transpose(1, 2, 0)  # (H, W, 3) BGR

        if scale != 1.0:
            img = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)

        bx = boxes[i]
        cl = classes[i]
        cf = confs[i] if confs is not None else None

        # Filter by confidence threshold for predictions
        if cf is not None and len(cf) > 0:
            mask = cf >= conf_thres
            bx = bx[mask]
            cl = cl[mask]
            cf = cf[mask]

        # Scale boxes to cell size
        if len(bx) > 0:
            bx_scaled = bx.copy()
            bx_scaled[:, [0, 2]] *= scale
            bx_scaled[:, [1, 3]] *= scale
        else:
            bx_scaled = bx

        # Draw boxes
        for j in range(len(bx_scaled)):
            x1, y1, x2, y2 = bx_scaled[j].astype(int)
            cls_id = int(cl[j])
            color = _PALETTE[cls_id % len(_PALETTE)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)

            label = names.get(cls_id, str(cls_id))
            if cf is not None:
                label = f"{label} {cf[j]:.2f}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
            bg_y1 = max(y1 - th - 6, 0)
            cv2.rectangle(img, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), ft, cv2.LINE_AA)

        row, col = divmod(i, ncols)
        mosaic[row * cell_h:(row + 1) * cell_h,
               col * cell_w:(col + 1) * cell_w] = img

    cv2.imwrite(str(fname), mosaic)
