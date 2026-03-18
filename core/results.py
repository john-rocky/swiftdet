"""Detection result containers for SwiftDet.

MIT License - Original implementation.
"""

import copy
from pathlib import Path

import cv2
import numpy as np


class Boxes:
    """Container for bounding box detections.

    Stores detection boxes in [x1, y1, x2, y2] format alongside confidence
    scores and class indices.  All data is kept as numpy arrays internally.

    Attributes:
        xyxy: (N, 4) array of boxes in [x1, y1, x2, y2] format.
        conf: (N,) array of confidence scores.
        cls:  (N,) array of integer class indices.
        data: (N, 6) raw array [x1, y1, x2, y2, conf, cls].
    """

    def __init__(self, data):
        """Create Boxes from an (N, 6) array [x1, y1, x2, y2, conf, cls].

        Args:
            data: numpy array of shape (N, 6) or (0, 6).
        """
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1 and data.shape[0] == 0:
            data = data.reshape(0, 6)
        if data.ndim != 2 or data.shape[1] != 6:
            raise ValueError(f"Boxes data must be (N, 6), got {data.shape}")
        self.data = data.astype(np.float32)

    # ---- Convenience properties ---- #

    @property
    def xyxy(self):
        """(N, 4) boxes in [x1, y1, x2, y2] format."""
        return self.data[:, :4]

    @property
    def conf(self):
        """(N,) confidence scores."""
        return self.data[:, 4]

    @property
    def cls(self):
        """(N,) class indices."""
        return self.data[:, 5]

    @property
    def xywh(self):
        """(N, 4) boxes in [cx, cy, w, h] format."""
        x1, y1, x2, y2 = self.xyxy.T
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        return np.stack([cx, cy, w, h], axis=1)

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return f"Boxes(n={len(self)}, fields=[xyxy, conf, cls])"


class Results:
    """Container for single-image detection results.

    Wraps the original image together with its detections so that
    visualization and serialization are convenient one-liners.

    Args:
        orig_img: Original image as a numpy array (H, W, 3) BGR.
        boxes:    Boxes instance holding detections for this image.
        names:    Dict mapping class index (int) to class name (str).
        path:     Optional filesystem path for the source image.
    """

    def __init__(self, orig_img, boxes, names, path=None):
        self.orig_img = orig_img
        self.boxes = boxes
        self.names = names or {}
        self.path = path

    # ---- Visualization ---- #

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

    def _color_for(self, cls_id):
        """Return a BGR color for the given class id."""
        return self._PALETTE[int(cls_id) % len(self._PALETTE)]

    def plot(self, line_width=None, font_scale=None, font_thickness=None):
        """Draw bounding boxes on the image and return the annotated copy.

        Args:
            line_width:     Rectangle border thickness (auto if None).
            font_scale:     Text scale factor (auto if None).
            font_thickness: Text thickness (auto if None).

        Returns:
            Annotated image as a numpy array (H, W, 3) BGR.
        """
        img = self.orig_img.copy()
        h, w = img.shape[:2]

        lw = line_width or max(1, int(min(h, w) / 300))
        fs = font_scale or max(0.4, lw / 3.0)
        ft = font_thickness or max(1, lw)

        for i in range(len(self.boxes)):
            x1, y1, x2, y2 = self.boxes.xyxy[i].astype(int)
            score = float(self.boxes.conf[i])
            cls_id = int(self.boxes.cls[i])
            color = self._color_for(cls_id)
            label = self.names.get(cls_id, str(cls_id))
            text = f"{label} {score:.2f}"

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)

            # Draw label background
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
            bg_y1 = max(y1 - th - 8, 0)
            cv2.rectangle(img, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                img, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), ft, cv2.LINE_AA,
            )
        return img

    def show(self):
        """Display annotated image using matplotlib."""
        import matplotlib.pyplot as plt

        annotated = self.plot()
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def save(self, path):
        """Save annotated image to *path*.

        Args:
            path: Destination file path (e.g. ``"output/det.jpg"``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        annotated = self.plot()
        cv2.imwrite(str(path), annotated)

    # ---- Convenience ---- #

    def __len__(self):
        return len(self.boxes)

    def __repr__(self):
        src = f", path={self.path}" if self.path else ""
        return f"Results(detections={len(self.boxes)}{src})"
