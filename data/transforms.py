# SwiftDet Image Transforms for Object Detection
# MIT License - Original implementation
#
# Standard computer vision image preprocessing techniques.
# LetterBox preserves aspect ratio (common practice in detection pipelines).
# HSV jittering, flipping, and normalization are standard augmentations.

import numpy as np
import cv2


class LetterBox:
    """Resize image preserving aspect ratio with padding to target size.

    This is a standard computer vision technique for feeding images of
    varying sizes into a fixed-size network input. The image is scaled
    to fit within the target dimensions, then padded with a fill value.

    Args:
        target_size: (h, w) target dimensions. Default (640, 640).
        fill_value: Pixel value used for padding regions. Default 114.
        auto: If True, pad to minimum stride-aligned rectangle instead
              of exact target_size. Default False.
        stride: Stride alignment when auto=True. Default 32.
    """

    def __init__(self, target_size=(640, 640), fill_value=114, auto=False, stride=32):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_h, self.target_w = target_size
        self.fill_value = fill_value
        self.auto = auto
        self.stride = stride

    def __call__(self, image, labels=None):
        """Apply letterbox resize and padding.

        Args:
            image: np.ndarray (H, W, 3) BGR image.
            labels: np.ndarray (N, 5) [cls, x1, y1, x2, y2] in pixel coords,
                    or None.

        Returns:
            image: np.ndarray (target_h, target_w, 3) letterboxed image.
            labels: np.ndarray (N, 5) with adjusted box coordinates, or None.
        """
        src_h, src_w = image.shape[:2]

        # Compute scale factor to fit image within target size
        scale = min(self.target_h / src_h, self.target_w / src_w)

        # New unpadded dimensions
        new_w = int(round(src_w * scale))
        new_h = int(round(src_h * scale))

        # Compute padding
        pad_w = self.target_w - new_w
        pad_h = self.target_h - new_h

        if self.auto:
            # Pad to minimum stride-aligned rectangle
            pad_w = pad_w % self.stride
            pad_h = pad_h % self.stride

        # Split padding evenly on both sides
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        # Resize the image
        if new_w != src_w or new_h != src_h:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Add border padding
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT,
            value=(self.fill_value, self.fill_value, self.fill_value),
        )

        # Adjust label coordinates
        if labels is not None and len(labels) > 0:
            labels = labels.copy()
            # Scale box coordinates
            labels[:, 1] = labels[:, 1] * scale + left  # x1
            labels[:, 2] = labels[:, 2] * scale + top   # y1
            labels[:, 3] = labels[:, 3] * scale + left  # x2
            labels[:, 4] = labels[:, 4] * scale + top   # y2

        return image, labels

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"target_size=({self.target_h}, {self.target_w}), "
            f"fill_value={self.fill_value}, auto={self.auto})"
        )


class RandomHSV:
    """Random HSV color space augmentation.

    Applies random multiplicative jitter independently to each HSV channel.
    This is a standard color augmentation technique in computer vision.

    Args:
        h_gain: Maximum fractional hue shift. Default 0.015.
        s_gain: Maximum fractional saturation scale. Default 0.7.
        v_gain: Maximum fractional value (brightness) scale. Default 0.4.
    """

    def __init__(self, h_gain=0.015, s_gain=0.7, v_gain=0.4):
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain

    def __call__(self, image, labels=None):
        """Apply random HSV augmentation.

        Args:
            image: np.ndarray (H, W, 3) BGR uint8 image.
            labels: np.ndarray (N, 5) or None. Passed through unchanged.

        Returns:
            image: np.ndarray (H, W, 3) augmented BGR uint8 image.
            labels: unchanged.
        """
        if self.h_gain == 0 and self.s_gain == 0 and self.v_gain == 0:
            return image, labels

        # Sample random gains uniformly from [-gain, +gain]
        r = np.random.uniform(-1, 1, 3) * np.array([self.h_gain, self.s_gain, self.v_gain]) + 1

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Build lookup tables for efficient per-pixel transformation
        # Hue channel wraps at 180 in OpenCV's uint8 HSV
        lut_h = np.arange(0, 256, dtype=np.float32)
        lut_s = lut_h.copy()
        lut_v = lut_h.copy()

        lut_h = ((lut_h * r[0]) % 180).astype(np.uint8)
        lut_s = np.clip(lut_s * r[1], 0, 255).astype(np.uint8)
        lut_v = np.clip(lut_v * r[2], 0, 255).astype(np.uint8)

        # Apply LUTs
        h = cv2.LUT(h, lut_h)
        s = cv2.LUT(s, lut_s)
        v = cv2.LUT(v, lut_v)

        # Merge and convert back
        hsv_aug = cv2.merge([h, s, v])
        image = cv2.cvtColor(hsv_aug, cv2.COLOR_HSV2BGR)

        return image, labels

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"h_gain={self.h_gain}, s_gain={self.s_gain}, v_gain={self.v_gain})"
        )


class RandomFlip:
    """Random horizontal and/or vertical flip.

    Standard augmentation for object detection. Adjusts bounding box
    coordinates accordingly.

    Args:
        p_horizontal: Probability of horizontal flip. Default 0.5.
        p_vertical: Probability of vertical flip. Default 0.0.
    """

    def __init__(self, p_horizontal=0.5, p_vertical=0.0):
        self.p_horizontal = p_horizontal
        self.p_vertical = p_vertical

    def __call__(self, image, labels=None):
        """Apply random flip.

        Args:
            image: np.ndarray (H, W, 3) image.
            labels: np.ndarray (N, 5) [cls, x1, y1, x2, y2] pixel coords,
                    or None.

        Returns:
            image: np.ndarray (H, W, 3) possibly flipped.
            labels: np.ndarray (N, 5) with adjusted coordinates, or None.
        """
        h, w = image.shape[:2]

        # Horizontal flip
        if np.random.random() < self.p_horizontal:
            image = np.ascontiguousarray(image[:, ::-1])
            if labels is not None and len(labels) > 0:
                labels = labels.copy()
                old_x1 = labels[:, 1].copy()
                labels[:, 1] = w - labels[:, 3]  # new x1 = w - old x2
                labels[:, 3] = w - old_x1         # new x2 = w - old x1

        # Vertical flip
        if np.random.random() < self.p_vertical:
            image = np.ascontiguousarray(image[::-1, :])
            if labels is not None and len(labels) > 0:
                labels = labels.copy()
                old_y1 = labels[:, 2].copy()
                labels[:, 2] = h - labels[:, 4]  # new y1 = h - old y2
                labels[:, 4] = h - old_y1         # new y2 = h - old y1

        return image, labels

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"p_horizontal={self.p_horizontal}, p_vertical={self.p_vertical})"
        )


class Normalize:
    """Normalize image to [0, 1] float32 and transpose HWC to CHW.

    Standard preprocessing step for feeding images into neural networks.
    Optionally applies mean/std normalization (e.g. ImageNet statistics).

    Args:
        mean: Per-channel mean for normalization. Default None (no mean sub).
        std: Per-channel std for normalization. Default None (no std div).
    """

    def __init__(self, mean=None, std=None):
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1) if mean is not None else None
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1) if std is not None else None

    def __call__(self, image, labels=None):
        """Normalize and transpose image.

        Args:
            image: np.ndarray (H, W, 3) uint8 or float image.
            labels: np.ndarray (N, 5) or None. Passed through unchanged.

        Returns:
            image: np.ndarray (3, H, W) float32 in [0, 1] (or mean/std normalized).
            labels: unchanged.
        """
        # Convert to float32 and scale to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Transpose HWC -> CHW
        image = image.transpose(2, 0, 1)

        # Ensure contiguous memory layout
        image = np.ascontiguousarray(image)

        # Apply mean/std normalization if specified
        if self.mean is not None:
            image = image - self.mean
        if self.std is not None:
            image = image / self.std

        return image, labels

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"mean={self.mean.flatten().tolist() if self.mean is not None else None}, "
            f"std={self.std.flatten().tolist() if self.std is not None else None})"
        )


class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms: List of transform callables. Each must accept
                    (image, labels) and return (image, labels).
    """

    def __init__(self, transforms):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, image, labels=None):
        """Apply all transforms in sequence."""
        for t in self.transforms:
            image, labels = t(image, labels)
        return image, labels

    def __repr__(self):
        lines = [f"{self.__class__.__name__}(["]
        for t in self.transforms:
            lines.append(f"    {t},")
        lines.append("])")
        return "\n".join(lines)
