# SwiftDet Training-Time Augmentations for Object Detection
# MIT License - Original implementation
#
# References (concepts only, all code written from scratch):
# - Mosaic: Bochkovskiy et al. "YOLOv4: Optimal Speed and Accuracy of Object
#   Detection" (2020). The mosaic concept combines 4 images into one.
# - CutMix: Yun et al. "CutMix: Regularization Strategy to Train Strong
#   Classifiers with Localizable Features" (2019). Related mixing concept.
# - MixUp: Zhang et al. "mixup: Beyond Empirical Risk Minimization" (2018).
#   Convex combination of image pairs and their labels.
# - Copy-Paste: Ghiasi et al. "Simple Copy-Paste is a Strong Data Augmentation
#   Method for Instance Segmentation" (2021). Copy instances between images.

import math
import random

import cv2
import numpy as np


class RandomAffine:
    """Random affine transformation: rotation, scale, translation, shear.

    Standard computer vision augmentation. Applies a combined affine
    transformation matrix to the image and adjusts bounding boxes.

    Args:
        degrees: Maximum rotation angle in degrees. Default 0.0.
        translate: Maximum translation as fraction of image size. Default 0.1.
        scale_range: (min_scale, max_scale) tuple. Default (0.5, 1.5).
        shear: Maximum shear angle in degrees. Default 0.0.
        border: Border size (negative means image is smaller). Default (0, 0).
        fill_value: Fill value for border pixels. Default 114.
    """

    def __init__(
        self,
        degrees=0.0,
        translate=0.1,
        scale_range=(0.5, 1.5),
        shear=0.0,
        border=(0, 0),
        fill_value=114,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale_min, self.scale_max = scale_range
        self.shear = shear
        self.border = border
        self.fill_value = fill_value

    def __call__(self, image, labels=None):
        """Apply random affine transformation.

        Args:
            image: np.ndarray (H, W, 3).
            labels: np.ndarray (N, 5) [cls, x1, y1, x2, y2] pixel coords,
                    or None.

        Returns:
            image: np.ndarray transformed image.
            labels: np.ndarray with transformed box coordinates.
        """
        h, w = image.shape[:2]

        # Output size accounts for border
        out_h = h + self.border[0] * 2
        out_w = w + self.border[1] * 2

        # -- Build affine matrix --
        # Center matrix: translate image center to origin
        center_mat = np.eye(3, dtype=np.float64)
        center_mat[0, 2] = -w / 2  # x offset
        center_mat[1, 2] = -h / 2  # y offset

        # Rotation and scale
        angle = random.uniform(-self.degrees, self.degrees)
        scale = random.uniform(self.scale_min, self.scale_max)
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        rot_mat = np.eye(3, dtype=np.float64)
        rot_mat[0, 0] = scale * cos_a
        rot_mat[0, 1] = scale * sin_a
        rot_mat[1, 0] = -scale * sin_a
        rot_mat[1, 1] = scale * cos_a

        # Shear
        shear_x = math.tan(math.radians(random.uniform(-self.shear, self.shear)))
        shear_y = math.tan(math.radians(random.uniform(-self.shear, self.shear)))

        shear_mat = np.eye(3, dtype=np.float64)
        shear_mat[0, 1] = shear_x
        shear_mat[1, 0] = shear_y

        # Translation
        tx = random.uniform(-self.translate, self.translate) * out_w
        ty = random.uniform(-self.translate, self.translate) * out_h

        trans_mat = np.eye(3, dtype=np.float64)
        trans_mat[0, 2] = tx + out_w / 2  # move center to output center
        trans_mat[1, 2] = ty + out_h / 2

        # Combined matrix: translate_center -> rotate/scale -> shear -> translate_out
        M = trans_mat @ shear_mat @ rot_mat @ center_mat

        # Apply affine warp (use 2x3 submatrix)
        image = cv2.warpAffine(
            image,
            M[:2],
            dsize=(out_w, out_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(self.fill_value, self.fill_value, self.fill_value),
        )

        # Transform bounding boxes
        if labels is not None and len(labels) > 0:
            labels = _warp_boxes(labels, M, out_w, out_h)

        return image, labels

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"degrees={self.degrees}, translate={self.translate}, "
            f"scale=({self.scale_min}, {self.scale_max}), shear={self.shear})"
        )


def _warp_boxes(labels, M, out_w, out_h):
    """Warp bounding boxes using an affine matrix and clip to image bounds.

    Each box is represented by its 4 corners, which are transformed
    independently. The new axis-aligned bounding box is computed from
    the transformed corners.

    Args:
        labels: np.ndarray (N, 5) [cls, x1, y1, x2, y2].
        M: 3x3 affine matrix.
        out_w: Output image width.
        out_h: Output image height.

    Returns:
        np.ndarray (M, 5) surviving labels after filtering.
    """
    n = len(labels)
    if n == 0:
        return labels

    cls = labels[:, 0]
    boxes = labels[:, 1:5]

    # Build 4 corner points for each box: (N, 4, 3)
    # corners: top-left, top-right, bottom-right, bottom-left
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    corners = np.zeros((n, 4, 3), dtype=np.float64)
    corners[:, 0] = np.stack([x1, y1, np.ones(n)], axis=1)  # top-left
    corners[:, 1] = np.stack([x2, y1, np.ones(n)], axis=1)  # top-right
    corners[:, 2] = np.stack([x2, y2, np.ones(n)], axis=1)  # bottom-right
    corners[:, 3] = np.stack([x1, y2, np.ones(n)], axis=1)  # bottom-left

    # Apply transformation: (N, 4, 3) @ (3, 3)^T -> (N, 4, 3)
    corners = corners @ M.T

    # Extract x, y from transformed corners
    cx = corners[:, :, 0]  # (N, 4)
    cy = corners[:, :, 1]  # (N, 4)

    # New axis-aligned bounding boxes
    new_x1 = cx.min(axis=1)
    new_y1 = cy.min(axis=1)
    new_x2 = cx.max(axis=1)
    new_y2 = cy.max(axis=1)

    # Clip to image boundaries
    new_x1 = np.clip(new_x1, 0, out_w)
    new_y1 = np.clip(new_y1, 0, out_h)
    new_x2 = np.clip(new_x2, 0, out_w)
    new_y2 = np.clip(new_y2, 0, out_h)

    # Filter out degenerate boxes (too small after clipping)
    orig_area = (x2 - x1) * (y2 - y1)
    new_area = (new_x2 - new_x1) * (new_y2 - new_y1)
    new_w = new_x2 - new_x1
    new_h = new_y2 - new_y1

    # Keep boxes that:
    # 1. Have sufficient area remaining (at least 10% of original)
    # 2. Have minimum width and height (at least 2 pixels)
    eps = 1e-6
    area_ratio = new_area / (orig_area + eps)
    keep = (area_ratio > 0.1) & (new_w > 2) & (new_h > 2)

    if not keep.any():
        return np.zeros((0, 5), dtype=labels.dtype)

    result = np.column_stack([
        cls[keep],
        new_x1[keep],
        new_y1[keep],
        new_x2[keep],
        new_y2[keep],
    ])

    return result.astype(labels.dtype)


class MosaicAugment:
    """4-image mosaic augmentation.

    Concept from Bochkovskiy et al. 2020 ("YOLOv4"): combine 4 training
    images into a single composite by placing them in quadrants around a
    randomly chosen center point. This expands the effective context and
    improves detection of small objects. Optionally uses 9 images.

    Related to CutMix (Yun et al. 2019), which mixes rectangular patches
    between two images.

    Args:
        dataset: Source dataset for sampling additional images. Must support
                 len() and indexing, returning (image, labels) pairs via
                 load_image() and load_label().
        target_size: Output image size (square). Default 640.
        mosaic9_prob: Probability of 9-image mosaic. Default 0.0.
        fill_value: Fill value for empty regions. Default 114.
        affine_degrees: Degrees for post-mosaic affine. Default 0.0.
        affine_translate: Translation for post-mosaic affine. Default 0.1.
        affine_scale: Scale range for post-mosaic affine. Default (0.5, 1.5).
        affine_shear: Shear for post-mosaic affine. Default 0.0.
    """

    def __init__(
        self,
        dataset,
        target_size=640,
        mosaic9_prob=0.0,
        fill_value=114,
        affine_degrees=0.0,
        affine_translate=0.1,
        affine_scale=(0.5, 1.5),
        affine_shear=0.0,
    ):
        self.dataset = dataset
        self.target_size = target_size
        self.mosaic9_prob = mosaic9_prob
        self.fill_value = fill_value
        self.affine = RandomAffine(
            degrees=affine_degrees,
            translate=affine_translate,
            scale_range=affine_scale,
            shear=affine_shear,
            border=(-target_size // 2, -target_size // 2),
            fill_value=fill_value,
        )

    def __call__(self, index):
        """Build a mosaic image from 4 (or 9) source images.

        Args:
            index: Index of the primary image in the dataset.

        Returns:
            image: np.ndarray (target_size, target_size, 3) mosaic image.
            labels: np.ndarray (N, 5) [cls, x1, y1, x2, y2] combined labels.
        """
        if random.random() < self.mosaic9_prob:
            return self._mosaic9(index)
        return self._mosaic4(index)

    def _mosaic4(self, index):
        """Create a 4-image mosaic.

        The canvas is 2x the target size. Four images are placed in
        quadrants around a random center point. A post-hoc affine
        transform crops and adjusts to the final target size.
        """
        s = self.target_size
        # Random center point within [0.5*s, 1.5*s]
        cx = random.randint(int(s * 0.5), int(s * 1.5))
        cy = random.randint(int(s * 0.5), int(s * 1.5))

        # Canvas is 2*s x 2*s
        canvas = np.full((s * 2, s * 2, 3), self.fill_value, dtype=np.uint8)
        all_labels = []

        # Sample 3 additional random indices
        n = len(self.dataset)
        indices = [index] + [random.randint(0, n - 1) for _ in range(3)]

        for i, idx in enumerate(indices):
            img = self.dataset.load_image(idx)
            lbl = self.dataset.load_label(idx)
            h, w = img.shape[:2]

            # Determine placement region for each quadrant
            if i == 0:  # top-left
                # Image region: right-bottom portion placed at (cx-w, cy-h) to (cx, cy)
                x1c, y1c, x2c, y2c = max(cx - w, 0), max(cy - h, 0), cx, cy
                # Corresponding crop from source image
                x1s = w - (x2c - x1c)
                y1s = h - (y2c - y1c)
                x2s, y2s = w, h
            elif i == 1:  # top-right
                x1c, y1c, x2c, y2c = cx, max(cy - h, 0), min(cx + w, s * 2), cy
                x1s = 0
                y1s = h - (y2c - y1c)
                x2s = x2c - x1c
                y2s = h
            elif i == 2:  # bottom-left
                x1c, y1c, x2c, y2c = max(cx - w, 0), cy, cx, min(cy + h, s * 2)
                x1s = w - (x2c - x1c)
                y1s = 0
                x2s = w
                y2s = y2c - y1c
            else:  # bottom-right
                x1c, y1c, x2c, y2c = cx, cy, min(cx + w, s * 2), min(cy + h, s * 2)
                x1s = 0
                y1s = 0
                x2s = x2c - x1c
                y2s = y2c - y1c

            # Place image crop onto canvas
            canvas[y1c:y2c, x1c:x2c] = img[y1s:y2s, x1s:x2s]

            # Offset labels to canvas coordinates
            if lbl is not None and len(lbl) > 0:
                lbl = lbl.copy()
                # Shift: canvas_pos - source_crop_origin
                dx = x1c - x1s
                dy = y1c - y1s
                lbl[:, 1] += dx  # x1
                lbl[:, 2] += dy  # y1
                lbl[:, 3] += dx  # x2
                lbl[:, 4] += dy  # y2
                all_labels.append(lbl)

        # Combine all labels
        if all_labels:
            labels = np.concatenate(all_labels, axis=0)
            # Clip to canvas boundaries
            labels[:, 1] = np.clip(labels[:, 1], 0, s * 2)
            labels[:, 2] = np.clip(labels[:, 2], 0, s * 2)
            labels[:, 3] = np.clip(labels[:, 3], 0, s * 2)
            labels[:, 4] = np.clip(labels[:, 4], 0, s * 2)
            # Remove degenerate boxes
            w_box = labels[:, 3] - labels[:, 1]
            h_box = labels[:, 4] - labels[:, 2]
            keep = (w_box > 2) & (h_box > 2)
            labels = labels[keep]
        else:
            labels = np.zeros((0, 5), dtype=np.float32)

        # Apply random affine to crop from 2*s canvas to s output
        image, labels = self.affine(canvas, labels)

        return image, labels

    def _mosaic9(self, index):
        """Create a 9-image mosaic.

        Arranges 9 images in a 3x3 grid on a large canvas, then
        applies an affine transform to crop to the target size.
        """
        s = self.target_size
        canvas = np.full((s * 3, s * 3, 3), self.fill_value, dtype=np.uint8)
        all_labels = []

        n = len(self.dataset)
        indices = [index] + [random.randint(0, n - 1) for _ in range(8)]

        for i, idx in enumerate(indices):
            img = self.dataset.load_image(idx)
            lbl = self.dataset.load_label(idx)
            h, w = img.shape[:2]

            # Grid position (row, col)
            row, col = divmod(i, 3)
            # Place each image centered in its grid cell
            gx = col * s + (s - w) // 2
            gy = row * s + (s - h) // 2

            # Compute overlap with canvas
            x1c = max(gx, 0)
            y1c = max(gy, 0)
            x2c = min(gx + w, s * 3)
            y2c = min(gy + h, s * 3)

            x1s = x1c - gx
            y1s = y1c - gy
            x2s = x2c - gx
            y2s = y2c - gy

            canvas[y1c:y2c, x1c:x2c] = img[y1s:y2s, x1s:x2s]

            if lbl is not None and len(lbl) > 0:
                lbl = lbl.copy()
                lbl[:, 1] += gx  # x offset
                lbl[:, 2] += gy  # y offset
                lbl[:, 3] += gx
                lbl[:, 4] += gy
                all_labels.append(lbl)

        if all_labels:
            labels = np.concatenate(all_labels, axis=0)
            labels[:, 1] = np.clip(labels[:, 1], 0, s * 3)
            labels[:, 2] = np.clip(labels[:, 2], 0, s * 3)
            labels[:, 3] = np.clip(labels[:, 3], 0, s * 3)
            labels[:, 4] = np.clip(labels[:, 4], 0, s * 3)
            w_box = labels[:, 3] - labels[:, 1]
            h_box = labels[:, 4] - labels[:, 2]
            keep = (w_box > 2) & (h_box > 2)
            labels = labels[keep]
        else:
            labels = np.zeros((0, 5), dtype=np.float32)

        # Crop center region with affine
        # Use a modified affine that targets the center of the 3x3 grid
        affine_9 = RandomAffine(
            degrees=self.affine.degrees,
            translate=self.affine.translate,
            scale_range=(self.affine.scale_min, self.affine.scale_max),
            shear=self.affine.shear,
            border=(-s, -s),
            fill_value=self.fill_value,
        )
        image, labels = affine_9(canvas, labels)

        return image, labels


class MixUpAugment:
    """MixUp augmentation for object detection.

    Based on Zhang et al. 2018 "mixup: Beyond Empirical Risk Minimization".
    Creates a convex combination of two images: mixed = lam * img1 + (1 - lam) * img2.
    For detection, labels from both images are concatenated (no label mixing).

    Args:
        dataset: Source dataset for sampling the second image.
        alpha: Beta distribution parameter. Default 1.5.
            When alpha=1.0, lambda is uniform on [0, 1].
            Higher alpha biases lambda towards 0.5 (stronger mixing).
    """

    def __init__(self, dataset, alpha=1.5):
        self.dataset = dataset
        self.alpha = alpha

    def __call__(self, image, labels, index=None):
        """Apply MixUp to an image with a random partner.

        Args:
            image: np.ndarray (H, W, 3) primary image.
            labels: np.ndarray (N, 5) [cls, x1, y1, x2, y2] primary labels.
            index: Optional index hint (unused, partner is random).

        Returns:
            image: np.ndarray (H, W, 3) mixed image.
            labels: np.ndarray (M, 5) concatenated labels from both images.
        """
        # Sample mixing ratio from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Sample a random partner image
        n = len(self.dataset)
        idx2 = random.randint(0, n - 1)
        img2 = self.dataset.load_image(idx2)
        lbl2 = self.dataset.load_label(idx2)

        h1, w1 = image.shape[:2]
        h2, w2 = img2.shape[:2]

        # Resize partner to match primary image size if needed
        if h1 != h2 or w1 != w2:
            img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_LINEAR)
            if lbl2 is not None and len(lbl2) > 0:
                lbl2 = lbl2.copy()
                lbl2[:, 1] *= w1 / w2  # x1
                lbl2[:, 2] *= h1 / h2  # y1
                lbl2[:, 3] *= w1 / w2  # x2
                lbl2[:, 4] *= h1 / h2  # y2

        # Blend images
        image = (image.astype(np.float32) * lam + img2.astype(np.float32) * (1 - lam))
        image = np.clip(image, 0, 255).astype(np.uint8)

        # Concatenate labels from both images
        parts = []
        if labels is not None and len(labels) > 0:
            parts.append(labels)
        if lbl2 is not None and len(lbl2) > 0:
            parts.append(lbl2)

        if parts:
            labels = np.concatenate(parts, axis=0)
        else:
            labels = np.zeros((0, 5), dtype=np.float32)

        return image, labels


class CopyPasteAugment:
    """Copy-Paste augmentation for object detection (bounding-box variant).

    Simplified version of Ghiasi et al. 2021 "Simple Copy-Paste is a Strong
    Data Augmentation Method for Instance Segmentation". Instead of using
    segmentation masks, this implementation copies rectangular bounding box
    regions from a source image and pastes them onto the target image.

    Args:
        dataset: Source dataset for sampling donor images.
        p: Probability of pasting each available object. Default 0.5.
        max_objects: Maximum number of objects to paste. Default 30.
    """

    def __init__(self, dataset, p=0.5, max_objects=30):
        self.dataset = dataset
        self.p = p
        self.max_objects = max_objects

    def __call__(self, image, labels):
        """Apply Copy-Paste augmentation.

        Args:
            image: np.ndarray (H, W, 3) target image.
            labels: np.ndarray (N, 5) [cls, x1, y1, x2, y2] target labels.

        Returns:
            image: np.ndarray (H, W, 3) image with pasted objects.
            labels: np.ndarray (M, 5) combined labels.
        """
        h, w = image.shape[:2]

        # Sample a random donor image
        n = len(self.dataset)
        idx2 = random.randint(0, n - 1)
        img2 = self.dataset.load_image(idx2)
        lbl2 = self.dataset.load_label(idx2)

        if lbl2 is None or len(lbl2) == 0:
            return image, labels

        h2, w2 = img2.shape[:2]
        image = image.copy()

        pasted = []
        count = 0

        for obj in lbl2:
            if count >= self.max_objects:
                break
            if random.random() > self.p:
                continue

            cls_id = obj[0]
            # Source box coordinates (in donor image)
            sx1 = int(np.clip(obj[1], 0, w2))
            sy1 = int(np.clip(obj[2], 0, h2))
            sx2 = int(np.clip(obj[3], 0, w2))
            sy2 = int(np.clip(obj[4], 0, h2))

            bw = sx2 - sx1
            bh = sy2 - sy1
            if bw < 3 or bh < 3:
                continue

            # Crop the object patch from donor
            patch = img2[sy1:sy2, sx1:sx2]

            # Scale patch if donor and target have different sizes
            if h2 != h or w2 != w:
                scale_x = w / w2
                scale_y = h / h2
                new_bw = int(bw * scale_x)
                new_bh = int(bh * scale_y)
                if new_bw < 3 or new_bh < 3:
                    continue
                patch = cv2.resize(patch, (new_bw, new_bh), interpolation=cv2.INTER_LINEAR)
                bw, bh = new_bw, new_bh

            # Random placement on target image
            if w - bw <= 0 or h - bh <= 0:
                continue
            tx1 = random.randint(0, w - bw)
            ty1 = random.randint(0, h - bh)
            tx2 = tx1 + bw
            ty2 = ty1 + bh

            # Alpha-blend the patch onto the target (simple overwrite)
            # Use a soft edge for more natural appearance
            image[ty1:ty2, tx1:tx2] = patch

            pasted.append([cls_id, tx1, ty1, tx2, ty2])
            count += 1

        # Combine original labels with pasted ones
        parts = []
        if labels is not None and len(labels) > 0:
            parts.append(labels)
        if pasted:
            parts.append(np.array(pasted, dtype=np.float32))

        if parts:
            labels = np.concatenate(parts, axis=0)
        else:
            labels = np.zeros((0, 5), dtype=np.float32)

        return image, labels
