# SwiftDet Dataset Classes for Object Detection
# MIT License - Original implementation
#
# Provides COCO-format detection dataset and ImageNet classification dataset.
# Label files use the standard text format: one object per line as
# "class center_x center_y width height" with normalized coordinates.

import os
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import yaml
except ImportError:
    yaml = None

from .transforms import LetterBox, RandomHSV, RandomFlip, RandomErasing, Normalize, Compose
from .augment import MosaicAugment, MixUpAugment, CopyPasteAugment


def _load_yaml(path):
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        dict: Parsed YAML contents.
    """
    if yaml is None:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _img_to_label_path(img_path):
    """Convert an image path to the corresponding label path.

    Replaces 'images' directory with 'labels' and changes extension to .txt.
    Example: /data/images/train/001.jpg -> /data/labels/train/001.txt

    Args:
        img_path: str or Path to an image file.

    Returns:
        str: Corresponding label file path.
    """
    p = str(img_path)
    # Replace the last occurrence of /images/ with /labels/
    parts = p.rsplit("/images/", 1)
    if len(parts) == 2:
        label_dir = parts[0] + "/labels/" + parts[1]
    else:
        # Fallback: same directory
        label_dir = p
    # Change extension to .txt
    label_path = os.path.splitext(label_dir)[0] + ".txt"
    return label_path


_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _scan_images(directory):
    """Recursively scan a directory for image files.

    Args:
        directory: str or Path to scan.

    Returns:
        list[str]: Sorted list of image file paths.
    """
    directory = Path(directory)
    images = []
    for p in sorted(directory.rglob("*")):
        if p.suffix.lower() in _IMG_EXTENSIONS and p.is_file():
            images.append(str(p))
    return images


class COCODetectionDataset(Dataset):
    """COCO-format object detection dataset.

    Loads images and annotations from a YAML config file.
    Expects the standard directory structure:
        path/
            images/
                train2017/
                val2017/
            labels/
                train2017/
                val2017/

    Each label text file contains one line per object:
        class_id center_x center_y width height
    where coordinates are normalized to [0, 1].

    The dataset converts labels to xyxy pixel coordinate format:
        [class_id, x1, y1, x2, y2]

    Args:
        data_yaml: Path to data YAML config file.
        split: Dataset split, 'train' or 'val'. Default 'train'.
        img_size: Target image size (square). Default 640.
        augment: Whether to apply training augmentations. Default False.
        mosaic: Probability of mosaic augmentation (0-1). Default 0.0.
        mosaic9_prob: Probability of 9-image mosaic. Default 0.0.
        mixup: Probability of MixUp augmentation (0-1). Default 0.0.
        copy_paste: Probability of Copy-Paste augmentation (0-1). Default 0.0.
        hsv_h: Hue augmentation gain. Default 0.015.
        hsv_s: Saturation augmentation gain. Default 0.7.
        hsv_v: Value augmentation gain. Default 0.4.
        flip_h: Horizontal flip probability. Default 0.5.
        flip_v: Vertical flip probability. Default 0.0.
        degrees: Rotation degrees for affine. Default 0.0.
        translate: Translation fraction for affine. Default 0.1.
        scale: Scale range for affine. Default (0.5, 1.5).
        shear: Shear degrees for affine. Default 0.0.
        mixup_alpha: Beta distribution alpha for MixUp. Default 1.5.
        cache_images: Cache images in RAM for faster training. Default False.
        mean: Normalization mean (BGR). Default None.
        std: Normalization std (BGR). Default None.
    """

    def __init__(
        self,
        data_yaml,
        split="train",
        img_size=640,
        augment=False,
        mosaic=0.0,
        mosaic9_prob=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flip_h=0.5,
        flip_v=0.0,
        degrees=0.0,
        translate=0.1,
        scale=(0.5, 1.5),
        shear=0.0,
        erasing=0.0,
        mixup_alpha=1.5,
        cache_images=False,
        mean=None,
        std=None,
    ):
        super().__init__()

        # Parse YAML config
        cfg = _load_yaml(data_yaml)
        base_path = Path(cfg.get("path", ""))

        # Resolve image directory
        split_dir = cfg.get(split, cfg.get("train" if split == "train" else "val"))
        if split_dir is None:
            raise ValueError(f"Split '{split}' not found in {data_yaml}")

        img_dir = base_path / split_dir
        if not img_dir.exists():
            # Try relative to YAML file location
            yaml_parent = Path(data_yaml).parent
            img_dir = yaml_parent / base_path / split_dir
            if not img_dir.exists():
                raise FileNotFoundError(f"Image directory not found: {img_dir}")

        self.img_dir = img_dir
        self.nc = cfg.get("nc", 80)
        self.names = cfg.get("names", {})

        # Scan for images
        self.img_paths = _scan_images(self.img_dir)
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

        # Resolve label directory
        # Supports two layouts:
        #   1. path/images/train2017/ + path/labels/train2017/  (YAML: train: images/train2017)
        #   2. path/train2017/        + path/labels/train2017/  (YAML: train: train2017)
        split_dir_str = str(split_dir)
        if split_dir_str.startswith("images/") or split_dir_str.startswith("images\\"):
            label_rel = split_dir_str.replace("images/", "labels/", 1).replace("images\\", "labels\\", 1)
        else:
            label_rel = "labels/" + split_dir_str
        self.label_dir = base_path / label_rel

        # Generate corresponding label paths
        self.label_paths = [
            str(self.label_dir / (Path(p).stem + ".txt")) for p in self.img_paths
        ]

        # Warn if no labels found
        found = sum(1 for lp in self.label_paths if os.path.exists(lp))
        print(f"  Labels: {found}/{len(self.img_paths)} found in {self.label_dir}")

        # Settings
        self.img_size = img_size
        self.augment = augment
        self.mosaic_prob = mosaic
        self._orig_mosaic_prob = mosaic
        self.mixup_prob = mixup
        self.copy_paste_prob = copy_paste

        # Image cache
        self.cache_images = cache_images
        self._image_cache = {}

        # Build augmentation pipeline
        self._mosaic = None
        self._mixup = None
        self._copy_paste = None

        if augment:
            if mosaic > 0:
                self._mosaic = MosaicAugment(
                    dataset=self,
                    target_size=img_size,
                    mosaic9_prob=mosaic9_prob,
                    affine_degrees=degrees,
                    affine_translate=translate,
                    affine_scale=scale,
                    affine_shear=shear,
                )
            if mixup > 0:
                self._mixup = MixUpAugment(dataset=self, alpha=mixup_alpha)
            if copy_paste > 0:
                self._copy_paste = CopyPasteAugment(dataset=self)

        # Standard transforms
        self.letterbox = LetterBox(target_size=(img_size, img_size))

        aug_transforms = []
        if augment:
            aug_transforms.append(RandomHSV(h_gain=hsv_h, s_gain=hsv_s, v_gain=hsv_v))
            aug_transforms.append(RandomFlip(p_horizontal=flip_h, p_vertical=flip_v))
            if erasing > 0:
                aug_transforms.append(RandomErasing(p=erasing))
        aug_transforms.append(Normalize(mean=mean, std=std))
        self.transforms = Compose(aug_transforms)

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.img_paths)

    def __getitem__(self, index):
        """Get a training sample.

        Returns:
            image: np.ndarray (3, H, W) float32 normalized.
            labels: np.ndarray (N, 5) [cls, x1, y1, x2, y2] in pixel coords.
        """
        mosaic_applied = False

        # Mosaic augmentation
        if self.augment and self._mosaic is not None and np.random.random() < self.mosaic_prob:
            image, labels = self._mosaic(index)
            mosaic_applied = True
        else:
            image = self.load_image(index)
            labels = self.load_label(index)
            # Letterbox resize
            image, labels = self.letterbox(image, labels)

        # Copy-Paste augmentation
        if self.augment and self._copy_paste is not None and np.random.random() < self.copy_paste_prob:
            image, labels = self._copy_paste(image, labels)

        # MixUp augmentation
        if self.augment and self._mixup is not None and np.random.random() < self.mixup_prob:
            image, labels = self._mixup(image, labels)

        # Standard transforms (HSV, flip, normalize)
        image, labels = self.transforms(image, labels)

        # Convert to tensors
        image = torch.from_numpy(image)
        if labels is None or len(labels) == 0:
            labels = torch.zeros((0, 5), dtype=torch.float32)
        else:
            labels = torch.from_numpy(labels.astype(np.float32))

        return image, labels

    def load_image(self, index):
        """Load an image by index.

        Uses cache if enabled. Returns the original-size image.

        Args:
            index: Dataset index.

        Returns:
            np.ndarray (H, W, 3) BGR uint8 image.
        """
        if self.cache_images and index in self._image_cache:
            return self._image_cache[index].copy()

        path = self.img_paths[index]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {path}")

        if self.cache_images:
            self._image_cache[index] = img.copy()

        return img

    def load_label(self, index):
        """Load annotation labels for an image.

        Reads the label text file (one object per line):
            class_id center_x center_y width height  (normalized)

        Converts to xyxy pixel coordinates:
            [class_id, x1, y1, x2, y2]

        Args:
            index: Dataset index.

        Returns:
            np.ndarray (N, 5) float32 labels, or empty (0, 5) array.
        """
        label_path = self.label_paths[index]

        if not os.path.exists(label_path):
            return np.zeros((0, 5), dtype=np.float32)

        try:
            with open(label_path, "r") as f:
                lines = f.read().strip().splitlines()
        except Exception:
            return np.zeros((0, 5), dtype=np.float32)

        if not lines:
            return np.zeros((0, 5), dtype=np.float32)

        # Load image dimensions for coordinate conversion
        img_path = self.img_paths[index]
        # Read only the header to get dimensions (faster than full decode)
        if self.cache_images and index in self._image_cache:
            img_h, img_w = self._image_cache[index].shape[:2]
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                return np.zeros((0, 5), dtype=np.float32)
            img_h, img_w = img.shape[:2]

        labels = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = float(parts[0])
            cx = float(parts[1])  # normalized center x
            cy = float(parts[2])  # normalized center y
            bw = float(parts[3])  # normalized width
            bh = float(parts[4])  # normalized height

            # Convert normalized xywh to pixel xyxy
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h

            # Clip to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            # Skip degenerate boxes
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue

            labels.append([cls_id, x1, y1, x2, y2])

        if labels:
            return np.array(labels, dtype=np.float32)
        return np.zeros((0, 5), dtype=np.float32)

    def set_mosaic(self, enabled):
        """Enable or disable mosaic augmentation (for mosaic closing)."""
        self.mosaic_prob = self._orig_mosaic_prob if enabled else 0.0

    def get_num_classes(self):
        """Return the number of object classes."""
        return self.nc

    def get_class_names(self):
        """Return the class name dictionary."""
        return self.names

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"images={len(self)}, nc={self.nc}, "
            f"img_size={self.img_size}, augment={self.augment})"
        )


class ImageNetDataset(Dataset):
    """ImageNet-style classification dataset for backbone pre-training.

    Loads images from the standard ImageNet directory structure:
        root/
            class_name_1/
                image1.jpg
                image2.jpg
            class_name_2/
                ...

    Args:
        root: Root directory containing class subdirectories.
        img_size: Target image size (square). Default 224.
        augment: Apply training augmentations. Default False.
        mean: Normalization mean (RGB). Default (0.485, 0.456, 0.406).
        std: Normalization std (RGB). Default (0.229, 0.224, 0.225).
    """

    def __init__(
        self,
        root,
        img_size=224,
        augment=False,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.root = Path(root)
        self.img_size = img_size
        self.augment = augment
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

        # Discover classes and images
        self.classes = sorted(
            d.name for d in self.root.iterdir() if d.is_dir()
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []  # list of (path, class_idx)
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            cls_idx = self.class_to_idx[cls_name]
            for img_path in sorted(cls_dir.rglob("*")):
                if img_path.suffix.lower() in _IMG_EXTENSIONS and img_path.is_file():
                    self.samples.append((str(img_path), cls_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, index):
        """Get a classification sample.

        Returns:
            image: torch.Tensor (3, img_size, img_size) float32.
            label: int class index.
        """
        path, label = self.samples[index]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augment:
            img = self._augment_train(img)
        else:
            img = self._preprocess_val(img)

        # Normalize
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.ascontiguousarray(img)
        img = (img - self.mean) / self.std

        return torch.from_numpy(img), label

    def _augment_train(self, img):
        """Training augmentation: random resized crop + horizontal flip.

        Args:
            img: np.ndarray (H, W, 3) RGB uint8.

        Returns:
            np.ndarray (img_size, img_size, 3) augmented image.
        """
        h, w = img.shape[:2]
        s = self.img_size

        # Random resized crop (scale 0.08 to 1.0, aspect ratio 3/4 to 4/3)
        area = h * w
        for _ in range(10):
            target_area = np.random.uniform(0.08, 1.0) * area
            aspect = np.exp(np.random.uniform(np.log(3 / 4), np.log(4 / 3)))
            crop_w = int(round(np.sqrt(target_area * aspect)))
            crop_h = int(round(np.sqrt(target_area / aspect)))
            if crop_w <= w and crop_h <= h:
                x1 = np.random.randint(0, w - crop_w + 1)
                y1 = np.random.randint(0, h - crop_h + 1)
                img = img[y1 : y1 + crop_h, x1 : x1 + crop_w]
                img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
                break
        else:
            # Fallback: center crop
            img = self._center_crop_resize(img, s)

        # Random horizontal flip
        if np.random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])

        return img

    def _preprocess_val(self, img):
        """Validation preprocessing: resize + center crop.

        Args:
            img: np.ndarray (H, W, 3) RGB uint8.

        Returns:
            np.ndarray (img_size, img_size, 3) preprocessed image.
        """
        s = self.img_size
        # Resize shorter side to img_size * 256/224 (standard practice)
        h, w = img.shape[:2]
        resize_size = int(s * 256 / 224)
        if h < w:
            new_h = resize_size
            new_w = int(w * resize_size / h)
        else:
            new_w = resize_size
            new_h = int(h * resize_size / w)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return self._center_crop_resize(img, s)

    @staticmethod
    def _center_crop_resize(img, size):
        """Center crop an image to a square of given size.

        Args:
            img: np.ndarray (H, W, 3).
            size: Target square size.

        Returns:
            np.ndarray (size, size, 3).
        """
        h, w = img.shape[:2]
        if h == size and w == size:
            return img

        # Center crop to square
        if h != w:
            crop = min(h, w)
            y1 = (h - crop) // 2
            x1 = (w - crop) // 2
            img = img[y1 : y1 + crop, x1 : x1 + crop]

        # Resize
        if img.shape[0] != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

        return img

    def get_num_classes(self):
        """Return the number of classes."""
        return len(self.classes)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"samples={len(self)}, classes={len(self.classes)}, "
            f"img_size={self.img_size}, augment={self.augment})"
        )


def detection_collate_fn(batch):
    """Custom collate function for object detection.

    Pads label tensors to the same size so they can be batched.
    Adds a batch index column to labels for downstream processing.

    Args:
        batch: List of (image, labels) tuples from the dataset.
            image: torch.Tensor (3, H, W) float32.
            labels: torch.Tensor (N_i, 5) [cls, x1, y1, x2, y2].

    Returns:
        images: torch.Tensor (B, 3, H, W) batched images.
        targets: torch.Tensor (B, max_labels, 6) padded labels.
            Each row: [batch_idx, cls, x1, y1, x2, y2].
            Padding rows are all zeros.
    """
    images, labels_list = zip(*batch)

    # Stack images (all should have the same spatial size)
    images = torch.stack(images, dim=0)

    # Find maximum number of labels in this batch
    max_labels = max(lbl.shape[0] for lbl in labels_list) if labels_list else 0
    max_labels = max(max_labels, 1)  # at least 1 to avoid empty tensor issues

    batch_size = len(labels_list)

    # Create padded target tensor: [cls, x1, y1, x2, y2]
    targets = torch.zeros((batch_size, max_labels, 5), dtype=torch.float32)

    for i, lbl in enumerate(labels_list):
        n = lbl.shape[0]
        if n > 0:
            targets[i, :n, :] = lbl[:, :5]  # cls, x1, y1, x2, y2

    return images, targets
