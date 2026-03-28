# SwiftDet - ImageNet pre-training for backbone
# MIT License - Original implementation
#
# References:
#   - ImageNet: Deng et al. 2009, "ImageNet: A Large-Scale Hierarchical Image Database"
#   - Transfer learning: Girshick et al. 2014 (R-CNN); He et al. 2019
#     "Rethinking ImageNet Pre-training"
#   - Cosine annealing: Loshchilov & Hutter 2017, "SGDR: Stochastic Gradient
#     Descent with Warm Restarts"
#   - Label smoothing: Szegedy et al. 2016, "Rethinking the Inception Architecture"

import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


class _ClassificationWrapper(nn.Module):
    """Wraps a detection backbone with a classification head.

    Adds global average pooling and a fully connected layer on top of the
    backbone's final feature map (highest-level feature) to produce class
    logits for standard ImageNet classification training.

    Args:
        backbone: The CNN backbone module. Must have an `out_channels` attribute
            that is a list of channel dimensions for each output level.
        num_classes: Number of classification categories (default 1000 for ImageNet).
    """

    def __init__(self, backbone, num_classes=1000):
        super().__init__()
        self.backbone = backbone
        # Use the last (highest-level) feature map's channel count
        final_channels = backbone.out_channels[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(final_channels, num_classes)

    def forward(self, x):
        """Forward pass: backbone features -> GAP -> FC logits.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            (B, num_classes) classification logits.
        """
        features = self.backbone(x)
        # Take the last (deepest) feature level
        feat = features[-1]  # (B, C, H, W)
        pooled = self.pool(feat).flatten(1)  # (B, C)
        return self.fc(pooled)


class ImageNetPretrainer:
    """ImageNet pre-training for the detection backbone.

    Performs standard classification pre-training: the backbone is wrapped
    with a global average pooling layer and a fully connected classifier,
    then trained on ImageNet with cross-entropy loss and cosine annealing.

    After pre-training, only the backbone weights are saved. The classification
    head is discarded since the detection head will be used instead.

    Args:
        model: SwiftDetector instance. The backbone attribute will be extracted
            and used for pre-training.
        data_path: Path to the ImageNet dataset root directory.
        epochs: Number of training epochs (default 100).
        batch_size: Training batch size (default 256).
        lr0: Initial learning rate (default 0.1).
        weight_decay: L2 regularization (default 0.0001).
        warmup_epochs: Linear warmup period (default 5).
        num_classes: Number of ImageNet classes (default 1000).
        label_smoothing: Cross-entropy label smoothing factor (default 0.1).
        device: Training device string (default None for auto-detect).
        save_dir: Directory for saving backbone weights (default 'runs/pretrain').
    """

    def __init__(
        self,
        model,
        data_path,
        epochs=100,
        batch_size=256,
        lr0=0.1,
        weight_decay=0.0001,
        warmup_epochs=5,
        num_classes=1000,
        label_smoothing=0.1,
        device=None,
        save_dir="runs/pretrain",
    ):
        self.model = model
        self.data_path = data_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr0 = lr0
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.save_dir = Path(save_dir)

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    def _build_dataloaders(self):
        """Build ImageNet train and val dataloaders.

        Uses the ImageNetDataset class which handles standard ImageNet directory
        structure and applies appropriate train/val augmentations (random crop
        + flip for train, center crop for val).

        Returns:
            Tuple of (train_loader, val_loader).
        """
        from swiftdet.data.dataset import ImageNetDataset

        data_root = Path(self.data_path)
        train_dataset = ImageNetDataset(
            str(data_root / "train"), img_size=224, augment=True
        )
        val_dataset = ImageNetDataset(
            str(data_root / "val"), img_size=224, augment=False
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=self.device.type in ("cuda", "mps"),
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=self.device.type in ("cuda", "mps"),
            drop_last=False,
        )
        return train_loader, val_loader

    def _cosine_lr(self, epoch):
        """Compute cosine annealing LR factor (Loshchilov & Hutter 2017).

        Args:
            epoch: Current epoch (0-based).

        Returns:
            Learning rate multiplier in [0, 1].
        """
        # Minimum LR factor: 1% of initial
        lrf = 0.01
        return lrf + 0.5 * (1.0 - lrf) * (1.0 + math.cos(math.pi * epoch / self.epochs))

    def train(self):
        """Pre-train the backbone on ImageNet.

        Creates a classification wrapper around the backbone, trains it with
        cross-entropy loss and cosine annealing LR, evaluates top-1/top-5
        accuracy, and saves the backbone weights.

        Returns:
            dict: Metrics including 'top1_acc', 'top5_acc', and 'best_top1'.
        """
        os.makedirs(self.save_dir, exist_ok=True)

        # Extract backbone from detector and wrap for classification
        backbone = self.model.backbone
        cls_model = _ClassificationWrapper(backbone, self.num_classes).to(self.device)
        cls_model.train()

        # Optimizer with separate weight decay handling
        decay_params = []
        no_decay_params = []
        for name, param in cls_model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "bn" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = optim.SGD(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.lr0,
            momentum=0.9,
            nesterov=True,
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        # AMP: autocast on CUDA and MPS; GradScaler only on CUDA
        use_amp = self.device.type in ("cuda", "mps")
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and self.device.type == "cuda"))

        # Build dataloaders
        train_loader, val_loader = self._build_dataloaders()
        n_batches = len(train_loader)

        best_top1 = 0.0

        for epoch in range(self.epochs):
            cls_model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            # Compute LR for this epoch
            if epoch < self.warmup_epochs:
                # Linear warmup
                lr = self.lr0 * (epoch + 1) / self.warmup_epochs
            else:
                lr = self.lr0 * self._cosine_lr(epoch)

            for pg in optimizer.param_groups:
                pg["lr"] = lr

            t_start = time.time()

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Ensure float and normalized
                if images.dtype == torch.uint8:
                    images = images.float() / 255.0

                with torch.amp.autocast(self.device.type, enabled=use_amp):
                    logits = cls_model(images)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Track training metrics
                total_loss += loss.detach().item() * images.shape[0]
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += images.shape[0]

            train_loss = total_loss / max(total_samples, 1)
            train_acc = total_correct / max(total_samples, 1)
            elapsed = time.time() - t_start

            # --- Validation ---
            top1, top5 = self._validate(cls_model, val_loader, use_amp)

            if top1 > best_top1:
                best_top1 = top1
                # Save best backbone weights
                torch.save(
                    backbone.state_dict(),
                    self.save_dir / "backbone_best.pt",
                )

            # Save last backbone weights
            torch.save(
                backbone.state_dict(),
                self.save_dir / "backbone_last.pt",
            )

            print(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"top1={top1:.4f} top5={top5:.4f} | "
                f"lr={lr:.6f} | time={elapsed:.1f}s"
            )

        print(f"Pre-training complete. Best Top-1: {best_top1:.4f}")
        return {"top1_acc": top1, "top5_acc": top5, "best_top1": best_top1}

    @torch.no_grad()
    def _validate(self, cls_model, val_loader, use_amp):
        """Run validation and compute top-1 and top-5 accuracy.

        Args:
            cls_model: The classification wrapper model.
            val_loader: Validation dataloader.
            use_amp: Whether to use automatic mixed precision.

        Returns:
            Tuple of (top1_accuracy, top5_accuracy) as floats in [0, 1].
        """
        cls_model.eval()
        total_correct_top1 = 0
        total_correct_top5 = 0
        total_samples = 0

        for images, labels in val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if images.dtype == torch.uint8:
                images = images.float() / 255.0

            with torch.amp.autocast(self.device.type, enabled=use_amp):
                logits = cls_model(images)

            # Top-1 accuracy
            preds_top1 = logits.argmax(dim=1)
            total_correct_top1 += (preds_top1 == labels).sum().item()

            # Top-5 accuracy
            k = min(5, logits.shape[1])
            _, preds_top5 = logits.topk(k, dim=1, largest=True, sorted=True)
            labels_expanded = labels.unsqueeze(1).expand_as(preds_top5)
            total_correct_top5 += (preds_top5 == labels_expanded).any(dim=1).sum().item()

            total_samples += images.shape[0]

        top1 = total_correct_top1 / max(total_samples, 1)
        top5 = total_correct_top5 / max(total_samples, 1)
        return top1, top5
