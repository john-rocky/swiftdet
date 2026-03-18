# SwiftDet - Detection model trainer
# MIT License - Original implementation
#
# References:
#   - Cosine annealing: Loshchilov & Hutter 2017, "SGDR: Stochastic Gradient
#     Descent with Warm Restarts"
#   - Mixed precision: Micikevicius et al. 2018, "Mixed Precision Training"
#   - EMA: Polyak & Juditsky 1992; Izmailov et al. 2018 (SWA)
#   - Mosaic augmentation: Bochkovskiy et al. 2020 (mosaic data augmentation)
#   - Label smoothing: Szegedy et al. 2016, "Rethinking the Inception Architecture"
#   - Linear warmup: Goyal et al. 2017, "Accurate, Large Minibatch SGD"

import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from swiftdet.losses import DetectionLoss
from swiftdet.utils.ema import ModelEMA


class DetectionTrainer:
    """Detection model trainer with AMP, EMA, cosine LR, and mosaic closing.

    Implements a standard anchor-free detection training pipeline with:
    - Cosine annealing learning rate schedule (Loshchilov & Hutter 2017)
    - Linear warmup (Goyal et al. 2017)
    - Automatic mixed precision training (Micikevicius et al. 2018)
    - Exponential moving average of model weights (Polyak & Juditsky 1992)
    - Mosaic augmentation with closing for final epochs (Bochkovskiy et al. 2020)
    - Gradient accumulation for effective large batch training

    Args:
        model: SwiftDetector model instance.
        data_yaml: Path to dataset YAML configuration file.
        epochs: Total training epochs (default 500).
        batch_size: Per-device batch size (default 64).
        img_size: Input image size in pixels (default 640).
        lr0: Initial learning rate (default 0.01).
        lrf: Final LR as a fraction of lr0 (default 0.01).
        optimizer: Optimizer type, one of 'sgd' or 'adamw' (default 'sgd').
        momentum: SGD momentum parameter (default 0.937).
        weight_decay: L2 regularization coefficient (default 0.0005).
        warmup_epochs: Number of warmup epochs with linear LR ramp (default 5).
        warmup_bias_lr: Warmup initial LR for bias parameters (default 0.1).
        close_mosaic: Disable mosaic augmentation for the last N epochs (default 15).
        amp: Enable automatic mixed precision (default True).
        device: Training device string, e.g. 'cuda:0' or 'cpu' (default None for auto).
        save_dir: Directory path for saving checkpoints (default 'runs/train').
        cls_gain: Classification loss weight (default 0.5).
        box_gain: Box regression loss weight (default 7.5).
        dfl_gain: Distribution focal loss weight (default 1.5).
        label_smoothing: Label smoothing factor (default 0.0).
        resume: Whether to resume from latest checkpoint (default False).
        grad_accum: Gradient accumulation steps (default 1, auto-computed when batch < 64).
    """

    def __init__(
        self,
        model,
        data_yaml,
        epochs=500,
        batch_size=64,
        img_size=640,
        lr0=0.01,
        lrf=0.01,
        optimizer="sgd",
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_bias_lr=0.1,
        close_mosaic=15,
        amp=True,
        device=None,
        save_dir="runs/train",
        cls_gain=0.5,
        box_gain=7.5,
        dfl_gain=1.5,
        label_smoothing=0.0,
        resume=False,
        grad_accum=1,
    ):
        self.model = model
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.lr0 = lr0
        self.lrf = lrf
        self.optimizer_type = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.warmup_bias_lr = warmup_bias_lr
        self.close_mosaic = close_mosaic
        self.amp = amp
        self.save_dir = Path(save_dir)
        self.cls_gain = cls_gain
        self.box_gain = box_gain
        self.dfl_gain = dfl_gain
        self.label_smoothing = label_smoothing
        self.resume = resume
        self.loss_fn = DetectionLoss(
            nc=model.nc, reg_max=model.reg_max,
            cls_gain=cls_gain, box_gain=box_gain, dfl_gain=dfl_gain,
        )

        # Auto-compute gradient accumulation: target nominal batch = 64
        nominal_batch = 64
        if grad_accum <= 1:
            self.grad_accum = max(1, nominal_batch // batch_size)
        else:
            self.grad_accum = grad_accum

        # Resolve device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def _build_optimizer(self):
        """Construct optimizer with per-parameter-group weight decay.

        Bias and normalization parameters do not receive weight decay,
        following standard practice for training deep networks.

        Returns:
            torch.optim.Optimizer instance.
        """
        # Separate parameters: weight-decay group vs no-decay group
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Bias and batch norm parameters: no weight decay
            if "bias" in name or "bn" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if self.optimizer_type == "adamw":
            return optim.AdamW(param_groups, lr=self.lr0, betas=(0.9, 0.999))
        else:
            return optim.SGD(
                param_groups,
                lr=self.lr0,
                momentum=self.momentum,
                nesterov=True,
            )

    def _cosine_lr(self, epoch):
        """Compute cosine annealing learning rate factor.

        Implements the schedule from Loshchilov & Hutter 2017 (SGDR):
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))

        Expressed as a multiplicative factor relative to lr0.

        Args:
            epoch: Current epoch index (0-based).

        Returns:
            Learning rate multiplier in [lrf, 1.0].
        """
        return self.lrf + 0.5 * (1.0 - self.lrf) * (
            1.0 + math.cos(math.pi * epoch / self.epochs)
        )

    def _warmup_lr_and_momentum(self, epoch, batch_idx, n_batches, optimizer):
        """Apply linear warmup to learning rate and momentum.

        During the warmup period (first warmup_epochs), LR is linearly
        ramped from near-zero to lr0, and SGD momentum is linearly ramped
        from 0.8 to the target value. Bias parameters use a separate
        warmup_bias_lr as the starting point.

        Reference: Goyal et al. 2017, "Accurate, Large Minibatch SGD"

        Args:
            epoch: Current epoch index (0-based).
            batch_idx: Current batch index within the epoch.
            n_batches: Total number of batches per epoch.
            optimizer: The optimizer whose param groups will be modified.
        """
        total_warmup_iters = self.warmup_epochs * n_batches
        current_iter = epoch * n_batches + batch_idx

        if current_iter >= total_warmup_iters:
            return

        # Linear interpolation factor [0, 1]
        xi = current_iter / max(total_warmup_iters, 1)

        for j, pg in enumerate(optimizer.param_groups):
            # Warmup LR: bias group gets warmup_bias_lr as start
            if j == 1:  # no-decay group (bias/norm params)
                pg["lr"] = self.warmup_bias_lr * (1.0 - xi) + self.lr0 * xi
            else:
                pg["lr"] = self.lr0 * xi

            # Warmup momentum (only for SGD)
            if "momentum" in pg:
                pg["momentum"] = 0.8 * (1.0 - xi) + self.momentum * xi

    def _compute_loss(self, outputs, targets):
        """Compute detection loss using the shared DetectionLoss module.

        Args:
            outputs: Dict from model forward pass.
            targets: (B, max_gt, 5) tensor [cls, x1, y1, x2, y2].

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        total_loss, loss_dict = self.loss_fn(outputs, targets)
        return total_loss, {
            "cls_loss": loss_dict["cls"].item(),
            "box_loss": loss_dict["box"].item(),
            "dfl_loss": loss_dict["dfl"].item(),
        }

    def train(self):
        """Run the full training loop.

        Performs model training with cosine annealing LR, linear warmup,
        gradient accumulation, AMP, EMA updates, mosaic closing, periodic
        validation, and checkpoint saving.

        Returns:
            dict: Final evaluation metrics including mAP50 and mAP50_95.
        """
        from swiftdet.data.dataset import COCODetectionDataset, detection_collate_fn
        from swiftdet.engine.evaluator import DetectionEvaluator

        os.makedirs(self.save_dir, exist_ok=True)

        # Move model and loss to device
        model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)
        model.train()

        # Build optimizer and EMA
        optimizer = self._build_optimizer()
        ema = ModelEMA(model, decay=0.9999, warmup_steps=2000)

        # AMP scaler (handles both CUDA and CPU gracefully)
        use_amp = self.amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # Build training dataloader
        train_dataset = COCODetectionDataset(
            self.data_yaml,
            split="train",
            img_size=self.img_size,
            augment=True,
            mosaic=1.0,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=self.device.type == "cuda",
            collate_fn=detection_collate_fn,
            drop_last=True,
        )

        # Build evaluator for validation
        evaluator = DetectionEvaluator(
            model=ema.ema_model,
            data_yaml=self.data_yaml,
            batch_size=self.batch_size,
            img_size=self.img_size,
            device=str(self.device),
        )

        # Resume from checkpoint if requested
        start_epoch = 0
        best_map = 0.0
        if self.resume:
            ckpt_path = self.save_dir / "last.pt"
            if ckpt_path.exists():
                start_epoch, best_map = self.load_checkpoint(
                    ckpt_path, model, optimizer, ema, scaler
                )

        n_batches = len(train_loader)

        # --- Print training configuration ---
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\n" + "=" * 60)
        print("SwiftDet Training")
        print("=" * 60)
        print(f"  Model:          {model.__class__.__name__} "
              f"({total_params / 1e6:.2f}M params, {trainable_params / 1e6:.2f}M trainable)")
        print(f"  Dataset:        {len(train_dataset)} train images, "
              f"{train_dataset.nc} classes")
        if train_dataset.names:
            names_preview = list(train_dataset.names.values())[:5]
            suffix = f", ... ({train_dataset.nc} total)" if train_dataset.nc > 5 else ""
            print(f"  Classes:        {names_preview}{suffix}")
        print(f"  Image size:     {self.img_size}")
        print(f"  Epochs:         {self.epochs}")
        print(f"  Batch size:     {self.batch_size} (x{self.grad_accum} accum = "
              f"{self.batch_size * self.grad_accum} effective)")
        print(f"  Optimizer:      {self.optimizer_type.upper()} lr={self.lr0} → {self.lr0 * self.lrf}")
        print(f"  AMP:            {use_amp}")
        print(f"  Device:         {self.device}")
        print(f"  Mosaic closing: last {self.close_mosaic} epochs")
        print(f"  Save dir:       {self.save_dir}")
        if self.resume and start_epoch > 0:
            print(f"  Resumed:        epoch {start_epoch}, best mAP={best_map:.4f}")
        print("=" * 60 + "\n")

        # --- Main Training Loop ---
        for epoch in range(start_epoch, self.epochs):
            model.train()
            epoch_losses = {"cls_loss": 0.0, "box_loss": 0.0, "dfl_loss": 0.0}

            # Mosaic closing: disable mosaic augmentation for last N epochs
            use_mosaic = epoch < (self.epochs - self.close_mosaic)
            if hasattr(train_dataset, "set_mosaic"):
                train_dataset.set_mosaic(use_mosaic)

            # Compute LR for this epoch via cosine schedule
            lr_factor = self._cosine_lr(epoch)

            t_start = time.time()
            optimizer.zero_grad()

            # Running averages for progress bar
            running_cls = 0.0
            running_box = 0.0
            running_dfl = 0.0

            pbar = tqdm(
                enumerate(train_loader),
                total=n_batches,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                bar_format="{l_bar}{bar:20}{r_bar}",
            )

            for batch_idx, (images, targets) in pbar:
                # Apply warmup during initial epochs
                self._warmup_lr_and_momentum(epoch, batch_idx, n_batches, optimizer)

                # After warmup, apply cosine schedule
                if epoch * n_batches + batch_idx >= self.warmup_epochs * n_batches:
                    for pg in optimizer.param_groups:
                        pg["lr"] = self.lr0 * lr_factor

                images = images.to(self.device, non_blocking=True).float()
                if images.ndim == 3:
                    images = images.unsqueeze(0)
                if images.max() > 1.0:
                    images = images / 255.0

                targets = targets.to(self.device, non_blocking=True)

                # Forward pass with optional AMP
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images)
                    loss, loss_dict = self._compute_loss(outputs, targets)
                    loss = loss / self.grad_accum

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Optimizer step (when accumulation is complete)
                if (batch_idx + 1) % self.grad_accum == 0 or (batch_idx + 1) == n_batches:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    ema.update(model)

                # Accumulate epoch losses
                for k in epoch_losses:
                    epoch_losses[k] += loss_dict.get(k, 0.0)

                # Update running averages and progress bar
                n = batch_idx + 1
                running_cls = epoch_losses["cls_loss"] / n
                running_box = epoch_losses["box_loss"] / n
                running_dfl = epoch_losses["dfl_loss"] / n
                mem = f"{torch.cuda.memory_reserved(self.device) / 1e9:.1f}G" if self.device.type == "cuda" else ""
                pbar.set_postfix_str(
                    f"cls={running_cls:.4f} box={running_box:.4f} dfl={running_dfl:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} {mem}"
                )

            pbar.close()

            # Average losses over batches
            for k in epoch_losses:
                epoch_losses[k] /= max(n_batches, 1)

            elapsed = time.time() - t_start

            # --- Validation ---
            metrics = {}
            do_val = (epoch + 1) % 10 == 0 or (epoch + 1) == self.epochs
            if do_val:
                evaluator.model = ema.ema_model
                metrics = evaluator.evaluate()

                current_map = metrics.get("mAP50_95", 0.0)

                if current_map > best_map:
                    best_map = current_map
                    self.save_checkpoint(
                        self.save_dir / "best.pt",
                        epoch=epoch,
                        best_map=best_map,
                        metrics=metrics,
                        model=model,
                        optimizer=optimizer,
                        ema=ema,
                        scaler=scaler,
                    )

            # Save last checkpoint every epoch
            self.save_checkpoint(
                self.save_dir / "last.pt",
                epoch=epoch,
                best_map=best_map,
                metrics=metrics,
                model=model,
                optimizer=optimizer,
                ema=ema,
                scaler=scaler,
            )

            # Epoch summary
            lr_current = optimizer.param_groups[0]["lr"]
            msg = (
                f"  >> cls={epoch_losses['cls_loss']:.4f} "
                f"box={epoch_losses['box_loss']:.4f} "
                f"dfl={epoch_losses['dfl_loss']:.4f} | "
                f"lr={lr_current:.6f} | {elapsed:.1f}s"
            )
            if metrics:
                msg += (
                    f" | mAP50={metrics.get('mAP50', 0.0):.4f}"
                    f" mAP50-95={metrics.get('mAP50_95', 0.0):.4f}"
                )
            print(msg)

        print(f"Training complete. Best mAP50-95: {best_map:.4f}")
        return metrics

    def save_checkpoint(self, path, epoch, best_map, metrics, model, optimizer, ema, scaler):
        """Save a training checkpoint to disk.

        The checkpoint contains model weights, optimizer state, EMA state,
        AMP scaler state, and metadata needed for resuming training.

        Args:
            path: File path for the checkpoint.
            epoch: Current epoch number.
            best_map: Best mAP achieved so far.
            metrics: Latest evaluation metrics dict.
            model: The training model.
            optimizer: The optimizer.
            ema: The ModelEMA instance.
            scaler: The AMP GradScaler.
        """
        checkpoint = {
            "epoch": epoch,
            "best_map": best_map,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, model, optimizer, ema, scaler):
        """Load a training checkpoint for resuming.

        Restores model weights, optimizer state, EMA state, and scaler state
        from a previously saved checkpoint.

        Args:
            path: File path to the checkpoint.
            model: The model to load weights into.
            optimizer: The optimizer to restore state for.
            ema: The ModelEMA instance to restore.
            scaler: The AMP GradScaler to restore.

        Returns:
            Tuple of (start_epoch, best_map) for resuming the training loop.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        ema.load_state_dict(checkpoint["ema_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_map = checkpoint.get("best_map", 0.0)

        print(f"Resumed from epoch {start_epoch}, best mAP50-95: {best_map:.4f}")
        return start_epoch, best_map
