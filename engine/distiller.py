# SwiftDet - Knowledge distillation trainer
# MIT License - Original implementation
#
# References:
#   - Hinton et al. 2015, "Distilling the Knowledge in a Neural Network"
#   - Romero et al. 2015, "FitNets: Hints for Thin Deep Nets"
#   - Cosine annealing: Loshchilov & Hutter 2017, "SGDR"
#   - EMA: Polyak & Juditsky 1992; Izmailov et al. 2018 (SWA)

import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from swiftdet.losses.distill import FeatureDistillLoss, LogitDistillLoss
from swiftdet.losses.dfl import BboxLoss
from swiftdet.losses.focal import VarifocalLoss
from swiftdet.utils.assigner import TaskAlignedAssigner
from swiftdet.utils.ema import ModelEMA


class DistillationTrainer:
    """Knowledge distillation trainer transferring knowledge from teacher to student.

    Combines standard detection training loss with feature-level and logit-level
    distillation losses. The teacher model is frozen and its outputs guide the
    student via:

    1. Feature distillation (Romero et al. 2015): MSE alignment of neck features.
    2. Logit distillation (Hinton et al. 2015): KL divergence on temperature-scaled
       classification logits plus L1 on regression outputs.

    The total loss is:
        L = L_det + feature_weight * L_feat + logit_weight * L_logit

    Args:
        student: SwiftDetector student model to be trained.
        teacher: SwiftDetector pre-trained teacher model (will be frozen).
        data_yaml: Path to dataset YAML configuration.
        epochs: Number of distillation training epochs (default 200).
        batch_size: Training batch size (default 64).
        img_size: Input image size in pixels (default 640).
        feature_weight: Weight for feature distillation loss (default 0.5).
        logit_weight: Weight for logit distillation loss (default 0.5).
        temperature: KD temperature for logit softening (default 4.0).
        lr0: Initial learning rate (default 0.01).
        lrf: Final LR as fraction of lr0 (default 0.01).
        optimizer: 'sgd' or 'adamw' (default 'sgd').
        momentum: SGD momentum (default 0.937).
        weight_decay: L2 regularization coefficient (default 0.0005).
        warmup_epochs: Linear warmup epochs (default 5).
        close_mosaic: Disable mosaic for last N epochs (default 15).
        amp: Enable automatic mixed precision (default True).
        device: Training device string (default None for auto).
        save_dir: Directory for saving checkpoints (default 'runs/distill').
        cls_gain: Classification loss weight (default 0.5).
        box_gain: Box regression loss weight (default 7.5).
        dfl_gain: DFL loss weight (default 1.5).
    """

    def __init__(
        self,
        student,
        teacher,
        data_yaml,
        epochs=200,
        batch_size=64,
        img_size=640,
        feature_weight=0.5,
        logit_weight=0.5,
        temperature=4.0,
        lr0=0.01,
        lrf=0.01,
        optimizer="sgd",
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        close_mosaic=15,
        amp=True,
        device=None,
        save_dir="runs/distill",
        cls_gain=0.5,
        box_gain=7.5,
        dfl_gain=1.5,
    ):
        self.student = student
        self.teacher = teacher
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.feature_weight = feature_weight
        self.logit_weight = logit_weight
        self.temperature = temperature
        self.lr0 = lr0
        self.lrf = lrf
        self.optimizer_type = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.close_mosaic = close_mosaic
        self.amp = amp
        self.save_dir = Path(save_dir)
        self.cls_gain = cls_gain
        self.box_gain = box_gain
        self.dfl_gain = dfl_gain

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Detection losses
        self.cls_loss_fn = VarifocalLoss()
        self.box_loss_fn = BboxLoss(reg_max=student.reg_max)
        self.assigner = TaskAlignedAssigner(topk=13, alpha=1.0, beta=6.0)

        # Auto gradient accumulation targeting nominal batch of 64
        self.grad_accum = max(1, 64 // batch_size)

    def _freeze_teacher(self):
        """Freeze all teacher parameters and set to eval mode.

        The teacher's weights remain fixed throughout distillation. Only the
        student model receives gradient updates.
        """
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def _build_optimizer(self):
        """Construct optimizer for student model with per-parameter-group weight decay.

        Returns:
            torch.optim.Optimizer instance.
        """
        decay_params = []
        no_decay_params = []
        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "bn" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if self.optimizer_type == "adamw":
            return torch.optim.AdamW(param_groups, lr=self.lr0, betas=(0.9, 0.999))
        else:
            return torch.optim.SGD(
                param_groups,
                lr=self.lr0,
                momentum=self.momentum,
                nesterov=True,
            )

    def _cosine_lr(self, epoch):
        """Compute cosine annealing LR factor.

        Args:
            epoch: Current epoch (0-based).

        Returns:
            LR multiplier in [lrf, 1.0].
        """
        return self.lrf + 0.5 * (1.0 - self.lrf) * (
            1.0 + math.cos(math.pi * epoch / self.epochs)
        )

    def _warmup_lr(self, epoch, batch_idx, n_batches, optimizer):
        """Apply linear warmup to learning rate during initial epochs.

        Args:
            epoch: Current epoch (0-based).
            batch_idx: Current batch index.
            n_batches: Total batches per epoch.
            optimizer: Optimizer to update.
        """
        total_warmup_iters = self.warmup_epochs * n_batches
        current_iter = epoch * n_batches + batch_idx

        if current_iter >= total_warmup_iters:
            return

        xi = current_iter / max(total_warmup_iters, 1)
        for pg in optimizer.param_groups:
            pg["lr"] = self.lr0 * xi
        if "momentum" in optimizer.param_groups[0]:
            for pg in optimizer.param_groups:
                pg["momentum"] = 0.8 * (1.0 - xi) + self.momentum * xi

    def _compute_detection_loss(self, outputs, targets):
        """Compute the standard detection loss for the student.

        Uses the same loss formulation as DetectionTrainer: Varifocal Loss for
        classification, CIoU + DFL for box regression, with task-aligned assignment.

        Args:
            outputs: Dict from student model forward pass.
            targets: Ground truth tensor (B, max_gt, 5).

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        cls_raw = outputs["cls"]
        box_dist = outputs["box_dist"]
        box_decoded = outputs["box_decoded"]
        anchors = outputs["anchors"]
        strides = outputs["strides"]

        B, N, nc = cls_raw.shape
        device = cls_raw.device

        gt_labels = targets[:, :, 0:1].long()
        gt_bboxes = targets[:, :, 1:5]
        mask_gt = (gt_bboxes.sum(dim=-1, keepdim=True) > 0).float()

        cls_scores_for_assign = cls_raw.detach().sigmoid()
        target_labels, target_bboxes, target_scores, fg_mask = self.assigner.forward(
            cls_scores=cls_scores_for_assign,
            bbox_preds=box_decoded.detach(),
            anchor_points=anchors,
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes,
            mask_gt=mask_gt,
        )

        target_scores_sum = target_scores.sum().clamp(min=1.0)

        cls_loss = self.cls_loss_fn(cls_raw, target_scores) / target_scores_sum

        if fg_mask.any():
            iou_loss, dfl_loss = self.box_loss_fn(
                pred_dist=box_dist,
                pred_bboxes=box_decoded,
                anchor_points=anchors,
                target_bboxes=target_bboxes,
                target_scores=target_scores,
                target_scores_sum=target_scores_sum,
                fg_mask=fg_mask,
                stride_tensor=strides,
            )
        else:
            iou_loss = torch.tensor(0.0, device=device)
            dfl_loss = torch.tensor(0.0, device=device)

        total_loss = (
            self.cls_gain * cls_loss
            + self.box_gain * iou_loss
            + self.dfl_gain * dfl_loss
        )

        loss_dict = {
            "cls_loss": cls_loss.detach().item(),
            "box_loss": iou_loss.detach().item(),
            "dfl_loss": dfl_loss.detach().item(),
        }

        return total_loss, loss_dict, fg_mask

    def _forward_with_features(self, model, images):
        """Run model forward pass and capture intermediate neck features.

        Performs a full forward pass through backbone, neck, and head,
        returning both the head outputs and the neck feature maps needed
        for feature distillation.

        Args:
            model: SwiftDetector model.
            images: Input image tensor (B, 3, H, W).

        Returns:
            Tuple of (head_outputs_dict, neck_features_list).
        """
        backbone_features = model.backbone(images)
        neck_features = model.neck(backbone_features)
        head_outputs = model.head(neck_features)
        return head_outputs, neck_features

    def train(self):
        """Run the distillation training loop.

        Trains the student model using a combination of standard detection loss
        and distillation losses from the frozen teacher model. Uses cosine
        annealing LR, gradient accumulation, AMP, and EMA.

        Returns:
            dict: Final evaluation metrics including mAP50 and mAP50_95.
        """
        from swiftdet.data.dataset import COCODetectionDataset, detection_collate_fn
        from swiftdet.engine.evaluator import DetectionEvaluator

        os.makedirs(self.save_dir, exist_ok=True)

        # Move models to device
        student = self.student.to(self.device)
        teacher = self.teacher.to(self.device)

        # Freeze teacher
        self._freeze_teacher()

        student.train()

        # Build optimizer and EMA for student
        optimizer = self._build_optimizer()
        ema = ModelEMA(student, decay=0.9999, warmup_steps=2000)

        # AMP scaler
        use_amp = self.amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # Build distillation loss modules
        # Determine channel dimensions for feature distillation adaptation layers
        student_neck_channels = self._get_neck_channels(student)
        teacher_neck_channels = self._get_neck_channels(teacher)

        feat_distill_loss = FeatureDistillLoss(
            student_channels=student_neck_channels,
            teacher_channels=teacher_neck_channels,
            normalize=True,
        ).to(self.device)

        logit_distill_loss = LogitDistillLoss(
            temperature=self.temperature,
            reg_weight=1.0,
        )

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

        # Build evaluator
        evaluator = DetectionEvaluator(
            model=ema.ema_model,
            data_yaml=self.data_yaml,
            batch_size=self.batch_size,
            img_size=self.img_size,
            device=str(self.device),
        )

        n_batches = len(train_loader)
        best_map = 0.0

        # --- Main Distillation Training Loop ---
        for epoch in range(self.epochs):
            student.train()
            epoch_losses = {
                "cls_loss": 0.0,
                "box_loss": 0.0,
                "dfl_loss": 0.0,
                "feat_loss": 0.0,
                "logit_loss": 0.0,
            }

            # Mosaic closing for last N epochs
            use_mosaic = epoch < (self.epochs - self.close_mosaic)
            if hasattr(train_dataset, "set_mosaic"):
                train_dataset.set_mosaic(use_mosaic)

            lr_factor = self._cosine_lr(epoch)
            t_start = time.time()
            optimizer.zero_grad()

            for batch_idx, (images, targets) in enumerate(train_loader):
                # Warmup
                self._warmup_lr(epoch, batch_idx, n_batches, optimizer)

                # Apply cosine schedule after warmup
                if epoch * n_batches + batch_idx >= self.warmup_epochs * n_batches:
                    for pg in optimizer.param_groups:
                        pg["lr"] = self.lr0 * lr_factor

                images = images.to(self.device, non_blocking=True).float()
                if images.max() > 1.0:
                    images = images / 255.0
                targets = targets.to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    # Forward pass through both teacher and student with features
                    with torch.no_grad():
                        teacher_outputs, teacher_neck_feats = self._forward_with_features(
                            teacher, images
                        )

                    student_outputs, student_neck_feats = self._forward_with_features(
                        student, images
                    )

                    # 1. Standard detection loss (student only)
                    det_loss, det_loss_dict, fg_mask = self._compute_detection_loss(
                        student_outputs, targets
                    )

                    # 2. Feature distillation loss (neck feature alignment)
                    feat_loss = feat_distill_loss(student_neck_feats, teacher_neck_feats)

                    # 3. Logit distillation loss (head output alignment)
                    logit_loss = logit_distill_loss(
                        student_cls=student_outputs["cls"],
                        student_reg=student_outputs["box_dist"],
                        teacher_cls=teacher_outputs["cls"],
                        teacher_reg=teacher_outputs["box_dist"],
                        fg_mask=fg_mask,
                    )

                    # Combined loss
                    total_loss = (
                        det_loss
                        + self.feature_weight * feat_loss
                        + self.logit_weight * logit_loss
                    )
                    total_loss = total_loss / self.grad_accum

                # Backward
                scaler.scale(total_loss).backward()

                # Optimizer step on accumulation boundary
                if (batch_idx + 1) % self.grad_accum == 0 or (batch_idx + 1) == n_batches:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    ema.update(student)

                # Accumulate losses for logging
                for k in det_loss_dict:
                    epoch_losses[k] += det_loss_dict[k]
                epoch_losses["feat_loss"] += feat_loss.detach().item()
                epoch_losses["logit_loss"] += logit_loss.detach().item()

            # Average epoch losses
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
                    self._save_checkpoint(
                        self.save_dir / "best.pt", epoch, best_map, metrics,
                        student, optimizer, ema, scaler,
                    )

            # Save last checkpoint
            self._save_checkpoint(
                self.save_dir / "last.pt", epoch, best_map, metrics,
                student, optimizer, ema, scaler,
            )

            # Print progress
            lr_current = optimizer.param_groups[0]["lr"]
            msg = (
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"cls={epoch_losses['cls_loss']:.4f} "
                f"box={epoch_losses['box_loss']:.4f} "
                f"dfl={epoch_losses['dfl_loss']:.4f} "
                f"feat={epoch_losses['feat_loss']:.4f} "
                f"logit={epoch_losses['logit_loss']:.4f} | "
                f"lr={lr_current:.6f} | time={elapsed:.1f}s"
            )
            if metrics:
                msg += (
                    f" | mAP50={metrics.get('mAP50', 0.0):.4f}"
                    f" mAP50-95={metrics.get('mAP50_95', 0.0):.4f}"
                )
            print(msg)

        print(f"Distillation complete. Best mAP50-95: {best_map:.4f}")
        return metrics

    @staticmethod
    def _get_neck_channels(model):
        """Extract the output channel dimensions from a model's neck.

        Inspects the neck module to determine per-level output channel counts,
        which are needed for constructing feature distillation adaptation layers.

        Args:
            model: SwiftDetector model instance.

        Returns:
            List of integers, one channel count per feature level.
        """
        neck = model.neck
        if hasattr(neck, "out_channels"):
            # If the neck explicitly stores output channels
            out_ch = neck.out_channels
            if isinstance(out_ch, int):
                # BiFPNLite uses a single unified channel width
                n_levels = len(model.backbone.out_channels)
                return [out_ch] * n_levels
            return list(out_ch)
        # Fallback: use the detection head's input channels
        head = model.head
        return [head.stems[i].conv.in_channels for i in range(head.n_levels)]

    @staticmethod
    def _save_checkpoint(path, epoch, best_map, metrics, model, optimizer, ema, scaler):
        """Save a distillation training checkpoint.

        Args:
            path: File path for checkpoint.
            epoch: Current epoch number.
            best_map: Best mAP achieved so far.
            metrics: Latest evaluation metrics dict.
            model: The student model.
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
