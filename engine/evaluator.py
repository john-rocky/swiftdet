# SwiftDet - COCO-style detection evaluator
# MIT License - Original implementation
#
# References:
#   - COCO evaluation protocol: Lin et al. 2014, "Microsoft COCO: Common
#     Objects in Context"
#   - NMS: Neubeck & Van Gool 2006, "Efficient Non-Maximum Suppression"
#   - DFL decoding: Li et al. 2020, "Generalized Focal Loss"

import os

import torch

from swiftdet.utils.metrics import APMetrics
from swiftdet.utils.nms import non_max_suppression


class DetectionEvaluator:
    """COCO-style detection evaluation.

    Evaluates a detection model on a validation dataset using the standard
    COCO protocol (Lin et al. 2014): 101-point interpolated precision-recall
    across IoU thresholds from 0.50 to 0.95 in steps of 0.05.

    Args:
        model: SwiftDetector model (or EMA model) in eval mode.
        data_yaml: Path to dataset YAML configuration file.
        batch_size: Evaluation batch size (default 32).
        img_size: Input image size in pixels (default 640).
        conf_thres: Confidence threshold for filtering detections (default 0.001).
        iou_thres: NMS IoU threshold (default 0.65).
        device: Evaluation device string (default None for auto-detect).
        max_det: Maximum detections per image after NMS (default 300).
    """

    def __init__(
        self,
        model,
        data_yaml,
        batch_size=32,
        img_size=640,
        conf_thres=0.001,
        iou_thres=0.65,
        device=None,
        max_det=300,
    ):
        self.model = model
        self.data_yaml = data_yaml
        self.batch_size = batch_size
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def evaluate(self):
        """Run evaluation on the validation set.

        Iterates through the validation dataloader, runs inference with no
        gradient computation, applies NMS, and accumulates COCO-style AP
        metrics.

        Returns:
            dict with keys:
                'mAP50': Mean AP at IoU threshold 0.50.
                'mAP50_95': Mean AP averaged over IoU thresholds 0.50:0.05:0.95.
                'ap_per_class': Per-class, per-threshold AP ndarray of shape
                    (nc, n_thresholds).
        """
        from swiftdet.data.dataset import COCODetectionDataset, detection_collate_fn

        model = self.model.to(self.device)
        model.eval()

        # Build validation dataloader (no augmentation, letterbox + normalize only)
        val_dataset = COCODetectionDataset(
            self.data_yaml,
            split="val",
            img_size=self.img_size,
            augment=False,
            mosaic=False,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=self.device.type == "cuda",
            collate_fn=detection_collate_fn,
            drop_last=False,
        )

        nc = model.nc if hasattr(model, "nc") else 80
        ap_metrics = APMetrics(nc=nc)

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device, non_blocking=True).float()
                if images.max() > 1.0:
                    images = images / 255.0

                targets = targets.to(self.device, non_blocking=True)

                # Forward pass
                outputs = model(images)

                # Build NMS input: concatenate decoded boxes and class scores
                # outputs['box_decoded']: (B, N, 4) xyxy format
                # outputs['cls']: (B, N, nc) raw logits -> sigmoid for confidence
                cls_scores = outputs["cls"].sigmoid()
                nms_input = torch.cat(
                    [outputs["box_decoded"], cls_scores], dim=-1
                )  # (B, N, nc+4)

                # Apply NMS per image
                detections = non_max_suppression(
                    nms_input,
                    conf_thres=self.conf_thres,
                    iou_thres=self.iou_thres,
                    max_det=self.max_det,
                )

                # Convert targets to per-image list format for APMetrics
                # targets: (B, max_gt, 5) where each row is [class_id, x1, y1, x2, y2]
                batch_targets = []
                for b in range(targets.shape[0]):
                    gt = targets[b]  # (max_gt, 5)
                    # Filter out zero-padded entries
                    valid_mask = gt[:, 1:5].sum(dim=-1) > 0
                    gt_valid = gt[valid_mask]
                    if gt_valid.numel() > 0:
                        # Reformat to [x1, y1, x2, y2, class_id]
                        gt_reformatted = torch.cat(
                            [gt_valid[:, 1:5], gt_valid[:, 0:1]], dim=-1
                        )
                    else:
                        gt_reformatted = torch.zeros((0, 5), device=self.device)
                    batch_targets.append(gt_reformatted)

                # Accumulate predictions and ground truths
                ap_metrics.update(detections, batch_targets)

        # Compute final mAP metrics
        results = ap_metrics.compute()
        print(
            f"Evaluation: mAP50={results['mAP50']:.4f}, "
            f"mAP50-95={results['mAP50_95']:.4f}"
        )
        return results
