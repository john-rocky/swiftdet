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
from tqdm import tqdm

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

        # Diagnostic counters
        total_dets = 0
        total_gts = 0
        max_conf_seen = 0.0
        first_batch_logged = False

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validating", bar_format="{l_bar}{bar:20}{r_bar}"):
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

                # Diagnostic: log first batch stats
                if not first_batch_logged:
                    logits = outputs["cls"]
                    boxes = outputs["box_decoded"]
                    max_score = cls_scores.max().item()
                    mean_score = cls_scores.max(dim=-1).values.mean().item()
                    logit_range = (logits.min().item(), logits.max().item())
                    box_range = (boxes.min().item(), boxes.max().item())
                    above_thresh = (cls_scores.max(dim=-1).values > self.conf_thres).sum().item()
                    total_anchors = cls_scores.shape[1]
                    print(
                        f"  [Eval debug] logit_range=({logit_range[0]:.2f}, {logit_range[1]:.2f})"
                        f" max_score={max_score:.4f} mean_max_score={mean_score:.4f}"
                        f" anchors_above_thresh={above_thresh}/{total_anchors * cls_scores.shape[0]}"
                        f" box_range=({box_range[0]:.1f}, {box_range[1]:.1f})"
                    )
                    # Show sample boxes for coordinate sanity check
                    b0_scores, b0_idx = cls_scores[0].max(dim=-1).values.topk(3)
                    for rank, (score_val, anc_idx) in enumerate(zip(b0_scores, b0_idx)):
                        box = boxes[0, anc_idx].tolist()
                        cls_id = cls_scores[0, anc_idx].argmax().item()
                        print(
                            f"    top-{rank+1} pred: box=[{box[0]:.1f},{box[1]:.1f},"
                            f"{box[2]:.1f},{box[3]:.1f}] cls={cls_id} score={score_val:.4f}"
                        )
                    # Show GT boxes for comparison
                    gt0 = targets[0]
                    gt0_valid = gt0[gt0[:, 1:5].sum(dim=-1) > 0]
                    for gi in range(min(3, gt0_valid.shape[0])):
                        gt_box = gt0_valid[gi, 1:5].tolist()
                        gt_cls = int(gt0_valid[gi, 0].item())
                        print(
                            f"    gt-{gi+1}:   box=[{gt_box[0]:.1f},{gt_box[1]:.1f},"
                            f"{gt_box[2]:.1f},{gt_box[3]:.1f}] cls={gt_cls}"
                        )
                    first_batch_logged = True

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

                # Track detection counts
                for det in detections:
                    n_det = det.shape[0]
                    total_dets += n_det
                    if n_det > 0:
                        batch_max = det[:, 4].max().item()
                        if batch_max > max_conf_seen:
                            max_conf_seen = batch_max

                # Convert targets to per-image list format for APMetrics
                # targets: (B, max_gt, 5) where each row is [class_id, x1, y1, x2, y2]
                batch_targets = []
                for b in range(targets.shape[0]):
                    gt = targets[b]  # (max_gt, 5)
                    # Filter out zero-padded entries
                    valid_mask = gt[:, 1:5].sum(dim=-1) > 0
                    gt_valid = gt[valid_mask]
                    total_gts += gt_valid.shape[0]
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

        print(
            f"  [Eval summary] total_dets={total_dets} total_gts={total_gts}"
            f" max_conf={max_conf_seen:.4f}"
        )

        # Compute final mAP metrics
        results = ap_metrics.compute()
        print(
            f"Evaluation: mAP50={results['mAP50']:.4f}, "
            f"mAP50-95={results['mAP50_95']:.4f}"
        )
        return results
