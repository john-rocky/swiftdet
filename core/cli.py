"""SwiftDet command-line interface.

MIT License - Original implementation.

Usage::

    swiftdet train  --model swiftdet-n --data coco.yaml --epochs 500
    swiftdet val    --model best.pt --data coco.yaml
    swiftdet predict --model best.pt --source image.jpg --conf 0.25
    swiftdet export --model best.pt --format coreml
    swiftdet distill --teacher swiftdet-l --student swiftdet-n --data coco.yaml
    swiftdet pretrain --model swiftdet-n --data imagenet --epochs 100
"""

import argparse
import sys


def _add_train_parser(subparsers):
    """Register the ``train`` subcommand."""
    p = subparsers.add_parser("train", help="Train a SwiftDet model")
    p.add_argument("--model", type=str, default="swiftdet-n", help="Model variant, YAML, or .pt checkpoint")
    p.add_argument("--data", type=str, required=True, help="Path to data YAML")
    p.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    p.add_argument("--batch", type=int, default=64, help="Batch size")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size")
    p.add_argument("--device", type=str, default=None, help="Device (e.g. cuda:0, mps, cpu)")
    p.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw", "adam"], help="Optimizer")
    p.add_argument("--amp", action="store_true", default=True, help="Use automatic mixed precision")
    p.add_argument("--no-amp", action="store_false", dest="amp", help="Disable AMP")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return p


def _add_val_parser(subparsers):
    """Register the ``val`` subcommand."""
    p = subparsers.add_parser("val", help="Validate a SwiftDet model")
    p.add_argument("--model", type=str, required=True, help="Model checkpoint (.pt)")
    p.add_argument("--data", type=str, default=None, help="Path to data YAML")
    p.add_argument("--batch", type=int, default=32, help="Batch size")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size")
    p.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    return p


def _add_predict_parser(subparsers):
    """Register the ``predict`` subcommand."""
    p = subparsers.add_parser("predict", help="Run inference with a SwiftDet model")
    p.add_argument("--model", type=str, required=True, help="Model checkpoint (.pt)")
    p.add_argument("--source", type=str, required=True, help="Image, directory, or video path")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size")
    p.add_argument("--max-det", type=int, default=300, help="Max detections per image")
    p.add_argument("--save", action="store_true", help="Save annotated images")
    p.add_argument("--show", action="store_true", help="Display annotated images")
    return p


def _add_export_parser(subparsers):
    """Register the ``export`` subcommand."""
    p = subparsers.add_parser("export", help="Export a SwiftDet model")
    p.add_argument("--model", type=str, required=True, help="Model checkpoint (.pt)")
    p.add_argument("--format", type=str, default="coreml", choices=["coreml", "onnx"], help="Export format")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size")
    p.add_argument("--half", action="store_true", default=True, help="FP16 quantization (CoreML)")
    p.add_argument("--no-half", action="store_false", dest="half", help="Keep FP32")
    p.add_argument("--int8", action="store_true", help="INT8 quantization")
    p.add_argument("--nms", action="store_true", help="Include NMS in exported model")
    return p


def _add_distill_parser(subparsers):
    """Register the ``distill`` subcommand."""
    p = subparsers.add_parser("distill", help="Knowledge distillation training")
    p.add_argument("--teacher", type=str, required=True, help="Teacher model (variant or .pt)")
    p.add_argument("--student", type=str, default="swiftdet-n", help="Student model")
    p.add_argument("--data", type=str, required=True, help="Path to data YAML")
    p.add_argument("--epochs", type=int, default=200, help="Training epochs")
    p.add_argument("--feature-weight", type=float, default=0.5, help="Feature distillation weight")
    p.add_argument("--logit-weight", type=float, default=0.5, help="Logit distillation weight")
    p.add_argument("--temperature", type=float, default=4.0, help="Distillation temperature")
    p.add_argument("--batch", type=int, default=64, help="Batch size")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size")
    p.add_argument("--lr0", type=float, default=0.005, help="Initial learning rate")
    p.add_argument("--device", type=str, default=None, help="Device")
    return p


def _add_pretrain_parser(subparsers):
    """Register the ``pretrain`` subcommand."""
    p = subparsers.add_parser("pretrain", help="Pre-train backbone on classification data")
    p.add_argument("--model", type=str, default="swiftdet-n", help="Model variant or YAML")
    p.add_argument("--data", type=str, required=True, help="Path to classification dataset")
    p.add_argument("--epochs", type=int, default=100, help="Pre-training epochs")
    p.add_argument("--batch", type=int, default=256, help="Batch size")
    p.add_argument("--lr0", type=float, default=0.1, help="Initial learning rate")
    p.add_argument("--device", type=str, default=None, help="Device")
    return p


def main():
    """CLI entry point — dispatches to the appropriate SwiftDet method."""
    parser = argparse.ArgumentParser(
        prog="swiftdet",
        description="SwiftDet — Fast and accurate object detection (MIT License)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _add_train_parser(subparsers)
    _add_val_parser(subparsers)
    _add_predict_parser(subparsers)
    _add_export_parser(subparsers)
    _add_distill_parser(subparsers)
    _add_pretrain_parser(subparsers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    from .model import SwiftDet

    # ---------- train ---------- #
    if args.command == "train":
        model = SwiftDet(args.model)
        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            lr0=args.lr0,
            optimizer=args.optimizer,
            amp=args.amp,
            resume=args.resume,
        )

    # ---------- val ---------- #
    elif args.command == "val":
        model = SwiftDet(args.model)
        metrics = model.val(
            data=args.data,
            batch=args.batch,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
        )
        print("Validation metrics:", metrics)

    # ---------- predict ---------- #
    elif args.command == "predict":
        model = SwiftDet(args.model)
        results = model.predict(
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            save=args.save,
            show=args.show,
        )
        for r in results:
            print(r)

    # ---------- export ---------- #
    elif args.command == "export":
        model = SwiftDet(args.model)
        path = model.export(
            format=args.format,
            imgsz=args.imgsz,
            half=args.half,
            int8=args.int8,
            nms=args.nms,
        )
        print(f"Exported to: {path}")

    # ---------- distill ---------- #
    elif args.command == "distill":
        student = SwiftDet(args.student)
        teacher = SwiftDet(args.teacher)
        student.distill(
            teacher=teacher,
            data=args.data,
            epochs=args.epochs,
            feature_weight=args.feature_weight,
            logit_weight=args.logit_weight,
            temperature=args.temperature,
            batch=args.batch,
            imgsz=args.imgsz,
            lr0=args.lr0,
        )

    # ---------- pretrain ---------- #
    elif args.command == "pretrain":
        model = SwiftDet(args.model)
        model.pretrain(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            lr0=args.lr0,
        )


if __name__ == "__main__":
    main()
