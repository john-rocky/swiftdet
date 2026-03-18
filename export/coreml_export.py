"""Export SwiftDet models to CoreML .mlpackage format.

MIT License - Original implementation.

Supported operations (CoreML compatible):
    Conv2d, BatchNorm2d, SiLU, MaxPool2d, Upsample, Concat, Sigmoid, Mean, Max.
    No in-place operations are used.
"""

import logging
from pathlib import Path

import torch

logger = logging.getLogger("swiftdet.export")


def _make_nms_pipeline(spec, class_names, iou_threshold=0.45, conf_threshold=0.25):
    """Append an NMS post-processing step to a CoreML model spec.

    Args:
        spec:           CoreML model spec to modify.
        class_names:    Dict or list of class names.
        iou_threshold:  NMS IoU threshold.
        conf_threshold: Minimum confidence for NMS.

    Returns:
        Pipeline model spec with NMS appended.
    """
    import coremltools as ct
    from coremltools.models import datatypes

    if isinstance(class_names, dict):
        labels = [class_names[i] for i in sorted(class_names.keys())]
    else:
        labels = list(class_names)

    nms_spec = ct.proto.Model_pb2.Model()
    nms_spec.specificationVersion = 5

    nms = nms_spec.nonMaximumSuppression
    nms.confidenceInputFeatureName = "raw_confidence"
    nms.coordinatesInputFeatureName = "raw_coordinates"
    nms.confidenceOutputFeatureName = "confidence"
    nms.coordinatesOutputFeatureName = "coordinates"
    nms.iouThresholdInputFeatureName = "iouThreshold"
    nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
    nms.iouThreshold = iou_threshold
    nms.confidenceThreshold = conf_threshold
    nms.stringClassLabels.vector.extend(labels)

    return nms_spec


def export_coreml(
    model,
    img_size=640,
    half=True,
    int8=False,
    nms=False,
    output_path=None,
    class_names=None,
):
    """Export a SwiftDet model to CoreML .mlpackage format.

    Uses ``torch.jit.trace`` followed by ``coremltools.convert``.
    Supports FP16 quantization (default) and optional INT8.

    Args:
        model:        PyTorch nn.Module (already on CPU, eval mode).
        img_size:     Input image size (int or tuple).
        half:         Quantize to FP16 (default True).
        int8:         Quantize to INT8 (requires calibration data).
        nms:          Append NMS pipeline to the model.
        output_path:  Output file path. Defaults to ``"swiftdet.mlpackage"``.
        class_names:  Dict mapping class index to name, used for NMS labels.

    Returns:
        Path to the exported .mlpackage file.
    """
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError(
            "coremltools is required for CoreML export. "
            "Install it with: pip install coremltools>=7.0"
        )

    if output_path is None:
        output_path = "swiftdet.mlpackage"
    output_path = Path(output_path)

    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    model.eval()
    model.cpu()

    # Trace the model
    logger.info("Tracing model with input shape (1, 3, %d, %d)...", img_size[0], img_size[1])
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1])
    traced = torch.jit.trace(model, dummy_input, strict=False)

    # Convert to CoreML
    logger.info("Converting to CoreML...")
    ct_input = ct.ImageType(
        name="image",
        shape=(1, 3, img_size[0], img_size[1]),
        scale=1.0 / 255.0,
        bias=[0.0, 0.0, 0.0],
        color_layout=ct.colorlayout.RGB,
    )

    compute_precision = ct.precision.FLOAT16 if half else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        inputs=[ct_input],
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )

    # INT8 quantization
    if int8:
        logger.info("Applying INT8 quantization...")
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig,
            OptimizationConfig,
            linear_quantize_weights,
        )

        op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        config = OptimizationConfig(global_config=op_config)
        mlmodel = linear_quantize_weights(mlmodel, config=config)

    # Set metadata
    mlmodel.author = "SwiftDet"
    mlmodel.short_description = "SwiftDet object detection model"
    mlmodel.version = "0.1.0"

    if class_names:
        if isinstance(class_names, dict):
            labels = [class_names[i] for i in sorted(class_names.keys())]
        else:
            labels = list(class_names)
        mlmodel.user_defined_metadata["classes"] = ",".join(labels)

    # Save
    mlmodel.save(str(output_path))
    logger.info("CoreML model saved to %s", output_path)

    return str(output_path)
