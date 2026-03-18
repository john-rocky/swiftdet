"""Export SwiftDet models to ONNX format.

MIT License - Original implementation.
"""

import logging
from pathlib import Path

import torch

logger = logging.getLogger("swiftdet.export")


def export_onnx(
    model,
    img_size=640,
    opset=17,
    simplify=True,
    output_path=None,
    dynamic_batch=False,
):
    """Export a SwiftDet model to ONNX format.

    Uses ``torch.onnx.export`` with optional ``onnxsim`` graph simplification.

    Args:
        model:         PyTorch nn.Module (eval mode, CPU).
        img_size:      Input image size (int or tuple).
        opset:         ONNX opset version (default 17).
        simplify:      Simplify the graph with onnxsim (default True).
        output_path:   Output file path. Defaults to ``"swiftdet.onnx"``.
        dynamic_batch: Export with dynamic batch dimension.

    Returns:
        Path to the exported .onnx file.
    """
    if output_path is None:
        output_path = "swiftdet.onnx"
    output_path = Path(output_path)

    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    model.eval()
    model.cpu()

    dummy_input = torch.randn(1, 3, img_size[0], img_size[1])

    # Build dynamic axes if requested
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch"},
            "output": {0: "batch"},
        }

    logger.info(
        "Exporting ONNX (opset=%d, input=1x3x%dx%d)...",
        opset, img_size[0], img_size[1],
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    # Validate and optionally simplify
    try:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validated successfully.")
    except ImportError:
        logger.warning("onnx package not installed; skipping model validation.")
    except Exception as exc:
        logger.warning("ONNX validation warning: %s", exc)

    if simplify:
        try:
            import onnxsim

            logger.info("Simplifying ONNX graph with onnxsim...")
            onnx_model = onnx.load(str(output_path))
            simplified, ok = onnxsim.simplify(onnx_model)
            if ok:
                onnx.save(simplified, str(output_path))
                logger.info("ONNX graph simplified.")
            else:
                logger.warning("onnxsim simplification failed; keeping original graph.")
        except ImportError:
            logger.info("onnxsim not installed; skipping simplification.")
        except Exception as exc:
            logger.warning("onnxsim error: %s", exc)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("ONNX model saved to %s (%.1f MB)", output_path, file_size_mb)

    return str(output_path)
