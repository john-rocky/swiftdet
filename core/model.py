"""High-level SwiftDet model interface.

MIT License - Original implementation.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from .results import Boxes, Results

logger = logging.getLogger("swiftdet")

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_VID_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

_VARIANT_CONFIGS = {
    "swiftdet-n": "swiftdet_n.yaml",
    "swiftdet-s": "swiftdet_s.yaml",
    "swiftdet-m": "swiftdet_m.yaml",
    "swiftdet-l": "swiftdet_l.yaml",
    "swiftdet2-n": "swiftdet2_n.yaml",
    "swiftdet2-s": "swiftdet2_s.yaml",
    "swiftdet2-m": "swiftdet2_m.yaml",
    "swiftdet2-l": "swiftdet2_l.yaml",
}


def _configs_dir():
    """Return the path to the built-in configs directory."""
    return Path(__file__).resolve().parent.parent / "configs"


def _load_yaml(path):
    """Load a YAML file and return a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _letterbox(img, new_shape=640, color=(114, 114, 114)):
    """Resize and pad image to *new_shape* while preserving aspect ratio."""
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def _nms(boxes, scores, iou_threshold):
    """Pure-numpy greedy NMS."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]
    return np.array(keep, dtype=np.int64)


def _postprocess(outputs, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """Post-process model dict output into final detections.

    Args:
        outputs: dict from model forward with 'box_decoded' and 'cls'.

    Returns:
        List of (M, 6) numpy arrays [x1, y1, x2, y2, conf, cls] per image.
    """
    boxes = outputs["box_decoded"].detach().cpu().numpy()
    cls_logits = outputs["cls"].detach().sigmoid().cpu().numpy()

    batch_out = []
    for b in range(boxes.shape[0]):
        b_boxes = boxes[b]          # (N, 4) xyxy
        b_scores = cls_logits[b]    # (N, nc)

        cls_id = b_scores.argmax(axis=1)
        cls_conf = b_scores[np.arange(len(b_scores)), cls_id]

        mask = cls_conf > conf_thres
        if not mask.any():
            batch_out.append(np.empty((0, 6), dtype=np.float32))
            continue

        b_boxes = b_boxes[mask]
        cls_conf = cls_conf[mask]
        cls_id = cls_id[mask]

        # Per-class NMS via coordinate offset
        offset = cls_id.astype(np.float32) * 7680.0
        shifted = b_boxes + offset[:, None]
        keep = _nms(shifted, cls_conf, iou_thres)

        if len(keep) > max_det:
            keep = keep[:max_det]

        det = np.concatenate(
            [b_boxes[keep], cls_conf[keep, None],
             cls_id[keep, None].astype(np.float32)],
            axis=1,
        )
        batch_out.append(det)
    return batch_out


class SwiftDet:
    """High-level model interface for training, validation, prediction, and export.

    Usage::

        model = SwiftDet("swiftdet-n")           # from variant name
        model = SwiftDet("path/to/best.pt")      # from checkpoint
        model.train(data="coco.yaml", epochs=500)
        metrics = model.val()
        results = model.predict("image.jpg")
        model.export(format="coreml")
    """

    def __init__(self, model="swiftdet-n", weights=None):
        self.model = None
        self.cfg = {}
        self.names = {}
        self.device = torch.device("cpu")
        self._ckpt_path = None
        self._data_yaml = None

        model_str = str(model)

        if model_str.endswith(".pt"):
            self._load_checkpoint(model_str)
        elif model_str.endswith(".yaml") or model_str.endswith(".yml"):
            self._load_from_yaml(model_str)
        elif model_str in _VARIANT_CONFIGS:
            cfg_path = _configs_dir() / _VARIANT_CONFIGS[model_str]
            self._load_from_yaml(str(cfg_path))
        else:
            raise ValueError(
                f"Unknown model specifier '{model_str}'. "
                f"Expected a variant name ({', '.join(_VARIANT_CONFIGS)}), "
                f"a .yaml config path, or a .pt checkpoint path."
            )

        if weights is not None:
            self._load_weights(weights)

    def _load_checkpoint(self, path):
        """Restore model from a .pt checkpoint."""
        self._ckpt_path = path
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict):
            self.cfg = ckpt.get("cfg", {})
            self.names = ckpt.get("names", {})
            self._data_yaml = ckpt.get("data_yaml", None)
            if "model_state_dict" in ckpt:
                self._build_model()
                self.model.load_state_dict(ckpt["model_state_dict"])
            elif "model" in ckpt and hasattr(ckpt["model"], "forward"):
                self.model = ckpt["model"]
            else:
                self._build_model()
        else:
            self.model = ckpt

    def _load_from_yaml(self, path):
        """Build model from YAML config."""
        self.cfg = _load_yaml(path)
        self._build_model()

    def _build_model(self):
        """Construct network from self.cfg."""
        model_cfg = self.cfg.get("model", self.cfg)
        version = model_cfg.get("version", 1)
        variant = model_cfg.get("variant", "n")
        nc = model_cfg.get("nc", 80)
        reg_max = model_cfg.get("reg_max", 16)

        if version == 2:
            from ..models.detector import build_swiftdet2
            self.model = build_swiftdet2(variant=variant, nc=nc, reg_max=reg_max)
        else:
            from ..models.detector import build_swiftdet
            self.model = build_swiftdet(variant=variant, nc=nc, reg_max=reg_max)
        if not self.names:
            self.names = {i: str(i) for i in range(nc)}

    def _load_weights(self, path):
        """Load state-dict weights."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=False)

    def _resolve_device(self, device=None):
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        return self.device

    def _to_device(self, device=None):
        dev = self._resolve_device(device)
        if self.model is not None:
            self.model = self.model.to(dev)
        return dev

    # ---- Training ---- #

    _UNSET = object()

    def train(self, data, epochs=_UNSET, batch=_UNSET, imgsz=_UNSET, device=None,
              lr0=_UNSET, optimizer=_UNSET, amp=_UNSET, resume=False, **kwargs):
        """Train the model on a detection dataset."""
        from ..engine.trainer import DetectionTrainer

        dev = self._to_device(device)
        self._data_yaml = data

        train_cfg = self.cfg.get("train", {})

        # Resolve explicit params: user arg > YAML train > hardcoded default
        epochs = train_cfg.get("epochs", 500) if epochs is self._UNSET else epochs
        batch = train_cfg.get("batch_size", 64) if batch is self._UNSET else batch
        imgsz = train_cfg.get("img_size", 640) if imgsz is self._UNSET else imgsz
        lr0 = train_cfg.get("lr0", 0.01) if lr0 is self._UNSET else lr0
        optimizer = train_cfg.get("optimizer", "sgd") if optimizer is self._UNSET else optimizer
        amp = train_cfg.get("amp", True) if amp is self._UNSET else amp

        # Merge remaining train params (lrf, momentum, weight_decay, etc.)
        _explicit_keys = {"epochs", "batch_size", "img_size", "lr0", "optimizer", "amp", "device"}
        for key, val in train_cfg.items():
            if key not in _explicit_keys and key not in kwargs:
                kwargs[key] = val

        # Merge augmentation config
        for key, val in self.cfg.get("augment", {}).items():
            if key not in kwargs:
                kwargs[key] = val

        # Merge loss config
        for key, val in self.cfg.get("loss", {}).items():
            if key not in kwargs:
                kwargs[key] = val

        trainer = DetectionTrainer(
            model=self.model,
            data_yaml=data,
            epochs=epochs,
            batch_size=batch,
            img_size=imgsz,
            lr0=lr0,
            optimizer=optimizer,
            amp=amp,
            resume=resume,
            device=str(dev),
            **kwargs,
        )
        metrics = trainer.train()
        return metrics

    # ---- Validation ---- #

    def val(self, data=None, batch=32, imgsz=640, conf=0.001, iou=0.7,
            save_dir=None, plots=False, **kwargs):
        """Validate the model."""
        from ..engine.evaluator import DetectionEvaluator

        dev = self._to_device()
        data = data or self._data_yaml
        if data is None:
            raise ValueError("No data YAML specified. Pass data= argument.")

        if plots and save_dir is None:
            save_dir = "runs/val"

        evaluator = DetectionEvaluator(
            model=self.model,
            data_yaml=data,
            batch_size=batch,
            img_size=imgsz,
            conf_thres=conf,
            iou_thres=iou,
            device=str(dev),
            save_dir=save_dir,
            plots=plots,
            **kwargs,
        )
        return evaluator.evaluate()

    # ---- Prediction ---- #

    def predict(self, source, conf=0.25, iou=0.45, imgsz=640, max_det=300,
                save=False, show=False, **kwargs):
        """Run inference on one or more images / a video."""
        dev = self._to_device()
        self.model.eval()

        images, paths = self._load_source(source)

        all_results = []
        for idx, img_bgr in enumerate(images):
            img_path = paths[idx] if idx < len(paths) else None

            img_input, ratio, pad = _letterbox(img_bgr, new_shape=imgsz)
            img_input = img_input[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
            img_input = np.ascontiguousarray(img_input, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(img_input).unsqueeze(0).to(dev)

            with torch.no_grad():
                pred = self.model(tensor)

            dets_list = _postprocess(pred, conf_thres=conf, iou_thres=iou,
                                     max_det=max_det)
            dets = dets_list[0]

            # Scale boxes back to original image
            if dets.shape[0] > 0:
                dets[:, [0, 2]] = (dets[:, [0, 2]] - pad[0]) / ratio
                dets[:, [1, 3]] = (dets[:, [1, 3]] - pad[1]) / ratio
                h0, w0 = img_bgr.shape[:2]
                dets[:, 0] = np.clip(dets[:, 0], 0, w0)
                dets[:, 1] = np.clip(dets[:, 1], 0, h0)
                dets[:, 2] = np.clip(dets[:, 2], 0, w0)
                dets[:, 3] = np.clip(dets[:, 3], 0, h0)

            boxes = Boxes(dets)
            result = Results(orig_img=img_bgr, boxes=boxes,
                             names=self.names, path=img_path)
            all_results.append(result)

            if show:
                result.show()

        if save:
            save_dir = Path("runs/predict")
            save_dir.mkdir(parents=True, exist_ok=True)
            for i, r in enumerate(all_results):
                fname = Path(r.path).name if r.path else f"result_{i}.jpg"
                r.save(str(save_dir / fname))

        return all_results

    def _load_source(self, source):
        """Normalise source into list of BGR numpy images and paths."""
        if isinstance(source, np.ndarray):
            return [source], [""]

        if isinstance(source, torch.Tensor):
            if source.ndim == 3:
                source = source.unsqueeze(0)
            imgs = []
            for t in source:
                arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                imgs.append(arr)
            return imgs, [""] * len(imgs)

        if isinstance(source, (list, tuple)):
            images, paths = [], []
            for s in source:
                im, pa = self._load_source(s)
                images.extend(im)
                paths.extend(pa)
            return images, paths

        source = str(source)
        p = Path(source)

        if p.is_file() and p.suffix.lower() in _VID_EXTENSIONS:
            return self._load_video(source)

        if p.is_file():
            img = cv2.imread(source)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
            return [img], [source]

        if p.is_dir():
            files = sorted(
                f for f in p.iterdir() if f.suffix.lower() in _IMG_EXTENSIONS
            )
            if not files:
                raise FileNotFoundError(f"No images in directory: {source}")
            images, paths = [], []
            for f in files:
                img = cv2.imread(str(f))
                if img is not None:
                    images.append(img)
                    paths.append(str(f))
            return images, paths

        # File path that doesn't exist → clear error
        if p.suffix.lower() in _IMG_EXTENSIONS | _VID_EXTENSIONS:
            raise FileNotFoundError(f"File not found: {source}")

        raise ValueError(f"Unsupported source type: {source}")

    @staticmethod
    def _load_video(path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        return frames, [path] * len(frames)

    # ---- Export ---- #

    def export(self, format="coreml", imgsz=640, half=True, int8=False,
               nms=False, **kwargs):
        """Export model to CoreML or ONNX."""
        self.model.eval()
        self.model.cpu()

        if format == "coreml":
            from ..export.coreml_export import export_coreml
            return export_coreml(self.model, img_size=imgsz, half=half,
                                 int8=int8, nms=nms, class_names=self.names,
                                 **kwargs)
        elif format == "onnx":
            from ..export.onnx_export import export_onnx
            return export_onnx(self.model, img_size=imgsz, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format!r}. Use 'coreml' or 'onnx'.")

    # ---- Knowledge distillation ---- #

    def distill(self, teacher, data, epochs=200, **kwargs):
        """Train using knowledge distillation from a teacher model."""
        from ..engine.distiller import DistillationTrainer

        if isinstance(teacher, (str, Path)):
            teacher = SwiftDet(str(teacher))

        dev = self._to_device()
        teacher_model = teacher.model if isinstance(teacher, SwiftDet) else teacher
        teacher_model.to(dev).eval()

        distiller = DistillationTrainer(
            student=self.model,
            teacher=teacher_model,
            data_yaml=data,
            epochs=epochs,
            device=str(dev),
            **kwargs,
        )
        return distiller.train()

    # ---- Pretraining ---- #

    def pretrain(self, data, epochs=100, **kwargs):
        """ImageNet pre-training for backbone."""
        from ..engine.pretrain import ImageNetPretrainer

        dev = self._to_device()
        pretrainer = ImageNetPretrainer(
            model=self.model,
            data_path=data,
            epochs=epochs,
            device=str(dev),
            **kwargs,
        )
        return pretrainer.train()

    # ---- Utilities ---- #

    def fuse(self):
        """Fuse Conv+BN layers for faster inference."""
        if hasattr(self.model, "fuse"):
            self.model.fuse()
        return self

    @property
    def info(self):
        """Print model summary."""
        if self.model is None:
            print("No model loaded.")
            return

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters()
                        if p.requires_grad)
        n_layers = len(list(self.model.modules()))

        summary = (
            f"SwiftDet model summary:\n"
            f"  Layers:           {n_layers}\n"
            f"  Parameters:       {total_params:,}\n"
            f"  Trainable params: {trainable:,}\n"
            f"  Device:           {self.device}\n"
        )
        print(summary)
        return summary

    def __repr__(self):
        variant = self.cfg.get("model", {}).get("variant", "?")
        return f"SwiftDet(variant={variant}, device={self.device})"
