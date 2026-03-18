# SwiftDet

**Fast, accurate object detection under the MIT License.**

SwiftDet is a from-scratch object detection library built entirely on published academic research. It offers an Ultralytics-style API (train in 3 lines) while being fully MIT-licensed and free to use in commercial products.

## Highlights

- **MIT License** — no AGPL/GPL restrictions, use it anywhere
- **3-line training** — `SwiftDet("swiftdet-n").train(data="coco.yaml", epochs=500)`
- **Anchor-free + DFL** — state-of-the-art detection head with Distribution Focal Loss
- **CSP + CBAM backbone** — efficient feature extraction with channel & spatial attention
- **Built-in distillation** — boost small models with knowledge from large ones
- **CoreML & ONNX export** — deploy to mobile and edge devices

## Architecture

```
Input (640x640)
  -> Backbone (CSPNet + CBAM Attention)
  -> Neck (Bidirectional FPN)
  -> Head (Decoupled Anchor-Free + DFL)
  -> Detections
```

All components are based on published papers:

| Component | Reference |
|-----------|-----------|
| CSP Backbone | Wang et al. 2020 (CSPNet) |
| CBAM Attention | Woo et al. 2018 |
| SPP | He et al. 2015 (SPPNet) |
| BiFPN Neck | Liu et al. 2018 (PANet) |
| Decoupled Head | Ge et al. 2021 (YOLOX) |
| DFL | Li et al. 2020 (Generalized Focal Loss) |
| Varifocal Loss | Zhang et al. 2021 |
| CIoU Loss | Zheng et al. 2019 |
| Task-Aligned Assigner | Feng et al. 2021 (TOOD) |

## Model Zoo

| Model | Params | mAP<sup>val 50-95</sup> | mAP<sup>val 50</sup> | Weights |
|-------|--------|------------------------|---------------------|---------|
| SwiftDet-N | 3.0M | TODO | TODO | TODO |
| SwiftDet-S | 7.1M | TODO | TODO | TODO |
| SwiftDet-M | 13.5M | TODO | TODO | TODO |
| SwiftDet-L | 29.2M | TODO | TODO | TODO |

> Trained on COCO train2017, evaluated on COCO val2017 at 640px. Weights coming soon.

## Quick Start

### Install

```bash
pip install torch torchvision
git clone https://github.com/john-rocky/swiftdet.git
cd swiftdet
pip install -e .
```

### Training

```python
from swiftdet import SwiftDet

model = SwiftDet("swiftdet-n")
model.train(data="coco.yaml", epochs=500)
```

### Inference

```python
model = SwiftDet("swiftdet-n", weights="swiftdet-n.pt")
results = model.predict("image.jpg", conf=0.25)

for r in results:
    print(r.boxes.xyxy)   # [[x1, y1, x2, y2], ...]
    print(r.boxes.conf)   # [0.95, 0.87, ...]
    print(r.boxes.cls)    # [0, 1, 2, ...]
    r.save("output.jpg")
```

### Validation

```python
metrics = model.val(data="coco.yaml")
print(f"mAP50-95: {metrics['mAP50_95']:.4f}")
```

### Export

```python
model.export(format="coreml")   # -> .mlpackage
model.export(format="onnx")     # -> .onnx
```

### CLI

```bash
swiftdet train --model swiftdet-n --data coco.yaml --epochs 500
swiftdet val --model runs/train/best.pt --data coco.yaml
swiftdet predict --model runs/train/best.pt --source image.jpg
swiftdet export --model runs/train/best.pt --format coreml
```

## Advanced

### Knowledge Distillation

Transfer knowledge from a large model to a small one for +1-2 mAP:

```python
teacher = SwiftDet("swiftdet-l", weights="swiftdet-l.pt")
student = SwiftDet("swiftdet-n")
student.distill(teacher=teacher, data="coco.yaml", epochs=200)
```

### ImageNet Pre-training

Pre-train the backbone for +2-3 mAP:

```python
model = SwiftDet("swiftdet-n")
model.pretrain(data="/path/to/imagenet", epochs=100)
model.train(data="coco.yaml", epochs=500)
```

### Resume Training

```python
model = SwiftDet("swiftdet-n")
model.train(data="coco.yaml", epochs=500, resume=True,
            save_dir="runs/swiftdet_n_500e")
```

## Data Format

SwiftDet uses the YOLO label format. Each image has a corresponding `.txt` file:

```
class_id center_x center_y width height
class_id center_x center_y width height
```

Coordinates are normalized to [0, 1]. Dataset YAML:

```yaml
path: /datasets/coco
train: images/train2017    # or just train2017
val: images/val2017        # or just val2017
nc: 80
names:
  0: person
  1: bicycle
  ...
```

## Model Variants

| Variant | width | depth | Neck ch | Target Use Case |
|---------|-------|-------|---------|-----------------|
| SwiftDet-N | 0.30 | 0.33 | 128 | Mobile / Edge |
| SwiftDet-S | 0.50 | 0.33 | 192 | Balanced |
| SwiftDet-M | 0.75 | 0.67 | 256 | High accuracy |
| SwiftDet-L | 1.00 | 1.00 | 384 | Maximum accuracy / Teacher |

## Training Recipe

Default settings for best results:

- **Epochs:** 500
- **Optimizer:** SGD (lr=0.01, momentum=0.937)
- **Scheduler:** Cosine annealing to lr x 0.01
- **Augmentation:** Mosaic, MixUp, HSV jitter, horizontal flip
- **Mosaic closing:** Disabled for last 15 epochs
- **AMP:** Enabled by default on CUDA
- **EMA:** Decay 0.9999

## Project Structure

```
swiftdet/
  models/      # Backbone, neck, head, detector
  losses/      # VFL, CIoU, DFL, distillation losses
  data/        # Dataset, augmentations, transforms
  engine/      # Trainer, evaluator, distiller, pretrainer
  core/        # High-level API, CLI, results
  export/      # CoreML, ONNX export
  configs/     # Model YAML configs
```

## License

MIT License. See [LICENSE](LICENSE) for details.
