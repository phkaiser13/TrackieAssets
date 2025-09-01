# Training a Model

This document outlines the standard procedure for training, fine-tuning, and evaluating models within the Trackie ecosystem. Adhering to this process ensures reproducibility and consistency.

## 1. Configuration

All training runs must be defined by a configuration file, typically in YAML format. This file should be versioned and stored alongside the training scripts.

### Example `config.yaml`:

```yaml
# General settings
project: trackie-assets
model_name: yolov5nu-v1.0.0
seed: 42

# Dataset
dataset_path: /path/to/coco-like-dataset
validation_split: 0.2
preprocessing:
  resize: [640, 640]
  normalize: [0, 1]
  augmentations:
    - hflip
    - color_jitter
    - mosaic

# Model Hyperparameters
architecture: yolov5nu
backbone: CSP_variant
epochs: 300
batch_size: 32
optimizer: AdamW
learning_rate:
  initial: 1e-3
  scheduler: cosine
  warmup_epochs: 3

# Export settings
export_formats:
  - onnx
  - tensorrt
onnx_opset: 14
tensorrt_precision: fp16
```

## 2. Environment Setup

To ensure reproducibility, the training environment should be captured. This can be done using a Dockerfile or a `requirements.txt` / `environment.yml` file.

**Example Docker command:**
```bash
docker build -t trackie-training-env -f Dockerfile.train .
docker run --gpus all -v /path/to/data:/data trackie-training-env python train.py --config configs/yolov5nu_v1.yaml
```

## 3. Running Training

Execute the main training script, passing the configuration file as an argument.

```bash
python /path/to/your/train_script.py --config /path/to/config.yaml
```

## 4. Exporting Artifacts

After training, models must be exported to the formats specified in the `format_priorities` for that model (e.g., ONNX, GGUF, TorchScript).

**Example export command:**
```bash
python tools/convert_to_onnx.py \
    --input weights/model.pt \
    --output weights/model.onnx \
    --opset 14 \
    --dynamic
```

## 5. Verification and Hashing

All final model artifacts must be verified for integrity. Generate SHA256 checksums for all weight files.

```bash
sha256sum weights/* > weights/SHA256SUMS
```
This file should be included with the release artifacts.
