# fast-face-python

Fast and easy face detection using ONNX runtime.

## Overview

fast-face-python provides a unified interface for running face detection and recognition models using ONNX Runtime. It handles model downloading, preprocessing, inference, and post-processing, including non-maximum suppression for detection and alignment for recognition. 

The library supports batched inference and configurable execution providers such as CPU, CUDA, ROCm (AMD GPU), and OpenVINO.

## Features

* Face Detection: YuNet, RetinaFace (MobileNet, ResNet50)
* Face Recognition: AdaFace (IR18, IR50, IR101)
* Hardware Acceleration: CPU, CUDA, ROCm (AMD GPU), and OpenVINO support via ONNX Runtime
* Batched Inference: Process multiple images simultaneously
* Automatic Model Management: Downloads required ONNX model weights on first use

## Requirements

* Python >= 3.11
* OpenCV
* NumPy
* Pillow
* ONNX Runtime

## Installation

Install the package using pip:

```bash
pip install fast-face-python
```

For NVIDIA GPU support via CUDA 12:

```bash
pip install "fast-face-python[gpu]"
```

For AMD GPU support via ROCm:

```bash
pip install "fast-face-python[rocm]"
```

For OpenVINO support:

```bash
pip install "fast-face-python[openvino]"
```

## Usage

### Face Detection

Instantiate a detection model using `FaceModelFactory` and call the `detect` method. The method accepts an image path, a NumPy array in RGB format, or a list of either.

```python
from fast_face import FaceModelFactory

# Initialize the model (weights are downloaded automatically if missing)
model = FaceModelFactory.get_model("YUNET", top_k=500, conf_threshold=0.6)

# Run detection on a local image file
results = model.detect("image.jpg", return_dict=True)

# Process the detections for the first image
for det in results[0]:
    bbox = det["bbox"]
    confidence = det["score"]
    landmarks = det["landmarks"]
    print(f"Face detected at {bbox} with confidence {confidence}")
```

To run inference on an existing OpenCV image, convert it to RGB first:

```python
import cv2

img = cv2.imread("image.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = model.detect(img_rgb, return_dict=True)
```

### Face Recognition

Recognition models extract L2-normalized embeddings from aligned face images.

```python
import numpy as np
from fast_face import FaceModelFactory

model = FaceModelFactory.get_model("ADAFACE_IR50")

# Provide pre-aligned 112x112 RGB face crops
face_crop1 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
face_crop2 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

# Extract embeddings for the batch
embeddings = model.extract([face_crop1, face_crop2])

print(f"Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
```

If you have unaligned images and face landmarks from a detection model, you can pass the landmarks to automatically align the faces before extraction:

```python
landmarks = np.array([
    [200.0, 200.0], [280.0, 200.0], [240.0, 250.0], 
    [210.0, 300.0], [270.0, 300.0]
], dtype=np.float32)

embeddings = model.extract([img_rgb], landmarks=[landmarks])
```

### Supported Models

The following model identifiers are supported by `FaceModelFactory`:

* `YUNET`
* `RETINAFACE_MOBILENET`
* `RETINAFACE_RESNET50`
* `ADAFACE_IR18`
* `ADAFACE_IR50`
* `ADAFACE_IR101`

## Development

To set up the repository for development, install the `dev` dependencies:

```bash
pip install -e ".[dev,cpu]"
```

### Testing

The project uses `pytest` for testing. Run the test suite:

```bash
pytest tests/
```

### Formatting and Linting

The project uses `ruff` for code formatting and linting:

```bash
ruff check .
ruff format .
```

## License

This project is licensed under the terms found in the `LICENSE` file.
