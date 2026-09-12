from typing import Any, Dict, Tuple, Union

MODEL_FILENAMES = {
    "YUNET": "yunet.onnx",
    "RETINAFACE_MOBILENET": "retinaface_mobilenet.onnx",
    "RETINAFACE_RESNET50": "retinaface_resnet50.onnx",
    # "SCRFD": "scrfd.onnx",
}

ProviderType = Union[str, tuple[str, dict[str, Any]]]

