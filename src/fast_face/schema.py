from typing import Any, Dict, Tuple, Union

MODEL_FILENAMES = {
    "YUNET": "yunet.onnx",
    "RETINAFACE": "retinaface.onnx",
    "SCRFD": "scrfd.onnx",
}

ProviderType = Union[str, tuple[str, dict[str, Any]]]

