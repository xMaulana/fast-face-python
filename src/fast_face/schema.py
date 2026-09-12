from typing import Any, Union

MODEL_FILENAMES = {
    "YUNET": "yunet.onnx",
    "RETINAFACE_MOBILENET": "retinaface_mobilenet.onnx",
    "RETINAFACE_RESNET50": "retinaface_resnet50.onnx",
    # "SCRFD": "scrfd.onnx",
    "ADAFACE_IR101": "adaface_ir101.onnx",
    "ADAFACE_IR50": "adaface_ir50.onnx",
    "ADAFACE_IR18": "adaface_ir18.onnx",
}

ProviderType = Union[str, tuple[str, dict[str, Any]]]
