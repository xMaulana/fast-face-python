import logging
import os
from typing import Any, Literal, Union, overload

from ..schema import MODEL_FILENAMES
from .adaface import AdaFace
from .base import BaseFaceModel
from .base_recognition import BaseRecognitionModel
from .downloader import download_model
from .retinaface import RetinaFace
from .yunet import YuNet

logger = logging.getLogger(__name__)


class FaceModelFactory:
    """Factory to instantiate face detection models."""

    _models = {
        "YUNET": YuNet,
        "RETINAFACE_MOBILENET": RetinaFace,
        "RETINAFACE_RESNET50": RetinaFace,
        "ADAFACE_IR101": AdaFace,
        "ADAFACE_IR50": AdaFace,
        "ADAFACE_IR18": AdaFace,
        # "SCRFD": SCRFD, # To be implemented
    }

    @overload
    @classmethod
    def get_model(cls, model_type: Literal["YUNET", "yunet"], **kwargs: Any) -> YuNet: ...

    @overload
    @classmethod
    def get_model(
        cls,
        model_type: Literal[
            "RETINAFACE_MOBILENET",
            "retinaface_mobilenet",
            "RETINAFACE_RESNET50",
            "retinaface_resnet50",
        ],
        **kwargs: Any,
    ) -> RetinaFace: ...

    @overload
    @classmethod
    def get_model(
        cls,
        model_type: Literal[
            "ADAFACE_IR101",
            "adaface_ir101",
            "ADAFACE_IR50",
            "adaface_ir50",
            "ADAFACE_IR18",
            "adaface_ir18",
        ],
        **kwargs: Any,
    ) -> AdaFace: ...

    @overload
    @classmethod
    def get_model(cls, model_type: str, **kwargs: Any) -> Union[BaseFaceModel, BaseRecognitionModel]: ...

    @classmethod
    def get_model(
        cls, model_type: str, **kwargs: Any
    ) -> Union[BaseFaceModel, BaseRecognitionModel]:
        """Get an instance of a face detection or recognition model.

        Args:
            model_type (str): The type of model to instantiate (e.g., "YUNET", "ADAFACE_IR50").
            **kwargs: Arguments to pass to the model's constructor.

        Returns:
            Union[BaseFaceModel, BaseRecognitionModel]: An instance of the requested model.
        """
        model_type = model_type.upper()
        if model_type not in cls._models:
            raise ValueError(
                f"Model type '{model_type}' is not supported. Supported models: {list(cls._models.keys())}"
            )

        canonical_filename = MODEL_FILENAMES.get(
            model_type, f"{model_type.lower()}.onnx"
        )
        if "model_path" not in kwargs:
            kwargs["model_path"] = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), canonical_filename
            )

        model_path = kwargs["model_path"]
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            download_model(canonical_filename, model_path)

        model_class = cls._models[model_type]
        return model_class(**kwargs)

    @classmethod
    def register_model(cls, model_type: str, model_class: type[BaseFaceModel]):
        """Register a custom model with the factory.

        Args:
            model_type (str): The identifier for the model.
            model_class (Type[BaseFaceModel]): The model class.
        """
        cls._models[model_type.upper()] = model_class
