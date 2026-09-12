import os
from typing import Type

from .base import BaseFaceModel
from .yunet import YuNet
from .downloader import download_model
from ..schema import MODEL_FILENAMES


class FaceModelFactory:
    """Factory to instantiate face detection models."""

    _models = {
        "YUNET": YuNet,
        # "RETINAFACE": RetinaFace, # To be implemented
        # "SCRFD": SCRFD, # To be implemented
    }

    @classmethod
    def get_model(cls, model_type: str, **kwargs) -> BaseFaceModel:
        """Get an instance of a face detection model.

        Args:
            model_type (str): The type of model to instantiate (e.g., "YUNET").
            **kwargs: Arguments to pass to the model's constructor.

        Returns:
            BaseFaceModel: An instance of the requested face model.
        """
        model_type = model_type.upper()
        if model_type not in cls._models:
            raise ValueError(
                f"Model type '{model_type}' is not supported. Supported models: {list(cls._models.keys())}"
            )

        canonical_filename = MODEL_FILENAMES.get(model_type, f"{model_type.lower()}.onnx")
        if "model_path" not in kwargs:
            kwargs["model_path"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), canonical_filename)

        model_path = kwargs["model_path"]
        if not os.path.exists(model_path):
            
            # Ensure the directory exists before downloading
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
