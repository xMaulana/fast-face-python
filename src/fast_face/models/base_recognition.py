from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import onnxruntime as ort

from ..schema import ProviderType
from .session import ONNXSession


class BaseRecognitionModel(ABC):
    """Base class for face recognition models that extract embeddings."""

    def __init__(
        self,
        model_path: str,
        providers: list[ProviderType] | None = None,
        sess_options: ort.SessionOptions | None = None,
    ):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ONNXSession(
            model_path=model_path, providers=providers, sess_options=sess_options
        )
        self.input_name = self.session.session.get_inputs()[0].name

    @abstractmethod
    def preprocess(self, imgs: list[np.ndarray]) -> np.ndarray:
        """Preprocess aligned face images for the specific model."""

    @abstractmethod
    def extract(
        self,
        imgs: list[np.ndarray],
        landmarks: list[np.ndarray | dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Extract face embeddings from images.

        Args:
            imgs: List of face images (RGB).
            landmarks: Optional list of landmarks for alignment.

        Returns:
            np.ndarray: L2-normalized embeddings of shape (N, embedding_dim).
        """
