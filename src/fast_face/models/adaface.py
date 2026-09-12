from typing import Any

import cv2
import numpy as np

from ..schema import ProviderType
from ..tools import align_face
from .base_recognition import BaseRecognitionModel


class AdaFace(BaseRecognitionModel):
    """AdaFace face recognition model for embedding extraction.

    Supports IR-18, IR-50, and IR-101 backbones.
    Expects RGB input images; converts to BGR internally as required by AdaFace.
    """

    def __init__(
        self,
        model_path: str,
        align_size: tuple[int, int] = (112, 112),
        providers: list[ProviderType] | None = None,
        **kwargs,
    ):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        super().__init__(
            model_path=model_path,
            providers=providers,
            sess_options=kwargs.get("sess_options"),
        )
        self.align_size = align_size

    def preprocess(self, imgs: list[np.ndarray]) -> np.ndarray:
        """Preprocess aligned face images: RGB->BGR, normalize to [-1, 1], transpose to NCHW."""
        processed = []
        for img in imgs:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            h, w = bgr.shape[:2]
            if (w, h) != self.align_size:
                bgr = cv2.resize(bgr, self.align_size)

            processed.append(bgr)

        batch = np.stack(processed, axis=0).astype(np.float32)

        batch = (batch / 255.0 - 0.5) / 0.5

        batch = np.transpose(batch, (0, 3, 1, 2))
        return batch

    def extract(
        self,
        imgs: list[np.ndarray] | np.ndarray,
        landmarks: list[np.ndarray | dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Extract L2-normalized face embeddings from images.

        Args:
            imgs: List of face images in RGB, or a single image array.
                  If landmarks are provided, images should be the full/cropped
                  frames from which faces will be aligned.
            landmarks: Optional list of landmarks for face alignment.
                       Each element can be an np.ndarray of shape (5, 2) or (10,),
                       or a dictionary with landmark keys.

        Returns:
            np.ndarray: L2-normalized embeddings of shape (N, embedding_dim).
        """
        if isinstance(imgs, np.ndarray) and imgs.ndim == 3:
            imgs = [imgs]

        if landmarks is not None:
            aligned_imgs = []
            for img, lmk in zip(imgs, landmarks):
                aligned = align_face(img, lmk, align_size=self.align_size)
                aligned_imgs.append(aligned)
        else:
            aligned_imgs = imgs

        batch = self.preprocess(aligned_imgs)
        outputs = self.session(batch)

        embeddings = outputs[0]

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        embeddings = embeddings / norms

        return embeddings
