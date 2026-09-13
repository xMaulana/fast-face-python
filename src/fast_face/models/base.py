import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

import cv2
import numpy as np
import onnxruntime as ort

from ..schema import ProviderType
from ..tools import nms, parse_det
from .session import ONNXSession


class BaseFaceModel(ABC):
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.4,
        top_k: int = 5000,
        keep_top_k: int = 1000,
        providers: list[ProviderType] | None = None,
        sess_options: ort.SessionOptions | None = None,
    ):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.keep_top_k = keep_top_k

        self.session = ONNXSession(
            model_path=model_path, providers=providers, sess_options=sess_options
        )
        self.input_name = self.session.session.get_inputs()[0].name

    @abstractmethod
    def preprocess(self, imgs: np.ndarray) -> np.ndarray:
        """Preprocess images for the specific model."""

    @abstractmethod
    def post_process(
        self,
        outputs: list[np.ndarray],
        original_shapes: list[tuple[int, int]],
        preprocessed_shape: tuple[int, int],
    ) -> list[np.ndarray]:
        """Post-process ONNX outputs to bounding boxes, confidence, and landmarks."""

    def detect(
        self,
        imgs: str | list[str] | np.ndarray | list[np.ndarray],
        return_dict: bool = False,
    ) -> list[np.ndarray | list[dict[str, Any]]]:
        """Run face detection inference on input images."""
        original_shapes = []
        logger.debug(f"Starting detection on {len(imgs) if isinstance(imgs, list) else 1} images")
        if isinstance(imgs, str):
            imgs = [imgs]

        if isinstance(imgs, list):
            raw_imgs = []
            for item in imgs:
                if isinstance(item, str):
                    img = cv2.imread(item)
                    if img is None:
                        logger.error(f"Could not load image from {item}")
                        raise ValueError(f"Could not load image from {item}")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = item
                original_shapes.append(img.shape)
                raw_imgs.append(img)
            logger.debug(f"Preprocessing {len(raw_imgs)} images from list")
            preprocessed_imgs = self.preprocess(raw_imgs)
        else:
            if len(imgs.shape) == 3:  # (H, W, C)
                original_shapes.append(imgs.shape)
                preprocessed_imgs = self.preprocess([imgs])
            else:
                for idx in range(imgs.shape[0]):
                    original_shapes.append(imgs[idx].shape)
                preprocessed_imgs = self.preprocess(
                    [imgs[idx] for idx in range(imgs.shape[0])]
                )

        _, _, h, w = preprocessed_imgs.shape
        preprocessed_shape = (h, w)

        logger.debug(f"Running ONNX session inference on shape {preprocessed_imgs.shape}")
        outputs = self.session(preprocessed_imgs)

        logger.debug("Running post-processing")
        batch_dets = self.post_process(outputs, original_shapes, preprocessed_shape)

        results = []
        for i in range(len(batch_dets)):
            dets = batch_dets[i]
            if dets.shape[0] == 0:
                results.append(
                    [] if return_dict else np.empty((0, 15), dtype=np.float32)
                )
                continue

            keep = nms(dets, self.nms_threshold)
            dets = dets[keep, :]

            dets = dets[: self.keep_top_k, :]

            if return_dict:
                dets = [parse_det(x) for x in dets]
            results.append(dets)

        logger.info(f"Detection completed. Found faces in {sum(len(d) > 0 for d in results)}/{len(results)} images")
        return results
