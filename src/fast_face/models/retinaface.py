import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from ..schema import ProviderType
from ..tools import decode, decode_landmark, get_priorbox
from .base import BaseFaceModel


class RetinaFace(BaseFaceModel):
    """RetinaFace detector supporting MobileNet and ResNet50 backbones."""

    def __init__(
        self,
        model_path: str,
        input_size: int = 640,
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.2,
        top_k: int = 10000,
        keep_top_k: int = 1000,
        variance: tuple[float, float] = (0.1, 0.2),
        providers: list[ProviderType] | None = None,
        **kwargs,
    ):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        super().__init__(
            model_path=model_path,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            keep_top_k=keep_top_k,
            providers=providers,
            sess_options=kwargs.get("sess_options"),
        )
        self.input_size = input_size
        self.variance = variance

    def _resize_single(self, image: np.ndarray) -> np.ndarray:
        """Letterbox-resize a single image to (input_size x input_size)."""
        h, w = image.shape[:2]
        scale = self.input_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        pad_w = self.input_size - new_w
        pad_h = self.input_size - new_h
        return cv2.copyMakeBorder(
            resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    def preprocess(self, imgs: list[np.ndarray]) -> np.ndarray:
        """Preprocess images: resize, normalize, transpose to NCHW."""
        logger.debug(f"Preprocessing {len(imgs)} images for RetinaFace")
        processed = []
        for img in imgs:
            resized = self._resize_single(img)
            processed.append(resized)

        batch = np.stack(processed, axis=0).astype(np.float32)

        # RetinaFace normalization: subtract mean, no std division
        mean = np.array([104.0, 117.0, 123.0], dtype=np.float32)
        batch -= mean

        # HWC -> CHW
        batch = np.transpose(batch, (0, 3, 1, 2))
        return batch

    def post_process(
        self,
        outputs: list[np.ndarray],
        original_shapes: list[tuple[int, int]],
        preprocessed_shape: tuple[int, int],
    ) -> list[np.ndarray]:
        """Decode RetinaFace outputs into [x1, y1, x2, y2, score, landmarks...] arrays."""
        loc, conf, landms = outputs
        h, w = preprocessed_shape
        prior_data = get_priorbox(image_size=(h, w))

        logger.debug("Decoding RetinaFace output bounding boxes and landmarks")
        results = []
        for i in range(loc.shape[0]):
            orig_h, orig_w = original_shapes[i][:2]
            padded_size = max(orig_h, orig_w)
            scale_box = np.array([padded_size] * 4, dtype=np.float32)
            scale_lm = np.array([padded_size] * 10, dtype=np.float32)

            boxes = decode(loc[i], prior_data, self.variance) * scale_box
            landmarks = decode_landmark(landms[i], prior_data, self.variance) * scale_lm
            scores = conf[i][:, 1]

            inds = np.where(scores > self.conf_threshold)[0]
            boxes = boxes[inds]
            landmarks = landmarks[inds]
            scores = scores[inds]

            order = scores.argsort()[::-1][: self.top_k]
            boxes = boxes[order]
            landmarks = landmarks[order]
            scores = scores[order]

            if boxes.shape[0] == 0:
                results.append(np.empty((0, 15), dtype=np.float32))
                continue

            dets = np.hstack((boxes, scores[:, np.newaxis], landmarks)).astype(
                np.float32, copy=False
            )

            results.append(dets)

        return results
