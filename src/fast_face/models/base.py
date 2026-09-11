import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Dict, Any, Optional
import onnxruntime as ort

from .session import ONNXSession
from ..schema import ProviderType
from ..tools import nms, parse_det


class BaseFaceModel(ABC):
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.4,
        top_k: int = 5000,
        keep_top_k: int = 1000,
        providers: List[ProviderType] = ["CPUExecutionProvider"],
        sess_options: Optional[ort.SessionOptions] = None,
    ):
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
        pass

    @abstractmethod
    def post_process(
        self,
        outputs: List[np.ndarray],
        original_shapes: List[Tuple[int, int]],
        preprocessed_shape: Tuple[int, int],
    ) -> List[np.ndarray]:
        """Post-process ONNX outputs to bounding boxes, confidence, and landmarks."""
        pass

    def detect(
        self,
        imgs: Union[str, List[str], np.ndarray, List[np.ndarray]],
        return_dict: bool = False,
    ) -> List[Union[np.ndarray, List[Dict[str, Any]]]]:
        """Run face detection inference on input images."""
        original_shapes = []
        if isinstance(imgs, str):
            imgs = [imgs]

        if isinstance(imgs, list):
            raw_imgs = []
            for item in imgs:
                if isinstance(item, str):
                    img = cv2.imread(item)
                    if img is None:
                        raise ValueError(f"Could not load image from {item}")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = item
                original_shapes.append(img.shape)
                raw_imgs.append(img)
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

        outputs = self.session(preprocessed_imgs)

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

        return results
