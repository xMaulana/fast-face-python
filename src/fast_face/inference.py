import os
import cv2
import numpy as np
from typing import List, Union, Tuple, Dict, Any, Literal, Optional
from onnxruntime import SessionOptions

from .models.session import ONNXSession, ProviderType
from .tools import (
    resize,
    normalize,
    get_priorbox,
    decode,
    decode_landmark,
    nms,
    parse_det,
)


class Inference:
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        top_k: int = 10000,
        nms_threshold: float = 0.2,
        keep_top_k: int = 1000,
        img_size: int = 640,
        variance: tuple[float, float] = (0.1, 0.2),
        return_original: bool = True,
        model: Literal["MOBILENET", "RESNET50"] = "MOBILENET",
        providers: List[ProviderType] = ["CPUExecutionProvider"],
        sess_options: Optional[SessionOptions] = None,
    ):
        assert model in ["MOBILENET", "RESNET50"]
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.img_size = img_size
        self.variance = variance
        self.return_original = return_original

        if model == "MOBILENET":
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "models",
                "mobilenet_retinaface.onnx",
            )
        else:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "models",
                "resnet50_retinaface.onnx",
            )
        self.sessions = ONNXSession(
            providers=providers, model_path=model_path, sess_options=sess_options
        )

    def preprocess(self, imgs: np.ndarray):
        """Preprocess images

        Args:
            imgs: Image input (B, H, W, C) or (H, W, C)

        Return:
            np.ndarray: Preprocessed images (B, H, W, C)
        """
        imgs = resize(imgs, img_size=self.img_size, expand_dims=len(imgs.shape) != 4)
        imgs = normalize(
            imgs, mean=(104.0, 117.0, 123.0), std=(1.0, 1.0, 1.0), max_pixel_value=1.0
        )
        imgs = np.transpose(imgs, (0, 3, 1, 2))

        return imgs

    def detect(
        self,
        imgs: Union[str, List[str], np.ndarray, List[np.ndarray]],
        return_dict: bool = False,
    ) -> List[Union[np.ndarray, List[Dict[str, Any]]]]:
        """Run face detection inference on input images.

        Args:
            imgs (Union[str, List[str], np.ndarray, List[np.ndarray]]): File path, list of file paths, image array, or list of image arrays.
            return_dict (bool): If True, returns results as structured dictionaries instead of raw arrays.

        Returns:
            List[Union[np.ndarray, List[Dict[str, Any]]]]: A list of detection results. Each result is an array
                of shape (num_dets, 15) or a list of dictionaries if return_dict is True.
        """
        original_shapes = []
        if isinstance(imgs, str):
            imgs = [imgs]

        if isinstance(imgs, list):
            preprocessed_list = []
            for item in imgs:
                if isinstance(item, str):
                    img = cv2.imread(item)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if img is None:
                        raise ValueError(f"Could not load image from {item}")
                else:
                    img = item
                original_shapes.append(img.shape)
                preprocessed_list.append(self.preprocess(img))

            preprocessed_imgs = np.concatenate(preprocessed_list, axis=0)
        else:
            if len(imgs.shape) == 3:  # (H, W, C)
                original_shapes.append(imgs.shape)
            else:
                for idx in range(imgs.shape[0]):
                    original_shapes.append(imgs[idx].shape)
            preprocessed_imgs = self.preprocess(imgs)

        _, _, h, w = preprocessed_imgs.shape
        self.prior_data = get_priorbox(image_size=(h, w))

        preprocessed_imgs = preprocessed_imgs.astype(np.float32)
        outputs = self.sessions(preprocessed_imgs)

        loc, conf, landms = outputs

        results = []
        for i in range(loc.shape[0]):
            dets = self.post_process(
                loc[i], conf[i], landms[i], original_shapes[i], (h, w)
            )
            if return_dict:
                dets = [parse_det(x) for x in dets]
            results.append(dets)

        return results

    def post_process(
        self,
        loc: np.ndarray,
        conf: np.ndarray,
        landms: np.ndarray,
        original_shape: Tuple[int, int],
        preprocessed_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Decode model outputs, apply NMS, and scale coordinates back to the original image.

        Args:
            loc (np.ndarray): Bounding box predictions of shape (num_anchors, 4).
            conf (np.ndarray): Confidence predictions of shape (num_anchors, 2).
            landms (np.ndarray): Landmark predictions of shape (num_anchors, 10).
            original_shape (Tuple[int, int]): The original image (height, width).
            preprocessed_shape (Tuple[int, int]): The preprocessed image (height, width).

        Returns:
            np.ndarray: Detections array of shape (num_dets, 15) containing:
                [x1, y1, x2, y2, score, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5].
        """
        boxes = decode(loc, self.prior_data, self.variance)

        if self.return_original:
            orig_h, orig_w = original_shape[:2]
            padded_size = max(orig_h, orig_w)
            scale = np.array([padded_size] * 4, dtype=np.float32)
        else:
            h, w = preprocessed_shape
            scale = np.array([w, h, w, h], dtype=np.float32)

        boxes = boxes * scale

        scores = conf[:, 1]

        landms_copy = decode_landmark(landms, self.prior_data, self.variance)

        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]

        if self.return_original:
            scale1 = np.array([padded_size] * 10, dtype=np.float32)
        else:
            scale1 = np.array([w, h] * 5, dtype=np.float32)

        landms_copy = landms_copy[inds] * scale1
        scores = scores[inds]

        order = scores.argsort()[::-1][: self.top_k]
        boxes = boxes[order]
        landms_copy = landms_copy[order]
        scores = scores[order]

        if boxes.shape[0] == 0:
            return np.empty((0, 15), dtype=np.float32)

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms_copy = landms_copy[keep]

        dets = dets[: self.keep_top_k, :]
        landms_copy = landms_copy[: self.keep_top_k, :]

        dets = np.concatenate((dets, landms_copy), axis=1)

        if dets.shape[0] > 0:
            dets = np.array(sorted(dets, key=lambda x: x[4], reverse=True))

        return dets
