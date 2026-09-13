import logging

import cv2
import numpy as np

from ..schema import ProviderType
from .base import BaseFaceModel

logger = logging.getLogger(__name__)

class YuNet(BaseFaceModel):
    OUTPUT_NAMES = [
        "cls_8",
        "cls_16",
        "cls_32",
        "obj_8",
        "obj_16",
        "obj_32",
        "bbox_8",
        "bbox_16",
        "bbox_32",
        "kps_8",
        "kps_16",
        "kps_32",
    ]
    STRIDES = (8, 16, 32)
    DIVISOR = 32

    def __init__(
        self,
        model_path: str,
        input_size: tuple = (320, 320),
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.4,
        top_k: int = 5000,
        keep_top_k: int = 1000,
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
        self._input_w, self._input_h = input_size
        self._update_pad_size()

    def _update_pad_size(self):
        self._pad_w = ((self._input_w - 1) // self.DIVISOR + 1) * self.DIVISOR
        self._pad_h = ((self._input_h - 1) // self.DIVISOR + 1) * self.DIVISOR

    def set_input_size(self, input_size: tuple[int, int]):
        self._input_w, self._input_h = input_size
        self._update_pad_size()

    def preprocess(self, imgs: list[np.ndarray]) -> np.ndarray:
        logger.debug(f"Preprocessing {len(imgs)} images for YuNet")
        preprocessed_imgs = []
        for img in imgs:
            h, w = img.shape[:2]
            scale = min(self._input_w / w, self._input_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = cv2.resize(img, (new_w, new_h))

            padded = np.zeros((self._pad_h, self._pad_w, 3), dtype=np.uint8)
            padded[:new_h, :new_w, :] = resized

            preprocessed_imgs.append(padded)

        batch_imgs = np.stack(preprocessed_imgs, axis=0)
        batch_imgs = np.transpose(batch_imgs, (0, 3, 1, 2)).astype(np.float32)
        return batch_imgs

    def _generate_priors(self, h: int, w: int) -> np.ndarray:
        priors = []
        for stride in self.STRIDES:
            feat_w = int(np.ceil(w / stride))
            feat_h = int(np.ceil(h / stride))
            for y in range(feat_h):
                for x in range(feat_w):
                    cx = x * stride
                    cy = y * stride
                    priors.append([cx, cy, stride, stride])
        return np.array(priors, dtype=np.float32)

    def post_process(
        self,
        outputs: list[np.ndarray],
        original_shapes: list[tuple[int, int]],
        preprocessed_shape: tuple[int, int],
    ) -> list[np.ndarray]:
        batch_size = outputs[0].shape[0]
        h, w = preprocessed_shape
        priors = self._generate_priors(h, w)

        cls_scores = []
        obj_scores = []
        bboxes = []
        kpss = []

        for i in range(3):
            cls_out = outputs[i]
            if len(cls_out.shape) == 3 and cls_out.shape[-1] == 2:
                cls_out = cls_out[:, :, 1:2]
            elif len(cls_out.shape) > 3:
                cls_out = cls_out.reshape((batch_size, cls_out.shape[1], -1)).transpose(
                    (0, 2, 1)
                )
                if cls_out.shape[-1] == 2:
                    cls_out = cls_out[:, :, 1:2]
            cls_scores.append(cls_out)

            obj_out = outputs[i + 3]
            if len(obj_out.shape) > 3:
                obj_out = obj_out.reshape((batch_size, 1, -1)).transpose((0, 2, 1))
            obj_scores.append(obj_out)

            bbox_out = outputs[i + 6]
            if len(bbox_out.shape) > 3:
                bbox_out = bbox_out.reshape((batch_size, 4, -1)).transpose((0, 2, 1))
            bboxes.append(bbox_out)

            kps_out = outputs[i + 9]
            if len(kps_out.shape) > 3:
                kps_out = kps_out.reshape((batch_size, 10, -1)).transpose((0, 2, 1))
            kpss.append(kps_out)

        cls_scores = np.concatenate(cls_scores, axis=1)
        obj_scores = np.concatenate(obj_scores, axis=1)
        bboxes = np.concatenate(bboxes, axis=1)
        kpss = np.concatenate(kpss, axis=1)

        scores = cls_scores * obj_scores

        results = []
        logger.debug("Decoding and filtering bounding boxes for each image")
        for i in range(batch_size):
            orig_h, orig_w = original_shapes[i][:2]
            scale_factor = min(self._input_w / orig_w, self._input_h / orig_h)

            batch_bbox = bboxes[i]
            cx = batch_bbox[:, 0:1] * priors[:, 2:3] + priors[:, 0:1]
            cy = batch_bbox[:, 1:2] * priors[:, 3:4] + priors[:, 1:2]
            w_box = np.exp(batch_bbox[:, 2:3]) * priors[:, 2:3]
            h_box = np.exp(batch_bbox[:, 3:4]) * priors[:, 3:4]

            x1 = cx - w_box / 2.0
            y1 = cy - h_box / 2.0
            x2 = cx + w_box / 2.0
            y2 = cy + h_box / 2.0

            batch_kps = kpss[i]
            decoded_kps = np.zeros_like(batch_kps)
            for k in range(5):
                decoded_kps[:, k * 2 : k * 2 + 1] = (
                    batch_kps[:, k * 2 : k * 2 + 1] * priors[:, 2:3] + priors[:, 0:1]
                )
                decoded_kps[:, k * 2 + 1 : k * 2 + 2] = (
                    batch_kps[:, k * 2 + 1 : k * 2 + 2] * priors[:, 3:4]
                    + priors[:, 1:2]
                )

            x1 /= scale_factor
            y1 /= scale_factor
            x2 /= scale_factor
            y2 /= scale_factor
            decoded_kps /= scale_factor

            batch_scores = scores[i].flatten()

            inds = np.where(batch_scores > self.conf_threshold)[0]

            if len(inds) == 0:
                results.append(np.empty((0, 15), dtype=np.float32))
                continue

            x1 = x1[inds]
            y1 = y1[inds]
            x2 = x2[inds]
            y2 = y2[inds]
            filtered_kps = decoded_kps[inds]
            filtered_scores = batch_scores[inds]

            order = filtered_scores.argsort()[::-1][: self.top_k]

            x1 = x1[order]
            y1 = y1[order]
            x2 = x2[order]
            y2 = y2[order]
            filtered_kps = filtered_kps[order]
            filtered_scores = filtered_scores[order]

            dets = np.hstack(
                (x1, y1, x2, y2, filtered_scores[:, np.newaxis], filtered_kps)
            ).astype(np.float32, copy=False)

            results.append(dets)

        return results
