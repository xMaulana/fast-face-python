# Adapted from https://github.com/elliottzheng/batch-face
from typing import Optional
import cv2
import numpy as np
from math import ceil
from typing import Union, Tuple, List, Dict, Any

FACIAL_REF_POINT = [
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 92.3655014],
    [62.72990036, 92.20410156],
]

DEFAULT_CROP_SIZE = (96, 112)


def resize_single(image: np.ndarray, img_size: int = 640) -> np.ndarray:
    """Resize a single image using letterbox padding to a fixed (img_size x img_size) square.

    The image is scaled proportionally so its longest dimension equals img_size,
    and black padding is added to the right or bottom to make it square.

    Args:
        image (np.ndarray): Input image array (H, W, C).
        img_size (int): Target width and height.

    Returns:
        np.ndarray: The resized image array of shape (img_size, img_size, C).
    """
    h, w = image.shape[:2]
    scale = img_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    pad_w = img_size - new_w
    pad_h = img_size - new_h

    return cv2.copyMakeBorder(
        resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


def resize(
    image: np.ndarray, img_size: int = 640, expand_dims: bool = False
) -> np.ndarray:
    """Resize input images using letterbox padding to a fixed square of size img_size.

    Args:
        image (np.ndarray): Input image array (B, H, W, C) or (H, W, C).
        img_size (int): Target size for the square dimensions.
        expand_dims (bool): Whether to add a batch dimension (for unbatched inputs).

    Returns:
        np.ndarray: The resized and padded image array.
    """
    if len(image.shape) == 4:
        out = np.stack(
            [resize_single(image[i], img_size) for i in range(image.shape[0])]
        )
        return out
    else:
        out = resize_single(image, img_size)
        if expand_dims:
            out = np.expand_dims(out, axis=0)
        return out


def normalize(
    image: np.ndarray,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    max_pixel_value=255.0,
) -> np.ndarray:
    """
    Normalize image value.

    Args:
        image (np.ndarray): Input image array (B, H, W, C) or (H, W, C)
        mean (tuple[float,float,float]): Mean value for every channel
        std (tuple[float,float,float]): Std deviation value for every channel
        max_pixel_value (float): Maximum pixel value
    Return:
        np.ndarray: Normalized image
    """
    img = image.astype(np.float32)

    if max_pixel_value != 1.0:
        img /= max_pixel_value

    mean_np = np.array(mean, dtype=np.float32)
    std_np = np.array(std, dtype=np.float32)

    img -= mean_np

    img /= std_np

    return img


def get_priorbox(
    image_size: tuple[int, int] = (640, 640),
    min_sizes: list[list[int]] = None,
    steps: list[int] = None,
    clip: bool = False,
) -> np.ndarray:
    """Generate prior anchor boxes for ONNX model inference.

    Args:
        image_size (tuple[int, int], optional): Tuple of (height, width) for model input image size. Defaults to (640, 640).
        min_sizes (list[list[int]], optional): Minimum anchor sizes per feature map layer. Defaults to RetinaFace standard.
        steps (list[int], optional): Strides/steps per feature map layer. Defaults to [8, 16, 32].
        clip (bool, optional): Whether to clip anchor coordinates to [0, 1]. Defaults to False.

    Returns:
        np.ndarray: Generated anchor boxes of shape (N, 4).
    """
    if min_sizes is None:
        min_sizes = [[16, 32], [64, 128], [256, 512]]
    if steps is None:
        steps = [8, 16, 32]

    feature_maps = [
        [ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps
    ]

    anchors = []
    for k, f in enumerate(feature_maps):
        layer_min_sizes = min_sizes[k]
        step = steps[k]

        xv, yv = np.meshgrid(np.arange(f[1]), np.arange(f[0]))

        cx = (xv.flatten() + 0.5) * step / image_size[1]
        cy = (yv.flatten() + 0.5) * step / image_size[0]

        num_min_sizes = len(layer_min_sizes)

        cx = np.repeat(cx, num_min_sizes)
        cy = np.repeat(cy, num_min_sizes)
        s_kx = np.tile([m / image_size[1] for m in layer_min_sizes], len(xv.flatten()))
        s_ky = np.tile([m / image_size[0] for m in layer_min_sizes], len(yv.flatten()))

        layer_anchors = np.stack((cx, cy, s_kx, s_ky), axis=1)
        anchors.append(layer_anchors)

    output = np.vstack(anchors).astype(np.float32)

    if clip:
        output = np.clip(output, 0, 1)

    return output


def decode(
    loc: np.ndarray,
    priors: np.ndarray,
    variances: Union[Tuple[float, float], List[float]] = (0.1, 0.2),
) -> np.ndarray:
    """Decode bounding box predictions using prior boxes and variance scaling.

    Args:
        loc (np.ndarray): Bounding box offset predictions from model of shape (N, 4).
        priors (np.ndarray): Prior anchor boxes of shape (N, 4) in (cx, cy, w, h) format.
        variances (Union[Tuple[float, float], List[float]]): Variance scaling factors for (center, size). Defaults to (0.1, 0.2).

    Returns:
        np.ndarray: Decoded bounding boxes of shape (N, 4) in (xmin, ymin, xmax, ymax) format.
    """
    boxes = np.concatenate(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
        ),
        axis=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landmark(
    pre: np.ndarray,
    priors: np.ndarray,
    variances: Union[Tuple[float, float], List[float]] = (0.1, 0.2),
) -> np.ndarray:
    """Decode facial landmark location predictions using prior boxes and variance scaling.

    Args:
        pre (np.ndarray): Landmark offset predictions from model of shape (N, 10).
        priors (np.ndarray): Prior anchor boxes of shape (N, 4) in (cx, cy, w, h) format.
        variances (Union[Tuple[float, float], List[float]]): Variance scaling factors for landmarks. Defaults to (0.1, 0.2).

    Returns:
        np.ndarray: Decoded facial landmarks of shape (N, 10) in (x1, y1, x2, y2, ..., x5, y5) format.
    """
    landms = np.concatenate(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        axis=1,
    )
    return landms


def nms(dets: np.ndarray, thresh: float) -> List[int]:
    """Perform Non-Maximum Suppression (NMS) on bounding boxes.

    Args:
        dets (np.ndarray): Bounding box detections with confidence scores of shape (N, 5),
            where each row represents (x1, y1, x2, y2, score).
        thresh (float): Intersection-over-Union (IoU) threshold for suppressing overlapping boxes.

    Returns:
        List[int]: List of kept bounding box indices after NMS.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def parse_det(det: np.ndarray) -> Dict[str, Any]:
    """Parse a raw detection array into a structured dictionary.

    Args:
        det (np.ndarray): Detection array of shape (15,) containing
            [x1, y1, x2, y2, score, l1_x, l1_y, ..., l5_x, l5_y].

    Returns:
        Dict[str, Any]: Structured dictionary with bounding box, confidence, and facial landmarks.
    """
    return {
        "bbox": [float(x) for x in det[0:4]],
        "confidence": float(det[4]),
        "landmarks": {
            "left_eye": [float(det[5]), float(det[6])],
            "right_eye": [float(det[7]), float(det[8])],
            "nose": [float(det[9]), float(det[10])],
            "left_mouth": [float(det[11]), float(det[12])],
            "right_mouth": [float(det[13]), float(det[14])],
        },
    }


def crop_face(
    img: np.ndarray,
    det: Union[np.ndarray, Dict[str, Any], List[Dict[str, Any]]],
    return_dict: bool = False,
) -> Tuple[List[np.ndarray], Union[np.ndarray, List[Dict[str, Any]]]]:
    """Crop face(s) from image and adjust landmarks relative to the cropped bounding box.

    Args:
        img (np.ndarray): Original image array (H, W, C).
        det (Union[np.ndarray, Dict, List[Dict]]): Detection output containing bbox and landmarks.
            Can be a single dict, a list of dicts, a 1D array (15,), or a 2D array (N, 15).
        return_dict (bool): If True, returns landmarks as a dictionary. Default False.

    Returns:
        Tuple[List[np.ndarray], Union[np.ndarray, List[Dict[str, Any]]]]: A list of cropped face images
        and their corresponding adjusted landmarks (as an array or a list of dictionaries).
    """
    if isinstance(det, list):
        dets_list = det
    elif isinstance(det, np.ndarray) and det.ndim == 2:
        dets_list = [det[i] for i in range(det.shape[0])]
    else:
        dets_list = [det]

    cropped_imgs = []
    adjusted_lmks_list = []
    h, w = img.shape[:2]

    for d in dets_list:
        if isinstance(d, dict):
            bbox = d["bbox"]
            landmarks = d["landmarks"]
            lmk = np.array(
                [
                    landmarks["left_eye"],
                    landmarks["right_eye"],
                    landmarks["nose"],
                    landmarks["left_mouth"],
                    landmarks["right_mouth"],
                ],
                dtype=np.float32,
            )
        elif isinstance(d, np.ndarray):
            if d.shape[0] < 15:
                raise ValueError("Detection array must have at least 15 elements")
            bbox = d[0:4]
            lmk = d[5:15].reshape((5, 2)).astype(np.float32)
        else:
            raise TypeError("det must be a dictionary or numpy array")

        x1, y1, x2, y2 = [int(x) for x in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        cropped_img = img[y1:y2, x1:x2]
        cropped_imgs.append(cropped_img)

        adjusted_lmk = lmk.copy()
        adjusted_lmk[:, 0] -= x1
        adjusted_lmk[:, 1] -= y1

        if return_dict:
            ret_landmarks = {
                "left_eye": adjusted_lmk[0].tolist(),
                "right_eye": adjusted_lmk[1].tolist(),
                "nose": adjusted_lmk[2].tolist(),
                "left_mouth": adjusted_lmk[3].tolist(),
                "right_mouth": adjusted_lmk[4].tolist(),
            }
            adjusted_lmks_list.append(ret_landmarks)
        else:
            adjusted_lmks_list.append(adjusted_lmk)

    if not return_dict:
        return cropped_imgs, np.array(adjusted_lmks_list)
    return cropped_imgs, adjusted_lmks_list


def _get_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Compute similarity transform matrix from src_pts to dst_pts.

    Args:
        src_pts (np.ndarray): Source points, shape (K, 2).
        dst_pts (np.ndarray): Destination points, shape (K, 2).

    Returns:
        np.ndarray: 2x3 affine transform matrix.
    """
    num = src_pts.shape[0]

    src_mean = np.mean(src_pts, axis=0)
    dst_mean = np.mean(dst_pts, axis=0)

    src_demean = src_pts - src_mean
    dst_demean = dst_pts - dst_mean

    A = np.zeros((2 * num, 4), dtype=np.float64)
    b = np.zeros((2 * num, 1), dtype=np.float64)

    for i in range(num):
        A[2 * i, 0] = src_demean[i, 0]
        A[2 * i, 1] = -src_demean[i, 1]
        A[2 * i, 2] = 1
        A[2 * i, 3] = 0
        A[2 * i + 1, 0] = src_demean[i, 1]
        A[2 * i + 1, 1] = src_demean[i, 0]
        A[2 * i + 1, 2] = 0
        A[2 * i + 1, 3] = 1
        b[2 * i] = dst_demean[i, 0]
        b[2 * i + 1] = dst_demean[i, 1]

    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    params = params.flatten()

    # params = [a, b, tx, ty] where the transform is:
    # [a, -b, tx]   [x]   [x']
    # [b,  a, ty] * [y] = [y']
    a, b_val, tx, ty = params

    tfm = np.float64(
        [
            [a, -b_val, dst_mean[0] - (a * src_mean[0] - b_val * src_mean[1]) + tx],
            [b_val, a, dst_mean[1] - (b_val * src_mean[0] + a * src_mean[1]) + ty],
        ]
    )

    return tfm


def _get_reference_facial_points(
    output_size: Optional[Tuple[int, int]] = None, default_square: bool = False
) -> np.ndarray:
    """Get reference facial points scaled to the given output size.

    Matches AdaFace's get_reference_facial_points behavior.

    Args:
        output_size (Tuple[int, int], optional): Target (w, h). If None, returns default points.
        default_square (bool): If True, pad default (96, 112) to (112, 112) before scaling.

    Returns:
        np.ndarray: Reference points, shape (5, 2).
    """
    tmp_5pts = np.array(FACIAL_REF_POINT, dtype=np.float64)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE, dtype=np.float64)

    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    if output_size is None:
        return tmp_5pts.astype(np.float32)

    output_size = np.array(output_size, dtype=np.float64)
    if output_size[0] == tmp_crop_size[0] and output_size[1] == tmp_crop_size[1]:
        return tmp_5pts.astype(np.float32)

    scale_factor = output_size / tmp_crop_size
    tmp_5pts[:, 0] *= scale_factor[0]
    tmp_5pts[:, 1] *= scale_factor[1]

    return tmp_5pts.astype(np.float32)


def align_face(
    img: np.ndarray,
    landmarks: Union[np.ndarray, Dict[str, Any]],
    align_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """Align a face image based on facial reference points.

    Uses a similarity transform matching AdaFace's warp_and_crop_face behavior.

    Args:
        img (np.ndarray): Original image array (H, W, C) or cropped face image.
        landmarks (Union[np.ndarray, Dict[str, Any]]): Facial landmarks. Can be an array of shape (5, 2)
            or a dictionary containing landmark coordinates.
        align_size (Tuple[int, int], optional): Target size (width, height) of the aligned face. Defaults to (112, 112).

    Returns:
        np.ndarray: Aligned face image array of shape (align_size[1], align_size[0], C).
    """
    default_square = align_size[0] == align_size[1]
    ref_pts = _get_reference_facial_points(
        output_size=align_size, default_square=default_square
    )

    if isinstance(landmarks, dict):
        if "landmarks" in landmarks:
            landmarks = landmarks["landmarks"]

        for i in ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]:
            if i not in landmarks:
                raise ValueError(f"Landmarks dictionary must contain '{i}' etc.")

        lmk = np.array(
            [
                landmarks["left_eye"],
                landmarks["right_eye"],
                landmarks["nose"],
                landmarks["left_mouth"],
                landmarks["right_mouth"],
            ],
            dtype=np.float32,
        )
    elif isinstance(landmarks, np.ndarray):
        if landmarks.shape == (5, 2):
            lmk = landmarks.astype(np.float32)
        elif landmarks.size == 10:
            lmk = landmarks.reshape((5, 2)).astype(np.float32)
        elif landmarks.size >= 15:
            lmk = landmarks[5:15].reshape((5, 2)).astype(np.float32)
        else:
            raise ValueError("Landmarks array must have shape (5, 2) or (10,)")
    else:
        raise TypeError("landmarks must be a dictionary or numpy array")

    tform = _get_similarity_transform(lmk, ref_pts)

    if tform is None:
        raise ValueError("Failed to estimate similarity transform for face alignment.")

    aligned_img = cv2.warpAffine(img, tform, align_size)
    return aligned_img
