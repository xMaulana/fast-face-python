import numpy as np
import pytest

from fast_face.models.factory import FaceModelFactory


def test_retinaface_mobilenet_batched_inference():
    model = FaceModelFactory.get_model("RETINAFACE_MOBILENET", top_k=500)

    img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    results = model.detect([img1, img2], return_dict=True)

    assert len(results) == 2, "Should return results for two images"

    assert isinstance(results[0], list), (
        "Result should be a list of dicts when return_dict=True"
    )
    assert isinstance(results[1], list), (
        "Result should be a list of dicts when return_dict=True"
    )


def test_retinaface_resnet50_batched_inference():
    model = FaceModelFactory.get_model("RETINAFACE_RESNET50", top_k=500)

    img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    results = model.detect([img1, img2], return_dict=True)

    assert len(results) == 2, "Should return results for two images"

    assert isinstance(results[0], list), (
        "Result should be a list of dicts when return_dict=True"
    )
    assert isinstance(results[1], list), (
        "Result should be a list of dicts when return_dict=True"
    )

