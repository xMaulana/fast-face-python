import os
import pytest
import numpy as np
from fast_face.models.factory import FaceModelFactory


def test_yunet_batched_inference():
    model_path = os.getenv(
        "FAST_FACE_DIR_PATH",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../src/fast_face/models/yunet.onnx",
        ),
    )

    if not os.path.exists(model_path):
        pytest.skip(f"Skipping test because model not found at {model_path}")

    model = FaceModelFactory.get_model("YUNET", model_path=model_path, top_k=500)

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
