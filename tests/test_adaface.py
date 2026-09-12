import numpy as np

from fast_face.models.factory import FaceModelFactory


def test_adaface_ir18_extract_without_landmarks():
    """Test AdaFace IR18 with pre-aligned images (no landmarks)."""
    model = FaceModelFactory.get_model("ADAFACE_IR18")

    face1 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    face2 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

    embeddings = model.extract([face1, face2])

    assert embeddings.shape[0] == 2, "Should return embeddings for two images"
    assert embeddings.shape[1] > 0, "Embedding dimension should be > 0"

    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_adaface_ir18_extract_with_landmarks():
    """Test AdaFace IR18 with landmark-based alignment."""
    model = FaceModelFactory.get_model("ADAFACE_IR18")

    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    landmarks = np.array(
        [
            [200.0, 200.0],
            [280.0, 200.0],
            [240.0, 250.0],
            [210.0, 300.0],
            [270.0, 300.0],
        ],
        dtype=np.float32,
    )

    embeddings = model.extract([img], landmarks=[landmarks])

    assert embeddings.shape[0] == 1, "Should return one embedding"
    assert embeddings.shape[1] > 0, "Embedding dimension should be > 0"

    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_adaface_ir50_extract_without_landmarks():
    """Test AdaFace IR50 with pre-aligned images (no landmarks)."""
    model = FaceModelFactory.get_model("ADAFACE_IR50")

    face1 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    face2 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

    embeddings = model.extract([face1, face2])

    assert embeddings.shape[0] == 2, "Should return embeddings for two images"
    assert embeddings.shape[1] > 0, "Embedding dimension should be > 0"

    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_adaface_ir50_extract_with_landmarks():
    """Test AdaFace IR50 with landmark-based alignment."""
    model = FaceModelFactory.get_model("ADAFACE_IR50")

    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    landmarks = np.array(
        [
            [200.0, 200.0],
            [280.0, 200.0],
            [240.0, 250.0],
            [210.0, 300.0],
            [270.0, 300.0],
        ],
        dtype=np.float32,
    )

    embeddings = model.extract([img], landmarks=[landmarks])

    assert embeddings.shape[0] == 1, "Should return one embedding"
    assert embeddings.shape[1] > 0, "Embedding dimension should be > 0"

    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_adaface_ir101_extract_without_landmarks():
    """Test AdaFace IR101 with pre-aligned images (no landmarks)."""
    model = FaceModelFactory.get_model("ADAFACE_IR101")

    face1 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    face2 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

    embeddings = model.extract([face1, face2])

    assert embeddings.shape[0] == 2, "Should return embeddings for two images"
    assert embeddings.shape[1] > 0, "Embedding dimension should be > 0"

    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_adaface_ir101_extract_with_landmarks():
    """Test AdaFace IR101 with landmark-based alignment."""
    model = FaceModelFactory.get_model("ADAFACE_IR101")

    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    landmarks = np.array(
        [
            [200.0, 200.0],
            [280.0, 200.0],
            [240.0, 250.0],
            [210.0, 300.0],
            [270.0, 300.0],
        ],
        dtype=np.float32,
    )

    embeddings = model.extract([img], landmarks=[landmarks])

    assert embeddings.shape[0] == 1, "Should return one embedding"
    assert embeddings.shape[1] > 0, "Embedding dimension should be > 0"

    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
