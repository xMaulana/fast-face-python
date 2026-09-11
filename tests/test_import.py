def test_import():
    try:
        import fast_face
        from fast_face import models

        assert fast_face is not None
        assert models is not None
    except ImportError as e:
        raise AssertionError(f"Failed to import fast_face: {e}")
