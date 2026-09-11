def test_import():
    try:
        import fast_face
        from fast_face import models
    except ImportError as e:
        assert False, f"Failed to import fast_face: {e}"
