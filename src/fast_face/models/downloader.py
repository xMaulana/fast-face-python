import os
import urllib.request
import logging
import shutil

logger = logging.getLogger(__name__)

BASE_URL = "https://mwmaulana.my.id/fast_face_onnx"


def download_model(model_filename: str, target_path: str):
    """Download the model from the remote server if it doesn't exist.

    Args:
        model_filename (str): The name of the file to download (e.g. 'yunet.onnx').
        target_path (str): The local path where the file should be saved.
    """
    if os.path.exists(target_path):
        return

    url = f"{BASE_URL}/{model_filename}"

    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    logger.info(f"Model not found locally. Downloading {model_filename} from {url}...")
    print(f"Downloading {model_filename} from {url}...")

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with (
            urllib.request.urlopen(req) as response,
            open(target_path, "wb") as out_file,
        ):
            shutil.copyfileobj(response, out_file)

        logger.info(f"Successfully downloaded to {target_path}")
        print(f"Successfully downloaded to {target_path}")
    except Exception as e:
        logger.error(f"Failed to download model {model_filename}: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)
        raise RuntimeError(
            f"Failed to download {model_filename} from {url}. Error: {e}"
        ) from e
