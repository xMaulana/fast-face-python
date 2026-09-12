import logging
import os
import urllib.request

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

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
    temp_path = f"{target_path}.tmp"

    logger.info(f"Model not found locally. Downloading {model_filename} from {url}...")

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req) as response:
            total_size = response.headers.get("Content-Length")
            total_bytes = int(total_size) if total_size is not None else None

            with open(temp_path, "wb") as out_file:
                chunk_size = 1024 * 1024  # 1 MB
                if tqdm is not None:
                    with tqdm(
                        total=total_bytes,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading {model_filename}",
                    ) as pbar:
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            out_file.write(chunk)
                            pbar.update(len(chunk))
                else:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)

        os.replace(temp_path, target_path)
        logger.info(f"Successfully downloaded to {target_path}")
    except Exception as e:
        logger.error(f"Failed to download model {model_filename}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(target_path):
            os.remove(target_path)
        raise RuntimeError(
            f"Failed to download {model_filename} from {url}. Error: {e}"
        ) from e
