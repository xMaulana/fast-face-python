import logging
import os
import urllib.request

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)

BASE_URL = "https://mwmaulana.my.id/fast_face_onnx"


_TRANSIENT_ERRORS = (
    BrokenPipeError,
    ConnectionResetError,
    ConnectionAbortedError,
    TimeoutError,
    OSError,
    urllib.error.URLError,
)

_DEFAULT_MAX_RETRIES = 5
_DEFAULT_BACKOFF_BASE = 2  # seconds


def _get_total_size(url: str) -> int | None:
    """Send a HEAD request to get the total file size.

    Returns:
        The file size in bytes, or None if the server doesn't report it.
    """
    req = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    try:
        with urllib.request.urlopen(req) as response:
            content_length = response.headers.get("Content-Length")
            return int(content_length) if content_length is not None else None
    except Exception:
        return None


def download_model(
    model_filename: str,
    target_path: str,
    max_retries: int = _DEFAULT_MAX_RETRIES,
):
    """Download the model from the remote server if it doesn't exist.

    Supports resuming partial downloads on transient network failures
    (e.g. BrokenPipeError, ConnectionResetError) using HTTP Range headers.
    Retries up to ``max_retries`` times with exponential backoff.

    Args:
        model_filename (str): The name of the file to download (e.g. 'yunet.onnx').
        target_path (str): The local path where the file should be saved.
        max_retries (int): Maximum number of retry attempts on transient errors.
    """
    if os.path.exists(target_path):
        return

    url = f"{BASE_URL}/{model_filename}"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    temp_path = f"{target_path}.tmp"

    total_bytes = _get_total_size(url)

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        downloaded = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0

        if total_bytes is not None and downloaded >= total_bytes:
            logger.debug(
                f"Temp file already has {downloaded} bytes (>= {total_bytes}), finalizing."
            )
            break

        if attempt == 1 and downloaded == 0:
            logger.info(
                f"Model not found locally. Downloading {model_filename} from {url}..."
            )
        elif downloaded > 0:
            logger.info(
                f"Resuming download of {model_filename} from byte {downloaded} "
                f"(attempt {attempt}/{max_retries})..."
            )
        else:
            logger.info(
                f"Retrying download of {model_filename} "
                f"(attempt {attempt}/{max_retries})..."
            )

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        if downloaded > 0:
            headers["Range"] = f"bytes={downloaded}-"

        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req) as response:
                if response.status == 200 and downloaded > 0:
                    logger.warning(
                        "Server does not support Range requests. Restarting download."
                    )
                    downloaded = 0

                chunk_size = 1024 * 1024  # 1 MB
                open_mode = "ab" if downloaded > 0 else "wb"

                with open(temp_path, open_mode) as out_file:
                    if tqdm is not None:
                        with tqdm(
                            total=total_bytes,
                            initial=downloaded,
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

            last_error = None
            break

        except _TRANSIENT_ERRORS as e:
            last_error = e
            backoff = _DEFAULT_BACKOFF_BASE**attempt
            logger.warning(
                f"Download interrupted: {e}. "
                f"Retrying in {backoff}s (attempt {attempt}/{max_retries})..."
            )
            import time

            time.sleep(backoff)
        except Exception as e:
            logger.error(f"Failed to download model {model_filename}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise RuntimeError(
                f"Failed to download {model_filename} from {url}. Error: {e}"
            ) from e

    if last_error is not None:
        logger.error(
            f"Download of {model_filename} failed after {max_retries} attempts."
        )
        raise RuntimeError(
            f"Failed to download {model_filename} from {url} "
            f"after {max_retries} attempts. Error: {last_error}"
        ) from last_error

    os.replace(temp_path, target_path)
    logger.info(f"Successfully downloaded to {target_path}")
