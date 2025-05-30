import os
import threading
import logging

from download_model import download_ministral_model
from download_model_optimized import download_ministral_model_optimized

_logger = logging.getLogger("model_manager")

# Internal state
_ready = False
_error = None
_lock = threading.Lock()
_thread = None


def start_background_download():
    """
    Starts the model download in a background thread.
    If a download is already in progress or completed, this is a no-op.
    """
    global _thread
    with _lock:
        if _thread is not None:
            _logger.debug("Download has already been started.")
            return
        _thread = threading.Thread(target=_run_download, daemon=True)
        _thread.start()
        _logger.info("Background download thread started.")


def is_ready() -> bool:
    """
    Returns True if the model download (and conversion) completed successfully.
    Returns False if still in progress or if it failed.
    """
    with _lock:
        return _ready


def get_error() -> str:
    """
    If the download failed or threw an exception, returns the error message.
    Otherwise returns None.
    """
    with _lock:
        return _error


def _run_download():
    """
    Internal helper to run the download logic and update state flags.
    Chooses between standard and optimized download based on DOWNLOAD_MODE env var.
    """
    global _ready, _error
    try:
        mode = os.environ.get("DOWNLOAD_MODE", "").strip().lower()
        if mode == "optimized":
            _logger.info("Using optimized download logic.")
            try:
                success = download_ministral_model_optimized()
            except TypeError:
                success = download_ministral_model_optimized()
        else:
            _logger.info("Using standard download logic.")
            try:
                success = download_ministral_model()
            except TypeError:
                success = download_ministral_model()

        with _lock:
            if success:
                _ready = True
                _logger.info("Model is ready for use.")
            else:
                _error = "Download logic returned failure."
                _logger.error("Model download failed.")
    except Exception as exc:
        with _lock:
            _error = str(exc)
        _logger.exception("Exception occurred during background download.")
