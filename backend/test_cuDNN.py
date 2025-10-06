# python - <<'PY'
import ctypes
import logging


def check_cudnn() -> None:
    """Attempt to load cuDNN and log a success message."""
    ctypes.CDLL("libcudnn_ops.so.9")
    logging.getLogger(__name__).info("cuDNN 9 OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_cudnn()
# PY
