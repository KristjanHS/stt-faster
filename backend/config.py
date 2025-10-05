import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv
from rich.logging import RichHandler

# Load environment variables from the project root .env (if present)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Model names (single source of truth)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_OLLAMA_MODEL = "cas/mistral-7b-instruct-v0.3"

# Working model names (with environment variable overrides)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
RERANKER_MODEL = os.getenv("RERANK_MODEL", DEFAULT_RERANKER_MODEL)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

# Model paths and caching
HF_CACHE_DIR = os.getenv("HF_HOME", "/data/hf")


# --- Logging Configuration ---
# runtime scripts should use logging.getLogger(...) and env LOG_LEVEL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_logging_configured = False


def _parse_level(value: str | None, default: int = logging.INFO) -> int:
    if not value:
        return default
    name = value.strip().upper()
    return getattr(logging, name, default)


def _build_console_handler(level: int) -> logging.Handler:
    """Return a console handler. Use Rich in TTY, plain stream otherwise."""
    if sys.stderr.isatty():
        handler = RichHandler(
            show_time=False,
            show_path=False,
            rich_tracebacks=True,
            show_level=True,
            log_time_format="[%X]",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.setLevel(level)
    return handler


def _setup_logging() -> None:
    """Configure the root logger once with console and optional file handler."""
    global _logging_configured
    if _logging_configured:
        return

    level = _parse_level(os.getenv("LOG_LEVEL", "INFO"))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(_build_console_handler(level))

    # Optional file logging only when APP_LOG_DIR is set
    app_log_dir = os.getenv("APP_LOG_DIR")
    if app_log_dir:
        log_dir = Path(app_log_dir)
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            backup_count_env = os.getenv("APP_LOG_BACKUP_COUNT", "5")
            try:
                backup_count = max(0, int(str(backup_count_env).strip()))
            except (TypeError, ValueError):
                backup_count = 5
                logging.warning(
                    "Invalid APP_LOG_BACKUP_COUNT=%r. Defaulting to 5.",
                    backup_count_env,
                )

            file_handler = TimedRotatingFileHandler(
                filename=log_dir / "rag_system.log",
                when="midnight",
                backupCount=backup_count,
                encoding="utf-8",
                utc=True,
            )
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            file_handler.setLevel(level)
            root.addHandler(file_handler)
        except (PermissionError, OSError) as e:
            logging.warning("Failed to configure file logging to '%s'. Error: %s", app_log_dir, e)

    # Reduce noise from common libraries
    for noisy in (
        "httpx",
        "urllib3",
        "requests",
        "sentence_transformers",
        "transformers",
        "torch",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Very noisy PDF internals
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    logging.getLogger("pypdf.generic._base").setLevel(logging.ERROR)

    # Forward warnings module messages to logging
    logging.captureWarnings(True)

    _logging_configured = True


# --- End Logging Configuration ---
_setup_logging()
