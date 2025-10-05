import importlib
import logging
from pathlib import Path


def _clear_root():
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    root.setLevel(logging.NOTSET)


def _console_handler() -> logging.Handler | None:
    for h in logging.getLogger().handlers:
        if not isinstance(h, logging.FileHandler):
            return h
    return None


def test_env_driven_log_level_on_init(monkeypatch):
    import backend.config as cfg

    _clear_root()
    # Set desired level via env and reload the module to re-init logging
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setattr(cfg.sys.stderr, "isatty", lambda: False, raising=False)
    importlib.reload(cfg)
    assert logging.getLogger().level == logging.DEBUG
    assert _console_handler().level == logging.DEBUG

    # Change env to WARNING and reload; levels should follow
    _clear_root()
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    importlib.reload(cfg)
    assert logging.getLogger().level == logging.WARNING
    assert _console_handler().level == logging.WARNING


def test_file_logging_creates_logfile(tmp_path, monkeypatch):
    import backend.config as cfg

    _clear_root()
    importlib.reload(cfg)
    monkeypatch.setenv("APP_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(cfg.sys.stderr, "isatty", lambda: False, raising=False)

    # Reload again to pick up APP_LOG_DIR and emit a record
    importlib.reload(cfg)
    logging.getLogger("t").info("hello")

    # Expect rotated file handler targeting rag_system.log
    assert (Path(tmp_path) / "rag_system.log").exists()
    assert any(isinstance(h, logging.handlers.TimedRotatingFileHandler) for h in logging.getLogger().handlers)
