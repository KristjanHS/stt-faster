import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_bash(snippet: str, env: dict | None = None, cwd: Path | None = None):
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", "-lc", snippet],
        cwd=str(cwd or REPO_ROOT),
        text=True,
        capture_output=True,
        env=full_env,
    )


def test_log_respects_level_and_format(tmp_path):
    # Non-TTY in CI: expect plain format "ts [LEVEL] msg"
    env = {
        "LOG_LEVEL": "WARN",
        "LOGS_DIR": str(tmp_path),
    }
    snippet = r"""
        source scripts/common.sh
        log INFO "hello-info"
        log WARN "hello-warn"
    """
    res = _run_bash(snippet, env=env)
    assert res.returncode == 0
    out = res.stdout
    assert "[WARN] hello-warn" in out
    assert "[INFO] hello-info" not in out


def test_init_script_logging_creates_symlink_and_file(tmp_path):
    env = {}
    snippet = rf'''
        source scripts/common.sh
        LOGS_DIR="{tmp_path}"
        f=$(init_script_logging "demo")
        echo "$f"
        if [ -L "$LOGS_DIR/demo.log" ]; then echo SYMLINK; fi
        if [ -f "$f" ]; then echo FILE_OK; fi
    '''
    res = _run_bash(snippet, env=env)
    assert res.returncode == 0
    out = res.stdout.splitlines()
    # first printed line is the absolute or relative path to the new log file
    log_path = Path(out[0]).resolve()
    assert (tmp_path / log_path.name).exists()
    assert "SYMLINK" in res.stdout
    assert "FILE_OK" in res.stdout
