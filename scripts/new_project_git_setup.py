#!/usr/bin/env python3
from __future__ import annotations

import logging
import pathlib
import re
import shutil
import subprocess  # nosec B404  # Used for legitimate git commands only

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Find git executable path for security
GIT_EXECUTABLE = shutil.which("git")


def validate_git_url(url: str) -> bool:
    """Validate that a URL looks like a valid git remote URL."""
    if not url or url == "skip":
        return True
    # Allow common git URL patterns: https://, git@, ssh://, file://
    patterns = [
        r"^https?://[\w\-\.]+/[\w\-\.]+/[\w\-\.]+\.git$",
        r"^git@[\w\-\.]+:[\w\-\.]+/[\w\-\.]+\.git$",
        r"^ssh://[\w\-\.@]+/[\w\-\.]+/[\w\-\.]+\.git$",
        r"^file:///.+\.git$",
    ]
    return any(re.match(pattern, url) for pattern in patterns)


def validate_branch_name(branch: str) -> bool:
    """Validate that a branch name is safe to use."""
    if not branch:
        return True
    # Disallow dangerous characters and patterns
    if any(char in branch for char in [" ", "\n", "\r", "\t", ";", "&", "|", "$", "`"]):
        return False
    # Disallow starting with - or .
    if branch.startswith(("-", ".")):
        return False
    # Disallow containing ..
    if ".." in branch:
        return False
    return True


def main() -> int:
    if not GIT_EXECUTABLE:
        logger.error("git executable not found in PATH")
        return 1
    root = pathlib.Path(".").resolve()
    pyproject = root / "pyproject.toml"
    origin_file = root / ".bootstrap_origin_url"

    if not pyproject.exists():
        logger.error("pyproject.toml not found; cannot derive project slug.")
        return 1

    import tomllib

    data = tomllib.loads(pyproject.read_text())
    slug = data.get("project", {}).get("name")
    if not slug:
        logger.error("Project name missing in pyproject.toml; cannot set git remote.")
        return 1

    old_url = origin_file.read_text().strip() if origin_file.exists() else ""
    new_url = ""
    if old_url:
        # Replace the repo name (last path component before .git) with the new slug
        match = re.match(r"^(.*/)([^/]+?)(\.git)?$", old_url)
        if match:
            base_url, _old_repo, git_ext = match.groups()
            new_url = f"{base_url}{slug}{git_ext or '.git'}"
    if not new_url:
        new_url = f"https://github.com/UNKNOWN/{slug}.git"

    logger.info("Planned git setup values:")
    logger.info(f"- repo root: {root}")
    logger.info(f"- project slug (from pyproject): {slug}")
    logger.info(f"- default origin URL: {new_url}")
    logger.info("- default branch: main")

    def prompt(message: str, default: str | None = None) -> str:
        suffix = f" [{default}]" if default is not None else ""
        try:
            reply = input(f"{message}{suffix}: ").strip()
        except EOFError:
            return default or ""
        if reply:
            return reply
        return default or ""

    proceed = prompt("Proceed with git setup using the values above?", "y").lower()
    if not proceed.startswith("y"):
        logger.info("Aborted.")
        return 0

    if not (root / ".git").exists():
        logger.info(f"Initializing git repository in {root}...")
        subprocess.run([GIT_EXECUTABLE, "init"], cwd=root, check=True)  # nosec B603

    existing_origin = ""
    try:
        existing_origin = subprocess.check_output(  # nosec B603
            [GIT_EXECUTABLE, "remote", "get-url", "origin"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        existing_origin = ""
    if existing_origin:
        remote_default = "skip"
        remote_prompt = f"Replace existing origin ({existing_origin}) with new URL (or press Enter to skip)"
    else:
        remote_default = new_url
        remote_prompt = "Set remote origin URL (press Enter to use default, or type new URL)"
    remote_reply = prompt(remote_prompt, remote_default or "skip")
    if remote_reply and remote_reply != "skip":
        if not validate_git_url(remote_reply):
            logger.error(f"Invalid git URL format: {remote_reply}")
            logger.info("URL must match: https://github.com/user/repo.git or git@github.com:user/repo.git")
            return 1
        if existing_origin:
            confirm = prompt(f"Really replace origin '{existing_origin}' with '{remote_reply}'?", "n").lower()
            if confirm.startswith("y"):
                subprocess.run(  # nosec B603
                    [GIT_EXECUTABLE, "remote", "set-url", "origin", remote_reply], cwd=root, check=True
                )
            else:
                logger.info("Origin update skipped.")
        else:
            subprocess.run(  # nosec B603
                [GIT_EXECUTABLE, "remote", "add", "origin", remote_reply], cwd=root, check=True
            )
    else:
        logger.info("Origin update skipped.")

    branch_reply = prompt("Default branch name (press Enter to use 'main', or type different name)", "main")
    if branch_reply:
        if not validate_branch_name(branch_reply):
            logger.error(f"Invalid branch name: {branch_reply}")
            logger.info("Branch names must not contain spaces, special characters, or start with - or .")
            return 1
        subprocess.run([GIT_EXECUTABLE, "branch", "-M", branch_reply], cwd=root, check=True)  # nosec B603

    if origin_file.exists():
        origin_file.unlink()

    logger.info("Git setup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
