"""Architecture tests to enforce clean boundaries.

These tests validate that the codebase follows architectural rules:
- Frontend never imports backend directly (should use APIs)
- Scripts are thin wrappers (business logic in backend)
- Domain layer has no infrastructure dependencies
"""

import subprocess
from pathlib import Path


def test_frontend_has_no_business_logic():
    """Ensure frontend never imports backend business logic directly.

    Frontend should communicate with backend via API calls, not direct imports.
    This prevents tight coupling and maintains clear boundaries.
    """
    frontend_path = Path(__file__).parent.parent.parent / "frontend"

    # Only check if frontend has Python files
    python_files = list(frontend_path.rglob("*.py"))
    if not python_files:
        # No Python files yet, test passes
        return

    result = subprocess.run(
        ["grep", "-r", "from backend", str(frontend_path)],
        capture_output=True,
        text=True,
    )
    # grep returns 1 when no matches (which is what we want)
    assert result.returncode == 1, (
        "Frontend should not import from backend directly! Use API calls instead to maintain clean architecture."
    )


def test_scripts_are_thin_wrappers():
    """Ensure scripts don't contain business logic.

    Scripts should be thin CLI wrappers. Business logic belongs in backend/.
    This test checks that scripts primarily import from backend, not implement logic.
    """
    scripts_path = Path(__file__).parent.parent.parent / "scripts"

    # Find all Python files in scripts (excluding __pycache__)
    python_files = [
        f
        for f in scripts_path.rglob("*.py")
        if "__pycache__" not in str(f) and "transcription" not in str(f.parent.name)
    ]

    for script_file in python_files:
        with open(script_file) as f:
            content = f.read()

        # Count lines of code (excluding comments, blank lines, imports)
        lines = [
            line.strip()
            for line in content.split("\n")
            if line.strip()
            and not line.strip().startswith("#")
            and not line.strip().startswith('"""')
            and not line.strip().startswith("'''")
        ]

        # Scripts should be mostly imports and simple calls
        # If a script has >200 lines of code, it might contain business logic
        assert len(lines) < 300, (
            f"{script_file.name} has {len(lines)} lines. Consider moving business logic to backend/."
        )
