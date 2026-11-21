"""Connect to the AutoE2E source code (the "upstream" repo) and check its version.

AutoE2E is not pip-installable, so we put its folders on Python's import path.
The released checkpoints only load with the exact source revision they were trained
with, so we refuse to run against any other commit.
"""
import os
import subprocess
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]

# Revision named in the Hugging Face release config.json ("source.revision").
EXPECTED_COMMIT = "40a75cf8a26b60f6b094125e3924fe0bdb427b02"

# Default: the auto_e2e clone sits next to this repo. Override with AUTOE2E_REPO=/path.
AUTOE2E_REPO = Path(os.environ.get("AUTOE2E_REPO", LAB_ROOT.parent / "auto_e2e")).resolve()


def _current_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def setup() -> None:
    """Check the AutoE2E clone and make its modules importable. Safe to call twice."""
    if str(AUTOE2E_REPO / "Model") in sys.path:
        return
    if not (AUTOE2E_REPO / "Model").is_dir():
        raise FileNotFoundError(
            f"AutoE2E source not found at {AUTOE2E_REPO}. Clone it there or set AUTOE2E_REPO."
        )
    commit = _current_commit(AUTOE2E_REPO)
    if commit != EXPECTED_COMMIT and not os.environ.get("AUTOE2E_ALLOW_ANY_COMMIT"):
        raise RuntimeError(
            f"AutoE2E is at commit {commit[:8]}, but the checkpoints need {EXPECTED_COMMIT[:8]}.\n"
            f"Fix:  git -C {AUTOE2E_REPO} switch v1-checkpoint\n"
            "(Set AUTOE2E_ALLOW_ANY_COMMIT=1 to experiment anyway.)"
        )
    sys.path.insert(0, str(AUTOE2E_REPO / "Model"))  # model_components, training, ...
    sys.path.insert(1, str(AUTOE2E_REPO))            # Tools.trajectory_visualization


setup()
