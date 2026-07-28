"""Cross-platform locations for the Quant workspace and its shared repos.

Environment variables take precedence so a non-standard checkout can be used
without editing source code.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path


def _default_github_root() -> Path:
    if platform.system() == "Windows":
        return Path(os.environ.get("USERPROFILE", Path.home())) / "Documents" / "GitHub"
    return Path("/Users/xinc/GitHub")


GITHUB_ROOT = Path(os.environ.get("GITHUB_ROOT", _default_github_root()))
QUANT_ROOT = Path(os.environ.get("QUANT_ROOT", GITHUB_ROOT / "Quant"))
NOTE_REPO_ROOT = Path(
    os.environ.get("DATA_DOWNLOAD_OWNER_ROOT", GITHUB_ROOT / "note")
)


def _default_data_root() -> Path:
    """Use the first shared-data folder that exists for this checkout."""
    candidates = (
        GITHUB_ROOT / "Data",
        GITHUB_ROOT / "google_drive" / "Data",
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


DATA_ROOT = Path(os.environ.get("DATA_ROOT", _default_data_root()))
