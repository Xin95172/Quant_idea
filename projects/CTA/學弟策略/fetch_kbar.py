"""Deprecated Quant-side entrypoint for 1-minute Taiwan stock k-bar updates.

Data downloads are owned by the note repo. Run this instead:

    python <note-repo>/scripts/data_updates/fetch_tw_stock_kbar_1min.py
"""

from __future__ import annotations

import sys
from pathlib import Path

QUANT_ROOT = Path(__file__).resolve().parents[3]
if str(QUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ROOT))

from project_paths import NOTE_REPO_ROOT


def main() -> None:
    raise RuntimeError(
        "Quant is read/write-only for data. "
        f"Run {NOTE_REPO_ROOT / 'scripts/data_updates/fetch_tw_stock_kbar_1min.py'} "
        "to download or refresh k-bar data."
    )


if __name__ == "__main__":
    main()
