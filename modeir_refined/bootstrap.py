from __future__ import annotations

import sys
from pathlib import Path


NEW_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = NEW_ROOT / "vendor"
PARENT_PROJECT_ROOT = NEW_ROOT.parent


def _has_portable_vendor(root: Path) -> bool:
    return (root / "ldm").is_dir() and (root / "universal_dataset.py").is_file()


PROJECT_ROOT = VENDOR_ROOT if _has_portable_vendor(VENDOR_ROOT) else PARENT_PROJECT_ROOT


def bootstrap_legacy_backend() -> None:
    """Expose the bundled or original LDM backend while keeping refined shims first."""
    new_root = str(NEW_ROOT)
    project_root = str(PROJECT_ROOT)
    if new_root in sys.path:
        sys.path.remove(new_root)
    sys.path.insert(0, new_root)
    if project_root not in sys.path:
        sys.path.insert(1, project_root)
