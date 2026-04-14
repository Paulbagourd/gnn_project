#!/usr/bin/env python
"""
Utility to gather all LaTeX assets inside docs/ so the folder can be uploaded to
Overleaf without manual copying.

It mirrors selected source directories (plots, figures, etc.) into docs/assets/
while preserving relative paths. Run it after regenerating plots.
"""
from __future__ import annotations

import argparse
import shutil
import os
import stat
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"

# Source directories whose contents (PDF/PNG/SVG images) we want to mirror.
ASSET_SOURCES = [
    PROJECT_ROOT / "data/usecase_cyberspace/03_graph/outputs/plots",
]

# File extensions worth copying for the LaTeX document.
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".svg"}


def _handle_remove_readonly(func, path, exc_info):
    # Make read-only files writable then retry.
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        raise exc_info[1]


def sync_assets(clean: bool = False) -> None:
    seen_roots: set[Path] = set()
    for source in ASSET_SOURCES:
        if not source.exists():
            continue

        dest_root = DOCS_DIR / source.relative_to(PROJECT_ROOT)
        seen_roots.add(dest_root)

        if clean and dest_root.exists():
            shutil.rmtree(dest_root, onerror=_handle_remove_readonly)

        for asset in source.rglob("*"):
            if asset.is_dir() or asset.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue

            relative = asset.relative_to(source)
            destination = dest_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(asset, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy generated plots/assets into docs/assets/ for Overleaf upload."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove docs/assets/ before copying to avoid stale files.",
    )
    args = parser.parse_args()

    DOCS_DIR.mkdir(exist_ok=True)
    sync_assets(clean=args.clean)


if __name__ == "__main__":
    main()
