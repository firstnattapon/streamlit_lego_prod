"""Render cheat_sheet/*.mmd to PNG next to them. Needs Node.js (npx).

    python tools/render_cheat_sheet.py

For every ``cheat_sheet/<name>.mmd`` this writes ``cheat_sheet/<name>.<sha8>.png`` where
``<sha8>`` is the first 8 hex digits of the SHA-256 of the .mmd text. The dashboard only
shows an image whose name matches the current source, so a .mmd that was edited but not
rendered again is reported instead of being shown stale. Older images of the same diagram
are deleted.

The first run downloads Mermaid CLI (and puppeteer's Chromium). In a container set
PUPPETEER_CONFIG to a JSON file such as {"args": ["--no-sandbox"]}.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

CHEAT_SHEET_DIR = Path(__file__).resolve().parents[1] / "cheat_sheet"
MERMAID_CLI = "@mermaid-js/mermaid-cli@11.17.0"   # pinned: it bundles Mermaid 11.17.x
SCALE = 3                                         # 3x pixels: the browser shrinks it to the column, so it stays sharp


def source_digest(source: str) -> str:
    """Same rule as ``mmd_digest`` in streamlit_app.py (a test keeps the two in step)."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:8]


def render(mmd: Path) -> Path:
    target = mmd.with_name(f"{mmd.stem}.{source_digest(mmd.read_text(encoding='utf-8'))}.png")
    npx = shutil.which("npx")
    if npx is None:
        sys.exit("npx not found - install Node.js first")
    command = [npx, "-y", MERMAID_CLI, "-i", str(mmd), "-o", str(target), "-s", str(SCALE),
               "-b", "white", "-c", str(CHEAT_SHEET_DIR / "mermaid.config.json")]
    if os.environ.get("PUPPETEER_CONFIG"):
        command += ["-p", os.environ["PUPPETEER_CONFIG"]]
    subprocess.run(command, check=True)
    for old in mmd.parent.glob(f"{mmd.stem}.*.png"):
        if old != target:
            old.unlink()
    return target


def main() -> None:
    for mmd in sorted(CHEAT_SHEET_DIR.glob("*.mmd")):
        print(render(mmd).relative_to(CHEAT_SHEET_DIR.parent))


if __name__ == "__main__":
    main()
