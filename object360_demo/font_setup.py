"""Register the bundled Thai font for this process before Tk starts."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path


FONT_FAMILY = "Noto Sans Thai"


def register_font() -> bool:
    if sys.platform != "win32":
        return False
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    # FR_PRIVATE makes the font available to this app without installing it
    # system-wide or requiring administrator access.
    registered = False
    for name in ("NotoSansThai-Regular.ttf", "NotoSansThai-Bold.ttf"):
        font_path = base / "fonts" / name
        if font_path.is_file():
            registered |= bool(ctypes.windll.gdi32.AddFontResourceExW(str(font_path), 0x10, 0))
    return registered
