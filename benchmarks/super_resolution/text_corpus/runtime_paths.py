"""Explicit portable tools for source rendering and OCR."""

import os
from pathlib import Path
import shutil


def chrome():
    configured = os.environ.get("ISS_CHROME")
    if configured:
        return configured
    mac = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if mac.is_file():
        return str(mac)
    # None tells Playwright to use its installed, versioned Chromium bundle.
    return shutil.which("google-chrome") or None


def tessdata():
    configured = os.environ.get("ISS_TESSDATA") or os.environ.get("TESSDATA_PREFIX")
    candidates = (
        [Path(configured)] if configured else [Path("/opt/homebrew/share/tessdata")]
    )
    executable = shutil.which("tesseract")
    if executable:
        candidates.append(Path(executable).resolve().parent.parent / "share/tessdata")
    candidates.extend(
        [Path("/usr/share/tesseract-ocr/5/tessdata"), Path("/usr/share/tessdata")]
    )
    for folder in candidates:
        if (folder / "eng.traineddata").is_file():
            return folder
    raise FileNotFoundError(
        "Set ISS_TESSDATA to the installed OCR language-model directory"
    )


def font():
    for name in [
        os.environ.get("ISS_PREVIEW_FONT", ""),
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if name and Path(name).is_file():
            return name
    return "DejaVuSans.ttf"
