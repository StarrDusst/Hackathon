"""Download the official, COCO-pretrained Ultralytics YOLOv8n checkpoint."""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
DEST = Path(__file__).resolve().parent / "yolov8n.pt"
EXPECTED_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if DEST.is_file() and DEST.stat().st_size > 1_000_000:
        actual = sha256(DEST)
        if not EXPECTED_SHA256 or actual == EXPECTED_SHA256:
            print(f"Using existing checkpoint: {DEST} ({DEST.stat().st_size:,} bytes, sha256={actual})")
            return
        print("Existing checkpoint hash differs; downloading a fresh copy.")
    temp = DEST.with_suffix(".pt.part")
    print(f"Downloading {URL}")
    urlretrieve(URL, temp)
    if temp.stat().st_size < 1_000_000:
        temp.unlink(missing_ok=True)
        raise RuntimeError("Downloaded checkpoint is unexpectedly small; URL/network response may be invalid.")
    actual = sha256(temp)
    if EXPECTED_SHA256 and actual != EXPECTED_SHA256:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual}")
    temp.replace(DEST)
    print(f"Saved {DEST} ({DEST.stat().st_size:,} bytes, sha256={actual})")


if __name__ == "__main__":
    main()
