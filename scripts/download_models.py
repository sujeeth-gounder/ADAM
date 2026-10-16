"""Download script for Piper TTS Amy English Voice models.
Downloads:
- en_US-amy-medium.onnx
- en_US-amy-medium.onnx.json
Into models/piper/
"""

import sys
import os
from pathlib import Path
import urllib.request
import urllib.error

# Project roots
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "piper"

VOICE_FILES = {
    "en_US-amy-medium.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx",
    "en_US-amy-medium.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
}

def download_file_with_progress(url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[EXISTS] {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
        return True

    print(f"[DOWNLOADING] {dest_path.name} from {url}...")
    temp_path = dest_path.with_suffix(".tmp")
    
    try:
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100.0, downloaded * 100.0 / total_size)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  -> Progress: {percent:5.1f}% [{mb_down:.1f} MB / {mb_total:.1f} MB]")
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\r  -> Downloaded: {downloaded / 1024:.1f} KB")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, temp_path, reporthook)
        print()  # newline
        temp_path.rename(dest_path)
        print(f"[OK] Saved {dest_path.name} successfully.")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to download {dest_path.name}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("ADAM — Piper TTS Amy English Voice Downloader")
    print(f"Destination: {MODELS_DIR}")
    print("=" * 60)

    success_all = True
    for filename, url in VOICE_FILES.items():
        dest = MODELS_DIR / filename
        ok = download_file_with_progress(url, dest)
        if not ok:
            success_all = False

    if success_all:
        print("\nAll Piper Amy voice models downloaded and ready!")
    else:
        print("\nWarning: Some voice files failed to download. You can re-run this script later.")

if __name__ == "__main__":
    main()
