"""Piper Amy Voice Model Downloader for ADAM (Portable).
Checks available storage before initiating download.
Downloads:
- en_US-amy-medium.onnx (~65 MB)
- en_US-amy-medium.onnx.json (~5 KB)
Into models/piper/
"""

import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "piper"

VOICE_FILES = {
    "en_US-amy-medium.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx",
    "en_US-amy-medium.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
}

def check_storage_space(required_mb: float = 100.0) -> bool:
    """Verify drive has enough free storage space."""
    try:
        stat = shutil.disk_usage(str(MODELS_DIR.parent))
        free_mb = stat.free / (1024 * 1024)
        if free_mb < required_mb:
            print(f"[ERROR] Insufficient storage: {free_mb:.1f} MB available, but {required_mb:.1f} MB required.")
            return False
        print(f"[STORAGE CHECK] {free_mb / 1024:.2f} GB free space available on active partition (Requirement: ~{required_mb:.0f} MB). OK.")
        return True
    except Exception as e:
        print(f"[WARNING] Could not determine disk space: {e}")
        return True

def download_file_with_progress(url: str, dest_path: Path) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"[EXISTS] {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
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
        print()
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
    print("=" * 65)
    print("ADAM Portable — Piper TTS Amy English Voice Downloader")
    print(f"Destination: {MODELS_DIR}")
    print("=" * 65)

    if not check_storage_space(required_mb=120.0):
        sys.exit(1)

    all_ok = True
    for fname, url in VOICE_FILES.items():
        dest = MODELS_DIR / fname
        if not download_file_with_progress(url, dest):
            all_ok = False

    if all_ok:
        print("\n✔ All Piper Amy voice models verified and ready for portable use!")
    else:
        print("\n⚠️  Some voice files failed to download. Check network connection and re-run.")

if __name__ == "__main__":
    main()
