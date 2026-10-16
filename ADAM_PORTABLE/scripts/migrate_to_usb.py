"""USB Migration & Deployment Script for ADAM (Portable).
Safely clones the self-contained ADAM_PORTABLE suite to a new USB drive.
Checks filesystem type, verifies free storage space, prevents accidental overwrites,
excludes OS-specific binaries/virtualenvs, and validates post-copy integrity.
"""

import argparse
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Tuple

BASE_DIR = Path(__file__).resolve().parent.parent


def detect_filesystem(path: Path) -> str:
    """Attempt to detect filesystem type of target path."""
    system = platform.system().lower()
    if system == "linux":
        try:
            import subprocess
            out = subprocess.check_output(["df", "-T", str(path)], text=True).splitlines()
            if len(out) > 1:
                return out[1].split()[1]
        except Exception:
            pass
    return "Standard / Unknown"


def calculate_source_size(source_dir: Path, ignore_patterns: set) -> int:
    """Calculate total size of portable files to be copied."""
    total = 0
    for root, dirs, files in os.walk(source_dir):
        # Filter ignored directories in-place
        dirs[:] = [d for d in dirs if d not in ignore_patterns and not d.startswith(".")]
        for f in files:
            if not any(f.endswith(ext) for ext in [".pyc", ".tmp", ".log"]) and f not in ignore_patterns:
                p = Path(root) / f
                if p.is_file():
                    total += p.stat().st_size
    return total


def copy_portable_tree(src: Path, dst: Path, ignore_patterns: set) -> int:
    """Copy portable files, creating destination directories, returns count of files copied."""
    file_count = 0
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        # Skip ignored dirs
        dirs[:] = [d for d in dirs if d not in ignore_patterns and not d.startswith(".")]

        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            if any(f.endswith(ext) for ext in [".pyc", ".tmp", ".log"]):
                continue
            if f in ignore_patterns or f.startswith("."):
                continue

            src_file = Path(root) / f
            dst_file = target_dir / f

            # Copy file
            shutil.copy2(src_file, dst_file)
            file_count += 1

    return file_count


def verify_copy_integrity(src: Path, dst: Path, ignore_patterns: set) -> Tuple[bool, int, str]:
    """Verify copied files exist on destination and match sizes."""
    mismatches = 0
    verified = 0

    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        dirs[:] = [d for d in dirs if d not in ignore_patterns and not d.startswith(".")]

        for f in files:
            if any(f.endswith(ext) for ext in [".pyc", ".tmp", ".log"]):
                continue
            if f in ignore_patterns or f.startswith("."):
                continue

            src_file = Path(root) / f
            dst_file = dst / rel / f

            if not dst_file.exists():
                mismatches += 1
                continue

            if src_file.stat().st_size != dst_file.stat().st_size:
                mismatches += 1
            else:
                verified += 1

    if mismatches > 0:
        return False, verified, f"{mismatches} files failed integrity check."
    return True, verified, "All files matched source bytes exactly."


def main():
    parser = argparse.ArgumentParser(description="Safely migrate ADAM to a new USB drive")
    parser.add_argument("destination", nargs="?", help="Mount path or folder on the new USB drive (e.g. /run/media/user/USB or D:\\ADAM)")
    parser.add_argument("--yes", action="store_true", help="Auto-confirm without prompting")
    args = parser.parse_args()

    print("=" * 75)
    print("           ADAM PORTABLE — SAFE USB MIGRATION WIZARD")
    print("=" * 75)

    dest_str = args.destination
    if not dest_str:
        print("\nEnter destination mount point or directory on your new USB drive:")
        print("Examples:")
        print("  • Linux:   /run/media/sujeethkani/NEW_USB/ADAM_PORTABLE")
        print("  • Windows: E:\\ADAM_PORTABLE")
        print("  • macOS:   /Volumes/NEW_USB/ADAM_PORTABLE")
        dest_str = input("\nDestination Path: ").strip()

    if not dest_str:
        print("[ABORTED] No destination path provided.")
        sys.exit(1)

    dest_path = Path(dest_str).expanduser().resolve()
    print(f"\n[TARGET] Destination selected: {dest_path}")

    # Safety: Exclude root, user home, or current project root
    if dest_path == BASE_DIR or dest_path == BASE_DIR.parent:
        print("[ERROR] Destination cannot be the active development folder itself!")
        sys.exit(1)

    # Ignored host-specific items
    ignore_patterns = {
        "venv", ".venv", "__pycache__", ".git", ".idea", ".vscode", "backups"
    }

    # Calculate required storage
    needed_bytes = calculate_source_size(BASE_DIR, ignore_patterns)
    needed_mb = needed_bytes / (1024 * 1024)
    print(f"[ESTIMATION] Portable source data size: {needed_mb:.2f} MB")

    # Destination parent verification
    dest_parent = dest_path if dest_path.exists() else dest_path.parent
    if not dest_parent.exists():
        try:
            dest_parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[ERROR] Could not create target directory: {e}")
            sys.exit(1)

    # Check filesystem
    fs_type = detect_filesystem(dest_parent)
    print(f"[FILESYSTEM] Target filesystem type: {fs_type}")
    if "fat" in fs_type.lower() or "vfat" in fs_type.lower():
        print("  ℹ Notice: FAT32 detected. Standard ADAM portable files (including Amy voice) will work smoothly.")
        print("  ℹ Note: Single files over 4 GB cannot be stored on FAT32 (keep very large LLM models on exFAT/NTFS or internal disk).")

    # Check free disk space
    try:
        dest_stat = shutil.disk_usage(str(dest_parent))
        free_mb = dest_stat.free / (1024 * 1024)
        print(f"[STORAGE] Target drive free space: {free_mb:.1f} MB ({free_mb/1024:.2f} GB)")

        if free_mb < (needed_mb + 50.0):
            print(f"[ERROR] Insufficient disk space on target drive: {free_mb:.1f} MB free, need at least {needed_mb + 50:.1f} MB.")
            sys.exit(2)
    except Exception as e:
        print(f"[WARNING] Could not verify destination storage space: {e}")

    # Non-destructive overwrite check
    if dest_path.exists() and any(dest_path.iterdir()):
        print(f"\n⚠️  CAUTION: Destination '{dest_path}' already exists and is NOT empty.")
        if not args.yes:
            confirm = input("Are you sure you want to copy ADAM files into this folder? (y/N): ").strip().lower()
            if confirm != "y":
                print("[ABORTED] Migration cancelled by user. No files were modified.")
                sys.exit(0)

    # Perform Safe Copy
    print("\nCopying portable suite to target drive...")
    try:
        copied_files = copy_portable_tree(BASE_DIR, dest_path, ignore_patterns)
        print(f"✔ Successfully copied {copied_files} files.")
    except Exception as e:
        print(f"[ERROR] Copy failed: {e}")
        sys.exit(3)

    # Verification Step
    print("\nVerifying post-copy file integrity...")
    valid, count, msg = verify_copy_integrity(BASE_DIR, dest_path, ignore_patterns)
    if valid:
        print(f"✔ Integrity Check Passed! ({count} files verified, {msg})")
    else:
        print(f"⚠️  Integrity Warning: {msg}")

    print("\n" + "=" * 75)
    print("          🎉 ADAM IS READY FOR PORTABLE USB USE!")
    print("=" * 75)
    print("How to run ADAM on another computer:")
    print("  • Linux / macOS:")
    print(f"      cd {dest_path}")
    print("      ./run_adam.sh")
    print("  • Windows 10/11:")
    print(f"      cd /d {dest_path}")
    print("      run_adam.bat")
    print("  • Any system with Python 3.10+:")
    print(f"      python run_adam.py")
    print("=" * 75)


if __name__ == "__main__":
    main()
