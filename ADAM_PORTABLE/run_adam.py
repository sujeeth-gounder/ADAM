"""Universal Cross-Platform Python Launcher for ADAM (Portable).
Works natively on Linux, Windows, and macOS without shell-specific dependencies.
Usage: python run_adam.py
"""

import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Base portable directory
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

def ensure_venv_execution():
    """If a local venv directory exists and we are not in it, re-exec using venv python."""
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        return

    # Check for local venv or root workspace venv
    venv_dir = ROOT_DIR / "venv"
    if not venv_dir.exists():
        venv_dir = ROOT_DIR.parent / "venv"
    if not venv_dir.exists():
        return

    # Determine venv python binary
    if platform.system().lower() == "windows":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    if venv_python.exists() and str(venv_python) != sys.executable:
        print(f"[LAUNCHER] Activating local virtualenv: {venv_python}")
        cmd = [str(venv_python), str(Path(__file__).resolve())] + sys.argv[1:]
        env = dict(os.environ)
        user_site = Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        if user_site.exists():
            existing_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{user_site}{os.pathsep}{existing_pp}" if existing_pp else str(user_site)
        try:
            sys.exit(subprocess.call(cmd, env=env))
        except Exception as e:
            print(f"[LAUNCHER] Notice: Could not re-exec into venv ({e}), continuing with {sys.executable}")

def check_and_start_ollama():
    """Detect if Ollama is installed and ensure the local service is running."""
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        return

    # Check if daemon is reachable
    import urllib.request
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                return  # already running
    except Exception:
        pass

    # Start in background
    print("[LAUNCHER] Starting local Ollama service in background...")
    try:
        if platform.system().lower() == "windows":
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
        else:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        time.sleep(1.5)
    except Exception as e:
        print(f"[LAUNCHER] Could not auto-start Ollama: {e}")

def main():
    print("=" * 65)
    print("      A D A M  —  Advanced Desktop AI Mate (Portable)")
    print(f"      Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print("=" * 65)

    ensure_venv_execution()
    check_and_start_ollama()

    # Inject user site packages for Fedora (e.g. Faster-Whisper)
    candidates = [
        Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        Path.home() / ".local" / "lib" / f"python3.{sys.version_info.minor}" / "site-packages",
    ]
    for c in candidates:
        if c.exists() and str(c) not in sys.path:
            sys.path.insert(0, str(c))

    # Import and launch GUI
    try:
        from app.main import main as app_main
        app_main()
    except ImportError as e:
        print(f"\n[ERROR] Missing required dependencies: {e}")
        print("Please install requirements using:")
        print("    pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
