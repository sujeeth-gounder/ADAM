"""ADAM — Advanced Desktop AI Mate
Universal Root Launcher & Preflight Environment Validator.
Supports Linux (Fedora, Ubuntu, Debian, Arch) and Windows 10/11 natively.

Usage:
    python run_adam.py
"""

import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def run_preflight_checks() -> dict:
    """Run non-blocking diagnostics on system dependencies."""
    status = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_ok": sys.version_info >= (3, 10),
        "tkinter_ok": False,
        "piper_voice_ok": False,
        "ollama_available": False,
        "ollama_running": False,
        "blender_available": False,
        "audio_playback_ok": False,
    }

    # Tkinter check
    try:
        import tkinter
        status["tkinter_ok"] = True
    except ImportError:
        status["tkinter_ok"] = False

    # Piper check
    model_paths = [
        ROOT_DIR / "models" / "piper" / "en_US-amy-medium.onnx",
        Path.home() / ".local" / "share" / "piper-voices" / "en" / "en_US" / "amy" / "medium" / "en_US-amy-medium.onnx",
    ]
    status["piper_voice_ok"] = any(p.exists() for p in model_paths)

    # Ollama check
    if shutil.which("ollama"):
        status["ollama_available"] = True
        import urllib.request
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    status["ollama_running"] = True
        except Exception:
            status["ollama_running"] = False

    # Blender check
    from app.platform_utils import find_blender_binary
    status["blender_available"] = bool(find_blender_binary())

    # Audio playback check
    if sys.platform == "win32":
        status["audio_playback_ok"] = True
    else:
        status["audio_playback_ok"] = bool(
            shutil.which("pw-play") or shutil.which("paplay") or shutil.which("aplay") or shutil.which("play")
        )

    return status


def ensure_venv_execution():
    """If a local venv directory exists and we are not in it, re-exec using venv python."""
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        return

    venv_dir = ROOT_DIR / "venv"
    if not venv_dir.exists():
        return

    if platform.system().lower() == "windows":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    if venv_python.exists() and str(venv_python) != sys.executable:
        print(f"[ADAM] Activating workspace virtualenv: {venv_python}")
        cmd = [str(venv_python), str(Path(__file__).resolve())] + sys.argv[1:]
        env = dict(os.environ)

        # Append user-site directory dynamically if present
        user_site = Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        if user_site.exists():
            existing_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{user_site}{os.pathsep}{existing_pp}" if existing_pp else str(user_site)

        try:
            sys.exit(subprocess.call(cmd, env=env))
        except Exception as e:
            print(f"[ADAM] Notice: Could not re-exec into venv ({e}), continuing with {sys.executable}")


def start_ollama_if_needed():
    """Ensure Ollama daemon is running if the binary is present."""
    if not shutil.which("ollama"):
        return

    import urllib.request
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                return  # already up
    except Exception:
        pass

    print("[ADAM] Starting local Ollama service in background...")
    try:
        if platform.system().lower() == "windows":
            subprocess.Popen(
                ["ollama", "serve"],
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        time.sleep(1.0)
    except Exception as e:
        print(f"[ADAM] Could not auto-start Ollama: {e}")


def main():
    print("=" * 68)
    print("          A D A M  —  Advanced Desktop AI Mate")
    print(f"          OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print("=" * 68)

    # 1. Enforce Virtual Environment if available
    ensure_venv_execution()

    # 2. Add user-site packages dynamically
    candidates = [
        Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        Path.home() / ".local" / "lib" / f"python3.{sys.version_info.minor}" / "site-packages",
    ]
    for c in candidates:
        if c.exists() and str(c) not in sys.path:
            sys.path.insert(0, str(c))

    # 3. Preflight checks
    checks = run_preflight_checks()
    if not checks["python_ok"]:
        print(f"[FATAL] Python 3.10+ required. Current version: {checks['python_version']}")
        sys.exit(1)

    if not checks["tkinter_ok"]:
        print("[FATAL] Python Tkinter GUI module not found.")
        if platform.system().lower() == "linux":
            print("        Install with: sudo dnf install python3-tkinter (Fedora) or sudo apt install python3-tk (Ubuntu)")
        sys.exit(1)

    # 4. Service startups
    start_ollama_if_needed()

    print(f"[*] Python: {checks['python_version']} (OK)")
    print(f"[*] Tkinter UI: {'Available' if checks['tkinter_ok'] else 'Missing'}")
    print(f"[*] Piper Voice: {'Amy Model Present' if checks['piper_voice_ok'] else 'Fallback/System'}")
    print(f"[*] Ollama: {'Installed & Accessible' if checks['ollama_available'] else 'Not Installed'}")
    print(f"[*] Blender 3D: {'Detected' if checks['blender_available'] else 'Standby / Viewport Only'}")
    print(f"[*] Audio Output: {'Ready' if checks['audio_playback_ok'] else 'Generic'}")
    print("-" * 68)
    print("[ADAM] Launching graphical interface...")

    # 5. Launch main desktop application
    from app.main import main as app_main
    app_main()


if __name__ == "__main__":
    main()
