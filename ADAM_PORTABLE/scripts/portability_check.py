"""Portability & Dependency Diagnostic Tool for ADAM.
Inspects operating system, hardware resources, audio devices, Ollama, Blender,
estimates storage requirements, and explains cross-platform portability.
"""

import os
import platform
import shutil
import sys
from pathlib import Path

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from app.os_adapter import OSAdapter
    from app.speech import list_input_microphones
    from app.three_d import get_blender_path, is_blender_available
    from app.voice import PiperTTS
    from config.settings_manager import SettingsManager
except ImportError as e:
    print(f"Import error during diagnostic setup: {e}")
    sys.exit(1)


def format_size_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def check_directory_size(path: Path) -> int:
    total = 0
    if path.exists():
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return total


def run_diagnostics():
    adapter = OSAdapter()
    settings = SettingsManager()

    print("=" * 75)
    print("           ADAM PORTABLE — SYSTEM & PORTABILITY DIAGNOSTICS")
    print("=" * 75)

    # 1. Operating System & Host Environment
    print("\n[1] HOST OPERATING SYSTEM")
    print(f"  • Detected OS:      {adapter.get_os().upper()} ({adapter.get_os_display_name()})")
    print(f"  • Architecture:     {platform.machine()}")
    print(f"  • Python Runtime:   {platform.python_version()} ({sys.executable})")

    # Tkinter check
    try:
        import tkinter
        print("  • Tkinter GUI:      Installed and available ✔")
    except ImportError:
        print("  • Tkinter GUI:      NOT FOUND (install python3-tkinter) ✖")

    # 2. Storage & Memory Inspection
    print("\n[2] STORAGE & MEMORY ESTIMATION")
    storage = adapter.get_storage_info()
    print(f"  • Partition Space:  {storage['free_gb']} GB Free of {storage['total_gb']} GB total ({storage['percent_used']}% used)")

    diag = adapter.get_system_diagnostics()
    if "ram_total" in diag:
        print(f"  • System RAM:       {diag['ram_total']} total ({diag.get('ram_avail', 'N/A')} available, {diag.get('ram_used_pct', 'N/A')} used)")

    app_size = check_directory_size(BASE_DIR / "app")
    models_size = check_directory_size(BASE_DIR / "models")
    data_size = check_directory_size(BASE_DIR / "data")
    total_portable_size = app_size + models_size + data_size

    print(f"  • Current Portable Footprint:")
    print(f"      - Core Application Source:  {format_size_mb(app_size)}")
    print(f"      - Piper Voice Models:       {format_size_mb(models_size)}")
    print(f"      - User Data & 3D Exports:   {format_size_mb(data_size)}")
    print(f"      - Total Portable Data:      {format_size_mb(total_portable_size)}")

    print("\n  • Estimated USB Storage Requirements:")
    print("      - ADAM Core Application & Scripts:   ~15 MB")
    print("      - Piper Amy Voice ONNX & Config:     ~65 MB")
    print("      - Ollama Qwen2.5 1.5B Model:         ~1.0 GB  (Optional on USB, can stay on host)")
    print("      - OS-specific Virtual Environment:   ~120 MB  (Recreated per OS)")
    print("      -> Recommended Minimum USB Capacity: 4 GB to 16 GB FAT32/exFAT Drive.")

    # 3. Audio & Microphone Hardware
    print("\n[3] AUDIO HARDWARE & INPUT CAPTURE")
    mics = list_input_microphones()
    if mics:
        print(f"  • Input Microphones Detected ({len(mics)} found):")
        for m in mics:
            saved_mark = " [CURRENT DEFAULT]" if m["index"] == settings.get("audio", "microphone_device_index", None) else ""
            print(f"      [{m['index']}] {m['name']} ({m['channels']} input channels, {m['sample_rate']}Hz){saved_mark}")
    else:
        print("  • Microphones:      No input devices detected (Text-only mode fallback active) ⚠️")

    # Speech Recognition library
    try:
        import speech_recognition
        print("  • Speech Engine:    SpeechRecognition library installed ✔")
    except ImportError:
        print("  • Speech Engine:    SpeechRecognition library NOT installed (in requirements.txt) ⚠️")

    # 4. Neural Text-To-Speech (Piper Amy Voice)
    print("\n[4] VOICE SYNTHESIS (TTS)")
    tts = PiperTTS()
    print(f"  • TTS Engine State: {tts.get_status_info()}")
    print(f"  • Amy Model ONNX:   {tts.onnx_path} ({'Found' if tts.onnx_path.exists() else 'Missing - run download_models.py'})")
    print(f"  • Amy Model Config: {tts.config_path} ({'Found' if tts.config_path.exists() else 'Missing'})")

    # 5. Local AI Inference (Ollama)
    print("\n[5] LOCAL NEURAL INFERENCE (OLLAMA)")
    import urllib.request
    ollama_host = settings.get("ollama", "host", "http://localhost:11434")
    try:
        req = urllib.request.Request(f"{ollama_host}/api/tags")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            import json
            models_data = json.loads(resp.read().decode())
            installed = [m.get("name") for m in models_data.get("models", [])]
            print(f"  • Ollama Daemon:    RUNNING at {ollama_host} ✔")
            print(f"  • Installed Models: {', '.join(installed) if installed else 'None pulled yet'}")
            if "qwen2.5:1.5b" in installed or any("qwen2.5:1.5b" in m for m in installed):
                print("  • 8GB RAM Model:    'qwen2.5:1.5b' is ready! ✔")
            else:
                print("  • 8GB RAM Model:    'qwen2.5:1.5b' pending (Run: ollama pull qwen2.5:1.5b) ℹ")
    except Exception:
        print(f"  • Ollama Daemon:    NOT REACHABLE at {ollama_host}")
        print("    ℹ To start: 'ollama serve &'")

    # 6. Blender & 3D Synthesis
    print("\n[6] 3D OBJECT SYNTHESIS & BLENDER INTEGRATION")
    blender_bin = get_blender_path()
    if blender_bin:
        print(f"  • Blender Executable: Found at {blender_bin} ✔")
        try:
            import subprocess
            out = subprocess.check_output([blender_bin, "--version"], stderr=subprocess.STDOUT, text=True).splitlines()[0]
            print(f"  • Blender Version:    {out} ✔")
        except Exception:
            pass
    else:
        print("  • Blender:            Not found in system PATH.")
        print("  • 3D Engine Fallback: Active! (ADAM's pure-Python STL engine generates valid 3D meshes without Blender) ✔")

    # 7. Multi-OS Portability Summary
    print("\n[7] PORTABILITY SEPARATION")
    print("  ✔ PORTABLE FILES (Safe to copy to USB drive):")
    print("      - app/ (Python application source code)")
    print("      - config/ (Settings template and configurations)")
    print("      - models/ (Piper Amy ONNX voice models)")
    print("      - data/ (Chat history and exported 3D STL/OBJ objects)")
    print("      - scripts/ (Blender generator, downloader, migration tool)")
    print("      - run_adam.sh, run_adam.bat, run_adam.py")
    print("  ✖ HOST-SPECIFIC ARTIFACTS (Excluded from USB / Recreated per OS):")
    print("      - venv/ (Compiled C extensions & OS-specific Python binaries)")
    print("      - __pycache__/ & .pyc files")
    print("      - System daemon sockets & temporary lock files")

    print("\n" + "=" * 75)
    print("Ready to migrate to a new USB drive? Run:")
    print("    python3 scripts/migrate_to_usb.py /path/to/new_usb_drive")
    print("=" * 75)


if __name__ == "__main__":
    run_diagnostics()
