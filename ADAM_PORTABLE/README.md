# ADAM — Advanced Desktop AI Mate (Portable Edition)

**ADAM** is a fully portable, cross-platform AI desktop assistant engineered for **Linux (Fedora, Ubuntu, Debian, Arch)**, **Windows 10/11**, and **macOS**.

---

## 🌟 Architecture & Portable Layout

```text
ADAM_PORTABLE/
├── app/
│   ├── __init__.py           # Package exports
│   ├── os_adapter.py         # Cross-platform OS abstraction (paths, audio, apps, Blender)
│   ├── main.py               # Glassmorphic UI with animated AI reactor visualizer
│   ├── speech.py             # Microphone enumeration, test tool & 'Hii' wake-word listener
│   ├── voice.py              # Piper TTS (Amy English voice) with multi-OS fallback
│   ├── integrations.py       # Ollama LLM, web & app launcher, diagnostics, safe file ops
│   └── three_d.py            # Blender auto-detection & pure-Python 3D engine fallback
├── config/
│   ├── settings_template.json# Default settings template
│   └── settings_manager.py   # Settings manager & theme definitions
├── models/
│   ├── piper/                # Offline Piper Amy voice ONNX models (~65 MB)
│   └── README.md             # Offline model transfer guide
├── data/
│   ├── chat_history.json     # Persistent conversation logs
│   ├── settings.json         # Active user preferences & selected microphone
│   └── exports/              # Exported 3D objects (.stl, .obj)
├── scripts/
│   ├── blender_gen.py        # Headless script for Blender (blender -b -P ...)
│   ├── download_models.py    # Downloader for Amy voice files with storage check
│   ├── portability_check.py  # System diagnostic tool & storage estimator
│   └── migrate_to_usb.py     # Safe USB migration wizard
├── run_adam.py               # Universal Python launcher (Linux, Windows, macOS)
├── run_adam.sh               # Native Bash launcher (Linux & macOS)
├── run_adam.bat              # Native Batch launcher (Windows 10/11)
├── setup_portable.sh         # Automated setup script (Linux & macOS)
├── setup_portable.bat        # Automated setup script (Windows)
├── requirements.txt          # Universal Python dependencies
└── README.md                 # Complete documentation & multi-OS manual
```

---

## 🌐 Multi-OS Setup & Launch Guide

### 🐧 Linux (Fedora, Ubuntu, Debian)

1. **System Dependencies**:
   ```bash
   # On Fedora:
   sudo dnf install -y python3 python3-pip python3-tkinter portaudio-devel alsa-utils pulseaudio-utils
   
   # On Ubuntu/Debian:
   sudo apt update && sudo apt install -y python3 python3-pip python3-tk portaudio19-dev alsa-utils
   ```

2. **Automated Setup**:
   ```bash
   cd ADAM_PORTABLE
   ./setup_portable.sh
   ```

3. **Launch**:
   ```bash
   ./run_adam.sh
   # Or: python3 run_adam.py
   ```

---

### 🪟 Windows 10 & 11

1. **Prerequisites**:
   - Install **Python 3.10+** from [python.org](https://www.python.org/downloads/) (Make sure to check *"Add python.exe to PATH"* during installation).
   - Install **Ollama for Windows** from [ollama.com/download/windows](https://ollama.com/download/windows).

2. **Automated Setup**:
   Double-click `setup_portable.bat` (or open Command Prompt):
   ```cmd
   cd ADAM_PORTABLE
   setup_portable.bat
   ```

3. **Launch**:
   Double-click `run_adam.bat` (or run `python run_adam.py`).

---

### 🍎 macOS (Apple Silicon M1/M2/M3 & Intel)

1. **Prerequisites**:
   - Install Python 3.10+ and Ollama:
     ```bash
     brew install python python-tk ollama
     ```

2. **Setup & Launch**:
   ```bash
   cd ADAM_PORTABLE
   ./setup_portable.sh
   ./run_adam.sh
   ```

---

## 🎙️ Microphone Detection & Audio Configuration

ADAM features an **Audio Input Setup & Signal Tester**:

1. **Fixing Linux Device 0 HDMI Issue**:
   - On Linux systems where Device 0 is an output-only HDMI channel, ADAM automatically excludes output-only devices and binds directly to active input hardware (such as analog stereo microphones or USB headsets).
2. **Interactive Audio Setup**:
   - Click the **`🎙️ Audio Setup`** button in ADAM's top navigation bar.
   - Select your desired microphone from the dropdown list.
   - Click **`▶ Test Signal`** and speak for 2 seconds: ADAM will sample audio and render a live RMS energy level meter to confirm sound reception.
   - Click **`Save Active Microphone`** to persist your choice to `data/settings.json`.

---

## ⚡ 3D Synthesis Lab & Blender Integration

- **Existing Blender Auto-Detection**:
  - ADAM detects your existing Blender installation on Fedora at `/usr/bin/blender` (Blender 5.2.2 LTS).
  - On Windows, it auto-detects `C:\Program Files\Blender Foundation\Blender *\blender.exe`.
  - On macOS, it auto-detects `/Applications/Blender.app/Contents/MacOS/Blender`.
- **Pure-Python 3D Engine Fallback**:
  - If Blender is not installed on a target computer, ADAM's built-in 3D engine generates valid binary **STL** and Wavefront **OBJ** files with exact triangle calculations and normals.
  - Supported shapes: `cube`, `sphere`, `cylinder`, `torus`, `scifi_drone`, `scifi_pedestal`, `pyramid`.
  - Exported meshes are saved directly into `data/exports/`.

---

## 📦 Safe USB Migration Guide

When you plug in your new USB drive tomorrow, run the migration wizard to copy ADAM safely:

```bash
python3 scripts/migrate_to_usb.py /run/media/sujeethkani/<NEW_USB_NAME>/ADAM_PORTABLE
```

*What the migration wizard does:*
1. **Drive & Filesystem Check**: Detects filesystem type (FAT32, exFAT, NTFS, ext4) and checks available disk space.
2. **Non-Destructive Guarantee**: Verifies target directory and will **never** overwrite existing files without explicit user confirmation.
3. **Clean Portable Copy**: Copies application source, configs, offline voice models, and data, while **excluding** OS-specific compiled binaries (`venv/`), cache files (`__pycache__`), and temporary locks.
4. **Post-Copy Verification**: Validates file counts and byte sizes to ensure complete copy integrity.

---

## 🔌 Offline vs Online Features Breakdown

| Feature | Offline? | Notes |
| :--- | :---: | :--- |
| **Local LLM Inference (Ollama)** | ✅ **100% Offline** | Runs locally with `qwen2.5:1.5b` (Uses <1.4 GB RAM). |
| **Piper Neural Voice (Amy)** | ✅ **100% Offline** | Synthesizes speech locally from `models/piper/en_US-amy-medium.onnx`. |
| **3D Mesh Generation & Export** | ✅ **100% Offline** | Generates STL/OBJ files using headless Blender or pure-Python engine. |
| **System Diagnostics & Launcher** | ✅ **100% Offline** | Opens local software, inspects RAM/CPU/Disk, lists local files safely. |
| **Destructive Safety Gate** | ✅ **100% Offline** | Interactive confirmation dialog rendered in Tkinter. |
| **Settings & Chat History** | ✅ **100% Offline** | Saved in local `data/` directory. |
| **Speech-to-Text Recognition** | 🌐 Online | Uses Google Speech API (Can use local Vosk/Sphinx for offline STT). |
| **YouTube & Web Navigation** | 🌐 Online | Requires internet connectivity for web queries. |
