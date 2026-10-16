# ADAM — Advanced Desktop AI Mate

**ADAM** (*Advanced Desktop AI Mate*) is a complete, modular, portable local AI desktop assistant built from scratch for **Fedora Linux** with cross-platform desktop compatibility.

---

## 🌟 Key Architecture & Features

### 1. Cyber Dark Glassmorphism Desktop UI
- **Animated AI Reactor Orb / Hologram**: A real-time 35 FPS gyroscopic visualizer on the Tkinter canvas with reactive states:
  - `IDLE`: Concentric breathing radar halo.
  - `LISTENING`: Sonic ripple waves reactive to live audio energy.
  - `THINKING`: Gyroscopic particles orbiting in counter-rotation while Ollama generates tokens.
  - `SPEAKING`: Real-time audio waveform plasma bursts during Piper TTS playback.
- **Modern Glass Cards**: Translucent dark slate surfaces (`#121826`), glowing cyan accents (`#00E5FF`), and crisp typography.
- **Thread-Safe Event Queue**: All speech recognition, LLM inference, and audio streams run on worker threads and pass messages via `queue.Queue` to Tkinter's main loop.

### 2. Local Neural LLM with Ollama (Optimized for 8 GB RAM)
- Native integration with the local Ollama API (`http://localhost:11434`).
- **Default Recommended Model**: `qwen2.5:1.5b` (Uses <1.4 GB RAM, ultra-responsive latency under 100ms on laptop CPUs, highly articulate reasoning).
- Also supports: `llama3.2:1b`, `llama3.2:3b`, `qwen2.5:3b`, and `phi3:mini` via the top-bar model selector.
- Streaming responses with real-time word-by-word typing effect.

### 3. Speech Recognition & "Hii" Wake Word
- English-only STT using `SpeechRecognition`.
- Background always-listening loop configured for the wake word **"Hii"** (also matches natural variants like "Hi", "Hey Adam", "Hi Adam").
- Instant push-to-talk (`🎙️ Listen Once`) button.
- Graceful degradation: If no microphone is detected, ADAM continues in text-only mode with zero crashes.

### 4. Neural Text-to-Speech (Piper TTS with Amy English Voice)
- High-fidelity local neural speech synthesis using Piper's natural female **Amy English voice** (`en_US-amy-medium`).
- Background speech queue with non-blocking audio playback (`aplay`, `paplay`, or `sounddevice`).
- Direct fallback to `pyttsx3` or system speech if voice files are still being downloaded.
- "Stop Voice" button to immediately halt playback.

### 5. Desktop Automation & Safety Gating
- **System Time & Date**: Instant spoken queries (e.g., *"What time is it?"*, *"Today's date"*).
- **App Launcher**: Launches local Linux software (`browser`, `terminal`, `files`, `calculator`, `text editor`, `blender`).
- **Web & Media**: Opens URLs and initiates direct YouTube queries (`"search youtube for lo-fi music"`).
- **Interactive Safety Confirmation**: Destructive operations (file unlinking, recursive directory deletion) are blocked until the user explicitly confirms via an interactive red-glow confirmation card directly in the chat feed!

### 6. 3D Object Synthesis Lab (Blender + Pure Python Fallback)
- **Blender Headless Mode**: Executes `blender -b -P scripts/blender_gen.py` to create parametric 3D models with bevel modifiers.
- **Built-in Pure-Python 3D Engine**: If Blender is not installed, ADAM's integrated pure-Python mesh engine computes triangles, normals, and exports valid binary **STL** and Wavefront **OBJ** files immediately.
- Shapes supported: `cube`, `sphere`, `cylinder`, `torus`, `scifi_drone`, `scifi_pedestal`, `pyramid`.

---

## 📁 Project Directory Structure

```text
ADAM/
├── app/
│   ├── __init__.py           # Package exports & versioning
│   ├── config.py             # Colors, paths, model settings, system prompts
│   ├── main.py               # Tkinter GUI, reactor animation, chat loop
│   ├── voice.py              # Piper TTS with Amy voice & non-blocking playback
│   ├── speech.py             # Speech recognition, "Hii" wake word listener
│   ├── integrations.py       # Ollama client, system tools, safety confirmation
│   └── three_d.py            # Blender headless exporter + pure-Python 3D engine
├── models/
│   └── piper/                # Local storage for Piper Amy voice ONNX models
├── exports/                  # Generated 3D files (.stl, .obj)
├── scripts/
│   ├── blender_gen.py        # Headless Blender generation script
│   └── download_models.py    # Python script to download Amy voice files
├── requirements.txt          # Python dependencies
├── setup.sh                  # Automated Fedora setup & dependency script
├── run.sh                    # One-click launcher
└── README.md                 # Complete documentation
```

---

## 🚀 Fedora Linux Installation Guide

### Step 1: System Dependencies (Fedora)
Ensure your system has the standard Python, audio, and development libraries:

```bash
sudo dnf install -y python3 python3-pip python3-tkinter portaudio-devel alsa-utils pulseaudio-utils
```

*(Optional: Install Blender if you want Blender-driven rendering)*
```bash
sudo dnf install -y blender
```

### Step 2: Automated Project Setup
Run the included setup script from the `ADAM` directory:

```bash
cd /run/media/sujeethkani/games/ADAM
./setup.sh
```

The script will automatically:
1. Verify system RAM and storage.
2. Create an isolated Python virtual environment (`venv`).
3. Install required Python packages (`ollama`, `SpeechRecognition`, `piper-tts`, etc.).
4. Download the **Piper Amy English voice** (`en_US-amy-medium.onnx` + `.json`) into `models/piper/`.
5. Check Ollama and pull the lightweight `qwen2.5:1.5b` model.

---

## 🎮 Launching ADAM

Simply run:
```bash
./run.sh
```

Or manually:
```bash
source venv/bin/activate
python3 app/main.py
```

---

## 🧪 Comprehensive Feature Testing Checklist

### 1. Test Wake Word & Voice Chat
- Say out loud: **"Hii, what time is it?"**
- Notice the AI reactor orb switch to **green** (Listening), then **purple** (Thinking), then **cyan** (Speaking).
- Hear the spoken response using the natural **Amy English Voice**.
- Alternatively, click the **"🎙️ Listen Once"** button for push-to-talk.

### 2. Test Desktop & Web Automations
- **App Launcher**: Type or speak `open terminal`, `open browser`, or `open calculator`.
- **Website Navigation**: Type or speak `open github.com` or `launch fedoraproject.org`.
- **YouTube Search**: Type or speak `search youtube for interstellar soundtrack`.
- **System Diagnostics**: Type or speak `system info` or `ram usage`.

### 3. Test 3D Object Synthesis Lab
- Click the **"⚡ 3D Lab"** button in the top navigation bar.
- Select `scifi_drone`, format `STL`, and click **"Generate & Export 3D Mesh"**.
- Or via chat: Type `create 3d scifi_drone as stl` or `generate 3d cube`.
- The file will be exported to the `exports/` folder immediately.

### 4. Test Destructive File Operation Safety
- Create a temporary dummy file to test safety:
  ```bash
  touch /tmp/adam_test.txt
  ```
- In ADAM's chat bar, type:
  ```text
  delete file /tmp/adam_test.txt
  ```
- Notice that ADAM **refuses to delete the file immediately**.
- Instead, a crimson **Destructive Action Confirmation Card** appears in the chat with **`[ ✅ Confirm & Delete ]`** and **`[ ❌ Cancel ]`** buttons.
- Click `[ ❌ Cancel ]` -> ADAM aborts the action and speaks *"Operation cancelled."*
- Try again and click `[ ✅ Confirm & Delete ]` -> The file is safely unlinked with user consent.

---

## 💾 USB Portability Guide

ADAM is designed to be self-contained and portable:
1. All path references in `app/config.py` use dynamic `Path(__file__).resolve().parent.parent` rather than hardcoded machine paths.
2. The AI voice models reside in `models/piper/` inside the folder.
3. To copy to a new USB drive:
   ```bash
   cp -r /run/media/sujeethkani/games/ADAM /run/media/sujeethkani/<NEW_USB_NAME>/
   ```
4. On any new Linux or Windows machine with Python 3.10+ and Ollama installed, simply run `./setup.sh` or `python app/main.py`.
