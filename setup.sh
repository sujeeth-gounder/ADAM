#!/usr/bin/env bash
# ==============================================================================
# ADAM (Advanced Desktop AI Mate) — Automated Setup Script for Fedora Linux
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo "          A D A M  —  Advanced Desktop AI Mate Installation"
echo "                 Fedora Linux & Portable Workstation"
echo "=============================================================================="

# 1. System Inspection
echo -e "\n[1/6] Inspecting System Resources..."
if [ -f /etc/fedora-release ]; then
    echo "  ✔ OS: $(cat /etc/fedora-release)"
else
    echo "  ℹ OS: Non-Fedora Linux detected. Proceeding with standard Linux setup."
fi

TOTAL_MEM=$(free -h | awk '/^Mem:/ {print $2}')
AVAIL_MEM=$(free -h | awk '/^Mem:/ {print $7}')
DISK_AVAIL=$(df -h . | awk 'NR==2 {print $4}')
echo "  ✔ RAM: Total $TOTAL_MEM (Available: $AVAIL_MEM)"
echo "  ✔ Storage available: $DISK_AVAIL in $SCRIPT_DIR"

# 2. Check Prerequisites
echo -e "\n[2/6] Checking System Packages..."
MISSING_PKGS=()

if ! command -v python3 &>/dev/null; then
    MISSING_PKGS+=("python3" "python3-pip")
fi

if ! python3 -c "import tkinter" &>/dev/null; then
    MISSING_PKGS+=("python3-tkinter")
fi

# Audio and Speech Recognition dependencies on Fedora
if ! command -v aplay &>/dev/null && ! command -v paplay &>/dev/null; then
    MISSING_PKGS+=("alsa-utils" "pulseaudio-utils")
fi

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    echo "  ⚠️  Missing system packages: ${MISSING_PKGS[*]}"
    echo "  To install them, run:"
    echo "      sudo dnf install -y ${MISSING_PKGS[*]} portaudio-devel"
    echo ""
    read -p "  Would you like to install them now with sudo dnf? (y/N): " -r CONFIRM_INSTALL
    if [[ "$CONFIRM_INSTALL" =~ ^[Yy]$ ]]; then
        sudo dnf install -y "${MISSING_PKGS[@]}" portaudio-devel
    else
        echo "  Skipping dnf install. Note: Some audio or GUI features might need these packages."
    fi
else
    echo "  ✔ Core system dependencies present (Python3, Tkinter, Audio utilities)."
fi

# 3. Create Python Virtual Environment
echo -e "\n[3/6] Configuring Python Virtual Environment..."
if [ ! -d "venv" ]; then
    echo "  Creating virtual environment in ./venv ..."
    python3 -m venv venv
else
    echo "  ✔ Virtual environment already exists."
fi

source venv/bin/activate
echo "  Upgrading pip..."
pip install --upgrade pip --quiet

# 4. Install Python Requirements
echo -e "\n[4/6] Installing Python Dependencies..."
pip install -r requirements.txt

# 5. Download Piper Amy Voice (Neural TTS)
echo -e "\n[5/6] Checking Piper TTS Amy English Voice Model..."
python3 scripts/download_models.py

# 6. Check and Configure Ollama
echo -e "\n[6/6] Checking Ollama Local LLM Engine..."
if command -v ollama &>/dev/null; then
    echo "  ✔ Ollama binary found at: $(which ollama)"
    # Check if daemon is active
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "  ✔ Ollama service is currently RUNNING."
    else
        echo "  ℹ Starting Ollama service in the background..."
        nohup ollama serve > /tmp/ollama_adam.log 2>&1 &
        sleep 2
    fi

    # Pull recommended 8GB-optimized model
    echo "  Pulling lightweight model 'qwen2.5:1.5b' (Optimized for 8GB RAM laptops)..."
    ollama pull qwen2.5:1.5b || {
        echo "  ⚠️  Could not pull qwen2.5:1.5b automatically. You can pull it manually with: ollama pull qwen2.5:1.5b"
    }
else
    echo "  ⚠️  Ollama is not installed. To install Ollama on Fedora:"
    echo "      curl -fsSL https://ollama.com/install.sh | sh"
fi

# Ensure run.sh is executable
chmod +x run.sh || true

echo -e "\n=============================================================================="
echo "          ✔ ADAM SETUP COMPLETE & READY TO LAUNCH!"
echo "=============================================================================="
echo "To start ADAM, run:"
echo "    ./run.sh"
echo "Or manually:"
echo "    source venv/bin/activate"
echo "    python3 app/main.py"
echo "=============================================================================="
