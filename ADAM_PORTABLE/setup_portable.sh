#!/usr/bin/env bash
# ==============================================================================
# ADAM PORTABLE — Automated Setup Script (Linux & macOS)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo "          A D A M  PORTABLE  —  Setup & Dependency Installer"
echo "=============================================================================="

# 1. Detect Environment
OS_TYPE="$(uname -s)"
echo -e "\n[1/5] Detecting Operating System: $OS_TYPE"
if [ "$OS_TYPE" = "Linux" ]; then
    if [ -f /etc/fedora-release ]; then
        echo "  ✔ Distribution: $(cat /etc/fedora-release)"
    fi
elif [ "$OS_TYPE" = "Darwin" ]; then
    echo "  ✔ macOS detected: $(sw_vers -productVersion)"
fi

# 2. Virtual Environment Setup
echo -e "\n[2/5] Creating Python Virtual Environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✔ Virtualenv created at ./venv"
else
    echo "  ✔ Existing virtualenv found."
fi

source venv/bin/activate
pip install --upgrade pip --quiet

# 3. Install Python Dependencies
echo -e "\n[3/5] Installing Python Dependencies..."
pip install -r requirements.txt

# 4. Download Piper Amy Neural Voice
echo -e "\n[4/5] Downloading Piper Amy English Voice Models..."
python3 scripts/download_models.py

# 5. Check Ollama Local AI Engine
echo -e "\n[5/5] Checking Ollama Local LLM..."
if command -v ollama &>/dev/null; then
    echo "  ✔ Ollama binary found at: $(which ollama)"
    if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "  Starting Ollama daemon in background..."
        nohup ollama serve > /tmp/ollama_adam_setup.log 2>&1 &
        sleep 2
    fi
    echo "  Pulling lightweight 8GB-optimized model 'qwen2.5:1.5b'..."
    ollama pull qwen2.5:1.5b || echo "  ℹ Please run 'ollama pull qwen2.5:1.5b' manually when convenient."
else
    echo "  ⚠️  Ollama is not yet installed on this computer."
    if [ "$OS_TYPE" = "Linux" ]; then
        echo "      Install via: curl -fsSL https://ollama.com/install.sh | sh"
    elif [ "$OS_TYPE" = "Darwin" ]; then
        echo "      Install via: brew install ollama"
    fi
fi

# Make scripts executable
chmod +x run_adam.sh setup_portable.sh scripts/*.py || true

echo -e "\n=============================================================================="
echo "          ✔ ADAM PORTABLE SETUP COMPLETE!"
echo "=============================================================================="
echo "To launch ADAM:"
echo "    ./run_adam.sh"
echo "Or with python directly:"
echo "    python3 run_adam.py"
echo "=============================================================================="
