#!/usr/bin/env bash
# ==============================================================================
# ADAM (Advanced Desktop AI Mate) — One-Click Launcher
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure venv exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not detected. Running quick setup..."
    bash setup.sh
    source venv/bin/activate
fi

# Add local portable ollama and library path if available
if [ -d "$SCRIPT_DIR/dist/linux-amd64/bin" ]; then
    export PATH="$SCRIPT_DIR/dist/linux-amd64/bin:$PATH"
fi
if [ -d "$SCRIPT_DIR/dist/linux-amd64/lib/ollama" ]; then
    export OLLAMA_LIBRARY_PATH="$SCRIPT_DIR/dist/linux-amd64/lib/ollama:$OLLAMA_LIBRARY_PATH"
fi

# Ensure Ollama daemon is up
if command -v ollama &>/dev/null; then
    if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "Starting Ollama server daemon in background..."
        nohup ollama serve > /tmp/ollama_adam.log 2>&1 &
        sleep 1.5
    fi
fi

# Set python path
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Launch ADAM GUI
python3 app/main.py
