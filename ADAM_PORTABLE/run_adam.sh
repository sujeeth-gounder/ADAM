#!/usr/bin/env bash
# ==============================================================================
# ADAM PORTABLE — Linux & macOS Launcher
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Check or activate Python virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
else
    echo "ℹ Virtual environment not found in $SCRIPT_DIR/venv."
    echo "Running quick setup..."
    bash setup_portable.sh
    source venv/bin/activate
fi

# Add local portable ollama and library path if available
if [ -d "$SCRIPT_DIR/dist/linux-amd64/bin" ]; then
    export PATH="$SCRIPT_DIR/dist/linux-amd64/bin:$PATH"
fi
if [ -d "$SCRIPT_DIR/dist/linux-amd64/lib/ollama" ]; then
    export OLLAMA_LIBRARY_PATH="$SCRIPT_DIR/dist/linux-amd64/lib/ollama:$OLLAMA_LIBRARY_PATH"
fi

# 2. Check and start Ollama in background if not already running
if command -v ollama &>/dev/null; then
    if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "Starting Ollama daemon in background..."
        nohup ollama serve > /tmp/ollama_adam_portable.log 2>&1 &
        sleep 1.5
    fi
fi

# 3. Launch ADAM
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
python3 run_adam.py "$@"
