#!/usr/bin/env bash
# ==============================================================================
# ADAM — Advanced Desktop AI Mate (Linux Launcher)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Activate Virtual Environment if available
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# 2. Add user-site packages dynamically
if [ -d "$HOME/.local/lib/python3.14/site-packages" ]; then
    export PYTHONPATH="$HOME/.local/lib/python3.14/site-packages:$PYTHONPATH"
fi

# 3. Check and start Ollama in background if not already running
if command -v ollama &>/dev/null; then
    if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "[ADAM] Starting Ollama daemon in background..."
        nohup ollama serve > /tmp/ollama_adam.log 2>&1 &
        sleep 1.0
    fi
fi

# 4. Launch ADAM
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
exec python3 run_adam.py "$@"
