# ADAM Models Directory

This directory stores offline AI models for portable execution.

## Piper Neural TTS Voice (Amy English)
- **Path**: `models/piper/en_US-amy-medium.onnx` and `models/piper/en_US-amy-medium.onnx.json`
- **Voice**: Clear, natural American English female voice (Amy).
- **Size**: ~65 MB.
- **Portability**: These ONNX models are 100% cross-platform and work identically on Linux, Windows, and macOS without recompilation!

## Ollama Local LLMs (qwen2.5:1.5b)
- **Default Storage Location by OS**:
  - **Linux**: `~/.ollama/models/` or `/usr/share/ollama/.ollama/models`
  - **Windows**: `C:\Users\<Username>\.ollama\models\`
  - **macOS**: `~/.ollama/models/`
- **Copying Models Directly Between Machines**:
  If you want to transfer Ollama models offline onto another computer without re-downloading:
  1. Copy the `manifests/` and `blobs/` folders from your laptop's `~/.ollama/models/` to the target computer's `.ollama/models/` directory.
  2. Run `ollama list` on the target machine; it will immediately recognize `qwen2.5:1.5b` without needing an internet connection.
