"""Central Configuration for ADAM (Advanced Desktop AI Mate).
Maintains cross-platform paths, theme palettes, model settings, and defaults.
"""

from pathlib import Path
import os
import sys

# Base directories (Portable: relative to this file's root)
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models" / "piper"
EXPORTS_DIR = PROJECT_ROOT / "exports"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Piper Voice Configuration (Amy English Voice)
PIPER_MODEL_NAME = "en_US-amy-medium"
PIPER_ONNX_PATH = MODELS_DIR / f"{PIPER_MODEL_NAME}.onnx"
PIPER_CONFIG_PATH = MODELS_DIR / f"{PIPER_MODEL_NAME}.onnx.json"

PIPER_MODEL_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/{PIPER_MODEL_NAME}.onnx"
PIPER_CONFIG_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/{PIPER_MODEL_NAME}.onnx.json"

# Ollama LLM Configuration
# qwen2.5:1.5b is exceptionally fast and coherent on 8GB RAM systems (<1.5 GB memory footprint)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("ADAM_MODEL", "qwen2.5:1.5b")
RECOMMENDED_MODELS = [
    "qwen2.5:1.5b",   # Default: ultra-light, high reasoning, 8GB RAM optimized
    "llama3.2:1b",    # Meta's 1B compact model
    "llama3.2:3b",    # Meta's 3B compact model
    "qwen2.5:3b",     # 3B high capability
    "phi3:mini"       # Microsoft Phi-3 Mini 3.8B
]

# Wake Word Settings
PRIMARY_WAKE_WORD = "hii"
WAKE_WORDS = ["hii", "hi", "hey adam", "hi adam", "adam", "hello adam"]

# UI Design Tokens — Cyber Dark Glassmorphism Theme
THEME = {
    "bg_main": "#0A0E17",          # Deep space obsidian
    "bg_card": "#121826",          # Translucent glass dark card
    "bg_card_hover": "#1A2338",    # Card hover
    "bg_input": "#0F1626",         # Text entry background
    "bg_header": "#0D1322",        # Glass navbar
    
    # Borders & Accents
    "border_glass": "#1E2B47",     # Glass card borders
    "border_focus": "#00E5FF",     # Active glow border
    "accent_cyan": "#00E5FF",      # Electric neon cyan
    "accent_purple": "#8B5CF6",    # Cyber violet
    "accent_pink": "#EC4899",      # Neon magenta
    "accent_green": "#10B981",     # Cyber emerald (listening/success)
    "accent_amber": "#F59E0B",     # Warning / Caution
    "accent_red": "#EF4444",       # Destructive / Danger
    
    # Typography
    "text_primary": "#F1F5F9",     # Crisp off-white
    "text_secondary": "#94A3B8",   # Soft slate gray
    "text_muted": "#64748B",       # Subtle timestamp text
    "text_cyan": "#38BDF8",        # Cyan highlighted text
    
    # Message bubbles
    "user_bubble_bg": "#1E293B",
    "user_bubble_border": "#334155",
    "adam_bubble_bg": "#0D1E36",
    "adam_bubble_border": "#0284C7",
    
    # Fonts
    "font_main": "Segoe UI" if sys.platform.startswith("win") else "Liberation Sans",
    "font_mono": "Consolas" if sys.platform.startswith("win") else "Liberation Mono",
}

# Persona System Prompt
ADAM_SYSTEM_PROMPT = """You are ADAM (Advanced Desktop AI Mate), a futuristic, direct, and capable local AI assistant running on Fedora Linux.
Your persona is polite, concise, intelligent, and focused on helping the user navigate their computer, manage tasks, explore ideas, and generate content.

Guidelines:
1. Keep answers concise, natural, and conversational when spoken, but provide accurate technical detail when asked.
2. If the user asks you to open a site, application, or perform a system task, indicate clearly what you are doing.
3. Be respectful of safety. Destructive file operations or terminations will trigger explicit confirmation prompts.
4. If you do not know something or a local service is unavailable, inform the user honestly without fabricating data.
"""
