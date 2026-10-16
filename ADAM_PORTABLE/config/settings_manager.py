"""Settings Manager for ADAM (Portable).
Manages configurable runtime preferences stored in data/settings.json,
with fallback to config/settings_template.json.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.os_adapter import OSAdapter

logger = logging.getLogger("ADAM.Settings")


class SettingsManager:
    """Singleton configuration manager with auto-persistence to data/settings.json."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SettingsManager, cls).__new__(cls)
            cls._instance._init_settings()
        return cls._instance

    def _init_settings(self):
        self.adapter = OSAdapter()
        self.template_file = self.adapter.config_dir / "settings_template.json"
        self.active_file = self.adapter.data_dir / "settings.json"
        self.settings: Dict[str, Any] = {}
        self.load()

    def load(self):
        """Load settings from data/settings.json or bootstrap from template."""
        if self.active_file.exists():
            try:
                with open(self.active_file, "r", encoding="utf-8") as f:
                    self.settings = json.load(f)
                    return
            except Exception as e:
                logger.error(f"Failed to read {self.active_file}: {e}")

        # Fallback to template
        if self.template_file.exists():
            try:
                with open(self.template_file, "r", encoding="utf-8") as f:
                    self.settings = json.load(f)
                    self.save()  # create initial data/settings.json
                    return
            except Exception as e:
                logger.error(f"Failed to read template {self.template_file}: {e}")

        # In-memory emergency defaults
        self.settings = {
            "version": "1.0.0",
            "ollama": {"host": "http://localhost:11434", "default_model": "qwen2.5:1.5b"},
            "audio": {"microphone_device_index": None, "wake_word": "hii"},
            "blender": {"custom_executable_path": None}
        }

    def save(self):
        """Persist current settings to data/settings.json."""
        try:
            self.adapter.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.active_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write settings to {self.active_file}: {e}")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.settings.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any, auto_save: bool = True):
        if section not in self.settings:
            self.settings[section] = {}
        self.settings[section][key] = value
        if auto_save:
            self.save()


# Theme & Visual Constants
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
    "font_main": "Segoe UI" if sys.platform.startswith("win") else "Liberation Sans" if sys.platform.startswith("linux") else "Helvetica Neue",
    "font_mono": "Consolas" if sys.platform.startswith("win") else "Liberation Mono" if sys.platform.startswith("linux") else "Menlo",
}

ADAM_SYSTEM_PROMPT = """You are ADAM (Advanced Desktop AI Mate), a futuristic, direct, and capable local AI desktop companion.
You are running natively on the user's computer with full cross-platform desktop integration.
Your persona is polite, concise, intelligent, and focused on helping the user navigate their computer, manage tasks, explore ideas, and generate content.

Guidelines:
1. Keep answers concise, natural, and conversational when spoken, but provide accurate technical detail when asked.
2. If the user asks you to open a site, application, or perform a system task, indicate clearly what you are doing.
3. Be respectful of safety. Destructive file operations or terminations will trigger explicit confirmation prompts.
4. If you do not know something or a local service is unavailable, inform the user honestly without fabricating data.
"""
