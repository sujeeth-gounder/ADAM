"""ADAM — Advanced Desktop AI Mate
Main Desktop Application with Tkinter Glassmorphism UI, Animated AI Reactor Orb,
Voice Recognition ("Hii" wake word), Piper TTS (Amy voice), Ollama integration, and 3D Generator.
"""

import datetime

import json
import math
import os
import queue
import re
import secrets
import sys
import threading
import time
import tkinter as tk
import urllib.parse
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from tkinter import messagebox, ttk
from pathlib import Path
from typing import Optional

# Ensure user site-packages and local venv site-packages are accessible
_ROOT_DIR = Path(__file__).resolve().parent.parent
_VENV_CANDIDATES = [
    Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    _ROOT_DIR / "venv",
    _ROOT_DIR.parent / "venv",
]
for cand in _VENV_CANDIDATES:
    if cand.is_dir():
        if cand.name == "site-packages":
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
        else:
            for sp in cand.glob("lib*/python*/site-packages"):
                if sp.exists() and str(sp) not in sys.path:
                    sys.path.insert(0, str(sp))

from app.config import (
    ADAM_SYSTEM_PROMPT,
    DEFAULT_OLLAMA_MODEL,
    EXPORTS_DIR,
    RECOMMENDED_MODELS,
    THEME,
)
from app.integrations import OllamaClient, PendingAction, SystemIntegrations
from app.speech import (
    SpeechListener,
    diagnose_microphone_system,
    list_input_microphones,
    test_microphone_level,
)
from app.storage import LocalStore
from app.system_monitor import SystemMonitor
from app.three_d import (
    GeneratedModelAsset,
    generate_3d_object,
    get_available_shapes,
    is_blender_available,
)
from app.three_d_viewer import Interactive3DViewport
from app.voice import PiperTTS
from app.vision import VisionService, list_available_cameras


import logging

logger = logging.getLogger("ADAM.App")

VOICE_STATES = {
    "MIC_OFF": {
        "title": "MIC OFF",
        "icon": "🔇",
        "text": "Audio capture is stopped. Click MIC button to activate.",
        "color": "#64748B",
        "bg": "#1E293B",
        "btn_text": "🔇 MIC OFF",
        "btn_bg": "#475569",
        "active_capture": False,
        "animate": False
    },
    "MIC_STARTING": {
        "title": "MIC ACTIVATING",
        "icon": "⏳",
        "text": "Microphone Activating",
        "color": "#F59E0B",
        "bg": "#451A03",
        "btn_text": "⏳ STARTING...",
        "btn_bg": "#D97706",
        "active_capture": False,
        "animate": False
    },
    "LISTENING_WAKE": {
        "title": "LISTENING FOR WAKE WORD",
        "icon": "🎙️",
        "text": "Listening...",
        "color": "#10B981",
        "bg": "#064E3B",
        "btn_text": "🟢 MIC: ON",
        "btn_bg": "#059669",
        "active_capture": True,
        "animate": False
    },
    "WAKE_DETECTED": {
        "title": "WAKE WORD DETECTED",
        "icon": "⚡",
        "text": "Wake word detected! Start speaking your command.",
        "color": "#06B6D4",
        "bg": "#083344",
        "btn_text": "⚡ WAKE DETECTED",
        "btn_bg": "#0891B2",
        "active_capture": True,
        "animate": False
    },
    "LISTENING_COMMAND": {
        "title": "LISTENING FOR COMMAND",
        "icon": "🌊",
        "text": "Listening...",
        "color": "#3B82F6",
        "bg": "#1E1B4B",
        "btn_text": "🎙️ LISTENING...",
        "btn_bg": "#2563EB",
        "active_capture": True,
        "animate": False
    },
    "SPEECH_DETECTED": {
        "title": "SPEECH DETECTED",
        "icon": "🎙️",
        "text": "Speech Detected",
        "color": "#3B82F6",
        "bg": "#1E1B4B",
        "btn_text": "🎙️ RECORDING",
        "btn_bg": "#2563EB",
        "active_capture": True,
        "animate": False
    },
    "RECORDING": {
        "title": "RECORDING",
        "icon": "●",
        "text": "Recording...",
        "color": "#3B82F6",
        "bg": "#1E1B4B",
        "btn_text": "🎙️ RECORDING",
        "btn_bg": "#2563EB",
        "active_capture": True,
        "animate": False
    },
    "TRANSCRIBING": {
        "title": "QUERY TRANSCRIBING",
        "icon": "📝",
        "text": "Query is Transcribing...",
        "color": "#8B5CF6",
        "bg": "#2E1065",
        "btn_text": "📝 TRANSCRIBING...",
        "btn_bg": "#7C3AED",
        "active_capture": False,
        "animate": False
    },
    "QUERY_READY": {
        "title": "TRANSCRIPT READY",
        "icon": "✓",
        "text": "Transcript Ready",
        "color": "#14B8A6",
        "bg": "#134E4A",
        "btn_text": "✓ TRANSCRIPT READY",
        "btn_bg": "#0D9488",
        "active_capture": False,
        "animate": False
    },
    "PROCESSING": {
        "title": "PROCESSING / THINKING",
        "icon": "⚙️",
        "text": "Thinking... processing your request",
        "color": "#A855F7",
        "bg": "#3B0764",
        "btn_text": "⚙️ THINKING...",
        "btn_bg": "#9333EA",
        "active_capture": False,
        "animate": False
    },
    "SPEAKING": {
        "title": "SPEAKING",
        "icon": "🗣️",
        "text": "ADAM is speaking...",
        "color": "#38BDF8",
        "bg": "#0C4A6E",
        "btn_text": "🗣️ SPEAKING",
        "btn_bg": "#0284C7",
        "active_capture": False,
        "animate": False
    },
    "READY": {
        "title": "READY",
        "icon": "✔",
        "text": "Ready for your next command.",
        "color": "#14B8A6",
        "bg": "#134E4A",
        "btn_text": "🟢 READY",
        "btn_bg": "#0D9488",
        "active_capture": False,
        "animate": False
    },
    "MIC_ERROR": {
        "title": "MICROPHONE ERROR",
        "icon": "⚠️",
        "text": "Microphone error. Check Audio Setup.",
        "color": "#EF4444",
        "bg": "#450A0A",
        "btn_text": "⚠️ MIC ERROR",
        "btn_bg": "#DC2626",
        "active_capture": False,
        "animate": False
    }
}


class AdamApp(tk.Tk):

    """Main Application Window for ADAM Desktop Mate."""

    def __init__(self):
        super().__init__()
        self.title("ADAM — Advanced Desktop AI Mate")
        self.geometry("1100x780")
        self.minsize(880, 600)
        self.configure(bg=THEME["bg_main"])

        # Thread-safe UI event queue
        self.ui_queue: queue.Queue = queue.Queue()

        # Authoritative Voice Pipeline Controller State
        self.voice_state = "MIC_STARTING"
        self._mic_toggle_in_progress = False
        self.processing_start_time = 0.0
        self._last_user_prompt = ""
        self._last_user_prompt_time = 0.0
        self._waiting_for_followup_command = False
        self._request_lock = threading.Lock()
        self._seen_request_ids = set()
        self._request_id_order = []
        self._active_request_id = None
        self._active_request_cancel = None

        # Reactor animation states & energy
        self.reactor_state = "idle"
        self.anim_tick = 0.0
        self.audio_energy = 0.0

        # Subsystems
        self.voice_tts = PiperTTS(
            on_start=self._on_tts_start,
            on_finish=self._on_tts_finish
        )

        self.speech_stt = SpeechListener(
            on_wake_detected=self._on_wake_detected,
            on_listening_state_change=self._on_listening_change,
            on_audio_energy=self._on_audio_energy,
            on_flow_state_change=self._on_flow_state_change,
            on_command_recognized=self._on_command_recognized,
            on_command_timeout=self._on_command_timeout,
            on_command_failed=self._on_command_failed,
        )

        self._vision_settings_path = _ROOT_DIR / "data" / "vision_settings.json"
        self._vision_settings = self._load_vision_settings()
        self.vision = VisionService(
            camera_index=self._vision_settings["camera_index"],
            confidence_threshold=self._vision_settings["confidence_threshold"],
        )
        self.integrations = SystemIntegrations(
            confirmation_request_handler=self._on_confirmation_requested,
            vision_service=self.vision,
            three_d_callback=self._on_3d_model_generated,
            three_d_viewport_command_callback=self._on_3d_viewport_command,
        )
        self.local_store = LocalStore()
        self._tasks_panel_refresh = None
        self._camera_photo = None
        self._camera_is_active = False
        self._camera_generation = 0
        self._system_metrics = {
            "cpu": "Unavailable",
            "memory": "Unavailable",
            "battery": "Unavailable",
            "temperature": "Unavailable",
        }
        self.system_monitor = SystemMonitor(
            lambda metrics: self.ui_queue.put(("SYSTEM_METRICS", metrics))
        )

        # Build UI Components
        self._setup_styles()
        self._build_header()
        self._build_center_stage()
        self._build_bottom_bar()
        self._restore_conversation_history()
        self.protocol("WM_DELETE_WINDOW", self._shutdown)

        # Initialize voice state display
        initial_state = "MIC_STARTING" if self.speech_stt.mic_available else "MIC_OFF"
        self.set_voice_state(initial_state)

        # Start background timers & workers
        self.after(30, self._process_ui_queue)
        self.system_monitor.start()

        # Auto-start wake word listener
        if self.speech_stt.mic_available:
            self.speech_stt.start_wake_word_loop()

        # Welcome message
        self.after(400, self._post_welcome_message)

    def _load_vision_settings(self):
        settings = {
            "camera_index": 0,
            "confidence_threshold": 0.4,
            "camera_permission_granted": False,
        }
        try:
            if self._vision_settings_path.is_file():
                saved = json.loads(self._vision_settings_path.read_text(encoding="utf-8"))
                settings["camera_index"] = max(0, min(8, int(saved.get("camera_index", 0))))
                settings["confidence_threshold"] = min(
                    0.8, max(0.2, float(saved.get("confidence_threshold", 0.4)))
                )
                settings["camera_permission_granted"] = bool(
                    saved.get("camera_permission_granted", False)
                )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            logger.exception("Could not load camera settings; using safe defaults.")
        return settings

    def _save_vision_settings(self):
        self._vision_settings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._vision_settings_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(self._vision_settings, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self._vision_settings_path)

    def _setup_styles(self):
        """Configure ttk styles to match cyber dark theme."""
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure(".", background=THEME["bg_main"], foreground=THEME["text_primary"])
        self.style.configure(
            "Vertical.TScrollbar",
            gripcount=0,
            background=THEME["bg_card"],
            darkcolor=THEME["bg_main"],
            lightcolor=THEME["border_glass"],
            troughcolor=THEME["bg_main"],
            bordercolor=THEME["border_glass"],
            arrowcolor=THEME["accent_cyan"]
        )

    # =========================================================================
    # UI BUILDERS: HEADER, STAGE, CONTROLS
    # =========================================================================

    def _build_header(self):
        """Top glass navigation bar with system badges and tool buttons."""
        self.header_frame = tk.Frame(self, bg=THEME["bg_header"], height=58, highlightthickness=1, highlightbackground=THEME["border_glass"])
        self.header_frame.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(10, 6))

        # Brand Logo
        logo_frame = tk.Frame(self.header_frame, bg=THEME["bg_header"])
        logo_frame.pack(side=tk.LEFT, padx=16, pady=8)

        logo_title = tk.Label(
            logo_frame,
            text="A D A M",
            font=(THEME["font_main"], 15, "bold"),
            fg=THEME["accent_cyan"],
            bg=THEME["bg_header"]
        )
        logo_title.pack(side=tk.LEFT)

        logo_sub = tk.Label(
            logo_frame,
            text="AI MATE",
            font=(THEME["font_main"], 8, "bold"),
            fg="#60A5FA",
            bg="#1E293B",
            padx=6,
            pady=1
        )
        logo_sub.pack(side=tk.LEFT, padx=(8, 0))

        # Center Status Badges
        status_frame = tk.Frame(self.header_frame, bg=THEME["bg_header"])
        status_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)

        self.ollama_badge = tk.Label(
            status_frame,
            text=self.integrations.ollama.get_status_info(),
            font=(THEME["font_mono"], 9),
            fg=THEME["text_secondary"],
            bg=THEME["bg_card"],
            padx=10,
            pady=3,
            relief=tk.FLAT
        )
        self.ollama_badge.pack(side=tk.LEFT, padx=4)

        self.voice_badge = tk.Label(
            status_frame,
            text=self.voice_tts.get_status_info(),
            font=(THEME["font_mono"], 9),
            fg=THEME["text_secondary"],
            bg=THEME["bg_card"],
            padx=10,
            pady=3,
            relief=tk.FLAT
        )
        self.voice_badge.pack(side=tk.LEFT, padx=4)

        self.mic_badge = tk.Label(
            status_frame,
            text=self.speech_stt.get_status_info(),
            font=(THEME["font_mono"], 9),
            fg=THEME["accent_green"] if self.speech_stt.mic_available else THEME["accent_amber"],
            bg=THEME["bg_card"],
            padx=10,
            pady=3,
            relief=tk.FLAT
        )
        self.mic_badge.pack(side=tk.LEFT, padx=4)

        # Right Action Buttons
        btn_frame = tk.Frame(self.header_frame, bg=THEME["bg_header"])
        btn_frame.pack(side=tk.RIGHT, padx=12, pady=8)

        # Prominent Microphone Toggle Button in Header
        initial_mic_text = "⏳ MIC ACTIVATING" if self.speech_stt.mic_available else "⚠️ MIC UNAVAILABLE"
        initial_mic_bg = "#D97706" if self.speech_stt.mic_available else "#9A3412"

        self.btn_header_mic = tk.Button(
            btn_frame,
            text=initial_mic_text,
            font=(THEME["font_main"], 9, "bold"),
            bg=initial_mic_bg,
            fg="#FFFFFF",
            activebackground="#047857",
            activeforeground="#FFFFFF",
            bd=0,
            padx=12,
            pady=4,
            cursor="hand2",
            command=self._toggle_microphone
        )
        self.btn_header_mic.pack(side=tk.LEFT, padx=(0, 6))

        btn_audio = tk.Button(
            btn_frame,
            text="🎙️ Audio Setup",
            font=(THEME["font_main"], 9, "bold"),
            bg="#064E3B",
            fg=THEME["accent_green"],
            activebackground="#065F46",
            bd=0,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._open_microphone_modal
        )
        btn_audio.pack(side=tk.LEFT, padx=4)

        btn_voice = tk.Button(
            btn_frame,
            text="🗣 Voice",
            font=(THEME["font_main"], 9, "bold"),
            bg="#1E293B",
            fg=THEME["accent_cyan"],
            activebackground="#334155",
            bd=0,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._open_voice_modal
        )
        btn_voice.pack(side=tk.LEFT, padx=4)

        btn_3d = tk.Button(
            btn_frame,
            text="⚡ 3D Lab",
            font=(THEME["font_main"], 9, "bold"),
            bg="#1E1B4B",
            fg="#A78BFA",
            activebackground="#2E1065",
            activeforeground="#DDD6FE",
            bd=0,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._open_3d_modal
        )
        btn_3d.pack(side=tk.LEFT, padx=4)

        btn_model = tk.Button(
            btn_frame,
            text="⚙ Models",
            font=(THEME["font_main"], 9, "bold"),
            bg="#0369A1",
            fg="#FFFFFF",
            activebackground="#0284C7",
            bd=0,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._open_model_selector
        )
        btn_model.pack(side=tk.LEFT, padx=4)

        btn_clear = tk.Button(
            btn_frame,
            text="🗑 Clear",
            font=(THEME["font_main"], 9),
            bg=THEME["bg_card"],
            fg=THEME["text_secondary"],
            activebackground=THEME["bg_card_hover"],
            bd=0,
            padx=8,
            pady=4,
            cursor="hand2",
            command=self._clear_chat
        )
        btn_clear.pack(side=tk.LEFT, padx=4)

    def _build_center_stage(self):
        """Builds reactor orb visualizer and scrollable chat conversation view."""
        self.stage_frame = tk.Frame(self, bg=THEME["bg_main"])
        self.stage_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=4)

        # Top Section: quiet branding and current microphone state
        self.visualizer_frame = tk.Frame(
            self.stage_frame,
            bg=THEME["bg_card"],
            height=152,
            highlightthickness=1,
            highlightbackground=THEME["border_glass"]
        )
        self.visualizer_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        self.visualizer_frame.pack_propagate(False)

        # Split into simple branding and the live voice status panel
        self.visualizer_inner = tk.Frame(self.visualizer_frame, bg=THEME["bg_card"])
        self.visualizer_inner.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self.reactor_canvas = tk.Label(
            self.visualizer_inner,
            text="ADAM",
            font=(THEME["font_main"], 22, "bold"),
            bg=THEME["bg_card"],
            fg=THEME["accent_cyan"],
            width=12,
            anchor="center"
        )
        self.reactor_canvas.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))

        # Right: Live Voice Activity Controller Panel
        self.voice_panel = tk.Frame(
            self.visualizer_inner,
            bg="#0B132B",
            highlightthickness=1,
            highlightbackground=THEME["border_glass"],
            padx=12,
            pady=6
        )
        self.voice_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Row 1: State Badge + Pipeline Step Breadcrumbs
        row_top = tk.Frame(self.voice_panel, bg="#0B132B")
        row_top.pack(fill=tk.X, pady=(0, 4))

        self.badge_voice_frame = tk.Frame(
            row_top,
            bg="#1E293B",
            highlightthickness=1,
            highlightbackground="#64748B",
            padx=10,
            pady=2
        )
        self.badge_voice_frame.pack(side=tk.LEFT)

        self.lbl_voice_icon = tk.Label(
            self.badge_voice_frame,
            text="🔇",
            font=(THEME["font_main"], 11),
            bg="#1E293B",
            fg="#FFFFFF"
        )
        self.lbl_voice_icon.pack(side=tk.LEFT, padx=(0, 6))

        self.lbl_voice_title = tk.Label(
            self.badge_voice_frame,
            text="MIC OFF",
            font=(THEME["font_main"], 9, "bold"),
            bg="#1E293B",
            fg="#FFFFFF"
        )
        self.lbl_voice_title.pack(side=tk.LEFT)

        # Pipeline Steps Breadcrumb
        steps_frame = tk.Frame(row_top, bg="#0B132B")
        steps_frame.pack(side=tk.RIGHT)

        self.pipeline_steps = {}
        stages = [("WAKE", "Wake"), ("LISTEN", "Listen"), ("TRANSCRIBE", "Transcribe"), ("THINK", "Think"), ("SPEAK", "Speak")]
        for idx, (s_key, s_label) in enumerate(stages):
            if idx > 0:
                tk.Label(steps_frame, text="➔", font=(THEME["font_mono"], 8), fg="#475569", bg="#0B132B").pack(side=tk.LEFT, padx=3)
            lbl = tk.Label(
                steps_frame,
                text=f"[{s_label}]",
                font=(THEME["font_mono"], 8),
                fg="#64748B",
                bg="#0B132B"
            )
            lbl.pack(side=tk.LEFT)
            self.pipeline_steps[s_key] = lbl

        # Row 2: Status Text + Duration Timer
        row_mid = tk.Frame(self.voice_panel, bg="#0B132B")
        row_mid.pack(fill=tk.X, pady=(2, 4))

        self.lbl_voice_text = tk.Label(
            row_mid,
            text="Audio capture is stopped. Click MIC button to activate.",
            font=(THEME["font_main"], 10, "bold"),
            fg="#94A3B8",
            bg="#0B132B",
            anchor="w"
        )
        self.lbl_voice_text.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.lbl_voice_timer = tk.Label(
            row_mid,
            text="",
            font=(THEME["font_mono"], 9, "bold"),
            fg=THEME["accent_purple"],
            bg="#0B132B"
        )
        self.lbl_voice_timer.pack(side=tk.RIGHT, padx=(6, 0))

        # Row 3: Microphone Level Meter (LED-style peak meter)
        row_meter = tk.Frame(self.voice_panel, bg="#0B132B")
        row_meter.pack(fill=tk.X, pady=(2, 0))

        tk.Label(
            row_meter,
            text="MIC LEVEL:",
            font=(THEME["font_mono"], 8, "bold"),
            fg="#64748B",
            bg="#0B132B"
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.meter_canvas = tk.Canvas(
            row_meter,
            height=14,
            bg="#020617",
            highlightthickness=1,
            highlightbackground="#1E293B"
        )
        self.meter_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.lbl_meter_val = tk.Label(
            row_meter,
            text="[ MUTED ]",
            font=(THEME["font_mono"], 8, "bold"),
            fg="#64748B",
            bg="#0F172A",
            width=10,
            anchor="e"
        )
        self.lbl_meter_val.pack(side=tk.RIGHT)

        self.system_status_label = tk.Label(
            self.voice_panel,
            text="CPU: Unavailable  |  RAM: Unavailable  |  Battery: Unavailable  |  Temp: Unavailable",
            font=(THEME["font_mono"], 8),
            fg="#94A3B8",
            bg="#0B132B",
            anchor="w",
        )
        self.system_status_label.pack(fill=tk.X, pady=(5, 0))

        # Content Splitter: Left = Studio (Vision + 3D), Right = Chat Stream
        self.stage_paned = tk.PanedWindow(
            self.stage_frame,
            orient=tk.HORIZONTAL,
            bg="#0B132B",
            bd=0,
            sashwidth=6,
            sashrelief=tk.RAISED,
        )
        self.stage_paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ---------------------------------------------------------------------
        # Left Pane: Studio Panel (Tabs: Live Vision & 3D Studio)
        # ---------------------------------------------------------------------
        self.studio_panel = tk.Frame(
            self.stage_paned,
            bg="#0B132B",
            highlightthickness=1,
            highlightbackground=THEME["border_glass"],
        )
        self.camera_panel = self.studio_panel  # Compatibility alias

        # Studio Tab Selector Bar
        self.studio_tabs_bar = tk.Frame(self.studio_panel, bg="#0B132B", height=34)
        self.studio_tabs_bar.pack(side=tk.TOP, fill=tk.X)

        self.btn_tab_vision = tk.Button(
            self.studio_tabs_bar,
            text="📹 Live Vision",
            font=(THEME["font_main"], 9, "bold"),
            bg="#1E293B",
            fg=THEME["accent_cyan"],
            bd=0,
            padx=12,
            pady=6,
            cursor="hand2",
            command=lambda: self._select_studio_tab("vision"),
        )
        self.btn_tab_vision.pack(side=tk.LEFT, padx=(2, 2), pady=2)

        self.btn_tab_3d = tk.Button(
            self.studio_tabs_bar,
            text="📦 3D Studio",
            font=(THEME["font_main"], 9, "bold"),
            bg="#0B132B",
            fg="#94A3B8",
            bd=0,
            padx=12,
            pady=6,
            cursor="hand2",
            command=lambda: self._select_studio_tab("3d"),
        )
        self.btn_tab_3d.pack(side=tk.LEFT, padx=(2, 6), pady=2)

        # Tab 1: Live Vision Container
        self.vision_container = tk.Frame(self.studio_panel, bg="#0B132B")
        self.vision_container.pack(fill=tk.BOTH, expand=True)

        # Vision Header Controls
        vision_ctrl = tk.Frame(self.vision_container, bg="#0B132B", padx=6, pady=4)
        vision_ctrl.pack(side=tk.TOP, fill=tk.X)

        # Camera Device selector
        tk.Label(
            vision_ctrl,
            text="CAM:",
            font=(THEME["font_mono"], 8, "bold"),
            fg="#64748B",
            bg="#0B132B",
        ).pack(side=tk.LEFT, padx=(2, 4))

        self.available_cams = list_available_cameras(max_probe=4)
        cam_names = [f"Camera {c['index']}: {c['name']}" for c in self.available_cams]
        if not cam_names:
            cam_names = ["Camera 0 (Default)"]
        self.cam_device_var = tk.StringVar(value=cam_names[0] if cam_names else "Camera 0")

        self.cam_dropdown = ttk.Combobox(
            vision_ctrl,
            textvariable=self.cam_device_var,
            values=cam_names,
            state="readonly",
            width=18,
        )
        self.cam_dropdown.pack(side=tk.LEFT, padx=(0, 6))
        self.cam_dropdown.bind("<<ComboboxSelected>>", self._on_camera_device_changed)

        self.btn_camera_studio_toggle = tk.Button(
            vision_ctrl,
            text="Start Camera",
            font=(THEME["font_main"], 8, "bold"),
            bg="#0284C7",
            fg="#FFFFFF",
            bd=0,
            padx=8,
            pady=3,
            cursor="hand2",
            command=self._toggle_camera,
        )
        self.btn_camera_studio_toggle.pack(side=tk.LEFT, padx=3)
        self.btn_camera_stop = self.btn_camera_studio_toggle  # Compatibility alias

        self.lbl_camera_telemetry = tk.Label(
            vision_ctrl,
            text="⚡ 0.0 FPS | Infer: 0ms | Latency: 0ms",
            font=(THEME["font_mono"], 8),
            fg="#38BDF8",
            bg="#0B132B",
            anchor="e",
        )
        self.lbl_camera_telemetry.pack(side=tk.RIGHT, padx=4)

        # Vision Preview Display Area
        camera_preview = tk.Frame(self.vision_container, bg="#020617")
        camera_preview.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.camera_preview_label = tk.Label(
            camera_preview,
            text="Camera preview is off.\nClick 'Start Camera' or say 'Turn on camera'.",
            font=(THEME["font_main"], 10),
            fg="#94A3B8",
            bg="#020617",
            justify=tk.CENTER,
        )
        self.camera_preview_label.pack(fill=tk.BOTH, expand=True)

        # Vision Status & Detections Bar
        vision_footer = tk.Frame(self.vision_container, bg="#0B132B", padx=6, pady=4)
        vision_footer.pack(side=tk.BOTTOM, fill=tk.X)

        self.camera_status_label = tk.Label(
            vision_footer,
            text="Camera Stopped",
            font=(THEME["font_main"], 9, "bold"),
            fg=THEME["text_secondary"],
            bg="#0B132B",
            anchor="w",
        )
        self.camera_status_label.pack(side=tk.TOP, fill=tk.X)

        self.camera_detections_label = tk.Label(
            vision_footer,
            text="No current detections.",
            font=(THEME["font_mono"], 8),
            fg="#CBD5E1",
            bg="#0B132B",
            anchor="w",
        )
        self.camera_detections_label.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))

        # Tab 2: 3D Studio Container
        self.three_d_container = tk.Frame(self.studio_panel, bg="#0B132B")
        self.three_d_viewport = Interactive3DViewport(self.three_d_container)
        self.three_d_viewport.pack(fill=tk.BOTH, expand=True)

        # ---------------------------------------------------------------------
        # Right Pane: Chat Scroll Area
        # ---------------------------------------------------------------------
        self.chat_container = tk.Frame(
            self.stage_paned,
            bg=THEME["bg_card"],
            highlightthickness=1,
            highlightbackground=THEME["border_glass"]
        )

        self.stage_paned.add(self.studio_panel, minsize=320)
        self.stage_paned.add(self.chat_container, minsize=320)

        # Canvas + Scrollbar for chat messages
        self.chat_canvas = tk.Canvas(
            self.chat_container,
            bg=THEME["bg_card"],
            bd=0,
            highlightthickness=0
        )
        self.chat_scrollbar = ttk.Scrollbar(
            self.chat_container,
            orient=tk.VERTICAL,
            command=self.chat_canvas.yview
        )
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)

        self.chat_scrollable_frame = tk.Frame(self.chat_canvas, bg=THEME["bg_card"])
        self.chat_window_id = self.chat_canvas.create_window(
            (0, 0),
            window=self.chat_scrollable_frame,
            anchor="nw"
        )

        self.chat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.chat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.chat_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        )
        self.chat_canvas.bind(
            "<Configure>",
            lambda e: self.chat_canvas.itemconfig(self.chat_window_id, width=e.width)
        )

        # Mousewheel scroll support
        self.chat_canvas.bind_all("<Button-4>", lambda e: self.chat_canvas.yview_scroll(-2, "units"))
        self.chat_canvas.bind_all("<Button-5>", lambda e: self.chat_canvas.yview_scroll(2, "units"))
        self.chat_canvas.bind_all("<MouseWheel>", lambda e: self.chat_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

    def _build_bottom_bar(self):
        """Bottom glass interaction bar: input box, send, voice toggle, push-to-talk."""
        self.bottom_frame = tk.Frame(
            self,
            bg=THEME["bg_header"],
            height=70,
            highlightthickness=1,
            highlightbackground=THEME["border_glass"]
        )
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(4, 12))

        inner = tk.Frame(self.bottom_frame, bg=THEME["bg_header"])
        inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Push to talk button
        self.btn_push_talk = tk.Button(
            inner,
            text="🎙️ Listen Once",
            font=(THEME["font_main"], 9),
            bg=THEME["bg_card"],
            fg=THEME["text_primary"],
            activebackground=THEME["bg_card_hover"],
            bd=0,
            padx=10,
            pady=8,
            cursor="hand2",
            command=self._on_push_to_talk
        )
        self.btn_push_talk.pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            inner,
            text="Tasks",
            font=(THEME["font_main"], 9),
            bg=THEME["bg_card"],
            fg=THEME["text_primary"],
            activebackground=THEME["bg_card_hover"],
            bd=0,
            padx=8,
            pady=8,
            cursor="hand2",
            command=self._open_tasks_panel,
        ).pack(side=tk.LEFT, padx=(0, 6))

        self.btn_camera = tk.Button(
            inner,
            text="Start Camera",
            font=(THEME["font_main"], 9),
            bg=THEME["bg_card"],
            fg=THEME["accent_cyan"],
            activebackground=THEME["bg_card_hover"],
            bd=0,
            padx=8,
            pady=8,
            cursor="hand2",
            command=self._toggle_camera,
        )
        self.btn_camera.pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            inner,
            text="Camera Settings",
            font=(THEME["font_main"], 8),
            bg=THEME["bg_card"],
            fg=THEME["text_secondary"],
            activebackground=THEME["bg_card_hover"],
            bd=0,
            padx=6,
            pady=8,
            cursor="hand2",
            command=self._open_vision_settings,
        ).pack(side=tk.LEFT, padx=(0, 6))
        self.btn_3d_studio_bottom = tk.Button(
            inner,
            text="📦 3D Studio",
            font=(THEME["font_main"], 9),
            bg=THEME["bg_card"],
            fg=THEME["accent_cyan"],
            activebackground=THEME["bg_card_hover"],
            bd=0,
            padx=8,
            pady=8,
            cursor="hand2",
            command=lambda: self._select_studio_tab("3d"),
        )
        self.btn_3d_studio_bottom.pack(side=tk.LEFT, padx=(0, 6))

        gmail_status = self.integrations.gmail_client.get_auth_status()
        self.btn_gmail = tk.Button(
            inner,
            text="Gmail Connected" if gmail_status["authorized"] else "Connect Gmail",
            font=(THEME["font_main"], 9),
            bg=THEME["bg_card"],
            fg=THEME["text_primary"],
            activebackground=THEME["bg_card_hover"],
            bd=0,
            padx=8,
            pady=8,
            cursor="hand2",
            command=self._open_gmail_connection,
        )
        self.btn_gmail.pack(side=tk.LEFT, padx=(0, 8))

        # Text input field
        self.input_entry = tk.Entry(
            inner,
            font=(THEME["font_main"], 11),
            bg=THEME["bg_input"],
            fg=THEME["text_primary"],
            insertbackground=THEME["accent_cyan"],
            bd=0,
            highlightthickness=1,
            highlightbackground=THEME["border_glass"],
            highlightcolor=THEME["accent_cyan"]
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 8))
        self.input_entry.bind("<Return>", lambda e: self._handle_user_submit())
        self.input_entry.focus_set()

        # Stop Voice button
        self.btn_stop_voice = tk.Button(
            inner,
            text="⏹ Stop Voice",
            font=(THEME["font_main"], 9),
            bg=THEME["bg_card"],
            fg=THEME["accent_red"],
            activebackground=THEME["bg_card_hover"],
            bd=0,
            padx=10,
            pady=8,
            cursor="hand2",
            command=self._cancel_active_request
        )
        self.btn_stop_voice.pack(side=tk.RIGHT, padx=(4, 0))

        # Send Button
        self.btn_send = tk.Button(
            inner,
            text="SEND  ➤",
            font=(THEME["font_main"], 9, "bold"),
            bg="#0369A1",
            fg="#FFFFFF",
            activebackground="#0284C7",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._handle_user_submit
        )
        self.btn_send.pack(side=tk.RIGHT)

    # =========================================================================
    # ANIMATED REACTOR CANVAS (FUTURISTIC AI VISUALIZER)
    # =========================================================================

    def _draw_level_meter(self, level: float, active: bool):
        """Draw a real-time 16-segment studio LED peak VU meter."""
        if not hasattr(self, "meter_canvas"):
            return
        w = self.meter_canvas.winfo_width()
        h = self.meter_canvas.winfo_height()
        if w < 20 or h < 5:
            return

        self.meter_canvas.delete("all")
        total_segments = 16
        gap = 2
        total_gaps = (total_segments - 1) * gap
        seg_w = max(2.0, (w - total_gaps) / total_segments)

        lit_count = int(round(min(1.0, max(0.0, level)) * total_segments)) if active else 0

        for i in range(total_segments):
            x1 = i * (seg_w + gap)
            x2 = x1 + seg_w
            y1 = 1
            y2 = h - 2

            if active and i < lit_count:
                if i < 10:
                    color = "#10B981"  # green / low-mid
                elif i < 13:
                    color = "#F59E0B"  # amber / high
                else:
                    color = "#EF4444"  # red / peak
            else:
                color = "#131C31"  # dim unlit LED

            self.meter_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

        # Update text badge
        if not hasattr(self, "lbl_meter_val"):
            return
        if not active:
            if self.voice_state == "SPEAKING":
                self.lbl_meter_val.config(text="[ SPEAKING ]", fg="#38BDF8")
            elif self.voice_state == "PROCESSING":
                self.lbl_meter_val.config(text="[ THINKING ]", fg="#A78BFA")
            elif self.voice_state == "TRANSCRIBING":
                self.lbl_meter_val.config(text="[ PARSING ]", fg="#C084FC")
            elif self.voice_state == "MIC_STARTING":
                self.lbl_meter_val.config(text="[ STARTING ]", fg="#F59E0B")
            else:
                self.lbl_meter_val.config(text="[ MUTED ]", fg="#64748B")
        else:
            pct = int(round(min(1.0, max(0.0, level)) * 100))
            color = "#10B981" if pct < 65 else ("#F59E0B" if pct < 85 else "#EF4444")
            self.lbl_meter_val.config(text=f"[ {pct:02d}% ]", fg=color)

    def _animation_loop(self):
        """30-40 FPS animation loop for holographic gyroscopic reactor and live meter."""
        self.anim_tick += 0.06
        w = self.reactor_canvas.winfo_width()
        h = self.reactor_canvas.winfo_height()

        active_capture = VOICE_STATES.get(self.voice_state, {}).get("active_capture", False)

        # 1. Update LED VU Level Meter
        self._draw_level_meter(self.audio_energy if active_capture else 0.0, active_capture)

        # 2. Update Processing Timer
        if self.voice_state == "PROCESSING" and self.processing_start_time > 0:
            elapsed = time.time() - self.processing_start_time
            self.lbl_voice_timer.config(text=f"({elapsed:.1f}s)")
        else:
            self.lbl_voice_timer.config(text="")

        # 3. Hologram canvas rendering
        if w > 20 and h > 20:
            self.reactor_canvas.delete("all")
            cx = w / 2.0
            cy = h / 2.0

            # Determine colors and pulse based on voice_state
            if self.voice_state == "MIC_OFF":
                core_color = "#475569"
                outer_color = "#1E293B"
                label_text = "ADAM // MIC MUTED (TYPED CHAT ACTIVE)"
                pulse = 1.0
            elif self.voice_state == "MIC_STARTING":
                core_color = "#F59E0B"
                outer_color = "#78350F"
                label_text = "ADAM // INITIALIZING MICROPHONE..."
                pulse = 1.0 + 0.1 * math.sin(self.anim_tick * 4.0)
            elif self.voice_state in ("LISTENING_WAKE", "LISTENING_COMMAND"):
                core_color = THEME["accent_green"] if self.voice_state == "LISTENING_WAKE" else "#38BDF8"
                outer_color = "#064E3B" if self.voice_state == "LISTENING_WAKE" else "#0C4A6E"
                label_text = "ADAM // LISTENING FOR WAKE WORD" if self.voice_state == "LISTENING_WAKE" else "ADAM // LISTENING FOR COMMAND"
                energy_factor = max(0.1, self.audio_energy)
                pulse = 1.0 + energy_factor * 0.45 + 0.15 * math.sin(self.anim_tick * 4.0)
            elif self.voice_state == "WAKE_DETECTED":
                core_color = "#06B6D4"
                outer_color = "#083344"
                label_text = "ADAM // WAKE WORD DETECTED!"
                pulse = 1.25 + 0.15 * math.sin(self.anim_tick * 6.0)
            elif self.voice_state == "TRANSCRIBING":
                core_color = "#A855F7"
                outer_color = "#3B0764"
                label_text = "ADAM // TRANSCRIBING SPEECH..."
                pulse = 1.1 + 0.15 * math.sin(self.anim_tick * 5.0)
            elif self.voice_state == "PROCESSING":
                core_color = THEME["accent_purple"]
                outer_color = "#3B0764"
                label_text = "ADAM // PROCESSING WITH OLLAMA NEURAL ENGINE..."
                pulse = 1.0 + 0.18 * math.sin(self.anim_tick * 4.0)
            elif self.voice_state == "SPEAKING":
                core_color = "#38BDF8"
                outer_color = "#0369A1"
                label_text = "ADAM // VOCALIZING RESPONSE (PIPER AMY VOICE)"
                pulse = 1.0 + 0.25 * abs(math.sin(self.anim_tick * 6.0))
            elif self.voice_state == "READY":
                core_color = "#14B8A6"
                outer_color = "#042F2E"
                label_text = "ADAM // READY FOR NEXT COMMAND"
                pulse = 1.0 + 0.05 * math.sin(self.anim_tick * 2.0)
            elif self.voice_state == "MIC_ERROR":
                core_color = "#EF4444"
                outer_color = "#450A0A"
                label_text = "ADAM // MICROPHONE ERROR"
                pulse = 1.0
            else:
                core_color = THEME["accent_cyan"]
                outer_color = "#004D5A"
                label_text = "ADAM // SYSTEM READY"
                pulse = 1.0

            # Draw outer radar glow ring
            r_outer = 40.0 * pulse
            self.reactor_canvas.create_oval(
                cx - r_outer, cy - r_outer,
                cx + r_outer, cy + r_outer,
                outline=outer_color, width=1.5
            )

            # Draw rotating orbital nodes
            nodes = 6
            speed = 1.5
            if self.voice_state == "PROCESSING":
                speed = 3.5
            elif self.voice_state == "MIC_OFF":
                speed = 0.5
            for i in range(nodes):
                angle = self.anim_tick * speed + (2 * math.pi * i / nodes)
                nx = cx + r_outer * math.cos(angle)
                ny = cy + (r_outer * 0.5) * math.sin(angle)
                self.reactor_canvas.create_oval(nx - 2, ny - 2, nx + 2, ny + 2, fill=core_color, outline="")

            # Inner pulsating core orb
            r_core = 16.0 * pulse
            self.reactor_canvas.create_oval(
                cx - r_core, cy - r_core,
                cx + r_core, cy + r_core,
                fill=core_color, outline="#E0F2FE" if self.voice_state != "MIC_OFF" else "#64748B", width=1.5
            )

            # Center focal star
            self.reactor_canvas.create_oval(
                cx - 3, cy - 3,
                cx + 3, cy + 3,
                fill="#FFFFFF" if self.voice_state != "MIC_OFF" else "#94A3B8", outline=""
            )

            # Audio waveforms or telemetry bars
            bars = 10
            for b in range(-bars, bars + 1):
                if b == 0:
                    continue
                dist = abs(b)
                bx = cx + b * 11
                if active_capture:
                    amp = (16.0 / (dist * 0.7 + 1)) * max(0.15, self.audio_energy * 2.0) * math.sin(self.anim_tick * 5.0 + dist)
                elif self.voice_state == "SPEAKING":
                    amp = (18.0 / (dist * 0.7 + 1)) * abs(math.sin(self.anim_tick * 6.0 + dist))
                elif self.voice_state == "PROCESSING":
                    amp = (12.0 / (dist * 0.7 + 1)) * abs(math.sin(self.anim_tick * 3.0 + dist))
                else:
                    amp = 0.5  # flat calm line
                by1 = cy - abs(amp)
                by2 = cy + abs(amp)
                self.reactor_canvas.create_line(bx, by1, bx, by2, fill=outer_color, width=1.5)

            # Status label underneath
            self.reactor_canvas.create_text(
                cx, cy + 44,
                text=label_text,
                font=(THEME["font_mono"], 7, "bold"),
                fill=THEME["text_secondary"]
            )

        self.after(35, self._animation_loop)

    # =========================================================================
    # CHAT FEED & MESSAGE CARDS
    # =========================================================================

    def _post_welcome_message(self):
        """Displays startup welcome and diagnostic summary."""
        welcome_text = (
            "**Greetings! I am ADAM (Advanced Desktop AI Mate).**\n\n"
            "I'm your local neural assistant running natively on Fedora Linux.\n"
            "• **Speech**: Always listening for wake-word **\"Hii\"**.\n"
            "• **Voice**: Piper Neural TTS with the natural **Amy English Voice**.\n"
            "• **Local LLM**: Powered by Ollama (`qwen2.5:1.5b` recommended for 8 GB RAM).\n"
            "• **3D Lab**: Blender headless OBJ/STL exporter & pure Python fallback.\n"
            "• **Safety**: Destructive file actions are gated behind interactive user confirmations.\n\n"
            "Ask me questions, tell me to open applications, search YouTube, or try: `create 3d scifi_drone`."
        )
        self._add_message_card("ADAM", welcome_text, persist=False)

    def _restore_conversation_history(self):
        """Restore recent local messages without duplicating them in the database."""
        for message in self.local_store.list_messages():
            self._add_message_card(
                message["role"],
                message["content"],
                source=message["source"],
                created_at=message["created_at"],
                persist=False,
            )

    def _add_message_card(
        self,
        sender: str,
        text: str,
        source: str = "text",
        created_at: Optional[str] = None,
        persist: bool = True,
    ) -> tk.Label:
        """Add a conversation card and persist user/assistant text locally."""
        is_user = sender.upper() == "USER"
        if persist and text.strip():
            try:
                self.local_store.add_message(sender, text, source)
            except Exception:
                logger.exception("Could not persist a conversation message.")

        row = tk.Frame(self.chat_scrollable_frame, bg=THEME["bg_card"])
        row.pack(fill=tk.X, padx=12, pady=6)

        bubble_bg = THEME["user_bubble_bg"] if is_user else THEME["adam_bubble_bg"]
        border_col = THEME["user_bubble_border"] if is_user else THEME["adam_bubble_border"]
        align_side = tk.RIGHT if is_user else tk.LEFT
        header_col = THEME["accent_cyan"] if is_user else THEME["accent_purple"]

        card = tk.Frame(
            row,
            bg=bubble_bg,
            highlightthickness=1,
            highlightbackground=border_col,
            padx=14,
            pady=10
        )
        card.pack(side=align_side, fill=tk.X, expand=False, anchor="e" if is_user else "w")

        # Sender header + timestamp
        hdr_frame = tk.Frame(card, bg=bubble_bg)
        hdr_frame.pack(fill=tk.X, anchor="w", pady=(0, 4))

        icon = "👤" if is_user else "ADAM"
        sender_title = f"{icon} {sender.upper()}"
        if is_user:
            sender_title += " · VOICE" if source == "voice" else " · TYPED"
        sender_lbl = tk.Label(
            hdr_frame,
            text=sender_title,
            font=(THEME["font_main"], 9, "bold"),
            fg=header_col,
            bg=bubble_bg
        )
        sender_lbl.pack(side=tk.LEFT)

        try:
            local_now = datetime.datetime.fromisoformat(created_at).astimezone() if created_at else datetime.datetime.now().astimezone()
        except ValueError:
            local_now = datetime.datetime.now().astimezone()
        tz_abbr = local_now.strftime("%Z")
        time_str = f"{local_now.strftime('%I:%M %p')} {tz_abbr}".strip()
        time_lbl = tk.Label(
            hdr_frame,
            text=time_str,
            font=(THEME["font_mono"], 8),
            fg=THEME["text_muted"],
            bg=bubble_bg
        )
        time_lbl.pack(side=tk.LEFT, padx=10)

        # Message Body
        msg_lbl = tk.Label(
            card,
            text=text,
            font=(THEME["font_main"], 10),
            fg=THEME["text_primary"],
            bg=bubble_bg,
            justify=tk.LEFT,
            wraplength=760
        )
        msg_lbl.pack(fill=tk.BOTH, expand=True, anchor="w")

        self._scroll_to_bottom()
        return msg_lbl

    def _scroll_to_bottom(self):
        """Scroll chat canvas to bottom."""
        self.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)

    # =========================================================================
    # DESTRUCTIVE ACTIONS CONFIRMATION CARD
    # =========================================================================

    def _show_confirmation_dialog(self, action: PendingAction):
        """Renders an interactive confirmation dialog directly inside the chat feed."""
        row = tk.Frame(self.chat_scrollable_frame, bg=THEME["bg_card"])
        row.pack(fill=tk.X, padx=12, pady=8)

        card = tk.Frame(
            row,
            bg="#2A1116",  # Dark Crimson Glass
            highlightthickness=2,
            highlightbackground=THEME["accent_red"],
            padx=16,
            pady=12
        )
        card.pack(side=tk.LEFT, fill=tk.X, expand=True)

        title_lbl = tk.Label(
            card,
            text="⚠️  USER CONFIRMATION REQUIRED — DESTRUCTIVE ACTION",
            font=(THEME["font_main"], 10, "bold"),
            fg=THEME["accent_red"],
            bg="#2A1116"
        )
        title_lbl.pack(anchor="w", pady=(0, 4))

        desc_lbl = tk.Label(
            card,
            text=f"Action: {action.action_type}\nTarget: {action.target}\n{action.description}\n\nAre you sure you want to execute this operation?",
            font=(THEME["font_main"], 10),
            fg=THEME["text_primary"],
            bg="#2A1116",
            justify=tk.LEFT,
            wraplength=700
        )
        desc_lbl.pack(anchor="w", pady=(0, 8))

        btn_row = tk.Frame(card, bg="#2A1116")
        btn_row.pack(anchor="w")

        def on_confirm():
            btn_confirm.config(state=tk.DISABLED)
            btn_cancel.config(state=tk.DISABLED)
            ok, msg = self.integrations.execute_pending_action(action.action_id)
            res_color = THEME["accent_green"] if ok else THEME["accent_red"]
            status_lbl = tk.Label(card, text=f"Result: {msg}", font=(THEME["font_main"], 9, "bold"), fg=res_color, bg="#2A1116")
            status_lbl.pack(anchor="w", pady=(6, 0))
            self.voice_tts.speak(f"Action confirmed. {msg}")

        def on_cancel():
            btn_confirm.config(state=tk.DISABLED)
            btn_cancel.config(state=tk.DISABLED)
            msg = self.integrations.cancel_pending_action(action.action_id)
            status_lbl = tk.Label(card, text=f"Result: {msg}", font=(THEME["font_main"], 9), fg=THEME["text_secondary"], bg="#2A1116")
            status_lbl.pack(anchor="w", pady=(6, 0))
            self.voice_tts.speak("Operation cancelled.")

        btn_confirm = tk.Button(
            btn_row,
            text="✅ Confirm & Delete",
            font=(THEME["font_main"], 9, "bold"),
            bg="#B91C1C",
            fg="#FFFFFF",
            activebackground="#DC2626",
            bd=0,
            padx=12,
            pady=6,
            cursor="hand2",
            command=on_confirm
        )
        btn_confirm.pack(side=tk.LEFT, padx=(0, 10))

        btn_cancel = tk.Button(
            btn_row,
            text="❌ Cancel",
            font=(THEME["font_main"], 9),
            bg=THEME["bg_card"],
            fg=THEME["text_secondary"],
            activebackground=THEME["bg_card_hover"],
            bd=0,
            padx=12,
            pady=6,
            cursor="hand2",
            command=on_cancel
        )
        btn_cancel.pack(side=tk.LEFT)

        self._scroll_to_bottom()

    # =========================================================================
    # EVENT HANDLERS & THREAD-SAFE QUEUE
    # =========================================================================

    def set_voice_state(self, state: str, details: str = ""):
        """
        Authoritative single-source controller for all voice pipeline states.
        Updates state badges, status text, breadcrumb stage tracker,
        header & bottom mic buttons, and audio processing coordinator.
        """
        alias_map = {
            "LISTENING": "LISTENING_WAKE",
            "HEARING_COMMAND": "LISTENING_COMMAND",
            "THINKING": "PROCESSING",
        }
        state = alias_map.get(state, state)
        if state not in VOICE_STATES:
            state = "MIC_OFF"

        prev_state = self.voice_state
        self.voice_state = state
        st_info = VOICE_STATES[state]

        # Synchronize STT engine processing state and reactor animation state
        if state == "PROCESSING":
            if prev_state != "PROCESSING" or self.processing_start_time == 0.0:
                self.processing_start_time = time.time()
            self.speech_stt.is_busy_processing = True
            self.reactor_state = "thinking"
        else:
            if prev_state == "PROCESSING":
                self.processing_start_time = 0.0
            if state == "SPEAKING":
                self.reactor_state = "speaking"
            elif state in ("LISTENING_WAKE", "LISTENING_COMMAND", "SPEECH_DETECTED", "RECORDING"):
                self.speech_stt.is_busy_processing = False
                self.reactor_state = "listening"
            else:
                self.speech_stt.is_busy_processing = False
                self.reactor_state = "idle"

        # Update Live Voice Activity Panel widgets
        if hasattr(self, "lbl_voice_icon"):
            self.lbl_voice_icon.config(text=st_info["icon"], bg=st_info["bg"], fg=st_info["color"])
        if hasattr(self, "lbl_voice_title"):
            self.lbl_voice_title.config(text=st_info["title"], bg=st_info["bg"], fg=st_info["color"])
        if hasattr(self, "badge_voice_frame"):
            self.badge_voice_frame.config(bg=st_info["bg"], highlightbackground=st_info["color"])

        # Status text line
        if hasattr(self, "lbl_voice_text"):
            display_text = details.strip() if details.strip() else st_info["text"]
            self.lbl_voice_text.config(
                text=display_text,
                fg=st_info["color"] if state in ("MIC_ERROR", "WAKE_DETECTED") else "#CBD5E1"
            )

        # Update Stage Breadcrumb tracker
        stage_map = {
            "LISTENING_WAKE": "WAKE",
            "WAKE_DETECTED": "WAKE",
            "LISTENING_COMMAND": "LISTEN",
            "SPEECH_DETECTED": "LISTEN",
            "RECORDING": "LISTEN",
            "TRANSCRIBING": "TRANSCRIBE",
            "QUERY_READY": "TRANSCRIBE",
            "PROCESSING": "THINK",
            "SPEAKING": "SPEAK",
            "READY": "SPEAK",
        }
        active_stage = stage_map.get(state, "")
        if hasattr(self, "pipeline_steps"):
            for s_key, lbl in self.pipeline_steps.items():
                if s_key == active_stage:
                    lbl.config(fg="#38BDF8", font=(THEME["font_mono"], 8, "bold"))
                else:
                    lbl.config(fg="#475569", font=(THEME["font_mono"], 8))

        # Update Both Header & Bottom Microphone Buttons synchronously
        for btn in [getattr(self, "btn_header_mic", None), getattr(self, "btn_mic_toggle", None)]:
            if btn is not None:
                try:
                    btn.config(text=st_info["btn_text"], bg=st_info["btn_bg"], fg="#FFFFFF")
                except Exception:
                    pass

        # Update header device status badge
        if hasattr(self, "mic_badge"):
            self.mic_badge.config(text=self.speech_stt.get_status_info())
        if self.__dict__.get("meter_canvas") is not None and not st_info["active_capture"]:
            self._draw_level_meter(0.0, False)

    def _process_ui_queue(self):
        """Consumes events from background threads and applies them to Tkinter main thread."""
        while not self.ui_queue.empty():
            try:
                event_type, data = self.ui_queue.get_nowait()
                if event_type == "STREAM_CHUNK":
                    request_id, card_lbl, chunk = data
                    if request_id != self._active_request_id:
                        continue
                    card_lbl.config(text=card_lbl.cget("text") + chunk)
                    self._scroll_to_bottom()
                elif event_type == "STREAM_DONE":
                    request_id, final_text = data
                    if request_id != self._active_request_id:
                        continue
                    if final_text.strip():
                        try:
                            self.local_store.add_message("ADAM", final_text, "assistant")
                        except Exception:
                            logger.exception("Could not persist an assistant response.")
                        self.voice_tts.speak(final_text)
                    with self._request_lock:
                        if self._active_request_id == request_id:
                            self._active_request_id = None
                            self._active_request_cancel = None
                elif event_type == "DIRECT_RESPONSE":
                    resp_text, pending_action, request_id = data
                    if request_id != self._active_request_id:
                        continue
                    self._add_message_card("ADAM", resp_text)
                    if resp_text == "Voice playback stopped.":
                        self.voice_tts.stop()
                        self.set_voice_state("READY", "Voice playback stopped.")
                    else:
                        self.voice_tts.speak(resp_text)
                    if pending_action:
                        self._show_confirmation_dialog(pending_action)
                    with self._request_lock:
                        if self._active_request_id == request_id:
                            self._active_request_id = None
                            self._active_request_cancel = None
                elif event_type == "TASKS_REFRESH":
                    refresh_tasks = self._tasks_panel_refresh
                    if refresh_tasks is not None:
                        refresh_tasks()
                elif event_type == "CAMERA_STATUS":
                    generation, state, message = data
                    if generation == self._camera_generation:
                        self._handle_vision_status(state, message)
                elif event_type == "CAMERA_FRAME":
                    generation, frame_rgb, detections = data
                    if generation == self._camera_generation:
                        self._show_camera_frame(frame_rgb, detections)
                elif event_type == "SYSTEM_METRICS":
                    self._update_system_metrics(data)
                elif event_type == "POST_CARD":
                    sender, text = data
                    self._add_message_card(sender, text, persist=False)
                elif event_type == "COMMAND_RECOGNIZED":
                    if isinstance(data, tuple) and len(data) == 3:
                        command, source, request_id = data
                    elif isinstance(data, tuple):
                        command, source = data
                        request_id = uuid.uuid4().hex
                    else:
                        command, source = data, "voice"
                        request_id = uuid.uuid4().hex
                    command = str(command).strip()
                    if command:
                        self._submit_command(command, str(source), str(request_id))

                elif event_type == "COMMAND_TIMEOUT":
                    msg = data
                    self._add_message_card("ADAM", f"⏱️ {msg}")
                    if self.speech_stt.is_listening and not self.speech_stt._paused:
                        self.set_voice_state("LISTENING_WAKE", 'Listening for "Hey Adam"...')
                    else:
                        self.set_voice_state("MIC_OFF", "Audio capture is stopped.")

                elif event_type == "COMMAND_FAILED":
                    msg = data
                    self._add_message_card("ADAM", f"⚠️ {msg}")
                    if str(msg).startswith("Microphone Error"):
                        self.set_voice_state("MIC_ERROR", str(msg))
                    elif self.speech_stt.is_listening and not self.speech_stt._paused:
                        self.set_voice_state("LISTENING_WAKE", 'Listening for "Hey Adam"...')
                    else:
                        self.set_voice_state("MIC_OFF", "Audio capture is stopped.")

                elif event_type == "WAKE_WORD_HEARD":
                    prompt = data
                    now = time.time()
                    if prompt and prompt == self._last_user_prompt and (now - self._last_user_prompt_time) < 2.0:
                        continue
                    if prompt:
                        self.ui_queue.put(("COMMAND_RECOGNIZED", (prompt, "voice")))
                    else:
                        self.set_voice_state("WAKE_DETECTED", "Wake word detected! Start listening — speak your command now")

                elif event_type == "TTS_FINISHED":
                    self.speech_stt.set_tts_speaking(False)
                    if self._active_request_id is None:
                        if self.speech_stt.is_listening and not self.speech_stt._paused:
                            self.set_voice_state("LISTENING_WAKE", 'Listening for "Hey Adam"...')
                        else:
                            self.set_voice_state("MIC_OFF", "Audio capture is stopped.")

                elif event_type == "SET_VOICE_STATE":
                    st, details = data
                    self.set_voice_state(st, details)
                elif event_type == "FLOW_STATE":
                    st, details = data
                    self.set_voice_state(st, details)
                elif event_type == "AUDIO_ENERGY":
                    self.audio_energy = data
                    self._draw_level_meter(
                        data,
                        VOICE_STATES.get(self.voice_state, {}).get("active_capture", False),
                    )
                elif event_type == "CONFIRMATION_REQ":
                    action = data
                    self._show_confirmation_dialog(action)
                elif event_type == "MIC_STATUS":
                    active, status_str = data
                    if hasattr(self, "mic_badge"):
                        self.mic_badge.config(text=status_str)
            except queue.Empty:
                break
            except Exception as e:
                print(f"UI Queue processing exception: {e}", file=sys.stderr)

        self.after(30, self._process_ui_queue)

    def _on_tts_start(self):
        self.speech_stt.set_tts_speaking(True)
        self.ui_queue.put(("SET_VOICE_STATE", ("SPEAKING", "ADAM is speaking...")))

    def _on_tts_finish(self):
        self.ui_queue.put(("TTS_FINISHED", None))

    def _on_flow_state_change(self, state: str, details: str = ""):
        self.ui_queue.put(("FLOW_STATE", (state, details)))

    def _on_wake_detected(self, prompt: str):
        self.ui_queue.put(("WAKE_WORD_HEARD", prompt))

    def _on_command_recognized(self, command: str):
        request_id = self.speech_stt.telemetry.get("last_processed_utterance_id") or uuid.uuid4().hex
        self.ui_queue.put(("COMMAND_RECOGNIZED", (command, "voice", request_id)))

    def _on_command_timeout(self, msg: str):
        self.ui_queue.put(("COMMAND_TIMEOUT", msg))

    def _on_command_failed(self, msg: str):
        self.ui_queue.put(("COMMAND_FAILED", msg))

    def _on_listening_change(self, active: bool, msg: str):
        self.ui_queue.put(("MIC_STATUS", (active, msg)))

    def _on_audio_energy(self, energy: float):
        self.ui_queue.put(("AUDIO_ENERGY", energy))

    def _on_confirmation_requested(self, action: PendingAction):
        self.ui_queue.put(("CONFIRMATION_REQ", action))

    def _on_vision_status(self, state: str, message: str, generation: int):
        self.ui_queue.put(("CAMERA_STATUS", (generation, state, message)))

    def _on_vision_frame(self, frame_rgb, detections, generation: int):
        self.ui_queue.put(("CAMERA_FRAME", (generation, frame_rgb, detections)))

    def _on_system_metrics(self, metrics):
        self.ui_queue.put(("SYSTEM_METRICS", metrics))

    def _toggle_camera(self):
        if self.vision.status in ("CAMERA_STARTING", "CAMERA_ACTIVE"):
            self._stop_camera()
            return False

        if not self._vision_settings["camera_permission_granted"]:
            approved = messagebox.askyesno(
                "Camera permission",
                "Allow ADAM to open the selected webcam for local object detection?\n\n"
                "Frames are processed on this computer only. ADAM does not save or upload "
                "camera images or video. This feature is not secure authentication.",
                parent=self,
            )
            if not approved:
                return False
            self._vision_settings["camera_permission_granted"] = True
            try:
                self._save_vision_settings()
            except OSError as exc:
                logger.exception("Could not save camera permission setting.")
                messagebox.showerror(
                    "Camera settings",
                    f"Could not save camera permission: {exc}",
                    parent=self,
                )
                return False

        self._select_studio_tab("vision")
        self.camera_status_label.config(text="Camera Starting", fg=THEME["accent_amber"])
        self.camera_preview_label.config(text="Opening local camera…", image="")
        self.camera_detections_label.config(text="Loading local detector…")
        self.btn_camera.config(text="Starting…", state=tk.DISABLED)
        if hasattr(self, "btn_camera_studio_toggle"):
            self.btn_camera_studio_toggle.config(text="Starting…", state=tk.DISABLED)
        self._camera_generation += 1
        generation = self._camera_generation
        started = self.vision.start(
            lambda frame, detections: self._on_vision_frame(frame, detections, generation),
            lambda state, message: self._on_vision_status(state, message, generation),
        )
        if not started:
            self.btn_camera.config(text="Start Camera", state=tk.NORMAL)
            if hasattr(self, "btn_camera_studio_toggle"):
                self.btn_camera_studio_toggle.config(text="Start Camera", state=tk.NORMAL)
        return started

    def _select_studio_tab(self, tab: str):
        """Switch between 'vision' and '3d' in the Studio pane."""
        if tab == "vision":
            self.three_d_container.pack_forget()
            self.vision_container.pack(fill=tk.BOTH, expand=True)
            self.btn_tab_vision.config(bg="#1E293B", fg=THEME["accent_cyan"])
            self.btn_tab_3d.config(bg="#0B132B", fg="#94A3B8")
        elif tab == "3d":
            self.vision_container.pack_forget()
            self.three_d_container.pack(fill=tk.BOTH, expand=True)
            self.btn_tab_vision.config(bg="#0B132B", fg="#94A3B8")
            self.btn_tab_3d.config(bg="#1E293B", fg=THEME["accent_cyan"])

    def _on_camera_device_changed(self, event=None):
        val = self.cam_device_var.get()
        match = re.search(r"Camera\s+(\d+)", val)
        if match:
            idx = int(match.group(1))
            self._vision_settings["camera_index"] = idx
            self.vision.camera_index = idx
            try:
                self._save_vision_settings()
            except Exception:
                pass
            if self._camera_is_active:
                self._stop_camera()
                self.after(300, self._toggle_camera)

    def _on_3d_model_generated(self, asset):
        """Callback from integrations when a 3D model is created."""
        def _update():
            self._select_studio_tab("3d")
            if hasattr(self, "three_d_viewport"):
                self.three_d_viewport.load_asset(asset)
        self.after(0, _update)

    def _on_3d_viewport_command(self, action: str):
        """Callback from integrations for 3D navigation commands."""
        def _exec():
            self._select_studio_tab("3d")
            if not hasattr(self, "three_d_viewport"):
                return
            if action == "rotate":
                self.three_d_viewport.rotate_relative(30.0)
            elif action == "zoom_in":
                self.three_d_viewport.zoom_in()
            elif action == "zoom_out":
                self.three_d_viewport.zoom_out()
            elif action == "reset":
                self.three_d_viewport.reset_view()
            elif action == "open_blender":
                self.three_d_viewport._open_in_blender()
            elif action == "export":
                self.three_d_viewport._export_dialog()
        self.after(0, _exec)

    def _stop_camera(self):
        self._camera_is_active = False
        self.btn_camera.config(text="Stopping…", state=tk.DISABLED)
        if hasattr(self, "btn_camera_studio_toggle"):
            self.btn_camera_studio_toggle.config(text="Stopping…", state=tk.DISABLED)
        self.camera_status_label.config(text="Stopping camera…", fg=THEME["accent_amber"])
        self.vision.stop()

    def _handle_vision_status(self, state: str, message: str):
        if not hasattr(self, "camera_status_label"):
            return
        if state == "CAMERA_STARTING":
            self.camera_status_label.config(text="Camera Starting", fg=THEME["accent_amber"])
            self.btn_camera.config(text="Starting…", state=tk.DISABLED)
            if hasattr(self, "btn_camera_studio_toggle"):
                self.btn_camera_studio_toggle.config(text="Starting…", state=tk.DISABLED)
        elif state == "CAMERA_ACTIVE":
            self._camera_is_active = True
            self.camera_status_label.config(text=message, fg=THEME["accent_green"])
            self.btn_camera.config(text="Stop Camera", state=tk.NORMAL)
            if hasattr(self, "btn_camera_studio_toggle"):
                self.btn_camera_studio_toggle.config(text="Stop Camera", state=tk.NORMAL, bg="#7F1D1D")
        elif state == "CAMERA_ERROR":
            self._camera_is_active = False
            self.camera_status_label.config(text=f"Camera Error: {message}", fg=THEME["accent_red"])
            self.camera_preview_label.config(text="Camera unavailable", image="")
            self.btn_camera.config(text="Start Camera", state=tk.NORMAL)
            if hasattr(self, "btn_camera_studio_toggle"):
                self.btn_camera_studio_toggle.config(text="Start Camera", state=tk.NORMAL, bg="#0284C7")
            self.camera_detections_label.config(text="No current detections.")
            self._add_message_card("ADAM", f"Camera Error: {message}")
            self.voice_tts.speak(f"Camera error. {message}")
        else:
            self._camera_is_active = False
            self.camera_status_label.config(text="Camera Stopped", fg=THEME["text_secondary"])
            self.camera_preview_label.config(text="Camera preview is off.\nClick 'Start Camera' or say 'Turn on camera'.", image="")
            self.camera_detections_label.config(text="No current detections.")
            self.btn_camera.config(text="Start Camera", state=tk.NORMAL)
            if hasattr(self, "btn_camera_studio_toggle"):
                self.btn_camera_studio_toggle.config(text="Start Camera", state=tk.NORMAL, bg="#0284C7")
        self._update_system_metrics(self._system_metrics)

    def _show_camera_frame(self, frame_rgb, detections):
        if not self._camera_is_active:
            return

        import cv2

        raw_h, raw_w = frame_rgb.shape[:2]
        avail_w = self.camera_preview_label.winfo_width()
        avail_h = self.camera_preview_label.winfo_height()

        if avail_w > 40 and avail_h > 40:
            scale = min(avail_w / raw_w, avail_h / raw_h)
            target_w = max(160, int(raw_w * scale))
            target_h = max(120, int(raw_h * scale))
            if abs(target_w - raw_w) > 4 or abs(target_h - raw_h) > 4:
                frame_to_show = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            else:
                frame_to_show = frame_rgb
        else:
            frame_to_show = frame_rgb

        height, width = frame_to_show.shape[:2]
        ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + frame_to_show.tobytes()
        self._camera_photo = tk.PhotoImage(
            data=ppm.decode("latin-1"),
            format="PPM",
        )
        self.camera_preview_label.config(image=self._camera_photo, text="")

        # Update telemetry
        fps = getattr(self.vision, "capture_fps", 0.0)
        infer_ms = getattr(self.vision, "inference_latency_ms", 0.0)
        lat_ms = getattr(self.vision, "total_pipeline_latency_ms", 0.0)
        if hasattr(self, "lbl_camera_telemetry"):
            self.lbl_camera_telemetry.config(
                text=f"⚡ {fps:.1f} FPS | Infer: {infer_ms:.0f}ms | Latency: {lat_ms:.0f}ms"
            )

        if detections:
            lines = [
                f"{item.label}: {item.confidence:.0%}"
                for item in detections[:8]
            ]
            self.camera_detections_label.config(text="  ·  ".join(lines))
            self.camera_status_label.config(
                text=f"Detection Active · {len(detections)} object(s) detected",
                fg=THEME["accent_green"],
            )
        else:
            self.camera_detections_label.config(text="No current detections.")
            self.camera_status_label.config(
                text="Detection Active · Clear view",
                fg=THEME["accent_cyan"],
            )

    @staticmethod
    def _vision_command_intent(command: str):
        cleaned = re.sub(r"[.!?]+$", "", command.strip().lower())
        if re.search(
            r"(?:turn on|start|enable|open)\s+(?:the\s+)?(?:object detection(?: mode)?|camera)",
            cleaned,
        ):
            return "start"
        if re.search(
            r"(?:stop|turn off|disable|close)\s+(?:the\s+)?(?:object detection(?: mode)?|camera)",
            cleaned,
        ):
            return "stop"
        if re.search(
            r"(?:what can you see|what do you see|what is in the camera|what's in the camera|what is in front of me|what's in front of me|what is in front of the camera|describe what you see|describe the camera view|tell me what you see)",
            cleaned,
        ):
            return "describe"
        if re.search(
            r"(?:who is in the camera|who's in the camera|who is in front of the camera|who do you see|who can you see|who is that|who is here|is there a person in front of me|is there a person in front of the camera|is there a person|is anyone in front of me|is anyone there|can you see me|do you see me)",
            cleaned,
        ):
            return "who"
        if re.search(
            r"(?:tell me about this phone|what phone is this|inspect the phone|inspect this phone|inspect the object|inspect this object|identify the object|what object am i holding|what am i holding|what is in my hand|what am i showing you)",
            cleaned,
        ):
            return "inspect_phone" if "phone" in cleaned else "inspect_object"
        return None

    def _handle_vision_command(self, command: str) -> Optional[str]:
        intent = self._vision_command_intent(command)
        if intent == "start":
            self._select_studio_tab("vision")
            if self.vision.status == "CAMERA_ACTIVE":
                return "Camera and object detection are already active."
            if self.vision.status == "CAMERA_STARTING":
                return "Camera Starting. The camera is opening."
            if not self._toggle_camera():
                return "Camera access was not granted. Object detection was not started."
            return "Camera Starting. Object detection will run locally."
        if intent == "stop":
            if self.vision.status == "CAMERA_STOPPED":
                return "Camera is already stopped."
            self._stop_camera()
            return "Stopping object detection and releasing the camera."
        if intent == "describe":
            if self.vision.status != "CAMERA_ACTIVE":
                return "The camera is currently stopped. Click 'Start Camera' or ask me to turn on the camera to see what is in view."
            return self.vision.describe_latest()
        if intent == "who":
            if self.vision.status != "CAMERA_ACTIVE":
                return "The camera is currently stopped. Click 'Start Camera' or ask me to turn on the camera to see who is in view."
            return self.vision.describe_person()
        if intent in ("inspect_phone", "inspect_object"):
            if self.vision.status != "CAMERA_ACTIVE":
                return "The camera is currently stopped. Please start the camera to inspect objects."
            target = "cell phone" if intent == "inspect_phone" else None
            return self.vision.inspect_object(target)["response"]
        return None

    def _open_vision_settings(self):
        panel = tk.Toplevel(self)
        panel.title("ADAM — Camera Settings")
        panel.geometry("440x300")
        panel.configure(bg=THEME["bg_card"])
        panel.transient(self)

        tk.Label(
            panel,
            text="Local Object Detection",
            font=(THEME["font_main"], 13, "bold"),
            fg=THEME["accent_cyan"],
            bg=THEME["bg_card"],
        ).pack(anchor="w", padx=18, pady=(18, 10))
        tk.Label(
            panel,
            text=(
                "Model: YOLOX-s COCO (CPU, local ONNX)\n"
                "Classes include person, phone, laptop, keyboard, mouse, bottle, cup and book.\n"
                "Glasses are not a COCO class and will not be identified."
            ),
            font=(THEME["font_main"], 9),
            fg=THEME["text_primary"],
            bg=THEME["bg_card"],
            justify=tk.LEFT,
            wraplength=400,
        ).pack(anchor="w", padx=18, pady=6)

        device_frame = tk.Frame(panel, bg=THEME["bg_card"])
        device_frame.pack(fill=tk.X, padx=18, pady=8)
        tk.Label(
            device_frame,
            text="Camera device index (restart detection to apply):",
            font=(THEME["font_main"], 9),
            fg=THEME["text_primary"],
            bg=THEME["bg_card"],
        ).pack(anchor="w")
        device_var = tk.IntVar(value=self._vision_settings["camera_index"])
        tk.Spinbox(device_frame, from_=0, to=8, textvariable=device_var, width=6).pack(anchor="w", pady=4)
        confidence_var = tk.DoubleVar(value=self._vision_settings["confidence_threshold"])
        tk.Label(
            panel,
            text="Detection confidence threshold",
            font=(THEME["font_main"], 9),
            fg=THEME["text_primary"],
            bg=THEME["bg_card"],
        ).pack(anchor="w", padx=18, pady=(4, 0))
        tk.Scale(
            panel,
            from_=0.2,
            to=0.8,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=confidence_var,
            bg=THEME["bg_card"],
            fg=THEME["text_primary"],
            highlightthickness=0,
        ).pack(fill=tk.X, padx=18)

        def save_settings():
            self._vision_settings["camera_index"] = int(device_var.get())
            self._vision_settings["confidence_threshold"] = float(confidence_var.get())
            try:
                self._save_vision_settings()
            except OSError as exc:
                messagebox.showerror("Camera settings", f"Could not save settings: {exc}", parent=panel)
                return
            self.vision.camera_index = self._vision_settings["camera_index"]
            self.vision.confidence_threshold = self._vision_settings["confidence_threshold"]
            panel.destroy()

        tk.Button(
            panel,
            text="Save Camera Settings",
            command=save_settings,
            bg="#0369A1",
            fg="#FFFFFF",
            bd=0,
            padx=12,
            pady=7,
        ).pack(anchor="e", padx=18, pady=12)

    def _shutdown(self):
        self.system_monitor.stop()
        self.vision.stop(wait=2.0)
        self.speech_stt.stop()
        self.voice_tts.stop()
        self.destroy()

    # =========================================================================
    # USER SUBMISSION & ORCHESTRATION
    # =========================================================================

    def _handle_user_submit(self):
        """Process user text or speech command input."""
        text = self.input_entry.get().strip()
        if not text:
            return

        self.input_entry.delete(0, tk.END)
        self._submit_command(text, "text", uuid.uuid4().hex)

    @staticmethod
    def _task_title_from_command(command: str) -> Optional[str]:
        """Extract only explicit task-creation requests from user input."""
        match = re.match(
            r"^(?:(?:please|can you|could you|i want to|i need to)\s+)?"
            r"(?:assign|create|add)\s+(?:me\s+)?(?:a\s+)?task"
            r"(?:\s+(?:called|named|to)\s+|\s*:\s*|\s+)?(.*?)\s*[.!?]*$",
            command.strip(),
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        title = match.group(1).strip(" \t\r\n.!?")
        return title

    def _submit_command(self, command: str, source: str, request_id: Optional[str] = None):
        """Submit one typed or recognized utterance and suppress duplicate request IDs."""
        command = str(command).strip()
        if not command:
            return
        request_id = request_id or uuid.uuid4().hex

        with self._request_lock:
            if request_id in self._seen_request_ids:
                logger.info("Ignoring duplicate command request [%s].", request_id)
                return
            self._seen_request_ids.add(request_id)
            self._request_id_order.append(request_id)
            if len(self._request_id_order) > 512:
                expired_id = self._request_id_order.pop(0)
                self._seen_request_ids.discard(expired_id)

            previous_cancel = self._active_request_cancel
            cancel_event = threading.Event()
            self._active_request_id = request_id
            self._active_request_cancel = cancel_event

        if previous_cancel is not None:
            previous_cancel.set()
        if self.voice_tts.is_speaking():
            self.voice_tts.stop()
        self._add_message_card("USER", command, source=source)
        self.input_entry.delete(0, tk.END)
        if source == "voice":
            self.input_entry.insert(0, command)
        self.set_voice_state("PROCESSING", "Thinking... processing your request")

        vision_response = self._handle_vision_command(command)
        if vision_response is not None:
            self.ui_queue.put(("DIRECT_RESPONSE", (vision_response, None, request_id)))
            return

        task_title = self._task_title_from_command(command)
        if task_title is not None:
            if not task_title:
                response = "I couldn't save the task: Please provide a task title."
            else:
                try:
                    self.local_store.add_task(task_title, request_id=request_id)
                except Exception as exc:
                    logger.exception("Could not save task for request [%s].", request_id)
                    response = f"I couldn't save the task: {exc}"
                else:
                    logger.info("Saved task for request [%s]: %s", request_id, task_title)
                    response = f"Task has been assigned: {task_title}."
                    self.ui_queue.put(("TASKS_REFRESH", None))
            self.ui_queue.put(("DIRECT_RESPONSE", (response, None, request_id)))
            return

        cmd_lower = command.lower()
        is_3d_create = any(kw in cmd_lower for kw in ["3d", "blender", "model", "create", "generate", "make", "build"]) and any(
            sh in cmd_lower for sh in ["scorpio", "suv", "bag", "backpack", "bottle", "chair", "mug", "cup", "car", "robot", "table", "cube", "box", "sphere", "cylinder", "torus", "pedestal", "drone"]
        )
        if is_3d_create:
            self.set_voice_state("PROCESSING", "Generating 3D model in Blender...")
            def worker_3d():
                try:
                    handled, response, pending_action = self.integrations.try_handle_command(command)
                    if handled:
                        self.ui_queue.put(("DIRECT_RESPONSE", (response or "", pending_action, request_id)))
                        return
                except Exception as exc:
                    logger.exception("3D generation worker error.")
                    self.ui_queue.put(("DIRECT_RESPONSE", (f"3D generation error: {exc}", None, request_id)))
                    return
            threading.Thread(target=worker_3d, daemon=True).start()
            return

        try:
            handled, response, pending_action = self.integrations.try_handle_command(command)
        except Exception as exc:
            logger.exception("Direct command execution failed.")
            self.ui_queue.put(("DIRECT_RESPONSE", (f"Command failed: {exc}", None, request_id)))
            return

        if handled:
            self.ui_queue.put(("DIRECT_RESPONSE", (response or "", pending_action, request_id)))
            return

        short_preview = (command[:24] + "...") if len(command) > 24 else command
        self.set_voice_state("PROCESSING", f"Thinking... {short_preview}")
        response_card = self._add_message_card("ADAM", "")

        def worker_ollama():
            response_parts = []
            request_deadline = time.monotonic() + 60.0
            try:
                for chunk in self.integrations.ollama.generate_stream(
                    command,
                    cancel_event=cancel_event,
                    timeout=60.0,
                    max_chars=12000,
                ):
                    if cancel_event.is_set():
                        return
                    if time.monotonic() >= request_deadline:
                        break
                    response_parts.append(chunk)
                    self.ui_queue.put(("STREAM_CHUNK", (request_id, response_card, chunk)))
                if cancel_event.is_set():
                    return
                if time.monotonic() >= request_deadline:
                    timeout_message = "\nRequest timed out. Please try again."
                    response_parts.append(timeout_message)
                    self.ui_queue.put(("STREAM_CHUNK", (request_id, response_card, timeout_message)))
                self.ui_queue.put(("STREAM_DONE", (request_id, "".join(response_parts))))
            except Exception as exc:
                logger.exception("Ollama request [%s] failed.", request_id)
                if not cancel_event.is_set():
                    error_text = f"Request failed: {exc}"
                    self.ui_queue.put(("STREAM_CHUNK", (request_id, response_card, error_text)))
                    self.ui_queue.put(("STREAM_DONE", (request_id, "".join(response_parts) + error_text)))

        threading.Thread(target=worker_ollama, daemon=True).start()

    def _cancel_active_request(self):
        """Cancel an active model stream and stop queued/current TTS output."""
        with self._request_lock:
            cancel_event = self._active_request_cancel
            self._active_request_cancel = None
            self._active_request_id = None
        if cancel_event is not None:
            cancel_event.set()
        self.voice_tts.stop()
        self.set_voice_state("READY", "Request cancelled.")

    def _update_mic_buttons_ui(self, active: Optional[bool] = None, status_str: str = "", flow_state: str = ""):
        """Compatibility wrapper: delegates to set_voice_state."""
        if flow_state:
            self.set_voice_state(flow_state, status_str)
        elif active is not None:
            self.set_voice_state("LISTENING_WAKE" if active else "MIC_OFF", status_str)

    def _toggle_microphone(self):
        """Toggle microphone capture and wake-word listening with immediate responsive feedback."""
        if self._mic_toggle_in_progress:
            return

        if not self.speech_stt.mic_available:
            self.set_voice_state("MIC_ERROR", "No microphone detected on this system.")
            messagebox.showinfo(
                "Microphone",
                "No microphone detected on this system. You can open 'Audio Setup' in the header to diagnose and select input devices."
            )
            return

        self._mic_toggle_in_progress = True

        # Check current running state
        is_running = self.speech_stt._capture_requested and not self.speech_stt._paused

        if is_running:
            # Transition to MIC_OFF immediately on UI
            self.set_voice_state("MIC_OFF", "Audio capture is stopped.")

            def worker_stop():
                try:
                    self.speech_stt.pause()
                finally:
                    self.ui_queue.put(("SET_VOICE_STATE", ("MIC_OFF", "Audio capture is stopped. Click MIC button to activate.")))
                    self.ui_queue.put(("POST_CARD", ("ADAM", "🔇 Microphone is **OFF**. Wake-word listening and background recording stopped completely. (Typed chat remains active)")))
                    self._mic_toggle_in_progress = False

            threading.Thread(target=worker_stop, daemon=True).start()
        else:
            # Transition to MIC_STARTING immediately on UI
            self.set_voice_state("MIC_STARTING", "Initializing audio capture subsystem...")

            def worker_start():
                try:
                    self.speech_stt.resume()
                except Exception as e:
                    self.ui_queue.put(("SET_VOICE_STATE", ("MIC_ERROR", f"Failed to start microphone: {e}")))
                finally:
                    self._mic_toggle_in_progress = False

            threading.Thread(target=worker_start, daemon=True).start()

    def _on_push_to_talk(self):
        """Single phrase listen (Push-to-talk)."""
        if not self.speech_stt.mic_available:
            messagebox.showinfo("Microphone", "No microphone detected on this system.")
            return

        self.input_entry.delete(0, tk.END)
        self.set_voice_state("MIC_STARTING", "Microphone Activating")

        def worker():
            try:
                phrase = self.speech_stt.listen_once()
            except Exception as exc:
                logger.exception("Listen Once failed.")
                self.ui_queue.put(("COMMAND_FAILED", f"Microphone Error: {exc}"))
                return
            if phrase:
                self.ui_queue.put(("COMMAND_RECOGNIZED", (phrase, "voice", uuid.uuid4().hex)))

        threading.Thread(target=worker, daemon=True).start()

    def _clear_chat(self):
        """Clear the visible and persisted conversation history."""
        if not messagebox.askyesno("Clear history", "Permanently clear saved conversation history?"):
            return
        for widget in self.chat_scrollable_frame.winfo_children():
            widget.destroy()
        try:
            self.local_store.clear_history()
        except Exception:
            logger.exception("Could not clear saved conversation history.")
            messagebox.showerror("History", "Could not clear saved conversation history. See the ADAM log for details.")
        self._post_welcome_message()

    def _open_tasks_panel(self):
        """Open the independent, locally persisted task list."""
        panel = tk.Toplevel(self)
        panel.title("ADAM Tasks")
        panel.geometry("440x420")
        panel.configure(bg=THEME["bg_card"])
        panel.transient(self)

        tk.Label(
            panel,
            text="Tasks",
            font=(THEME["font_main"], 16, "bold"),
            fg=THEME["text_primary"],
            bg=THEME["bg_card"],
        ).pack(anchor="w", padx=16, pady=(14, 8))

        entry = tk.Entry(
            panel,
            font=(THEME["font_main"], 10),
            bg=THEME["bg_input"],
            fg=THEME["text_primary"],
            insertbackground=THEME["text_primary"],
        )
        entry.pack(fill=tk.X, padx=16, pady=(0, 8))

        task_list = tk.Listbox(
            panel,
            font=(THEME["font_main"], 10),
            bg=THEME["bg_main"],
            fg=THEME["text_primary"],
            selectbackground=THEME["accent_cyan"],
            activestyle="none",
        )
        task_list.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 8))
        visible_tasks = []

        def refresh():
            try:
                tasks = self.local_store.list_tasks()
            except Exception as exc:
                logger.exception("Could not load persisted tasks.")
                messagebox.showerror("Tasks", f"Could not load tasks: {exc}", parent=panel)
                return
            visible_tasks[:] = tasks
            task_list.delete(0, tk.END)
            for task in visible_tasks:
                marker = "✓" if task["completed"] else "○"
                task_list.insert(tk.END, f"{marker}  {task['title']}")

        self._tasks_panel_refresh = refresh

        def clear_refresh_callback(event):
            if event.widget is panel and self._tasks_panel_refresh is refresh:
                self._tasks_panel_refresh = None

        panel.bind("<Destroy>", clear_refresh_callback)

        def add_task():
            title = entry.get().strip()
            if not title:
                return
            try:
                self.local_store.add_task(title)
            except Exception as exc:
                messagebox.showerror("Tasks", f"Could not save task: {exc}", parent=panel)
                return
            entry.delete(0, tk.END)
            refresh()

        def selected_task():
            selection = task_list.curselection()
            return visible_tasks[selection[0]] if selection else None

        def toggle_task():
            task = selected_task()
            if task:
                try:
                    self.local_store.set_task_completed(int(task["id"]), not bool(task["completed"]))
                except Exception as exc:
                    messagebox.showerror("Tasks", f"Could not update task: {exc}", parent=panel)
                    return
                refresh()

        def delete_task():
            task = selected_task()
            if task:
                try:
                    self.local_store.delete_task(int(task["id"]))
                except Exception as exc:
                    messagebox.showerror("Tasks", f"Could not delete task: {exc}", parent=panel)
                    return
                refresh()

        actions = tk.Frame(panel, bg=THEME["bg_card"])
        actions.pack(fill=tk.X, padx=16, pady=(0, 14))
        for label, callback in (
            ("Add", add_task),
            ("Complete / Reopen", toggle_task),
            ("Delete", delete_task),
        ):
            tk.Button(
                actions,
                text=label,
                command=callback,
                bg=THEME["bg_header"],
                fg=THEME["text_primary"],
                bd=0,
                padx=8,
                pady=6,
            ).pack(side=tk.LEFT, padx=(0, 6))
        entry.bind("<Return>", lambda _event: add_task())
        refresh()

    def _open_gmail_connection(self):
        """Connect Gmail using Google's OAuth authorization-code callback flow."""
        client = self.integrations.gmail_client
        status = client.get_auth_status()
        panel = tk.Toplevel(self)
        panel.title("Connect Gmail")
        panel.geometry("560x300")
        panel.configure(bg=THEME["bg_card"])
        panel.transient(self)

        status_label = tk.Label(
            panel,
            text=status["status_message"],
            font=(THEME["font_main"], 10),
            fg=THEME["text_primary"],
            bg=THEME["bg_card"],
            justify=tk.LEFT,
            wraplength=510,
        )
        status_label.pack(anchor="w", padx=18, pady=(18, 12))

        if status["authorized"]:
            tk.Button(
                panel,
                text="Disconnect Gmail",
                command=lambda: self._disconnect_gmail(panel, status_label),
                bg="#7F1D1D",
                fg="#FFFFFF",
                bd=0,
                padx=12,
                pady=7,
            ).pack(anchor="w", padx=18)
            return

        credential_file = client.client_secret_file
        if not credential_file.is_file():
            setup = (
                "OAuth setup is required before Gmail can connect:\n"
                "1. In Google Cloud Console, create a project and enable the Gmail API.\n"
                "2. Configure the OAuth consent screen and create an OAuth client ID for a Desktop app.\n"
                "3. Download the OAuth client JSON to:\n"
                f"   {credential_file}\n"
                "4. Ensure Fedora's Secret Service keyring is unlocked and the Python 'keyring' package is installed.\n"
                "ADAM requests Gmail read-only access only. It never asks for your Google password."
            )
            status_label.config(text=setup, anchor="w")
            tk.Button(
                panel,
                text="Open Google Cloud Console",
                command=lambda: webbrowser.open("https://console.cloud.google.com/apis/credentials"),
                bg=THEME["bg_header"],
                fg=THEME["text_primary"],
                bd=0,
                padx=12,
                pady=7,
            ).pack(anchor="w", padx=18)
            return

        try:
            with credential_file.open("r", encoding="utf-8") as stream:
                credentials = json.load(stream)
            oauth_config = credentials.get("installed") or credentials.get("web") or {}
            client_id = oauth_config["client_id"]
            client_secret = oauth_config["client_secret"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            status_label.config(
                text=f"OAuth client file is invalid ({exc}). Download a Desktop app client JSON from Google Cloud Console."
            )
            return

        redirect_uri = "http://localhost:8080/"
        expected_state = secrets.token_urlsafe(32)
        auth_url = client.generate_auth_url(
            client_id,
            client_secret,
            redirect_uri=redirect_uri,
            state=expected_state,
        )
        completion = threading.Event()
        app = self

        class OAuthCallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                received_state = query.get("state", [""])[0]
                code = query.get("code", [""])[0]
                oauth_error = query.get("error", [""])[0]
                if not secrets.compare_digest(received_state, expected_state):
                    result = (False, "OAuth callback state did not match. Please retry.")
                    self.send_response(400)
                elif oauth_error:
                    result = (False, f"Google authorization failed: {oauth_error}")
                    self.send_response(400)
                elif not code:
                    result = (False, "Google returned no authorization code.")
                    self.send_response(400)
                else:
                    result = client.exchange_code_for_tokens(code, redirect_uri=redirect_uri)
                    self.send_response(200 if result[0] else 400)

                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    ("ADAM Gmail authorization succeeded. You may close this page."
                     if result[0] else result[1]).encode("utf-8")
                )
                completion.set()
                app.after(0, lambda: app._finish_gmail_connection(panel, status_label, result))

            def log_message(self, _format, *_args):
                return

        try:
            server = HTTPServer(("127.0.0.1", 8080), OAuthCallbackHandler)
            server.timeout = 180
        except OSError as exc:
            status_label.config(text=f"Could not start the local OAuth callback on port 8080: {exc}")
            return

        status_label.config(text="Waiting for Google OAuth consent in your browser...")

        def serve_callback():
            deadline = time.monotonic() + 180
            try:
                while not completion.is_set() and time.monotonic() < deadline:
                    server.handle_request()
            finally:
                server.server_close()
            if not completion.is_set():
                app.after(
                    0,
                    lambda: app._finish_gmail_connection(
                        panel,
                        status_label,
                        (False, "Google authorization timed out. Click Connect Gmail to try again."),
                    ),
                )

        threading.Thread(target=serve_callback, daemon=True).start()
        if not webbrowser.open(auth_url):
            status_label.config(text=f"Could not open a browser. Open this URL manually:\n{auth_url}")

    def _finish_gmail_connection(self, panel, status_label, result):
        if not panel.winfo_exists():
            return
        success, message = result
        status = self.integrations.gmail_client.get_auth_status()
        status_label.config(text=status["status_message"] if success else message)
        if success and hasattr(self, "btn_gmail"):
            self.btn_gmail.config(text="Gmail Connected")
        elif not success:
            self.btn_gmail.config(text="Connect Gmail")

    def _disconnect_gmail(self, panel, status_label):
        try:
            self.integrations.gmail_client.disconnect()
        except Exception as exc:
            status_label.config(text=f"Could not disconnect Gmail: {exc}")
            return
        status_label.config(text="Gmail is disconnected.")
        self.btn_gmail.config(text="Connect Gmail")

    # =========================================================================
    # MODALS: MICROPHONE SETUP, 3D LAB & MODEL SELECTOR
    # =========================================================================

    def _open_microphone_modal(self):
        """Audio Setup, Speech Recognition Diagnostics & Configurable Wake Words."""
        modal = tk.Toplevel(self)
        modal.title("ADAM — Audio, Speech Recognition & Wake Words")
        modal.geometry("680x820")
        modal.configure(bg=THEME["bg_card"])
        modal.transient(self)
        modal.grab_set()

        hdr = tk.Label(
            modal,
            text="🎙️ Audio Setup, Faster-Whisper STT & Wake Words",
            font=(THEME["font_main"], 12, "bold"),
            fg=THEME["accent_cyan"],
            bg=THEME["bg_card"],
            pady=10
        )
        hdr.pack(fill=tk.X)

        container = tk.Frame(modal, bg=THEME["bg_card"])
        container.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 6))

        canvas = tk.Canvas(container, bg=THEME["bg_card"], bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        form = tk.Frame(canvas, bg=THEME["bg_card"], padx=10)
        win_id = canvas.create_window((0, 0), window=form, anchor="nw")

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        form.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))

        # 1. System Subsystem Diagnostics Box
        diag_box = tk.LabelFrame(
            form,
            text="Audio Subsystem Diagnostics (Fedora / PipeWire / ALSA)",
            font=(THEME["font_main"], 9, "bold"),
            bg="#0F172A",
            fg=THEME["accent_cyan"],
            padx=12,
            pady=8
        )
        diag_box.pack(fill=tk.X, pady=(0, 10))

        diag_info = diagnose_microphone_system()
        status_color = THEME["accent_green"] if diag_info["device_count"] > 0 else THEME["accent_amber"]

        diag_lbl = tk.Label(
            diag_box,
            text=f"• Status: {diag_info['diagnosis_message']}\n"
                 f"• PyAudio Driver: {'Installed & Ready' if diag_info['pyaudio_available'] else 'Not Found'}\n"
                 f"• sounddevice: {'Available' if diag_info['sounddevice_available'] else 'Not Found'}\n"
                 f"• Current Active Device: {self.speech_stt.device_name}",
            font=(THEME["font_mono"], 9),
            fg=status_color,
            bg="#0F172A",
            justify=tk.LEFT
        )
        diag_lbl.pack(anchor="w")

        # Real-time Voice Pipeline Telemetry (Diagnostic Mode)
        telemetry_box = tk.LabelFrame(
            form,
            text="Live Voice Pipeline Telemetry (Diagnostic Mode)",
            font=(THEME["font_main"], 9, "bold"),
            bg="#0B132B",
            fg="#38BDF8",
            padx=12,
            pady=8
        )
        telemetry_box.pack(fill=tk.X, pady=(0, 10))

        telemetry_lbl = tk.Label(
            telemetry_box,
            text="Initializing live telemetry...",
            font=(THEME["font_mono"], 8),
            fg="#94A3B8",
            bg="#0B132B",
            justify=tk.LEFT
        )
        telemetry_lbl.pack(anchor="w")

        def update_telemetry_loop():
            if not modal.winfo_exists():
                return
            snap = self.speech_stt.get_diagnostics_snapshot()
            silent_str = "🔇 SILENT / AMBIENT" if snap.get("is_silent", True) else "🟢 ACTIVE SPEECH"
            wake_act = "Active" if snap.get("wake_detector_active") else "Standby"
            cmd_rec = "Active" if snap.get("command_recording_started") and not snap.get("command_recording_ended") else (
                "Completed" if snap.get("command_recording_ended") else "Idle"
            )
            utt_id = snap.get("current_utterance_id") or "None (Waiting for speech)"
            last_utt = snap.get("last_processed_utterance_id") or "None"
            t_res = snap.get("last_transcription_result") or "N/A"
            t_dur = snap.get("last_transcription_duration_sec", 0.0)
            count = snap.get("processed_utterance_count", 0)

            telemetry_text = (
                f"• Audio Device: [{snap.get('device_index')}] {snap.get('selected_device')} ({snap.get('sample_rate')}Hz, {snap.get('channels')}ch)\n"
                f"• Live Signal: RMS {snap.get('live_rms', 0.0):.1f} | Peak: {snap.get('live_peak', 0.0)*100:.1f}% | State: {silent_str}\n"
                f"• Wake Detector: {wake_act} | Last Result: {snap.get('last_wake_result', 'None')}\n"
                f"• Command Session: {cmd_rec} (Duration: {snap.get('command_duration_sec', 0.0):.1f}s)\n"
                f"• Faster-Whisper: {snap.get('whisper_model', 'base')} | Last Transcribed: \"{t_res}\" ({t_dur:.2f}s)\n"
                f"• Active Utterance ID: {utt_id}\n"
                f"• Verification: Processed {count} unique utterance(s) [Last: {last_utt}] (Never duplicated)"
            )
            telemetry_lbl.config(text=telemetry_text)
            modal.after(120, update_telemetry_loop)

        modal.after(100, update_telemetry_loop)

        # 2. Detected Input Microphones
        tk.Label(form, text="Detected Audio Input Devices:", font=(THEME["font_main"], 10, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(anchor="w", pady=(2, 2))

        mics = list_input_microphones()
        options_map = {}
        for m in mics:
            backend_str = f" [{m.get('backend', 'pyaudio')}]"
            label = f"[Index {m['index']}] {m['name']} ({m['channels']} ch, {m['sample_rate']}Hz){backend_str}"
            options_map[label] = m['index']

        current_label = ""
        for lbl, idx in options_map.items():
            if idx == self.speech_stt.device_index:
                current_label = lbl
                break
        if not current_label and options_map:
            current_label = list(options_map.keys())[0]

        mic_var = tk.StringVar(value=current_label if current_label else "No input microphones detected")
        combo = ttk.Combobox(form, values=list(options_map.keys()), textvariable=mic_var, state="readonly" if options_map else "disabled", font=(THEME["font_main"], 9))
        combo.pack(fill=tk.X, pady=(0, 10))

        # 3. Test Signal Meter Section
        test_frame = tk.Frame(form, bg="#1E293B", padx=12, pady=10, highlightthickness=1, highlightbackground=THEME["border_glass"])
        test_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(test_frame, text="Live Signal Meter Test (Speak for 2 seconds):", font=(THEME["font_main"], 9, "bold"), fg=THEME["accent_cyan"], bg="#1E293B").pack(anchor="w")

        meter_canvas = tk.Canvas(test_frame, height=18, bg="#0F172A", highlightthickness=1, highlightbackground=THEME["border_glass"])
        meter_canvas.pack(fill=tk.X, pady=6)

        status_lbl = tk.Label(test_frame, text="Click 'Test Signal' to measure live audio peak energy.", font=(THEME["font_main"], 9), fg=THEME["text_secondary"], bg="#1E293B")
        status_lbl.pack(anchor="w")

        def run_test():
            sel_label = mic_var.get()
            chosen_index = options_map.get(sel_label, None)
            meter_width = meter_canvas.winfo_width()
            status_lbl.config(text="Listening and measuring RMS audio level... Speak now!", fg=THEME["accent_amber"])
            modal.update()

            def worker_test():
                ok, peak, msg = test_microphone_level(chosen_index, duration_sec=2.0)
                fill_w = int(meter_width * min(1.0, peak))
                meter_color = THEME["accent_green"] if peak > 0.05 else THEME["accent_amber"]
                message = f"✔ {msg}" if ok else f"✖ {msg}"
                color = THEME["accent_green"] if ok else THEME["accent_red"]

                def update_result():
                    if not modal.winfo_exists():
                        return
                    meter_canvas.delete("all")
                    meter_canvas.create_rectangle(0, 0, fill_w, 18, fill=meter_color, outline="")
                    status_lbl.config(text=message, fg=color)

                modal.after(0, update_result)

            threading.Thread(target=worker_test, daemon=True).start()

        btn_test = tk.Button(
            test_frame,
            text="▶ Test Microphone Signal",
            font=(THEME["font_main"], 9, "bold"),
            bg="#0369A1",
            fg="#FFFFFF",
            bd=0,
            padx=12,
            pady=4,
            cursor="hand2",
            command=run_test
        )
        btn_test.pack(anchor="w", pady=(2, 0))

        # 4. Speech Recognition Settings (Faster-Whisper Models & Real Audio Test)
        stt_box = tk.LabelFrame(
            form,
            text="Speech Recognition Engine (Faster-Whisper on CPU int8)",
            font=(THEME["font_main"], 9, "bold"),
            bg="#0F172A",
            fg=THEME["accent_cyan"],
            padx=12,
            pady=8
        )
        stt_box.pack(fill=tk.X, pady=(0, 10))

        whisper_model_var = tk.StringVar(value=self.speech_stt.whisper_model_name)
        language_values = {"English": "en"}
        language_var = tk.StringVar(value="English")

        model_opts_frame = tk.Frame(stt_box, bg="#0F172A")
        model_opts_frame.pack(fill=tk.X, pady=(0, 4))
        language_row = tk.Frame(model_opts_frame, bg="#0F172A")
        language_row.pack(anchor="w", pady=(2, 5))
        tk.Label(
            language_row,
            text="Language:",
            font=(THEME["font_main"], 9),
            fg=THEME["text_primary"],
            bg="#0F172A",
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Combobox(
            language_row,
            textvariable=language_var,
            values=list(language_values),
            state="readonly",
            width=22,
        ).pack(side=tk.LEFT)

        rb_base = tk.Radiobutton(
            model_opts_frame,
            text="Faster-Whisper 'base' model (~140MB, Fast, Optimal for 8GB RAM)",
            variable=whisper_model_var,
            value="base",
            font=(THEME["font_main"], 9),
            fg=THEME["text_primary"],
            bg="#0F172A",
            selectcolor="#1E293B",
            activebackground="#0F172A"
        )
        rb_base.pack(anchor="w")

        rb_small = tk.Radiobutton(
            model_opts_frame,
            text="Faster-Whisper 'small' model (~460MB, Higher accuracy)",
            variable=whisper_model_var,
            value="small",
            font=(THEME["font_main"], 9),
            fg=THEME["text_primary"],
            bg="#0F172A",
            selectcolor="#1E293B",
            activebackground="#0F172A"
        )
        rb_small.pack(anchor="w")

        # Live Speech Recognition Test Box
        stt_test_frame = tk.Frame(stt_box, bg="#1E293B", padx=10, pady=8, highlightthickness=1, highlightbackground=THEME["border_glass"])
        stt_test_frame.pack(fill=tk.X, pady=(4, 0))

        stt_result_lbl = tk.Label(
            stt_test_frame,
            text="Click 'Test Speech Recognition' and speak a short phrase.",
            font=(THEME["font_mono"], 9),
            fg=THEME["text_secondary"],
            bg="#1E293B",
            wraplength=580,
            justify=tk.LEFT
        )
        stt_result_lbl.pack(anchor="w", pady=(0, 6))

        def run_stt_test():
            sel_label = mic_var.get()
            chosen_index = options_map.get(sel_label, None)
            target_model = whisper_model_var.get()
            target_language = language_values[language_var.get()]
            stt_result_lbl.config(text="Listening for 3 seconds... Speak your test phrase now!", fg=THEME["accent_amber"])
            modal.update()

            def worker_stt_test():
                if target_model != self.speech_stt.whisper_model_name:
                    self.speech_stt.set_stt_model(target_model)
                self.speech_stt.set_stt_language(target_language)
                ok, trans_res = self.speech_stt.test_transcription_live(duration_sec=3.0, device_index=chosen_index)
                if ok:
                    message = f'✔ Transcribed by Faster-Whisper: "{trans_res}"'
                    color = THEME["accent_green"]
                else:
                    message = f"✖ {trans_res}"
                    color = THEME["accent_red"]
                modal.after(0, lambda: stt_result_lbl.config(text=message, fg=color))

            threading.Thread(target=worker_stt_test, daemon=True).start()

        btn_stt_test = tk.Button(
            stt_test_frame,
            text="🗣 Test Speech Recognition (3s)",
            font=(THEME["font_main"], 9, "bold"),
            bg="#0369A1",
            fg="#FFFFFF",
            bd=0,
            padx=12,
            pady=4,
            cursor="hand2",
            command=run_stt_test
        )
        btn_stt_test.pack(anchor="w")

        # 5. Wake Word & Activation Phrases Management
        ww_box = tk.LabelFrame(
            form,
            text="Activation Phrases & Wake Word Configuration",
            font=(THEME["font_main"], 9, "bold"),
            bg="#0F172A",
            fg=THEME["accent_cyan"],
            padx=12,
            pady=8
        )
        ww_box.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            ww_box,
            text="ADAM detects these phrases (case-insensitive) to activate command capture.\n"
                 "Examples: 'Hi', 'Hey', 'Oi', 'Hi Adam', 'Hey Adam', 'Hello Adam', 'Hey Assistant', 'Wake up Adam'",
            font=(THEME["font_main"], 8),
            fg=THEME["text_secondary"],
            bg="#0F172A",
            justify=tk.LEFT
        ).pack(anchor="w", pady=(0, 6))

        # Quick Multi-Wake-Word toggle chips
        chips_frame = tk.Frame(ww_box, bg="#0F172A")
        chips_frame.pack(fill=tk.X, pady=(0, 6))

        chip_buttons = {}

        def update_chip_buttons():
            curr = [w.lower() for w in self.speech_stt.get_wake_words()]
            for p_key, b in chip_buttons.items():
                if p_key in curr:
                    b.config(bg="#0284C7", fg="#FFFFFF", text=f"✓ {p_key.title()}")
                else:
                    b.config(bg="#1E293B", fg="#94A3B8", text=f"+ {p_key.title()}")

        def toggle_chip(phrase: str):
            curr = [w.lower() for w in self.speech_stt.get_wake_words()]
            p_lower = phrase.lower().strip()
            if p_lower in curr:
                self.speech_stt.remove_wake_word(p_lower)
            else:
                self.speech_stt.add_wake_word(p_lower)
            refresh_ww_listbox()

        for chip_text in ["Hey Adam", "Drake", "Hi", "Oi", "Adam"]:
            btn = tk.Button(
                chips_frame,
                text=chip_text,
                font=(THEME["font_main"], 8, "bold"),
                bd=0,
                padx=8,
                pady=3,
                cursor="hand2",
                command=lambda p=chip_text: toggle_chip(p),
            )
            btn.pack(side=tk.LEFT, padx=3)
            chip_buttons[chip_text.lower()] = btn

        # Phrases Listbox with scrollbar
        list_frame = tk.Frame(ww_box, bg="#0F172A")
        list_frame.pack(fill=tk.X, pady=(0, 6))

        ww_listbox = tk.Listbox(
            list_frame,
            height=5,
            font=(THEME["font_mono"], 9),
            bg="#1E293B",
            fg=THEME["accent_cyan"],
            selectbackground=THEME["accent_cyan"],
            selectforeground="#000000",
            highlightthickness=1,
            highlightbackground=THEME["border_glass"]
        )
        ww_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=ww_listbox.yview)
        ww_listbox.configure(yscrollcommand=ww_scroll.set)

        ww_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ww_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        def refresh_ww_listbox():
            ww_listbox.delete(0, tk.END)
            for w in self.speech_stt.get_wake_words():
                ww_listbox.insert(tk.END, f"  ⚡  {w}")
            update_chip_buttons()

        refresh_ww_listbox()

        # Add / Remove controls
        ctrl_frame = tk.Frame(ww_box, bg="#0F172A")
        ctrl_frame.pack(fill=tk.X, pady=(0, 6))

        new_ww_entry = tk.Entry(
            ctrl_frame,
            font=(THEME["font_main"], 9),
            bg=THEME["bg_input"],
            fg=THEME["text_primary"],
            insertbackground=THEME["accent_cyan"],
            bd=0,
            highlightthickness=1,
            highlightbackground=THEME["border_glass"]
        )
        new_ww_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(0, 6))

        def add_phrase_action():
            phrase = new_ww_entry.get().strip()
            if phrase:
                if self.speech_stt.add_wake_word(phrase):
                    new_ww_entry.delete(0, tk.END)
                    refresh_ww_listbox()

        btn_add_phrase = tk.Button(
            ctrl_frame,
            text="➕ Add Phrase",
            font=(THEME["font_main"], 8, "bold"),
            bg="#065F46",
            fg="#FFFFFF",
            bd=0,
            padx=10,
            pady=4,
            cursor="hand2",
            command=add_phrase_action
        )
        btn_add_phrase.pack(side=tk.LEFT, padx=(0, 6))

        def remove_phrase_action():
            sel = ww_listbox.curselection()
            if sel:
                current_words = self.speech_stt.get_wake_words()
                idx = sel[0]
                if 0 <= idx < len(current_words):
                    to_remove = current_words[idx]
                    self.speech_stt.remove_wake_word(to_remove)
                    refresh_ww_listbox()

        btn_rem_phrase = tk.Button(
            ctrl_frame,
            text="❌ Remove Selected",
            font=(THEME["font_main"], 8),
            bg="#991B1B",
            fg="#FFFFFF",
            bd=0,
            padx=10,
            pady=4,
            cursor="hand2",
            command=remove_phrase_action
        )
        btn_rem_phrase.pack(side=tk.LEFT, padx=(0, 6))

        def reset_defaults_action():
            self.speech_stt.reset_wake_words_to_default()
            refresh_ww_listbox()

        btn_reset_defaults = tk.Button(
            ctrl_frame,
            text="🔄 Reset Defaults",
            font=(THEME["font_main"], 8),
            bg="#374151",
            fg=THEME["text_secondary"],
            bd=0,
            padx=10,
            pady=4,
            cursor="hand2",
            command=reset_defaults_action
        )
        btn_reset_defaults.pack(side=tk.LEFT)

        # Timeout & Cooldown Sliders
        tuning_frame = tk.Frame(ww_box, bg="#0F172A")
        tuning_frame.pack(fill=tk.X, pady=(4, 6))

        tk.Label(tuning_frame, text="Wake Timeout (sec):", font=(THEME["font_main"], 8), fg=THEME["text_primary"], bg="#0F172A").pack(side=tk.LEFT)
        timeout_scale = tk.Scale(
            tuning_frame,
            from_=3.0,
            to=25.0,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            bg="#0F172A",
            fg=THEME["accent_cyan"],
            highlightthickness=0,
            length=130
        )
        timeout_scale.set(self.speech_stt.wake_timeout)
        timeout_scale.pack(side=tk.LEFT, padx=(4, 16))

        tk.Label(tuning_frame, text="Cooldown (sec):", font=(THEME["font_main"], 8), fg=THEME["text_primary"], bg="#0F172A").pack(side=tk.LEFT)
        cooldown_scale = tk.Scale(
            tuning_frame,
            from_=0.5,
            to=5.0,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            bg="#0F172A",
            fg=THEME["accent_cyan"],
            highlightthickness=0,
            length=130
        )
        cooldown_scale.set(self.speech_stt.wake_cooldown)
        cooldown_scale.pack(side=tk.LEFT, padx=(4, 0))

        clap_frame = tk.Frame(ww_box, bg="#0F172A")
        clap_frame.pack(fill=tk.X, pady=(0, 6))
        clap_enabled_var = tk.BooleanVar(value=self.speech_stt.clap_activation_enabled)
        tk.Checkbutton(
            clap_frame,
            text="Enable double-clap activation (optional; not secure authentication)",
            variable=clap_enabled_var,
            font=(THEME["font_main"], 8),
            fg=THEME["text_primary"],
            selectcolor="#020617",
            activebackground="#0F172A",
            activeforeground=THEME["text_primary"],
            bg="#0F172A",
        ).pack(side=tk.LEFT)
        tk.Label(
            clap_frame,
            text="Sensitivity",
            font=(THEME["font_main"], 8),
            fg=THEME["text_secondary"],
            bg="#0F172A",
        ).pack(side=tk.LEFT, padx=(10, 4))
        clap_sensitivity_scale = tk.Scale(
            clap_frame,
            from_=0.0,
            to=1.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            bg="#0F172A",
            fg=THEME["accent_cyan"],
            highlightthickness=0,
            length=110,
        )
        clap_sensitivity_scale.set(self.speech_stt.clap_sensitivity)
        clap_sensitivity_scale.pack(side=tk.LEFT)

        # Live Wake-Word Detection Test Box
        ww_test_frame = tk.Frame(ww_box, bg="#1E293B", padx=10, pady=8, highlightthickness=1, highlightbackground=THEME["border_glass"])
        ww_test_frame.pack(fill=tk.X, pady=(4, 0))

        ww_test_lbl = tk.Label(
            ww_test_frame,
            text="Click 'Test Wake Word Mode' and say e.g. 'Hey Adam, open WhatsApp' or 'Oi'.",
            font=(THEME["font_mono"], 9),
            fg=THEME["text_secondary"],
            bg="#1E293B",
            wraplength=580,
            justify=tk.LEFT
        )
        ww_test_lbl.pack(anchor="w", pady=(0, 6))

        def run_ww_test():
            sel_label = mic_var.get()
            chosen_index = options_map.get(sel_label, None)
            ww_test_lbl.config(text="Listening for 4 seconds... Say a wake phrase now (e.g. 'Hey Adam' or 'Oi')!", fg=THEME["accent_amber"])
            modal.update()

            def worker_ww_test():
                detected, matched_phrase, remaining_cmd, full_trans = self.speech_stt.test_wake_word_live(
                    duration_sec=4.0,
                    device_index=chosen_index
                )
                if detected:
                    cmd_part = f" | Inline Command: '{remaining_cmd}'" if remaining_cmd else " (Wake phrase alone)"
                    ww_test_lbl.config(
                        text=f"✔ WAKE WORD DETECTED: '{matched_phrase}'{cmd_part}\nTranscript: \"{full_trans}\"",
                        fg=THEME["accent_green"]
                    )
                else:
                    if full_trans and full_trans != "No speech detected":
                        ww_test_lbl.config(
                            text=f"ℹ Speech Heard: \"{full_trans}\"\n(No configured wake phrase matched. Try 'Hey Adam' or add phrase above)",
                            fg=THEME["accent_amber"]
                        )
                    else:
                        ww_test_lbl.config(text=f"✖ {full_trans or 'No speech detected'}", fg=THEME["accent_red"])

            threading.Thread(target=worker_ww_test, daemon=True).start()

        btn_ww_test = tk.Button(
            ww_test_frame,
            text="⚡ Test Wake Word Mode (4s)",
            font=(THEME["font_main"], 9, "bold"),
            bg="#D97706",
            fg="#FFFFFF",
            bd=0,
            padx=12,
            pady=4,
            cursor="hand2",
            command=run_ww_test
        )
        btn_ww_test.pack(anchor="w")

        # Save & Apply All Settings
        def apply_audio_settings():
            sel_label = mic_var.get()
            chosen_index = options_map.get(sel_label, None)
            if chosen_index is not None:
                self.speech_stt.switch_microphone(chosen_index)

            chosen_model = whisper_model_var.get()
            if chosen_model != self.speech_stt.whisper_model_name:
                self.speech_stt.set_stt_model(chosen_model)
            self.speech_stt.set_stt_language(language_values[language_var.get()])

            self.speech_stt.wake_timeout = float(timeout_scale.get())
            self.speech_stt.wake_cooldown = float(cooldown_scale.get())
            self.speech_stt.clap_activation_enabled = bool(clap_enabled_var.get())
            self.speech_stt.clap_sensitivity = float(clap_sensitivity_scale.get())
            if not self.speech_stt.save_stt_settings():
                messagebox.showerror(
                    "Audio settings",
                    "Could not save speech and clap activation settings. See the ADAM log for details.",
                    parent=modal,
                )
                return

            self.mic_badge.config(text=self.speech_stt.get_status_info())
            if self.speech_stt.mic_available:
                if self.speech_stt.is_listening and not self.speech_stt._paused:
                    self.set_voice_state("LISTENING_WAKE")
                else:
                    self.set_voice_state("MIC_OFF")
            else:
                self.set_voice_state("MIC_ERROR", "No microphone detected.")
            self._add_message_card(
                "ADAM",
                f"Audio & Wake Word settings saved: Device `{self.speech_stt.device_name}`, Model `{self.speech_stt.whisper_model_name}`, Language `{language_var.get()}`, Active Wake Words: {len(self.speech_stt.get_wake_words())} phrases."
            )
            modal.destroy()

        btn_apply = tk.Button(
            modal,
            text="Save Audio & Wake Word Settings",
            font=(THEME["font_main"], 10, "bold"),
            bg="#059669",
            fg="#FFFFFF",
            bd=0,
            pady=10,
            cursor="hand2",
            command=apply_audio_settings
        )
        btn_apply.pack(fill=tk.X, padx=20, pady=(0, 16))

    def _open_voice_modal(self):
        """Voice Settings Panel for Piper TTS, voice selection, speed & volume."""
        modal = tk.Toplevel(self)
        modal.title("ADAM — Voice Settings & TTS Engine")
        modal.geometry("580x520")
        modal.configure(bg=THEME["bg_card"])
        modal.transient(self)
        modal.grab_set()

        hdr = tk.Label(
            modal,
            text="🗣 Voice Selection & Piper Settings",
            font=(THEME["font_main"], 12, "bold"),
            fg=THEME["accent_cyan"],
            bg=THEME["bg_card"],
            pady=12
        )
        hdr.pack(fill=tk.X)

        form = tk.Frame(modal, bg=THEME["bg_card"], padx=20)
        form.pack(fill=tk.BOTH, expand=True)

        # Rescan voices
        self.voice_tts.refresh_voices()
        available = self.voice_tts.available_voices

        # Status header
        active_prof = available.get(self.voice_tts.active_voice_id)
        current_name = active_prof.name if active_prof else self.voice_tts.active_voice_id
        
        status_box = tk.Frame(form, bg="#0F172A", padx=12, pady=8, highlightthickness=1, highlightbackground=THEME["border_glass"])
        status_box.pack(fill=tk.X, pady=(0, 12))
        tk.Label(
            status_box,
            text=f"• Active Provider: {self.voice_tts.engine_type.upper()}\n"
                 f"• Current Voice: {current_name}\n"
                 f"• Amy English Voice: {'Found & Ready' if any('amy' in k.lower() for k in available) else 'Not Found'}",
            font=(THEME["font_mono"], 9),
            fg=THEME["accent_green"],
            bg="#0F172A",
            justify=tk.LEFT
        ).pack(anchor="w")

        # Voice dropdown selector
        tk.Label(form, text="Choose Spoken Voice:", font=(THEME["font_main"], 10, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(anchor="w", pady=(4, 2))

        options_map = {prof.name: v_id for v_id, prof in available.items()}
        current_opt = current_name if current_name in options_map else list(options_map.keys())[0] if options_map else "No voices found"
        
        voice_var = tk.StringVar(value=current_opt)
        combo = ttk.Combobox(form, values=list(options_map.keys()), textvariable=voice_var, state="readonly" if options_map else "disabled", font=(THEME["font_main"], 9))
        combo.pack(fill=tk.X, pady=(0, 14))

        # Speed slider
        speed_frame = tk.Frame(form, bg=THEME["bg_card"])
        speed_frame.pack(fill=tk.X, pady=4)
        tk.Label(speed_frame, text="Speech Speed Factor:", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(side=tk.LEFT)
        speed_scale = tk.Scale(
            speed_frame,
            from_=0.5,
            to=1.8,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            bg=THEME["bg_card"],
            fg=THEME["accent_cyan"],
            highlightthickness=0,
            length=250
        )
        speed_scale.set(self.voice_tts.speed)
        speed_scale.pack(side=tk.RIGHT)

        # Volume slider
        vol_frame = tk.Frame(form, bg=THEME["bg_card"])
        vol_frame.pack(fill=tk.X, pady=4)
        tk.Label(vol_frame, text="Speech Volume:", font=(THEME["font_main"], 9, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(side=tk.LEFT)
        vol_scale = tk.Scale(
            vol_frame,
            from_=0.1,
            to=1.5,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            bg=THEME["bg_card"],
            fg=THEME["accent_cyan"],
            highlightthickness=0,
            length=250
        )
        vol_scale.set(self.voice_tts.volume)
        vol_scale.pack(side=tk.RIGHT)

        # Preview Button
        def preview_selected():
            chosen_name = voice_var.get()
            chosen_id = options_map.get(chosen_name)
            if chosen_id:
                self.voice_tts.set_voice(chosen_id)
            self.voice_tts.set_speed(speed_scale.get())
            self.voice_tts.set_volume(vol_scale.get())
            self.voice_tts.preview_voice("Hello! This is a preview of my synthesized speech.")

        btn_preview = tk.Button(
            form,
            text="🔊 Preview Selected Voice",
            font=(THEME["font_main"], 9, "bold"),
            bg="#0369A1",
            fg="#FFFFFF",
            bd=0,
            padx=12,
            pady=6,
            cursor="hand2",
            command=preview_selected
        )
        btn_preview.pack(anchor="w", pady=(10, 6))

        def open_wake_config():
            modal.destroy()
            self._open_microphone_modal()

        btn_wake_config = tk.Button(
            form,
            text="⚡ Configure Activation Phrases & Wake Words",
            font=(THEME["font_main"], 9, "bold"),
            bg="#1E293B",
            fg=THEME["accent_cyan"],
            activebackground="#334155",
            bd=0,
            padx=12,
            pady=6,
            cursor="hand2",
            command=open_wake_config
        )
        btn_wake_config.pack(anchor="w", pady=(0, 10))

        # Save & Apply
        def apply_voice():
            chosen_name = voice_var.get()
            chosen_id = options_map.get(chosen_name)
            if chosen_id:
                self.voice_tts.set_voice(chosen_id)
            self.voice_tts.set_speed(speed_scale.get())
            self.voice_tts.set_volume(vol_scale.get())
            self.voice_badge.config(text=self.voice_tts.get_status_info())
            self._add_message_card("ADAM", f"Switched active voice to `{chosen_name}`.")
            modal.destroy()

        btn_apply = tk.Button(
            modal,
            text="Save Voice Settings",
            font=(THEME["font_main"], 10, "bold"),
            bg="#059669",
            fg="#FFFFFF",
            bd=0,
            pady=10,
            cursor="hand2",
            command=apply_voice
        )
        btn_apply.pack(fill=tk.X, padx=20, pady=(0, 16))

    def _open_3d_modal(self):
        """Opens futuristic 3D Generation dialog."""
        modal = tk.Toplevel(self)
        modal.title("ADAM — 3D Object Synthesis Lab")
        modal.geometry("520x460")
        modal.configure(bg=THEME["bg_card"])
        modal.transient(self)
        modal.grab_set()

        hdr = tk.Label(
            modal,
            text="⚡ Parametric 3D Mesh Generator",
            font=(THEME["font_main"], 13, "bold"),
            fg=THEME["accent_cyan"],
            bg=THEME["bg_card"],
            pady=12
        )
        hdr.pack(fill=tk.X)

        blender_status = "Detected" if is_blender_available() else "Not Installed (Using ADAM Fallback Engine)"
        status_lbl = tk.Label(
            modal,
            text=f"Blender Engine: {blender_status}",
            font=(THEME["font_mono"], 9),
            fg=THEME["accent_green"] if is_blender_available() else THEME["accent_amber"],
            bg=THEME["bg_card"]
        )
        status_lbl.pack(fill=tk.X, pady=(0, 10))

        # Shape picker
        form = tk.Frame(modal, bg=THEME["bg_card"], padx=20)
        form.pack(fill=tk.BOTH, expand=True)

        tk.Label(form, text="Target Geometry:", font=(THEME["font_main"], 10, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(anchor="w", pady=(4, 2))
        shapes = get_available_shapes()
        shape_var = tk.StringVar(value=shapes[0])
        shape_combo = ttk.Combobox(form, values=shapes, textvariable=shape_var, state="readonly", font=(THEME["font_main"], 10))
        shape_combo.pack(fill=tk.X, pady=(0, 10))

        # Format picker
        tk.Label(form, text="Export Format:", font=(THEME["font_main"], 10, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(anchor="w", pady=(4, 2))
        fmt_var = tk.StringVar(value="stl")
        fmt_frame = tk.Frame(form, bg=THEME["bg_card"])
        fmt_frame.pack(fill=tk.X, pady=(0, 10))

        rb_stl = tk.Radiobutton(fmt_frame, text="STL (Standard 3D Print / CAD)", variable=fmt_var, value="stl", bg=THEME["bg_card"], fg=THEME["text_primary"], selectcolor="#1E293B", activebackground=THEME["bg_card"])
        rb_stl.pack(side=tk.LEFT, padx=(0, 14))

        rb_obj = tk.Radiobutton(fmt_frame, text="OBJ (Wavefront 3D Asset)", variable=fmt_var, value="obj", bg=THEME["bg_card"], fg=THEME["text_primary"], selectcolor="#1E293B", activebackground=THEME["bg_card"])
        rb_obj.pack(side=tk.LEFT)

        # Scale slider
        tk.Label(form, text="Model Scale:", font=(THEME["font_main"], 10, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(anchor="w", pady=(4, 2))
        scale_var = tk.DoubleVar(value=1.0)
        scale_scale = tk.Scale(form, variable=scale_var, from_=0.5, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, bg=THEME["bg_card"], fg=THEME["text_primary"], highlightthickness=0, troughcolor="#1E293B")
        scale_scale.pack(fill=tk.X, pady=(0, 14))

        # Result display
        res_lbl = tk.Label(form, text="", font=(THEME["font_main"], 9), fg=THEME["accent_green"], bg=THEME["bg_card"], wraplength=460)
        res_lbl.pack(fill=tk.X, pady=(4, 10))

        def do_generate():
            target_shape = shape_var.get()
            target_fmt = fmt_var.get()
            target_scale = scale_var.get()

            ok, out_path, msg = generate_3d_object(
                shape_type=target_shape,
                export_format=target_fmt,
                size=target_scale
            )
            if ok:
                res_lbl.config(text=f"Exported: {out_path.name}\nPath: {out_path}", fg=THEME["accent_green"])
                self._add_message_card("ADAM", f"✨ Generated 3D model `{target_shape}` ({target_fmt.upper()}) at `{out_path}`.")
                self.voice_tts.speak(f"Generated 3D model {target_shape} successfully.")
            else:
                res_lbl.config(text=f"Error: {msg}", fg=THEME["accent_red"])

        btn_gen = tk.Button(
            modal,
            text="🔨 Generate & Export 3D Mesh",
            font=(THEME["font_main"], 10, "bold"),
            bg="#4338CA",
            fg="#FFFFFF",
            activebackground="#4F46E5",
            bd=0,
            padx=16,
            pady=10,
            cursor="hand2",
            command=do_generate
        )
        btn_gen.pack(fill=tk.X, padx=20, pady=(0, 16))

    def _open_model_selector(self):
        """Ollama Model Manager: switch models dynamically, install new models with progress bar."""
        modal = tk.Toplevel(self)
        modal.title("ADAM — AI Model Manager")
        modal.geometry("560x540")
        modal.configure(bg=THEME["bg_card"])
        modal.transient(self)
        modal.grab_set()

        hdr = tk.Label(
            modal,
            text="🧠 Ollama Local AI Model Manager",
            font=(THEME["font_main"], 12, "bold"),
            fg=THEME["accent_cyan"],
            bg=THEME["bg_card"],
            pady=12
        )
        hdr.pack(fill=tk.X)

        form = tk.Frame(modal, bg=THEME["bg_card"], padx=20)
        form.pack(fill=tk.BOTH, expand=True)

        self.integrations.ollama.check_connection()
        is_online = self.integrations.ollama.is_connected
        available = self.integrations.ollama.available_models
        models_list = available if available else RECOMMENDED_MODELS

        # 1. Connection status box
        status_box = tk.Frame(form, bg="#0F172A", padx=12, pady=8, highlightthickness=1, highlightbackground=THEME["border_glass"])
        status_box.pack(fill=tk.X, pady=(0, 10))
        status_txt = f"• Ollama Status: {'Online (Ready)' if is_online else 'Offline (run `ollama serve`)'}\n• Active Chat Model: {self.integrations.ollama.model}"
        tk.Label(status_box, text=status_txt, font=(THEME["font_mono"], 9), fg=THEME["accent_green"] if is_online else THEME["accent_amber"], bg="#0F172A", justify=tk.LEFT).pack(anchor="w")

        # 2. Installed & Recommended Model Picker
        tk.Label(form, text="Select Active Model:", font=(THEME["font_main"], 10, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(anchor="w", pady=(4, 2))

        model_var = tk.StringVar(value=self.integrations.ollama.model)
        combo_models = ttk.Combobox(form, values=models_list, textvariable=model_var, state="readonly" if models_list else "disabled", font=(THEME["font_main"], 10))
        combo_models.pack(fill=tk.X, pady=(0, 12))

        def apply_model():
            selected = model_var.get().strip()
            if selected:
                self.integrations.ollama.set_model(selected)
                self.ollama_badge.config(text=self.integrations.ollama.get_status_info())
                self._add_message_card("ADAM", f"Switched active Ollama model to `{selected}`.")
                modal.destroy()

        btn_apply = tk.Button(
            form,
            text="✔ Set as Active Model",
            font=(THEME["font_main"], 9, "bold"),
            bg="#0369A1",
            fg="#FFFFFF",
            bd=0,
            padx=12,
            pady=6,
            cursor="hand2",
            command=apply_model
        )
        btn_apply.pack(anchor="w", pady=(0, 14))

        # 3. Pull / Download New Model Section
        pull_frame = tk.LabelFrame(
            form,
            text="Install New Ollama Model",
            font=(THEME["font_main"], 9, "bold"),
            bg="#0F172A",
            fg=THEME["accent_cyan"],
            padx=12,
            pady=10
        )
        pull_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(pull_frame, text="Enter Ollama Model Tag (e.g. llama3.2:1b, qwen2.5:3b, mistral):", font=(THEME["font_main"], 8), fg=THEME["text_secondary"], bg="#0F172A").pack(anchor="w", pady=(0, 4))

        entry_frame = tk.Frame(pull_frame, bg="#0F172A")
        entry_frame.pack(fill=tk.X, pady=(0, 6))

        new_model_entry = tk.Entry(entry_frame, font=(THEME["font_main"], 9), bg=THEME["bg_input"], fg=THEME["text_primary"], insertbackground=THEME["accent_cyan"])
        new_model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        new_model_entry.insert(0, "llama3.2:1b")

        pull_status_lbl = tk.Label(pull_frame, text="Ready to install model.", font=(THEME["font_mono"], 8), fg=THEME["text_secondary"], bg="#0F172A")
        pull_status_lbl.pack(anchor="w", pady=(2, 4))

        progress_bar = ttk.Progressbar(pull_frame, mode="determinate", length=300)
        progress_bar.pack(fill=tk.X, pady=(0, 6))

        def run_pull():
            target_tag = new_model_entry.get().strip()
            if not target_tag:
                return

            if not self.integrations.ollama.is_connected:
                pull_status_lbl.config(text="Cannot pull: Ollama service is offline. Run 'ollama serve' first.", fg=THEME["accent_red"])
                return

            btn_pull.config(state="disabled")
            pull_status_lbl.config(text=f"Initiating pull for {target_tag}...", fg=THEME["accent_amber"])

            def worker():
                try:
                    for update in self.integrations.ollama.pull_model_stream(target_tag):
                        status = update.get("status", "")
                        percent = update.get("percent", 0.0)
                        modal.after(0, lambda s=status, p=percent: update_ui(s, p))
                        if update.get("done", False):
                            break
                    modal.after(0, finish_pull)
                except Exception as e:
                    modal.after(0, lambda: pull_status_lbl.config(text=f"Pull error: {e}", fg=THEME["accent_red"]))
                    modal.after(0, lambda: btn_pull.config(state="normal"))

            def update_ui(status_text, percent):
                pull_status_lbl.config(text=status_text, fg=THEME["accent_cyan"])
                progress_bar["value"] = percent

            def finish_pull():
                btn_pull.config(state="normal")
                progress_bar["value"] = 100
                pull_status_lbl.config(text=f"✔ Successfully installed {target_tag}!", fg=THEME["accent_green"])
                # Refresh model selector
                self.integrations.ollama.check_connection()
                new_list = self.integrations.ollama.available_models
                combo_models["values"] = new_list
                model_var.set(target_tag)

            threading.Thread(target=worker, daemon=True).start()

        btn_pull = tk.Button(
            entry_frame,
            text="⬇ Install Model",
            font=(THEME["font_main"], 9, "bold"),
            bg="#4338CA",
            fg="#FFFFFF",
            activebackground="#4F46E5",
            bd=0,
            padx=10,
            pady=4,
            cursor="hand2",
            command=run_pull
        )
        btn_pull.pack(side=tk.RIGHT)


def main():
    """Main application entry point."""
    app = AdamApp()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        app.speech_stt.stop()
        app.voice_tts.stop()


if __name__ == "__main__":
    main()
