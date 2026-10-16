"""ADAM — Advanced Desktop AI Mate
Main Desktop Application with Tkinter Glassmorphism UI, Animated AI Reactor Orb,
Voice Recognition ("Hii" wake word), Piper TTS (Amy voice), Ollama integration, and 3D Generator.
"""

import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
from typing import Optional

# Ensure local venv site-packages is accessible even if launched with system python
_ROOT_DIR = Path(__file__).resolve().parent.parent
_VENV_PACKAGES = [
    _ROOT_DIR / "venv" / "lib" / "python3.14" / "site-packages",
    _ROOT_DIR / "venv" / "lib64" / "python3.14" / "site-packages",
]
for p in _VENV_PACKAGES:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.config import (
    ADAM_SYSTEM_PROMPT,
    DEFAULT_OLLAMA_MODEL,
    EXPORTS_DIR,
    RECOMMENDED_MODELS,
    THEME,
)
from app.integrations import OllamaClient, PendingAction, SystemIntegrations
from app.speech import SpeechListener, list_input_microphones, test_microphone_level
from app.three_d import (
    generate_3d_object,
    get_available_shapes,
    is_blender_available,
)
from app.voice import PiperTTS


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

        # Reactor animation states: 'idle', 'listening', 'thinking', 'speaking'
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
            on_audio_energy=self._on_audio_energy
        )

        self.integrations = SystemIntegrations(
            confirmation_request_handler=self._on_confirmation_requested
        )

        # Build UI Components
        self._setup_styles()
        self._build_header()
        self._build_center_stage()
        self._build_bottom_bar()

        # Start background timers & workers
        self.after(40, self._animation_loop)
        self.after(30, self._process_ui_queue)

        # Auto-start wake word listener
        self.speech_stt.start_wake_word_loop()

        # Welcome message
        self.after(400, self._post_welcome_message)

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
            font=(THEME["font_main"], 9),
            bg=THEME["bg_card"],
            fg=THEME["text_primary"],
            activebackground=THEME["bg_card_hover"],
            bd=0,
            padx=8,
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

        # Top Section: Animated AI Hologram / Reactor Canvas
        self.visualizer_frame = tk.Frame(
            self.stage_frame,
            bg=THEME["bg_card"],
            height=110,
            highlightthickness=1,
            highlightbackground=THEME["border_glass"]
        )
        self.visualizer_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        self.visualizer_frame.pack_propagate(False)

        # Canvas for animated reactor
        self.reactor_canvas = tk.Canvas(
            self.visualizer_frame,
            bg=THEME["bg_card"],
            bd=0,
            highlightthickness=0
        )
        self.reactor_canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom Section: Chat Scroll Area
        self.chat_container = tk.Frame(
            self.stage_frame,
            bg=THEME["bg_card"],
            highlightthickness=1,
            highlightbackground=THEME["border_glass"]
        )
        self.chat_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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

        # Microphone Toggle Button
        self.btn_mic_toggle = tk.Button(
            inner,
            text="🟢 Wake: 'Hii'",
            font=(THEME["font_main"], 9, "bold"),
            bg="#064E3B",
            fg=THEME["accent_green"],
            activebackground="#065F46",
            bd=0,
            padx=12,
            pady=8,
            cursor="hand2",
            command=self._toggle_microphone
        )
        self.btn_mic_toggle.pack(side=tk.LEFT, padx=(0, 6))

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
            command=self.voice_tts.stop
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

    def _animation_loop(self):
        """30-40 FPS animation loop for holographic gyroscopic reactor."""
        self.anim_tick += 0.06
        w = self.reactor_canvas.winfo_width()
        h = self.reactor_canvas.winfo_height()

        if w > 20 and h > 20:
            self.reactor_canvas.delete("all")
            cx = w / 2.0
            cy = h / 2.0

            # State styling
            if self.reactor_state == "idle":
                core_color = THEME["accent_cyan"]
                outer_color = "#004D5A"
                label_text = "ADAM // SYSTEM READY — SAY 'HII' OR TYPE COMMAND"
                pulse = 1.0 + 0.12 * math.sin(self.anim_tick * 2.0)
            elif self.reactor_state == "listening":
                core_color = THEME["accent_green"]
                outer_color = "#064E3B"
                label_text = "ADAM // LISTENING TO AUDIO INPUT..."
                energy_factor = max(0.2, self.audio_energy)
                pulse = 1.0 + energy_factor * 0.4 + 0.2 * math.sin(self.anim_tick * 5.0)
            elif self.reactor_state == "thinking":
                core_color = THEME["accent_purple"]
                outer_color = "#3B0764"
                label_text = "ADAM // PROCESSING WITH OLLAMA NEURAL ENGINE..."
                pulse = 1.0 + 0.18 * math.sin(self.anim_tick * 4.0)
            elif self.reactor_state == "speaking":
                core_color = "#38BDF8"
                outer_color = "#0369A1"
                label_text = "ADAM // VOCALIZING RESPONSE (PIPER AMY VOICE)"
                pulse = 1.0 + 0.25 * math.abs(math.sin(self.anim_tick * 6.0)) if hasattr(math, 'abs') else 1.0 + 0.25 * abs(math.sin(self.anim_tick * 6.0))
            else:
                core_color = THEME["accent_cyan"]
                outer_color = "#004D5A"
                label_text = "ADAM // SYSTEM READY"
                pulse = 1.0

            # Draw outer radar glow ring
            r_outer = 42.0 * pulse
            self.reactor_canvas.create_oval(
                cx - r_outer, cy - r_outer,
                cx + r_outer, cy + r_outer,
                outline=outer_color, width=1.5
            )

            # Draw rotating orbital nodes
            nodes = 6
            for i in range(nodes):
                angle = self.anim_tick * (1.5 if self.reactor_state != "thinking" else 3.5) + (2 * math.pi * i / nodes)
                nx = cx + r_outer * math.cos(angle)
                ny = cy + (r_outer * 0.5) * math.sin(angle)
                self.reactor_canvas.create_oval(nx - 2, ny - 2, nx + 2, ny + 2, fill=core_color, outline="")

            # Inner pulsating core orb
            r_core = 18.0 * pulse
            self.reactor_canvas.create_oval(
                cx - r_core, cy - r_core,
                cx + r_core, cy + r_core,
                fill=core_color, outline="#E0F2FE", width=1.5
            )

            # Center focal star
            self.reactor_canvas.create_oval(
                cx - 4, cy - 4,
                cx + 4, cy + 4,
                fill="#FFFFFF", outline=""
            )

            # Horizontal audio waveforms or telemetry bars
            bars = 14
            for b in range(-bars, bars + 1):
                if b == 0:
                    continue
                dist = abs(b)
                bx = cx + b * 24
                # Amplitude based on tick, distance, and state
                amp = (20.0 / (dist * 0.6 + 1)) * (pulse - 0.7)
                if self.reactor_state == "speaking":
                    amp *= 1.8
                elif self.reactor_state == "listening":
                    amp *= 1.4

                by1 = cy - amp
                by2 = cy + amp
                self.reactor_canvas.create_line(bx, by1, bx, by2, fill=outer_color, width=1.5)

            # Status label underneath
            self.reactor_canvas.create_text(
                cx, cy + 44,
                text=label_text,
                font=(THEME["font_mono"], 8, "bold"),
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
        self._add_message_card("ADAM", welcome_text)

    def _add_message_card(self, sender: str, text: str) -> tk.Label:
        """Add a glassmorphism message card to chat stream."""
        is_user = sender.upper() == "USER"

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

        icon = "👤" if is_user else "💠"
        sender_lbl = tk.Label(
            hdr_frame,
            text=f"{icon} {sender}",
            font=(THEME["font_main"], 9, "bold"),
            fg=header_col,
            bg=bubble_bg
        )
        sender_lbl.pack(side=tk.LEFT)

        time_str = time.strftime("%I:%M %p")
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

    def _process_ui_queue(self):
        """Consumes events from background threads and applies them to Tkinter main thread."""
        while not self.ui_queue.empty():
            try:
                event_type, data = self.ui_queue.get_nowait()
                if event_type == "STREAM_CHUNK":
                    card_lbl, chunk = data
                    card_lbl.config(text=card_lbl.cget("text") + chunk)
                    self._scroll_to_bottom()
                elif event_type == "STREAM_DONE":
                    final_text = data
                    self.reactor_state = "idle"
                    # Speak response
                    self.voice_tts.speak(final_text)
                elif event_type == "DIRECT_RESPONSE":
                    resp_text, pending_action = data
                    self._add_message_card("ADAM", resp_text)
                    self.reactor_state = "idle"
                    if resp_text == "Voice playback stopped.":
                        self.voice_tts.stop()
                    else:
                        self.voice_tts.speak(resp_text)
                    if pending_action:
                        self._show_confirmation_dialog(pending_action)
                elif event_type == "WAKE_WORD_HEARD":
                    prompt = data
                    if prompt:
                        self.input_entry.delete(0, tk.END)
                        self.input_entry.insert(0, prompt)
                        self._handle_user_submit()
                    else:
                        greeting = "Hii! I am listening. How can I help you?"
                        self._add_message_card("ADAM", greeting)
                        self.voice_tts.speak(greeting)
                elif event_type == "CONFIRMATION_REQ":
                    action = data
                    self._show_confirmation_dialog(action)
                elif event_type == "SET_REACTOR_STATE":
                    self.reactor_state = data
                elif event_type == "AUDIO_ENERGY":
                    self.audio_energy = data
                elif event_type == "MIC_STATUS":
                    active, status_str = data
                    self.mic_badge.config(text=status_str)
                    if not active:
                        self.btn_mic_toggle.config(
                            text="🔴 Mic Muted",
                            bg="#7F1D1D",
                            fg="#FCA5A5"
                        )
                    else:
                        self.btn_mic_toggle.config(
                            text="🟢 Wake: 'Hii'",
                            bg="#064E3B",
                            fg=THEME["accent_green"]
                        )
            except queue.Empty:
                break
            except Exception as e:
                print(f"UI Queue processing exception: {e}", file=sys.stderr)

        self.after(30, self._process_ui_queue)

    def _on_tts_start(self):
        self.ui_queue.put(("SET_REACTOR_STATE", "speaking"))

    def _on_tts_finish(self):
        self.ui_queue.put(("SET_REACTOR_STATE", "idle"))

    def _on_wake_detected(self, prompt: str):
        self.ui_queue.put(("WAKE_WORD_HEARD", prompt))

    def _on_listening_change(self, active: bool, msg: str):
        self.ui_queue.put(("MIC_STATUS", (active, msg)))

    def _on_audio_energy(self, energy: float):
        self.ui_queue.put(("AUDIO_ENERGY", energy))

    def _on_confirmation_requested(self, action: PendingAction):
        self.ui_queue.put(("CONFIRMATION_REQ", action))

    # =========================================================================
    # USER SUBMISSION & ORCHESTRATION
    # =========================================================================

    def _handle_user_submit(self):
        """Process user text input."""
        text = self.input_entry.get().strip()
        if not text:
            return

        self.input_entry.delete(0, tk.END)
        self._add_message_card("USER", text)

        # Stop previous voice
        self.voice_tts.stop()

        # Step 1: Check direct system integration commands
        handled, resp, pending_action = self.integrations.try_handle_command(text)
        if handled:
            self.ui_queue.put(("DIRECT_RESPONSE", (resp, pending_action)))
            return

        # Step 2: Route to Ollama LLM
        self.reactor_state = "thinking"
        response_card_lbl = self._add_message_card("ADAM", "")

        def worker_ollama():
            full_response = ""
            for chunk in self.integrations.ollama.generate_stream(text):
                full_response += chunk
                self.ui_queue.put(("STREAM_CHUNK", (response_card_lbl, chunk)))
            self.ui_queue.put(("STREAM_DONE", full_response))

        threading.Thread(target=worker_ollama, daemon=True).start()

    def _toggle_microphone(self):
        """Toggle microphone listening / mute state."""
        if not self.speech_stt.mic_available:
            messagebox.showinfo("Microphone", "No microphone detected on this system. You can continue typing prompts.")
            return

        is_now_listening = self.speech_stt.toggle_pause()
        if is_now_listening:
            self.btn_mic_toggle.config(text="🟢 Wake: 'Hii'", bg="#064E3B", fg=THEME["accent_green"])
        else:
            self.btn_mic_toggle.config(text="🔴 Mic Muted", bg="#7F1D1D", fg="#FCA5A5")

    def _on_push_to_talk(self):
        """Single phrase listen (Push-to-talk)."""
        if not self.speech_stt.mic_available:
            messagebox.showinfo("Microphone", "No microphone detected on this system.")
            return

        self.reactor_state = "listening"

        def worker():
            phrase = self.speech_stt.listen_once()
            if phrase:
                self.ui_queue.put(("WAKE_WORD_HEARD", phrase))
            else:
                self.ui_queue.put(("SET_REACTOR_STATE", "idle"))

        threading.Thread(target=worker, daemon=True).start()

    def _clear_chat(self):
        """Clear message cards from chat view."""
        for widget in self.chat_scrollable_frame.winfo_children():
            widget.destroy()
        self._post_welcome_message()

    # =========================================================================
    # MODALS: MICROPHONE SETUP, 3D LAB & MODEL SELECTOR
    # =========================================================================

    def _open_microphone_modal(self):
        """Interactive audio input selector and signal level meter."""
        modal = tk.Toplevel(self)
        modal.title("ADAM — Microphone Configuration")
        modal.geometry("560x480")
        modal.configure(bg=THEME["bg_card"])
        modal.transient(self)
        modal.grab_set()

        hdr = tk.Label(
            modal,
            text="🎙️ Audio Device Configuration & Signal Test",
            font=(THEME["font_main"], 12, "bold"),
            fg=THEME["accent_cyan"],
            bg=THEME["bg_card"],
            pady=12
        )
        hdr.pack(fill=tk.X)

        mics = list_input_microphones()

        form = tk.Frame(modal, bg=THEME["bg_card"], padx=20)
        form.pack(fill=tk.BOTH, expand=True)

        tk.Label(form, text="Detected Input Microphones:", font=(THEME["font_main"], 10, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"]).pack(anchor="w", pady=(6, 2))

        options_map = {}
        for m in mics:
            label = f"[Index {m['index']}] {m['name']} ({m['channels']} ch, {m['sample_rate']}Hz)"
            options_map[label] = m['index']

        current_label = ""
        for lbl, idx in options_map.items():
            if idx == self.speech_stt.device_index:
                current_label = lbl
                break
        if not current_label and options_map:
            current_label = list(options_map.keys())[0]

        mic_var = tk.StringVar(value=current_label)
        combo = ttk.Combobox(form, values=list(options_map.keys()), textvariable=mic_var, state="readonly", font=(THEME["font_main"], 9))
        combo.pack(fill=tk.X, pady=(0, 14))

        # Test audio signal section
        test_frame = tk.Frame(form, bg="#1E293B", padx=14, pady=12, highlightthickness=1, highlightbackground=THEME["border_glass"])
        test_frame.pack(fill=tk.X, pady=8)

        tk.Label(test_frame, text="Signal Meter Test (Speak for 2 seconds):", font=(THEME["font_main"], 9, "bold"), fg=THEME["accent_cyan"], bg="#1E293B").pack(anchor="w")

        meter_canvas = tk.Canvas(test_frame, height=20, bg="#0F172A", highlightthickness=1, highlightbackground=THEME["border_glass"])
        meter_canvas.pack(fill=tk.X, pady=8)

        status_lbl = tk.Label(test_frame, text="Click 'Test Signal' to measure live microphone audio.", font=(THEME["font_main"], 9), fg=THEME["text_secondary"], bg="#1E293B")
        status_lbl.pack(anchor="w")

        def run_test():
            sel_label = mic_var.get()
            chosen_index = options_map.get(sel_label, None)
            status_lbl.config(text="Listening and measuring RMS audio level... Speak now!", fg=THEME["accent_amber"])
            modal.update()

            def worker_test():
                ok, peak, msg = test_microphone_level(chosen_index, duration_sec=2.0)
                meter_w = meter_canvas.winfo_width()
                meter_canvas.delete("all")
                fill_w = int(meter_w * min(1.0, peak))
                meter_color = THEME["accent_green"] if peak > 0.05 else THEME["accent_amber"]
                meter_canvas.create_rectangle(0, 0, fill_w, 20, fill=meter_color, outline="")

                if ok:
                    status_lbl.config(text=f"✔ {msg}", fg=THEME["accent_green"])
                else:
                    status_lbl.config(text=f"✖ {msg}", fg=THEME["accent_red"])

            threading.Thread(target=worker_test, daemon=True).start()

        btn_test = tk.Button(
            test_frame,
            text="▶ Test Signal",
            font=(THEME["font_main"], 9, "bold"),
            bg="#0369A1",
            fg="#FFFFFF",
            bd=0,
            padx=10,
            pady=4,
            cursor="hand2",
            command=run_test
        )
        btn_test.pack(anchor="w", pady=(4, 0))

        def apply_mic():
            sel_label = mic_var.get()
            chosen_index = options_map.get(sel_label, None)
            if chosen_index is not None:
                self.speech_stt.switch_microphone(chosen_index)
                self.mic_badge.config(text=self.speech_stt.get_status_info())
                self._add_message_card("ADAM", f"Switched active microphone to: `{self.speech_stt.device_name}`.")
            modal.destroy()

        btn_apply = tk.Button(
            modal,
            text="Save Active Microphone",
            font=(THEME["font_main"], 10, "bold"),
            bg="#059669",
            fg="#FFFFFF",
            bd=0,
            pady=8,
            cursor="hand2",
            command=apply_mic
        )
        btn_apply.pack(fill=tk.X, padx=20, pady=16)

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
        """Allows switching Ollama model."""
        modal = tk.Toplevel(self)
        modal.title("Ollama Model Selector")
        modal.geometry("420x340")
        modal.configure(bg=THEME["bg_card"])
        modal.transient(self)
        modal.grab_set()

        tk.Label(
            modal,
            text="Select Ollama Neural Model",
            font=(THEME["font_main"], 12, "bold"),
            fg=THEME["accent_cyan"],
            bg=THEME["bg_card"],
            pady=12
        ).pack(fill=tk.X)

        self.integrations.ollama.check_connection()
        available = self.integrations.ollama.available_models
        models_list = available if available else RECOMMENDED_MODELS

        model_var = tk.StringVar(value=self.integrations.ollama.model)

        box_frame = tk.Frame(modal, bg=THEME["bg_card"], padx=20)
        box_frame.pack(fill=tk.BOTH, expand=True)

        for m in models_list:
            rb = tk.Radiobutton(
                box_frame,
                text=f"{m} {'(Active)' if m == self.integrations.ollama.model else ''}",
                variable=model_var,
                value=m,
                font=(THEME["font_main"], 10),
                bg=THEME["bg_card"],
                fg=THEME["text_primary"],
                selectcolor="#1E293B",
                activebackground=THEME["bg_card"]
            )
            rb.pack(anchor="w", pady=4)

        def apply_model():
            selected = model_var.get()
            self.integrations.ollama.model = selected
            self.ollama_badge.config(text=self.integrations.ollama.get_status_info())
            modal.destroy()
            self._add_message_card("ADAM", f"Switched active Ollama model to `{selected}`.")

        btn_apply = tk.Button(
            modal,
            text="Apply Model",
            font=(THEME["font_main"], 10, "bold"),
            bg="#0369A1",
            fg="#FFFFFF",
            bd=0,
            pady=8,
            command=apply_model
        )
        btn_apply.pack(fill=tk.X, padx=20, pady=16)


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
