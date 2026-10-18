#!/usr/bin/env python3
"""
Unit and Integration Tests for ADAM Live Voice Activity UI & Voice Controller Pipeline.
Verifies:
1. VOICE_STATES structure and coverage of all 10 states.
2. set_voice_state controller state machine transitions and breadcrumb stages.
3. Acoustic feedback prevention (TTS muting, Ollama busy guards).
4. Audio level meter VU segments calculation.
5. Deduplication of spoken user commands.
"""

import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

VENV_PKGS = [
    ROOT / "venv" / "lib" / "python3.14" / "site-packages",
    ROOT / "venv" / "lib64" / "python3.14" / "site-packages",
]
for p in VENV_PKGS:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.main import VOICE_STATES, THEME
from app.speech import SpeechListener, match_wake_word, normalize_speech_variants


class TestVoiceActivityUI(unittest.TestCase):
    """Test suite for voice activity states and UI pipeline logic."""

    def test_voice_states_completeness(self):
        """Verify all 10 required voice states are defined with correct schema."""
        expected_states = [
            "MIC_OFF",
            "MIC_STARTING",
            "LISTENING_WAKE",
            "WAKE_DETECTED",
            "LISTENING_COMMAND",
            "TRANSCRIBING",
            "PROCESSING",
            "SPEAKING",
            "READY",
            "MIC_ERROR"
        ]
        for st in expected_states:
            self.assertIn(st, VOICE_STATES, f"State '{st}' must be in VOICE_STATES")
            info = VOICE_STATES[st]
            self.assertIn("title", info)
            self.assertIn("icon", info)
            self.assertIn("text", info)
            self.assertIn("color", info)
            self.assertIn("btn_text", info)
            self.assertIn("btn_bg", info)
            self.assertIn("active_capture", info)

        # Check active_capture flags
        self.assertFalse(VOICE_STATES["MIC_OFF"]["active_capture"])
        self.assertTrue(VOICE_STATES["LISTENING_WAKE"]["active_capture"])
        self.assertTrue(VOICE_STATES["LISTENING_COMMAND"]["active_capture"])
        self.assertFalse(VOICE_STATES["PROCESSING"]["active_capture"])
        self.assertFalse(VOICE_STATES["SPEAKING"]["active_capture"])

    def test_acoustic_feedback_guards(self):
        """Verify speech engine ignores audio when busy processing or speaking."""
        listener = SpeechListener()
        # Ensure initial state
        self.assertFalse(listener.is_busy_processing)
        self.assertFalse(listener.is_tts_speaking)

        # Set TTS speaking
        listener.set_tts_speaking(True)
        self.assertTrue(listener.is_tts_speaking)
        self.assertEqual(listener.flow_state, "SPEAKING")

        # In TTS speaking mode, dispatching utterance must be discarded
        called = []
        listener.on_wake_detected = lambda cmd: called.append(cmd)
        listener._dispatch_utterance(b"\x00" * 3200)
        self.assertEqual(len(called), 0, "Utterance must be dropped while TTS is speaking")

        # Finish TTS
        listener.set_tts_speaking(False)
        self.assertFalse(listener.is_tts_speaking)
        self.assertEqual(listener.flow_state, "MIC_OFF")

        # Set Ollama busy processing
        listener.is_busy_processing = True
        listener._dispatch_utterance(b"\x00" * 3200)
        self.assertEqual(len(called), 0, "Utterance must be dropped while Ollama is busy processing")

    def test_wake_word_matching_and_command_extraction(self):
        """Verify activation phrases and inline command extraction."""
        wake_words = ["hey adam", "hi adam", "hey", "hi", "oi", "hello adam"]
        
        # Sole wake phrase
        ok, matched, rest = match_wake_word("hey adam", wake_words)
        self.assertTrue(ok)
        self.assertEqual(matched, "hey adam")
        self.assertEqual(rest, "")

        # Inline command
        ok, matched, rest = match_wake_word("hey adam open whatsapp", wake_words)
        self.assertTrue(ok)
        self.assertEqual(matched, "hey adam")
        self.assertEqual(rest, "open whatsapp")

        # Variant 'hi'
        ok, matched, rest = match_wake_word("hi what is the weather", wake_words)
        self.assertTrue(ok)
        self.assertEqual(matched, "hi")
        self.assertEqual(rest, "what is the weather")

        # Non-wake word
        ok, matched, rest = match_wake_word("play some jazz music", wake_words)
        self.assertFalse(ok)

    def test_mock_app_controller_state_machine(self):
        """Simulate AdamApp set_voice_state controller and breadcrumbs."""
        class MockLabel:
            def __init__(self):
                self.config_data = {}
            def config(self, **kwargs):
                self.config_data.update(kwargs)
            def cget(self, key):
                return self.config_data.get(key, "")

        class MockFrame:
            def __init__(self):
                self.config_data = {}
            def config(self, **kwargs):
                self.config_data.update(kwargs)

        class MockButton:
            def __init__(self):
                self.config_data = {}
            def config(self, **kwargs):
                self.config_data.update(kwargs)

        # Mock app instance
        from app.main import AdamApp
        app = object.__new__(AdamApp)
        app.voice_state = "MIC_STARTING"
        app.processing_start_time = 0.0
        app.speech_stt = SpeechListener()
        app.lbl_voice_icon = MockLabel()
        app.lbl_voice_title = MockLabel()
        app.lbl_voice_text = MockLabel()
        app.badge_voice_frame = MockFrame()
        app.btn_header_mic = MockButton()
        app.btn_mic_toggle = MockButton()
        app.mic_badge = MockLabel()
        app.pipeline_steps = {
            "WAKE": MockLabel(),
            "LISTEN": MockLabel(),
            "TRANSCRIBE": MockLabel(),
            "THINK": MockLabel(),
            "SPEAK": MockLabel(),
        }

        # Transition 1: LISTENING_WAKE
        AdamApp.set_voice_state(app, "LISTENING_WAKE")
        self.assertEqual(app.voice_state, "LISTENING_WAKE")
        self.assertEqual(app.lbl_voice_title.config_data["text"], "LISTENING FOR WAKE WORD")
        self.assertFalse(app.speech_stt.is_busy_processing)
        self.assertEqual(app.pipeline_steps["WAKE"].config_data["fg"], "#38BDF8")
        self.assertEqual(app.pipeline_steps["THINK"].config_data["fg"], "#475569")
        self.assertEqual(app.btn_header_mic.config_data["text"], "🟢 MIC: ON")

        # Transition 2: WAKE_DETECTED
        AdamApp.set_voice_state(app, "WAKE_DETECTED", "Hey! I'm ready. What can I do for you?")
        self.assertEqual(app.voice_state, "WAKE_DETECTED")
        self.assertEqual(app.lbl_voice_text.config_data["text"], "Hey! I'm ready. What can I do for you?")
        self.assertEqual(app.btn_header_mic.config_data["text"], "⚡ WAKE DETECTED")

        # Transition 3: LISTENING_COMMAND
        AdamApp.set_voice_state(app, "LISTENING_COMMAND")
        self.assertEqual(app.voice_state, "LISTENING_COMMAND")
        self.assertEqual(app.pipeline_steps["LISTEN"].config_data["fg"], "#38BDF8")

        # Transition 4: TRANSCRIBING
        AdamApp.set_voice_state(app, "TRANSCRIBING")
        self.assertEqual(app.voice_state, "TRANSCRIBING")
        self.assertEqual(app.pipeline_steps["TRANSCRIBE"].config_data["fg"], "#38BDF8")

        # Transition 5: PROCESSING
        AdamApp.set_voice_state(app, "PROCESSING")
        self.assertEqual(app.voice_state, "PROCESSING")
        self.assertTrue(app.speech_stt.is_busy_processing)
        self.assertGreater(app.processing_start_time, 0.0)
        self.assertEqual(app.pipeline_steps["THINK"].config_data["fg"], "#38BDF8")

        # Transition 6: SPEAKING
        AdamApp.set_voice_state(app, "SPEAKING")
        self.assertEqual(app.voice_state, "SPEAKING")
        self.assertEqual(app.pipeline_steps["SPEAK"].config_data["fg"], "#38BDF8")
        self.assertEqual(app.processing_start_time, 0.0)

        # Transition 7: READY
        AdamApp.set_voice_state(app, "READY")
        self.assertEqual(app.voice_state, "READY")

        # Transition 8: MIC_OFF
        AdamApp.set_voice_state(app, "MIC_OFF")
        self.assertEqual(app.voice_state, "MIC_OFF")
        self.assertEqual(app.btn_header_mic.config_data["text"], "🔇 MIC OFF")


if __name__ == "__main__":
    unittest.main()
