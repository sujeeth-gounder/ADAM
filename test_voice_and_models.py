"""Unit tests for Voice Selection and Model Manager."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.voice import PiperTTS, VoiceProfile
from app.integrations import OllamaClient
from app.speech import SpeechListener, list_input_microphones, diagnose_microphone_system


class TestVoiceAndModelFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.shared_listener = SpeechListener()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "shared_listener") and cls.shared_listener:
            cls.shared_listener.stop()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_voice_discovery_and_selection(self):
        voice_engine = PiperTTS()
        # Check that voices are discovered
        voices = voice_engine.available_voices
        self.assertGreater(len(voices), 0)

        # Check Amy profile exists or can be activated
        amy_key = "piper:en_US-amy-medium"
        if amy_key in voices:
            self.assertEqual(voices[amy_key].provider, "piper")
            self.assertTrue(voice_engine.set_voice(amy_key))
            self.assertEqual(voice_engine.active_voice_id, amy_key)

    def test_voice_speed_and_volume(self):
        voice_engine = PiperTTS()
        voice_engine.set_speed(1.3)
        self.assertAlmostEqual(voice_engine.speed, 1.3)

        voice_engine.set_volume(0.8)
        self.assertAlmostEqual(voice_engine.volume, 0.8)

    def test_microphone_diagnostics(self):
        diag = diagnose_microphone_system()
        self.assertIn("device_count", diag)
        self.assertIn("diagnosis_message", diag)
        self.assertIsInstance(diag["devices"], list)

    def test_microphone_toggle_listening_and_states(self):
        listener = self.shared_listener
        state = listener.get_state()
        self.assertIn(state, ["UNAVAILABLE", "MUTED", "LISTENING", "ACTIVE"])

        with patch.object(listener, "start_wake_word_loop"):
            listener.mic_available = True
            listener.resume()
            self.assertFalse(listener._paused)

            listener.pause()
            self.assertTrue(listener._paused)

    def test_ollama_client_model_methods(self):
        ollama = OllamaClient()
        self.assertTrue(ollama.set_model("qwen2.5:1.5b"))
        self.assertEqual(ollama.model, "qwen2.5:1.5b")

    def test_faster_whisper_transcriber_settings(self):
        from app.speech import FasterWhisperTranscriber
        transcriber = FasterWhisperTranscriber(model_size="base", device="cpu", compute_type="int8")
        self.assertEqual(transcriber.model_size, "base")
        self.assertEqual(transcriber.device, "cpu")
        self.assertEqual(transcriber.compute_type, "int8")

    def test_stt_model_switching_and_persistence(self):
        listener = self.shared_listener
        self.assertTrue(listener.set_stt_model("small") or listener.whisper_model_name == "small")
        self.assertEqual(listener.whisper_model_name, "small")

        self.assertTrue(listener.set_stt_model("base") or listener.whisper_model_name == "base")
        self.assertEqual(listener.whisper_model_name, "base")

        # Test invalid model rejected
        self.assertFalse(listener.set_stt_model("nonexistent_model"))

    def test_mic_button_sync_ui_mock(self):
        listener = self.shared_listener
        with patch.object(listener, "start_wake_word_loop"):
            listener.mic_available = True
            listener.pause()
            self.assertTrue(listener._paused)

            # Toggle should turn ON
            state_on = listener.toggle_listening()
            self.assertTrue(state_on)
            self.assertFalse(listener._paused)

            # Toggle should turn OFF
            state_off = listener.toggle_listening()
            self.assertFalse(state_off)
            self.assertTrue(listener._paused)


if __name__ == "__main__":
    unittest.main()

