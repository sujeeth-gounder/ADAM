"""Unit and integration tests for ADAM Wake Word Detection and Flow Pipeline."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.config import DEFAULT_WAKE_WORDS, DEFAULT_WAKE_TIMEOUT_SEC, DEFAULT_WAKE_COOLDOWN_SEC
from app.speech import (
    SpeechListener,
    normalize_speech_variants,
    match_wake_word,
)


class TestWakeWordPipeline(unittest.TestCase):
    """Test suite verifying wake-word detection, multi-phrase support, and listening flow."""

    def setUp(self):
        self.wake_words = list(DEFAULT_WAKE_WORDS)

    def test_all_required_activation_phrases(self):
        """Verify that all required phrases trigger wake word detection."""
        required_phrases = [
            "Hi",
            "Hey",
            "Oi",
            "Hi Adam",
            "Hey Adam",
            "Hi ADAM",
            "Hey ADAM",
            "Hello Adam",
            "Hey Assistant",
            "Wake up Adam",
        ]
        for phrase in required_phrases:
            detected, matched, remaining = match_wake_word(phrase, self.wake_words)
            self.assertTrue(
                detected,
                f"Required activation phrase failed to trigger: '{phrase}'"
            )
            self.assertEqual(remaining, "", f"Expected empty remaining command for solo wake phrase: '{phrase}'")

    def test_one_shot_commands(self):
        """Verify that wake phrases followed by immediate commands are correctly split."""
        test_cases = [
            ("Hey Adam, open WhatsApp", "hey adam", "open whatsapp"),
            ("Hi, what time is it?", "hi", "what time is it"),
            ("Hey Adam, play Believer by Imagine Dragons", "hey adam", "play believer by imagine dragons"),
            ("Oi, launch terminal", "oi", "launch terminal"),
            ("Wake up Adam, open YouTube in Brave", "wake up adam", "open youtube in brave"),
            ("Hello Adam! How is the weather today?", "hello adam", "how is the weather today"),
            ("Hey Assistant: search for quantum computing", "hey assistant", "search for quantum computing"),
            ("Hi ADAM, create 3d scifi_drone", "hi adam", "create 3d scifi_drone"),
            ("Hey ADAM, open Settings", "hey adam", "open settings"),
        ]
        for input_text, expected_match, expected_cmd in test_cases:
            detected, matched, remaining = match_wake_word(input_text, self.wake_words)
            self.assertTrue(detected, f"Failed detection on one-shot command: '{input_text}'")
            self.assertEqual(matched, expected_match)
            self.assertEqual(remaining, expected_cmd)

    def test_punctuation_casing_and_phonetic_variations(self):
        """Verify normalization handles punctuation, multiple letters ('Hiii', 'Heyyy'), and casing."""
        variations = [
            ("Hii Adam...", "hi adam"),
            ("HEYYY ADAM!", "hey adam"),
            ("oi!!", "oi"),
            ("HI, what is your name?", "hi"),
            ("wAkE   Up   aDaM", "wake up adam"),
        ]
        for var_text, expected_match in variations:
            detected, matched, _ = match_wake_word(var_text, self.wake_words)
            self.assertTrue(detected, f"Failed variation tolerance: '{var_text}'")
            self.assertEqual(matched, expected_match)

    def test_unrelated_phrases_do_not_trigger(self):
        """Verify unrelated words and sentences do not falsely trigger wake detection."""
        false_positives = [
            "highway traffic is heavy today",
            "history of computers",
            "the oil price changed",
            "today is a beautiful sunny morning",
            "what should we eat for dinner?",
            "adamant about the decision",
            "oily pizza",
        ]
        for phrase in false_positives:
            detected, matched, _ = match_wake_word(phrase, self.wake_words)
            self.assertFalse(
                detected,
                f"Unrelated phrase falsely triggered wake detection: '{phrase}' (matched: '{matched}')"
            )

    def test_custom_wake_word_add_and_remove(self):
        """Verify adding and removing activation phrases dynamically."""
        listener = SpeechListener()
        # Test add
        self.assertTrue(listener.add_wake_word("jarvis"))
        self.assertIn("jarvis", listener.get_wake_words())

        # Test duplicate add ignored
        self.assertFalse(listener.add_wake_word("jarvis"))

        # Test detection of newly added phrase
        detected, matched, rem = match_wake_word("Jarvis, open Calculator", listener.get_wake_words())
        self.assertTrue(detected)
        self.assertEqual(matched, "jarvis")
        self.assertEqual(rem, "open calculator")

        # Test remove
        self.assertTrue(listener.remove_wake_word("jarvis"))
        self.assertNotIn("jarvis", listener.get_wake_words())

        # Reset defaults
        listener.reset_wake_words_to_default()
        self.assertIn("hey adam", listener.get_wake_words())
        self.assertIn("oi", listener.get_wake_words())

    def test_persistence_of_settings(self):
        """Verify wake words and timeout/cooldown settings persist to JSON."""
        listener = SpeechListener()
        listener.wake_timeout = 14.5
        listener.wake_cooldown = 2.5
        listener.add_wake_word("computer")
        listener.save_stt_settings()

        # Create fresh listener instance and verify loaded settings
        listener_reloaded = SpeechListener()
        self.assertIn("computer", listener_reloaded.get_wake_words())
        self.assertAlmostEqual(listener_reloaded.wake_timeout, 14.5)
        self.assertAlmostEqual(listener_reloaded.wake_cooldown, 2.5)

        # Cleanup
        listener_reloaded.remove_wake_word("computer")
        listener_reloaded.wake_timeout = DEFAULT_WAKE_TIMEOUT_SEC
        listener_reloaded.wake_cooldown = DEFAULT_WAKE_COOLDOWN_SEC
        listener_reloaded.save_stt_settings()

    def test_flow_state_callbacks_and_tts_muting(self):
        """Verify state flow notifications and TTS mute suppression."""
        states_recorded = []

        def on_flow_change(state, details):
            states_recorded.append((state, details))

        listener = SpeechListener(on_flow_state_change=on_flow_change)
        listener.set_flow_state("LISTENING")
        listener.set_flow_state("WAKE_DETECTED", "hey adam")
        listener.set_flow_state("HEARING_COMMAND", "Listening for your prompt...")
        listener.set_flow_state("PROCESSING", "open whatsapp")
        listener.set_flow_state("MIC_OFF")

        expected_states = [
            "LISTENING",
            "WAKE_DETECTED",
            "HEARING_COMMAND",
            "PROCESSING",
            "MIC_OFF",
        ]
        self.assertEqual([s[0] for s in states_recorded], expected_states)

        # Verify TTS muting suppresses capture
        listener.set_tts_speaking(True)
        self.assertTrue(listener.is_tts_speaking)
        self.assertGreater(listener.last_cooldown_until, time.time() + 1000)

        # Verify unmuting restores cooldown
        listener.set_tts_speaking(False)
        self.assertFalse(listener.is_tts_speaking)
        self.assertLessEqual(listener.last_cooldown_until, time.time() + listener.wake_cooldown + 0.1)

    def test_one_shot_and_wake_alone_processing_logic(self):
        """Verify _process_recognized_phrase dispatches one-shot or opens command window."""
        detected_prompts = []

        def on_wake(prompt):
            detected_prompts.append(prompt)

        listener = SpeechListener(on_wake_detected=on_wake)
        listener.last_cooldown_until = 0.0
        listener.is_tts_speaking = False

        # 1. Wake phrase alone -> empty prompt, starts wake session
        listener._process_recognized_phrase("Hey Adam")
        self.assertEqual(len(detected_prompts), 1)
        self.assertEqual(detected_prompts[0], "")
        self.assertGreater(listener._wake_session_until, time.time())

        # 2. Subsequent command within session window
        listener._process_recognized_phrase("what time is it")
        self.assertEqual(len(detected_prompts), 2)
        self.assertEqual(detected_prompts[1], "what time is it")
        self.assertEqual(listener._wake_session_until, 0.0)

        # 3. One-shot command
        listener._process_recognized_phrase("Hey Adam, open WhatsApp")
        self.assertEqual(len(detected_prompts), 3)
        self.assertEqual(detected_prompts[2], "open whatsapp")


if __name__ == "__main__":
    unittest.main()
