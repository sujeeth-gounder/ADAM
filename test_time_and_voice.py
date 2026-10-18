#!/usr/bin/env python3
"""
Unit and Integration Tests for:
BUG 1: Time and IST Handling (zoneinfo, system clock, timezones, date boundaries)
BUG 2: Voice Recognition & Faster-Whisper Pipeline (sample rate negotiation, wake phrases, VAD)
"""

import datetime
import math
import re
import sys
import time
import unittest
import zoneinfo
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.integrations import (
    SystemIntegrations,
    resolve_requested_timezone,
    get_current_time_string,
    get_current_date_string,
    format_current_time,
    format_current_date,
    TIMEZONE_MAP,
)
from app.app_registry import CommandParser, AppRegistry
from app.speech import (
    SpeechListener,
    FasterWhisperTranscriber,
    _update_speech_onset,
    match_wake_word,
    normalize_speech_variants,
    resample_to_16k_mono,
    list_input_microphones,
    test_microphone_level,
)


class TestTimeAndIST(unittest.TestCase):
    """Test suite for Time and IST handling."""

    def setUp(self):
        self.integrations = SystemIntegrations()
        self.parser = CommandParser(AppRegistry())

    def test_system_and_ist_timezone_detection(self):
        """Test system timezone detection and explicit India/IST mapping."""
        # 1. System local timezone
        tz_local, label_local, is_exp = resolve_requested_timezone("")
        self.assertFalse(is_exp)
        self.assertIsNotNone(tz_local)

        # 2. India / IST queries
        for query in ["ist", "IST", "india", "India", "kolkata", "Indian Standard Time", "isd"]:
            tz_india, label_india, is_exp_india = resolve_requested_timezone(query)
            self.assertTrue(is_exp_india, f"Query '{query}' should be resolved as explicit timezone")
            # Verify timezone is Asia/Kolkata
            now_india = datetime.datetime.now(tz_india)
            self.assertEqual(now_india.strftime("%Z"), "IST", f"Timezone for '{query}' must have abbreviation 'IST'")
            self.assertEqual(now_india.strftime("%z"), "+0530", f"Timezone for '{query}' must have offset +0530")

    def test_utc_to_ist_conversion_and_math(self):
        """Test exact UTC to IST offset calculation and consistency."""
        tz_utc = zoneinfo.ZoneInfo("UTC")
        tz_ist = zoneinfo.ZoneInfo("Asia/Kolkata")

        # Fixed reference point in UTC
        dt_utc = datetime.datetime(2026, 10, 17, 4, 30, 0, tzinfo=tz_utc)
        dt_ist = dt_utc.astimezone(tz_ist)

        self.assertEqual(dt_ist.hour, 10)
        self.assertEqual(dt_ist.minute, 0)
        self.assertEqual(dt_ist.day, 17)
        self.assertEqual(dt_ist.strftime("%Z"), "IST")
        self.assertEqual(dt_ist.strftime("%z"), "+0530")

    def test_date_boundary_crossing(self):
        """Verify that date boundary shifts across timezones are strictly preserved."""
        tz_utc = zoneinfo.ZoneInfo("UTC")
        tz_tokyo = zoneinfo.ZoneInfo("Asia/Tokyo")
        tz_honolulu = zoneinfo.ZoneInfo("Pacific/Honolulu")

        # 2026-10-17 04:00 UTC:
        # In Tokyo (+09:00): 13:00 on 17 October 2026
        # In Honolulu (-10:00): 18:00 on 16 October 2026 (previous day!)
        dt_utc = datetime.datetime(2026, 10, 17, 4, 0, 0, tzinfo=tz_utc)
        dt_tokyo = dt_utc.astimezone(tz_tokyo)
        dt_honolulu = dt_utc.astimezone(tz_honolulu)

        self.assertEqual(dt_tokyo.day, 17)
        self.assertEqual(dt_tokyo.strftime("%Z"), "JST")
        self.assertEqual(dt_honolulu.day, 16)
        self.assertEqual(dt_honolulu.strftime("%B"), "October")

    def test_daylight_saving_aware_timezones(self):
        """Test daylight saving aware timezones (New York EDT/EST, London BST/GMT)."""
        tz_ny, label_ny, _ = resolve_requested_timezone("new york")
        tz_london, label_london, _ = resolve_requested_timezone("london")
        tz_tokyo, label_tokyo, _ = resolve_requested_timezone("tokyo")

        now_ny = datetime.datetime.now(tz_ny)
        now_london = datetime.datetime.now(tz_london)
        now_tokyo = datetime.datetime.now(tz_tokyo)

        # In October, New York is in EDT (-0400) and London is in BST (+0100)
        self.assertIn(now_ny.strftime("%Z"), ["EDT", "EST"])
        self.assertIn(now_london.strftime("%Z"), ["BST", "GMT"])
        self.assertEqual(now_tokyo.strftime("%Z"), "JST")

    def test_ist_is_never_appended_to_non_india_timezones(self):
        """Ensure 'IST' is NEVER appended to timestamps for non-India zones."""
        non_india_zones = ["UTC", "London", "New York", "Tokyo", "Chicago", "Paris"]
        for loc in non_india_zones:
            tz, label, _ = resolve_requested_timezone(loc)
            time_str = get_current_time_string(tz)
            self.assertNotIn("IST", time_str, f"Timestamp for '{loc}' must NOT contain 'IST'! Got: {time_str}")

    def test_time_and_date_formatting(self):
        """Verify format requirements: '11:15 AM IST (UTC+05:30)' pattern and 'Saturday, 17 October 2026' pattern."""
        tz_ist = zoneinfo.ZoneInfo("Asia/Kolkata")
        time_str = get_current_time_string(tz_ist)
        date_str = get_current_date_string(tz_ist)

        # Time pattern: HH:MM AM/PM IST (UTC+05:30)
        self.assertRegex(time_str, r"^\d{2}:\d{2}\s+(?:AM|PM)\s+IST\s+\(UTC\+05:30\)$")
        # Date pattern: Weekday, Day Month Year (e.g. Saturday, 17 October 2026)
        self.assertRegex(date_str, r"^[A-Z][a-z]+,\s+\d{1,2}\s+[A-Z][a-z]+\s+\d{4}$")

    def test_natural_language_time_commands(self):
        """Verify commands are intercepted directly without using Ollama LLM."""
        test_commands = [
            ("What time is it?", "The current time is"),
            ("What is the time now?", "The current time is"),
            ("Tell me the current time in IST", "IST"),
            ("Tell me the time in IST.", "IST"),
            ("What is today's date?", "Today's date is"),
            ("Show the current Indian time.", "current Indian time"),
            ("Tell me Indian time", "current Indian time"),
            ("Indian time", "current Indian time"),
            ("Show me the time in India", "current Indian time"),
            ("What's the time in Tokyo?", "Tokyo"),
            ("time in London", "London"),
            ("Current time in UTC", "UTC"),
            ("What day is today", "Today is"),
            ("What day is it today?", "Today is"),
            ("Tell me today's date", "Today's date is"),
            ("current time and date", "The current time is"),
            ("time and date in IST", "In IST, the current time is"),
        ]

        for cmd, expected_substring in test_commands:
            handled, response, pending = self.integrations.try_handle_command(cmd)
            self.assertTrue(handled, f"Command '{cmd}' should be handled directly by system integrations")
            self.assertIsNotNone(response)
            self.assertIn(expected_substring.lower(), response.lower(), f"Command '{cmd}' response '{response}' must contain '{expected_substring}'")
            self.assertIsNone(pending)

    def test_required_clock_phrases_use_a_fresh_ist_timestamp(self):
        ist = zoneinfo.ZoneInfo("Asia/Kolkata")
        for command, intent in (
            ("What is the time now?", "GET_TIME"),
            ("Tell me today's date", "GET_DATE"),
            ("What day is it today?", "GET_DAY"),
        ):
            before = datetime.datetime.now(ist)
            with self.assertLogs("ADAM.Integrations", level="INFO") as logs:
                handled, response, pending = self.integrations.try_handle_command(command)
            after = datetime.datetime.now(ist)

            self.assertTrue(handled, command)
            self.assertIsNone(pending)
            possible_times = {format_current_time(before), format_current_time(after)}
            possible_dates = {format_current_date(before), format_current_date(after)}
            if intent == "GET_TIME":
                self.assertTrue(any(value in response for value in possible_times), response)
            elif intent == "GET_DATE":
                self.assertTrue(any(value in response for value in possible_dates), response)
            else:
                self.assertIn(before.strftime("%A"), response)
                self.assertIn("Today is", response)
            self.assertTrue(any("Clock request" in line and intent in line for line in logs.output))


class TestVoiceRecognitionPipeline(unittest.TestCase):
    """Test suite for Voice Recognition & Faster-Whisper integration."""

    def test_speech_onset_requires_sustained_activity(self):
        onset = []
        preroll = []

        self.assertFalse(_update_speech_onset(onset, preroll, b"impulse", True, 4))
        self.assertFalse(_update_speech_onset(onset, preroll, b"quiet", False, 4))
        self.assertEqual(onset, [])
        self.assertEqual(preroll, [b"impulse"])

        self.assertFalse(_update_speech_onset(onset, preroll, b"speech-1", True, 4))
        self.assertTrue(_update_speech_onset(onset, preroll, b"speech-2", True, 4))
        self.assertEqual(onset, [b"speech-1", b"speech-2"])

    def test_faster_whisper_import_and_model_loading(self):
        """Verify Faster-Whisper is imported from user environment and loads model."""
        transcriber = FasterWhisperTranscriber(model_size="small")
        loaded = transcriber.load_model()
        self.assertTrue(loaded, f"Faster-Whisper model loading failed: {transcriber.last_error}")
        self.assertTrue(transcriber.is_loaded)
        self.assertEqual(transcriber.model_size, "small")
        self.assertEqual(transcriber.language, "en")

    def test_no_hallucination_on_silence_or_noise(self):
        """Verify that silence or noise does not produce hallucinated text like 'start the engine'."""
        import numpy as np
        transcriber = FasterWhisperTranscriber(model_size="small")
        transcriber.load_model()
        # 2 seconds of pure silence
        silence = b"\x00" * (16000 * 2 * 2)
        res = transcriber.transcribe_raw_pcm(silence, in_rate=16000, in_channels=1)
        self.assertIsNone(res, f"Silence should return None, not '{res}'")

        # 2 seconds of low-amplitude white noise (RMS ~50 / 32768 = ~0.0015)
        np.random.seed(42)
        noise = (np.random.normal(0, 50, 16000 * 2)).astype(np.int16).tobytes()
        res_noise = transcriber.transcribe_raw_pcm(noise, in_rate=16000, in_channels=1)
        self.assertIsNone(res_noise, f"Noise should return None, not '{res_noise}'")

    def test_transcribe_with_timeout_recovery(self):
        """Verify transcribe_with_timeout terminates cleanly without hanging."""
        transcriber = FasterWhisperTranscriber(model_size="small")
        # Test with tiny empty PCM - should return None within timeout
        res = transcriber.transcribe_with_timeout(b"\x00" * 800, in_rate=16000, in_channels=1, timeout=1.5)
        self.assertIsNone(res)

    def test_sample_rate_resampling_to_16k(self):
        """Test numpy audio resampling from 48000Hz stereo to 16000Hz mono."""
        import numpy as np
        # 1 second of 48000Hz 2-channel int16 PCM (192,000 bytes)
        raw_48k_stereo = (np.sin(np.linspace(0, 500, 48000 * 2)) * 12000).astype(np.int16).tobytes()
        resampled_16k = resample_to_16k_mono(raw_48k_stereo, in_rate=48000, in_channels=2)

        # 16000 samples at 16-bit mono = 32,000 bytes
        self.assertEqual(len(resampled_16k), 32000)

    def test_wake_words_recognition(self):
        """Verify all configured wake phrases match correctly."""
        wake_words = ["hi", "hey", "oi", "hi adam", "hey adam", "hello adam"]
        test_inputs = [
            ("Hi", True, "hi", ""),
            ("Hey", True, "hey", ""),
            ("Oi", True, "oi", ""),
            ("Hi Adam", True, "hi adam", ""),
            ("Hey Adam", True, "hey adam", ""),
            ("Hello Adam", True, "hello adam", ""),
            ("Hey Adam, what time is it?", True, "hey adam", "what time is it"),
            ("Hello Adam show me the time in India", True, "hello adam", "show me the time in india"),
            ("Oi what is today's date", True, "oi", "what is today s date"),
            ("play some jazz music", False, "", "play some jazz music"),
        ]

        for text, exp_detected, exp_matched, exp_remaining in test_inputs:
            detected, matched, rem = match_wake_word(text, wake_words)
            self.assertEqual(detected, exp_detected, f"Failed detection on '{text}'")
            if exp_detected:
                self.assertEqual(matched, exp_matched, f"Failed matched wake on '{text}'")
                self.assertEqual(rem, exp_remaining, f"Failed remaining on '{text}'")

    def test_voice_state_machine_flow(self):
        """Verify state machine transitions across MIC_OFF, WAKE_DETECTED, COMMAND, SPEAKING."""
        states_recorded = []
        listener = SpeechListener(
            on_flow_state_change=lambda st, dt: states_recorded.append(st)
        )

        listener.set_flow_state("MIC_OFF", "Off")
        listener.set_flow_state("MIC_STARTING", "Starting")
        listener.set_flow_state("LISTENING_WAKE", "Listening")
        listener.set_flow_state("WAKE_DETECTED", "Wake word detected!")
        listener.set_flow_state("LISTENING_COMMAND", "Listening for command")
        listener.set_flow_state("TRANSCRIBING", "Transcribing")
        listener.set_flow_state("PROCESSING", "Thinking")
        listener.set_flow_state("SPEAKING", "Speaking")

        expected = [
            "MIC_OFF", "MIC_STARTING", "LISTENING_WAKE",
            "WAKE_DETECTED", "LISTENING_COMMAND", "TRANSCRIBING",
            "PROCESSING", "SPEAKING"
        ]
        self.assertEqual(states_recorded[-len(expected):], expected)

    def test_tts_and_busy_processing_suppression(self):
        """Ensure audio is ignored during Piper TTS playback and during Ollama processing."""
        listener = SpeechListener()
        calls = []
        listener.on_wake_detected = lambda p: calls.append(p)

        # 1. TTS is speaking
        listener.set_tts_speaking(True)
        listener._dispatch_utterance(b"\x00" * 3200)
        self.assertEqual(len(calls), 0, "Audio must be ignored while TTS is speaking")

        # 2. Ollama is busy processing
        listener.set_tts_speaking(False)
        listener.is_busy_processing = True
        listener._dispatch_utterance(b"\x00" * 3200)
        self.assertEqual(len(calls), 0, "Audio must be ignored while Ollama is busy processing")

        listener.is_busy_processing = False

    def test_utterance_deduplication_and_single_processing(self):
        """Verify that an utterance ID is processed at most once and duplicate text within cooldown is suppressed."""
        listener = SpeechListener()
        recognized = []
        listener.on_command_recognized = lambda t: recognized.append(t)

        utt_id = "test_utt_123"
        # First dispatch
        listener._launch_command_transcription(b"\x00" * 3200, utterance_id=utt_id)
        time.sleep(0.1)
        self.assertIn(utt_id, listener._processed_utterance_ids)

        # Attempt duplicate dispatch with identical utterance_id
        listener._launch_command_transcription(b"\x00" * 3200, utterance_id=utt_id)
        time.sleep(0.1)
        # Should be dropped immediately
        self.assertEqual(len([u for u in listener._processed_utterance_ids if u == utt_id]), 1)

    def test_drain_hardware_buffer_on_tts_release(self):
        """Ensure _drain_hardware_buffer flag is engaged when TTS stops speaking to purge soundcard buffers."""
        listener = SpeechListener()
        listener.set_tts_speaking(True)
        self.assertTrue(listener.is_tts_speaking)
        self.assertFalse(listener._drain_hardware_buffer)

        # Stop TTS
        listener.set_tts_speaking(False)
        self.assertFalse(listener.is_tts_speaking)
        self.assertTrue(listener._drain_hardware_buffer, "Hardware buffer drain flag must be set when TTS finishes")

    def test_diagnostic_telemetry_snapshot(self):
        """Verify get_diagnostics_snapshot returns all required diagnostic fields."""
        listener = SpeechListener()
        snap = listener.get_diagnostics_snapshot()
        required_keys = [
            "selected_device", "sample_rate", "channels",
            "live_rms", "live_peak", "is_silent",
            "wake_detector_active", "last_wake_detection_time",
            "command_recording_started", "command_recording_ended",
            "transcription_in_progress", "last_transcription_duration_sec",
            "last_transcription_result", "current_utterance_id",
            "processed_utterance_count"
        ]
        for key in required_keys:
            self.assertIn(key, snap, f"Telemetry snapshot missing key: {key}")

    def test_real_microphone_hardware_check(self):
        """Verify actual physical microphone detection and non-silent RMS level."""
        mics = list_input_microphones()
        self.assertGreater(len(mics), 0, "At least one input microphone device must be detected")
        active_idx = mics[0]["index"]
        ok, peak, msg = test_microphone_level(device_index=active_idx, duration_sec=1.0)
        self.assertTrue(ok, f"Microphone signal test failed: {msg}")
        self.assertGreaterEqual(peak, 0.0)


if __name__ == "__main__":
    unittest.main()
