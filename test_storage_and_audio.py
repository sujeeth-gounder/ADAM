"""Tests for local persistence and audio conversion."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from app.speech import FasterWhisperTranscriber, record_diagnostic_sample, resample_to_16k_mono
from app.storage import LocalStore


class LocalStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "adam.db"
        self.store = LocalStore(self.database_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_messages_and_tasks_persist_between_instances(self):
        self.store.add_message("USER", "What is the time now?", "voice")
        self.store.add_message("ADAM", "It is 10:00 AM.")
        task_id = self.store.add_task("Review the meeting notes")
        self.store.set_task_completed(task_id, True)

        restarted_store = LocalStore(self.database_path)
        messages = restarted_store.list_messages()
        self.assertEqual([row["role"] for row in messages], ["USER", "ADAM"])
        self.assertEqual(messages[0]["source"], "voice")
        self.assertEqual(messages[0]["content"], "What is the time now?")
        tasks = restarted_store.list_tasks()
        self.assertEqual(tasks[0]["title"], "Review the meeting notes")
        self.assertEqual(tasks[0]["completed"], 1)

        restarted_store.clear_history()
        self.assertEqual(restarted_store.list_messages(), [])
        self.assertEqual(len(restarted_store.list_tasks()), 1)

    def test_empty_task_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.add_task("  ")

    def test_task_request_id_is_idempotent_and_survives_store_reopen(self):
        first_id = self.store.add_task("  Review   the meeting notes  ", "request-123")
        repeated_id = self.store.add_task("Review the meeting notes", "request-123")
        self.assertEqual(first_id, repeated_id)

        reopened = LocalStore(self.database_path)
        tasks = reopened.list_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["id"], first_id)
        self.assertEqual(tasks[0]["title"], "Review the meeting notes")

    def test_task_title_length_is_validated_before_write(self):
        with self.assertRaises(ValueError):
            self.store.add_task("x" * 201)
        self.assertEqual(self.store.list_tasks(), [])


class AudioConversionTests(unittest.TestCase):
    def test_diagnostic_capture_requests_signed_16_bit_pcm(self):
        stream = MagicMock()
        stream.read.return_value = b"\x00\x00" * 1024
        pyaudio = MagicMock()
        pyaudio.get_format_from_width.return_value = 8
        pyaudio.get_device_info_by_index.return_value = {
            "name": "Test microphone",
            "defaultSampleRate": 16000,
            "maxInputChannels": 1,
        }
        pyaudio.open.return_value = stream

        with patch("app.speech.get_global_pyaudio", return_value=pyaudio):
            result = record_diagnostic_sample(device_index=0, duration_sec=0.07)

        self.assertTrue(result["success"], result["message"])
        self.assertEqual(pyaudio.open.call_args.kwargs["format"], 8)
        self.assertEqual(result["samples_count"], 1024)
        self.assertEqual(result["duration_sec"], 0.06)
        self.assertEqual(len(result["raw_pcm"]), 1024 * 2)

    def test_resample_downmixes_stereo_and_keeps_duration(self):
        rate = 44100
        time = np.arange(rate, dtype=np.float32) / rate
        left = np.sin(2 * np.pi * 440 * time) * 12000
        right = np.sin(2 * np.pi * 440 * time) * 6000
        pcm = np.column_stack((left, right)).astype("<i2").tobytes()

        converted = resample_to_16k_mono(pcm, in_rate=rate, in_channels=2)
        samples = np.frombuffer(converted, dtype="<i2")
        self.assertEqual(len(samples), 16000)
        self.assertGreater(float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))), 1000)

    def test_resampling_applies_anti_alias_filter(self):
        rate = 44100
        time = np.arange(rate, dtype=np.float32) / rate
        above_target_nyquist = (np.sin(2 * np.pi * 10000 * time) * 12000).astype("<i2")
        converted = resample_to_16k_mono(above_target_nyquist.tobytes(), rate, 1)
        samples = np.frombuffer(converted, dtype="<i2").astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        self.assertLess(rms, 120)

    def test_silence_never_produces_a_transcript(self):
        transcriber = FasterWhisperTranscriber()
        transcriber.is_loaded = True
        transcriber.model = object()
        self.assertIsNone(transcriber.transcribe_raw_pcm(b"\x00" * 32000))


if __name__ == "__main__":
    unittest.main()
