"""Regression tests for bounded voice-command execution."""

import json
import queue
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from app.app_registry import AppRegistry
from app.integrations import OllamaClient, format_current_time
from app.main import AdamApp
from app.storage import LocalStore


class VoiceCommandSafetyTests(unittest.TestCase):
    def test_duplicate_request_id_executes_only_once(self):
        app = AdamApp.__new__(AdamApp)
        app._request_lock = threading.Lock()
        app._seen_request_ids = set()
        app._request_id_order = []
        app._active_request_id = None
        app._active_request_cancel = None
        app.voice_tts = Mock()
        app.input_entry = Mock()
        app.ui_queue = Mock()
        app._add_message_card = Mock()
        app.set_voice_state = Mock()
        app.integrations = Mock()
        app.integrations.try_handle_command.return_value = (True, "Opened one search.", None)

        app._submit_command("Play Vaadi Pulla Vaadi", "voice", "utterance-1")
        app._submit_command("Play Vaadi Pulla Vaadi", "voice", "utterance-1")

        app.integrations.try_handle_command.assert_called_once_with("Play Vaadi Pulla Vaadi")
        self.assertEqual(app._add_message_card.call_count, 1)
        app.ui_queue.put.assert_called_once()

    def _make_task_submission_app(self, database_path):
        app = AdamApp.__new__(AdamApp)
        app._request_lock = threading.Lock()
        app._seen_request_ids = set()
        app._request_id_order = []
        app._active_request_id = None
        app._active_request_cancel = None
        app.voice_tts = Mock()
        app.voice_tts.is_speaking.return_value = False
        app.input_entry = Mock()
        app.ui_queue = queue.Queue()
        app._add_message_card = Mock()
        app.set_voice_state = Mock()
        app.integrations = Mock()
        app.local_store = LocalStore(database_path)
        app._tasks_panel_refresh = Mock()
        app.after = Mock()
        return app

    def test_task_command_persists_and_refreshes_tasks_page(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app = self._make_task_submission_app(Path(temp_dir) / "adam.db")
            app._submit_command("Assign a task to call the dentist", "text", "task-request-1")
            app._process_ui_queue()

            self.assertEqual(
                [task["title"] for task in app.local_store.list_tasks()],
                ["call the dentist"],
            )
            app._tasks_panel_refresh.assert_called_once()
            app.integrations.try_handle_command.assert_not_called()
            self.assertTrue(
                any(
                    call.args[0] == "ADAM"
                    and "Task has been assigned: call the dentist." in call.args[1]
                    for call in app._add_message_card.call_args_list
                )
            )
            app.voice_tts.speak.assert_called_once_with("Task has been assigned: call the dentist.")

    def test_task_save_failure_never_reports_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app = self._make_task_submission_app(Path(temp_dir) / "adam.db")
            app.local_store.add_task = Mock(side_effect=OSError("database unavailable"))

            app._submit_command("Create a task to call the dentist", "text", "task-request-fail")
            app._process_ui_queue()

            spoken = app.voice_tts.speak.call_args.args[0]
            self.assertIn("couldn't save", spoken.lower())
            self.assertNotIn("Task has been assigned", spoken)
            app._tasks_panel_refresh.assert_not_called()

    def test_task_request_without_title_does_not_fall_through_to_ollama(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app = self._make_task_submission_app(Path(temp_dir) / "adam.db")

            app._submit_command("Create a task", "text", "task-request-empty")
            app._process_ui_queue()

            self.assertEqual(app.local_store.list_tasks(), [])
            app.integrations.try_handle_command.assert_not_called()
            self.assertIn("provide a task title", app.voice_tts.speak.call_args.args[0].lower())

    def test_time_ui_and_tts_receive_same_response(self):
        response = (
            f"The current time is "
            f"{format_current_time(datetime.now(ZoneInfo('Asia/Kolkata')))}."
        )
        app = AdamApp.__new__(AdamApp)
        app.ui_queue = queue.Queue()
        app.ui_queue.put(("DIRECT_RESPONSE", (response, None, "time-1")))
        app._active_request_id = "time-1"
        app._request_lock = threading.Lock()
        app._active_request_cancel = None
        app._add_message_card = Mock()
        app.voice_tts = Mock()
        app.after = Mock()

        app._process_ui_queue()

        app._add_message_card.assert_called_once_with("ADAM", response)
        app.voice_tts.speak.assert_called_once_with(response)

    def test_music_opens_one_search_without_rewriting_title(self):
        registry = AppRegistry()
        with patch.object(registry, "open_url", return_value=(True, "opened")) as open_url:
            success, message = registry.play_music("Vaadi Pulla Vaadi")

        self.assertTrue(success)
        self.assertIn("one YouTube search", message)
        open_url.assert_called_once()
        self.assertEqual(
            open_url.call_args.args[0],
            "https://www.youtube.com/results?search_query=Vaadi+Pulla+Vaadi",
        )

    def test_cancelled_ollama_stream_stops_without_fallback_retry(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                return iter(
                    [
                        json.dumps({"message": {"content": "English response"}, "done": False}).encode() + b"\n",
                        json.dumps({"message": {"content": "should not appear"}, "done": False}).encode() + b"\n",
                    ]
                )

        client = OllamaClient.__new__(OllamaClient)
        client.host = "http://localhost:11434"
        client.model = "test-model"
        event = threading.Event()
        fake_response = FakeResponse()

        with patch.object(client, "check_connection", return_value=True), patch(
            "urllib.request.urlopen", return_value=fake_response
        ) as open_url:
            response = client.generate_stream("hello", cancel_event=event)
            self.assertEqual(next(response), "English response")
            event.set()
            self.assertEqual(list(response), [])

        open_url.assert_called_once()


if __name__ == "__main__":
    unittest.main()
