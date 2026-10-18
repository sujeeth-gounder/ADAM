"""Unit tests for ADAM Email Service foundation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.email_service import GmailClient, EmailMessage


class TestGmailClient(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_path = Path(self.temp_dir.name)
        self.keyring_data = {}
        import keyring
        self._keyring_get = patch(
            "keyring.get_password",
            side_effect=lambda service, username: self.keyring_data.get((service, username)),
        )
        self._keyring_set = patch(
            "keyring.set_password",
            side_effect=lambda service, username, password: self.keyring_data.__setitem__((service, username), password),
        )
        self._keyring_delete = patch(
            "keyring.delete_password",
            side_effect=lambda service, username: self.keyring_data.pop((service, username), None),
        )
        self._keyring_get.start()
        self._keyring_set.start()
        self._keyring_delete.start()
        self.client = GmailClient(data_dir=self.data_path)

    def tearDown(self):
        self._keyring_get.stop()
        self._keyring_set.stop()
        self._keyring_delete.stop()
        self.temp_dir.cleanup()

    def test_initial_auth_status(self):
        status = self.client.get_auth_status()
        self.assertFalse(status["authorized"])
        self.assertIn("not connected", status["status_message"])

    def test_save_and_load_credentials(self):
        cred = {
            "access_token": "mock_access_token_12345",
            "refresh_token": "mock_refresh_token_67890",
            "client_id": "test_client_id.apps.googleusercontent.com",
            "client_secret": "test_secret_abc",
            "user_email": "user@gmail.com"
        }
        self.client.save_credentials(cred)
        self.assertTrue(self.client.is_authorized)
        self.assertEqual(self.client.user_email, "user@gmail.com")

        # Reload from newly created file
        new_client = GmailClient(data_dir=self.data_path)
        self.assertTrue(new_client.is_authorized)
        self.assertEqual(new_client.access_token, "mock_access_token_12345")
        self.assertEqual(new_client.user_email, "user@gmail.com")

    def test_generate_auth_url(self):
        url = self.client.generate_auth_url("my_client_id", "my_secret", "http://localhost:8080/")
        self.assertTrue(url.startswith("https://accounts.google.com/o/oauth2/v2/auth"))
        self.assertIn("client_id=my_client_id", url)
        self.assertIn("scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fgmail.readonly", url)

    def test_paginated_message_iteration(self):
        # Mock _api_get to simulate multiple pages
        def mock_api_get(endpoint, params=None):
            if endpoint == "messages":
                page = (params or {}).get("pageToken")
                if not page:
                    return {
                        "messages": [{"id": "msg_1", "threadId": "t1"}, {"id": "msg_2", "threadId": "t2"}],
                        "nextPageToken": "page_2_token"
                    }
                elif page == "page_2_token":
                    return {
                        "messages": [{"id": "msg_3", "threadId": "t3"}],
                        "nextPageToken": None
                    }
            return {}

        self.client.is_authorized = True
        self.client.access_token = "valid_token"
        with patch.object(self.client, "_api_get", side_effect=mock_api_get):
            msgs = list(self.client.iter_all_messages(limit=10))
            self.assertEqual(len(msgs), 3)
            self.assertEqual(msgs[0]["id"], "msg_1")
            self.assertEqual(msgs[2]["id"], "msg_3")

    def test_extract_message_detail(self):
        # Mock message detail response with Base64 payload
        import base64
        sample_text = "Important hackathon meeting tomorrow at 10 AM."
        encoded_data = base64.urlsafe_b64encode(sample_text.encode("utf-8")).decode("utf-8")

        mock_resp = {
            "id": "18a1b2c3d4",
            "threadId": "thread_xyz",
            "labelIds": ["INBOX", "UNREAD", "IMPORTANT"],
            "snippet": "Important hackathon meeting...",
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "Subject", "value": "Hackathon Schedule"},
                    {"name": "From", "value": "organizer@hackathon.org"},
                    {"name": "To", "value": "user@gmail.com"},
                    {"name": "Date", "value": "Fri, 16 Oct 2026 10:00:00 +0000"}
                ],
                "body": {"data": encoded_data}
            }
        }

        self.client.is_authorized = True
        self.client.access_token = "valid_token"
        with patch.object(self.client, "_api_get", return_value=mock_resp):
            msg = self.client.get_message_detail("18a1b2c3d4")
            self.assertIsNotNone(msg)
            self.assertEqual(msg.subject, "Hackathon Schedule")
            self.assertEqual(msg.sender, "organizer@hackathon.org")
            self.assertTrue(msg.is_unread)
            self.assertIn("Important hackathon meeting", msg.body_text)


if __name__ == "__main__":
    unittest.main()
