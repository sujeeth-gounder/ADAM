"""Unit and Integration Tests for ADAM Personal Memory & Identity Subsystem."""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.identity import (
    ASSISTANT_NAME,
    ASSISTANT_FULL_NAME,
    ASSISTANT_CREATOR,
    get_assistant_name,
    get_assistant_identity_response,
    get_creator_response,
    get_user_name_response,
)
from app.memory import MemoryManager
from app.storage import LocalStore
from app.integrations import SystemIntegrations


class TestIdentityConstants(unittest.TestCase):
    def test_canonical_identity(self):
        self.assertEqual(get_assistant_name(), "ADAM")
        self.assertEqual(ASSISTANT_FULL_NAME, "Advanced Desktop AI Mate")
        self.assertEqual(ASSISTANT_CREATOR, "Sujeeth Kani")
        self.assertIn("Sujeeth Kani", get_creator_response())
        self.assertIn("ADAM", get_assistant_identity_response())

    def test_user_name_response(self):
        resp = get_user_name_response("Sujith")
        self.assertIn("Sujith", resp)
        self.assertNotIn("ADAM", resp)

        resp_unknown = get_user_name_response(None)
        self.assertIn("I don't know your name yet", resp_unknown)


class TestPersonalMemoryStorageAndManager(unittest.TestCase):
    def setUp(self):
        # Use an isolated test database
        self.test_db = ROOT / "data" / "test_memory.db"
        if self.test_db.exists():
            self.test_db.unlink()
        self.store = LocalStore(db_path=self.test_db)
        self.memory = MemoryManager(store=self.store)

    def tearDown(self):
        if self.test_db.exists():
            self.test_db.unlink()

    def test_set_and_get_user_name(self):
        handled, msg = self.memory.try_handle_memory_command("My name is Sujith")
        self.assertTrue(handled)
        self.assertIn("Sujith", msg)
        self.assertEqual(self.memory.get_user_name(), "Sujith")

        # Now ask: What is my name?
        handled, msg = self.memory.try_handle_memory_command("What is my name?")
        self.assertTrue(handled)
        self.assertIn("Sujith", msg)
        self.assertNotIn("My name is Adam", msg)

    def test_update_user_name(self):
        self.memory.try_handle_memory_command("My name is Sujith")
        self.assertEqual(self.memory.get_user_name(), "Sujith")

        handled, msg = self.memory.try_handle_memory_command("Update my name to John")
        self.assertTrue(handled)
        self.assertEqual(self.memory.get_user_name(), "John")

    def test_call_me_name_intent(self):
        handled, msg = self.memory.try_handle_memory_command("Call me Alex")
        self.assertTrue(handled)
        self.assertEqual(self.memory.get_user_name(), "Alex")

    def test_assistant_identity_intents(self):
        handled, msg = self.memory.try_handle_memory_command("What is your name?")
        self.assertTrue(handled)
        self.assertIn("ADAM", msg)

        handled, msg = self.memory.try_handle_memory_command("Who are you?")
        self.assertTrue(handled)
        self.assertIn("ADAM", msg)

        handled, msg = self.memory.try_handle_memory_command("Who created you?")
        self.assertTrue(handled)
        self.assertIn("Sujeeth Kani", msg)

    def test_remember_fact_and_recall(self):
        handled, msg = self.memory.try_handle_memory_command("Remember that I prefer dark mode and coffee")
        self.assertTrue(handled)
        self.assertIn("Noted", msg)

        # What do you remember about me?
        handled, msg = self.memory.try_handle_memory_command("What do you remember about me?")
        self.assertTrue(handled)
        self.assertIn("dark mode and coffee", msg)

    def test_forget_memory(self):
        self.memory.try_handle_memory_command("My name is Sujith")
        self.assertEqual(self.memory.get_user_name(), "Sujith")

        handled, msg = self.memory.try_handle_memory_command("Forget my name")
        self.assertTrue(handled)
        self.assertIsNone(self.memory.get_user_name())

    def test_sensitive_credential_block(self):
        handled, msg = self.memory.try_handle_memory_command("Remember that my password is SecretPassword123")
        self.assertTrue(handled)
        self.assertIn("security", msg.lower())
        self.assertNotIn("SecretPassword123", str(self.store.list_memories()))

        handled, msg = self.memory.try_handle_memory_command("Remember my api key is sk-1234567890abcdef")
        self.assertTrue(handled)
        self.assertIn("security", msg.lower())

    def test_memory_context_prompt_injection(self):
        self.memory.try_handle_memory_command("My name is Sujith")
        self.memory.try_handle_memory_command("Remember that I am a Python developer")

        prompt_block = self.memory.get_memory_context_prompt()
        self.assertIn("Stored User Facts & Preferences", prompt_block)
        self.assertIn("Sujith", prompt_block)
        self.assertIn("Python developer", prompt_block)

    def test_system_integrations_routing(self):
        integrations = SystemIntegrations()
        integrations.memory = self.memory

        handled, resp, _ = integrations.try_handle_command("My name is Sujith")
        self.assertTrue(handled)
        self.assertIn("Sujith", resp)

        handled, resp, _ = integrations.try_handle_command("What is my name?")
        self.assertTrue(handled)
        self.assertIn("Sujith", resp)

        handled, resp, _ = integrations.try_handle_command("Who are you?")
        self.assertTrue(handled)
        self.assertIn("ADAM", resp)


if __name__ == "__main__":
    unittest.main()
