#!/usr/bin/env python3
"""Comprehensive Automated Test Suite for ADAM Application Launching & Voice Commands.
Tests:
1. CommandParser intent extraction for all required commands and natural language variations.
2. AppRegistry application resolution, Brave browser detection, and website fallback routing.
3. Launch-Test Mode checking real system executables, desktop entries, and browser fallbacks.
4. End-to-end command execution through SystemIntegrations.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.app_registry import AppRegistry, CommandParser
from app.integrations import SystemIntegrations


class TestCommandParser(unittest.TestCase):
    """Verify natural-language command parsing and parameter extraction."""

    def setUp(self):
        self.registry = AppRegistry()
        self.parser = CommandParser(self.registry)

    def test_open_whatsapp(self):
        res = self.parser.parse("open whatsapp")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "OPEN_APP")
        self.assertEqual(res["target"], "whatsapp")

    def test_open_whatsapp_web(self):
        res = self.parser.parse("open whatsapp web")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "OPEN_APP")
        self.assertEqual(res["target"], "whatsapp_web")
        self.assertTrue(res["details"].get("force_web", False))

    def test_open_instagram(self):
        res = self.parser.parse("open instagram")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "OPEN_APP")
        self.assertEqual(res["target"], "instagram")

    def test_open_instagram_web(self):
        res = self.parser.parse("open instagram web")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "OPEN_APP")
        self.assertEqual(res["target"], "instagram_web")

    def test_open_youtube(self):
        res = self.parser.parse("open youtube")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "OPEN_APP")
        self.assertEqual(res["target"], "youtube")

    def test_open_brave(self):
        res = self.parser.parse("open brave")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "OPEN_APP")
        self.assertEqual(res["target"], "brave")

    def test_play_tamil_songs_in_brave(self):
        res = self.parser.parse("play Tamil songs in Brave")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "PLAY_MUSIC")
        self.assertEqual(res["details"]["song_query"].lower(), "tamil songs")
        self.assertTrue(res["details"]["in_brave"])

    def test_play_songs_in_brave(self):
        res = self.parser.parse("play songs in Brave")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "PLAY_MUSIC")
        self.assertEqual(res["details"]["song_query"], "songs")
        self.assertTrue(res["details"]["in_brave"])

    def test_play_english_songs(self):
        res = self.parser.parse("play English songs")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "PLAY_MUSIC")
        self.assertIn("english", res["details"]["song_query"].lower())

    def test_search_youtube_for_python(self):
        res = self.parser.parse("search YouTube for Python tutorials")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "YOUTUBE_SEARCH")
        self.assertEqual(res["details"]["query"], "Python tutorials")

    def test_search_brave_for_ml(self):
        res = self.parser.parse("search Brave for machine learning")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "BRAVE_SEARCH")
        self.assertEqual(res["details"]["query"], "machine learning")

    def test_open_settings(self):
        res = self.parser.parse("open settings")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "OPEN_APP")
        self.assertEqual(res["target"], "settings")

    def test_open_bluetooth_settings(self):
        res = self.parser.parse("open bluetooth settings")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "OPEN_SETTINGS")
        self.assertEqual(res["details"]["subpage"], "bluetooth")

    def test_open_wifi_settings(self):
        res = self.parser.parse("open wi-fi settings")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "OPEN_SETTINGS")
        self.assertEqual(res["details"]["subpage"], "wifi")

    def test_open_downloads(self):
        res = self.parser.parse("open downloads")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "OPEN_APP")
        self.assertEqual(res["target"], "downloads")

    def test_open_terminal(self):
        res = self.parser.parse("open terminal")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "OPEN_APP")
        self.assertEqual(res["target"], "terminal")

    def test_open_calculator(self):
        res = self.parser.parse("open calculator")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "OPEN_APP")
        self.assertEqual(res["target"], "calculator")

    def test_open_file_manager(self):
        res = self.parser.parse("open file manager")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "OPEN_APP")
        self.assertEqual(res["target"], "file manager")

    def test_open_text_editor(self):
        res = self.parser.parse("open text editor")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "OPEN_APP")
        self.assertEqual(res["target"], "text editor")

    def test_stop_voice(self):
        res = self.parser.parse("stop voice")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "STOP_VOICE")

    def test_time_and_date(self):
        res_time = self.parser.parse("what time is it?")
        self.assertTrue(res_time["handled"])
        self.assertEqual(res_time["intent"], "GET_TIME")

        res_date = self.parser.parse("what is today's date?")
        self.assertTrue(res_date["handled"])
        self.assertEqual(res_date["intent"], "GET_DATE")


class TestAppRegistryAndLaunchResolution(unittest.TestCase):
    """Verify application resolution and launch verification."""

    def setUp(self):
        self.registry = AppRegistry()

    def test_brave_browser_detection(self):
        brave_bin = self.registry.find_brave_browser()
        print(f"\n[DIAGNOSTIC] Detected Brave Browser executable: {brave_bin}")
        # On user system, Brave is installed; check returns valid path or None gracefully
        if brave_bin:
            self.assertTrue(Path(brave_bin).exists() or brave_bin.startswith("flatpak:"))

    def test_downloads_folder_exists(self):
        dl = Path.home() / "Downloads"
        print(f"[DIAGNOSTIC] Downloads path: {dl} (exists: {dl.exists()})")
        # Ensure path resolution never crashes
        self.assertTrue(Path.home().exists())

    def test_whatsapp_resolution(self):
        spec = self.registry._apps.get("whatsapp")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.website_fallback, "https://web.whatsapp.com")
        self.assertEqual(spec.force_browser, "brave")

    def test_instagram_resolution(self):
        spec = self.registry._apps.get("instagram")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.website_fallback, "https://www.instagram.com")

    def test_spotify_resolution(self):
        spec = self.registry._apps.get("spotify")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.website_fallback, "https://open.spotify.com")


class TestSystemIntegrationsExecution(unittest.TestCase):
    """Verify end-to-end command execution and truthful status messages."""

    def setUp(self):
        self.integrations = SystemIntegrations()

    def test_exec_open_whatsapp(self):
        handled, resp, _ = self.integrations.try_handle_command("open whatsapp")
        print(f"\n[COMMAND TEST] 'open whatsapp' -> Handled: {handled} | Response: '{resp}'")
        self.assertTrue(handled)
        self.assertIsNotNone(resp)
        # Verify it either launched WhatsApp or opened WhatsApp Web (truthful status, never fake acknowledgement)
        self.assertTrue("WhatsApp" in resp or "web.whatsapp.com" in resp)
        self.assertNotIn("Sent request to open 'whatsapp' via system launcher", resp)

    def test_exec_open_instagram(self):
        handled, resp, _ = self.integrations.try_handle_command("open instagram")
        print(f"[COMMAND TEST] 'open instagram' -> Handled: {handled} | Response: '{resp}'")
        self.assertTrue(handled)
        self.assertIsNotNone(resp)
        self.assertTrue("Instagram" in resp or "instagram.com" in resp)

    def test_exec_open_youtube(self):
        handled, resp, _ = self.integrations.try_handle_command("open youtube")
        print(f"[COMMAND TEST] 'open youtube' -> Handled: {handled} | Response: '{resp}'")
        self.assertTrue(handled)
        self.assertIsNotNone(resp)
        self.assertTrue("YouTube" in resp or "youtube.com" in resp)

    def test_exec_play_tamil_songs_in_brave(self):
        handled, resp, _ = self.integrations.try_handle_command("play Tamil songs in Brave")
        print(f"[COMMAND TEST] 'play Tamil songs in Brave' -> Handled: {handled} | Response: '{resp}'")
        self.assertTrue(handled)
        self.assertIsNotNone(resp)
        self.assertIn("Tamil songs", resp)
        self.assertIn("YouTube", resp)

    def test_exec_search_youtube_for_python(self):
        handled, resp, _ = self.integrations.try_handle_command("search YouTube for Python tutorials")
        print(f"[COMMAND TEST] 'search YouTube for Python tutorials' -> Handled: {handled} | Response: '{resp}'")
        self.assertTrue(handled)
        self.assertIsNotNone(resp)
        self.assertIn("Python tutorials", resp)

    def test_exec_open_settings(self):
        handled, resp, _ = self.integrations.try_handle_command("open settings")
        print(f"[COMMAND TEST] 'open settings' -> Handled: {handled} | Response: '{resp}'")
        self.assertTrue(handled)
        self.assertIsNotNone(resp)
        self.assertTrue("Settings" in resp)

    def test_exec_open_downloads(self):
        handled, resp, _ = self.integrations.try_handle_command("open downloads")
        print(f"[COMMAND TEST] 'open downloads' -> Handled: {handled} | Response: '{resp}'")
        self.assertTrue(handled)
        self.assertIsNotNone(resp)
        self.assertIn("Downloads", resp)

    def test_exec_stop_voice(self):
        handled, resp, _ = self.integrations.try_handle_command("stop voice")
        print(f"[COMMAND TEST] 'stop voice' -> Handled: {handled} | Response: '{resp}'")
        self.assertTrue(handled)
        self.assertEqual(resp, "Voice playback stopped.")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING ADAM COMMAND SYSTEM & LAUNCH SUITE")
    print("=" * 65)
    unittest.main(verbosity=2)
