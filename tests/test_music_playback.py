"""Unit and Integration Tests for ADAM Music Search & Direct Playback Pipeline."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.app_registry import CommandParser, AppRegistry
from app.music import MusicService


class TestMusicCommandParser(unittest.TestCase):
    def setUp(self):
        self.registry = AppRegistry()
        self.parser = CommandParser(self.registry)

    def test_parse_play_vadi_pulla_vadi(self):
        res = self.parser.parse("Play Vadi Pulla Vadi")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "PLAY_MUSIC")
        self.assertIn("vadi pulla vadi", res["details"]["song_query"].lower())

    def test_parse_play_artist(self):
        res = self.parser.parse("Play Believer by Imagine Dragons")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "PLAY_MUSIC")
        self.assertIn("believer", res["details"]["song_query"].lower())

    def test_parse_forced_spotify(self):
        res = self.parser.parse("Play Vaadi Pulla Vaadi on Spotify")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "PLAY_MUSIC")
        self.assertEqual(res["details"].get("force_service"), "spotify")

    def test_parse_forced_youtube(self):
        res = self.parser.parse("Play Starboy on YouTube")
        self.assertTrue(res["handled"])
        self.assertEqual(res["intent"], "PLAY_MUSIC")
        self.assertEqual(res["details"].get("force_service"), "youtube")


class TestMusicService(unittest.TestCase):
    def setUp(self):
        self.mock_registry = MagicMock(spec=AppRegistry)
        self.music = MusicService(registry=self.mock_registry)

    def test_youtube_video_id_search_parsing(self):
        """Verify video resolver extracts 11-char video ID and title from YouTube search HTML."""
        sample_html = (
            '<html><body>'
            '{"videoId":"7mWd8fOG7qI","thumbnail":{"thumbnails":[]},'
            '"title":{"runs":[{"text":"Vaadi Pulla Vaadi Official Video"}]}}'
            '</body></html>'
        ).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = sample_html
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.music.search_youtube_first_video("Vaadi Pulla Vaadi")
            self.assertIsNotNone(result)
            video_id, title = result
            self.assertEqual(video_id, "7mWd8fOG7qI")
            self.assertIn("Vaadi Pulla Vaadi", title)

    def test_play_youtube_direct_autoplay_url(self):
        self.mock_registry.find_brave_browser.return_value = "/usr/bin/brave-browser"
        self.mock_registry.open_url.return_value = (True, "Opened in Brave")

        # Mock direct video resolver
        with patch.object(self.music, "_resolve_youtube_video", return_value=("7mWd8fOG7qI", "Vaadi Pulla Vaadi")):
            with patch.object(self.music, "is_spotify_installed", return_value=False):
                ok, msg = self.music.play_song("Vaadi Pulla Vaadi", force_service="youtube")
                self.assertTrue(ok)
                self.assertIn("YouTube", msg)
                self.assertTrue(self.mock_registry.open_url.called)

                opened_url = self.mock_registry.open_url.call_args[0][0]
                self.assertEqual(opened_url, "https://www.youtube.com/watch?v=7mWd8fOG7qI&autoplay=1")

    def test_debounce_cooldown(self):
        self.mock_registry.find_brave_browser.return_value = "/usr/bin/brave-browser"
        self.mock_registry.open_url.return_value = (True, "Opened")

        with patch.object(self.music, "_resolve_youtube_video", return_value=("vid12345678", "Believer")):
            with patch.object(self.music, "is_spotify_installed", return_value=False):
                # First call initiates
                ok1, _ = self.music.play_song("Believer")
                self.assertTrue(ok1)

                # Immediate second call within 4s window is debounced
                ok2, msg2 = self.music.play_song("Believer")
                self.assertTrue(ok2)
                self.assertIn("Already initiating playback", msg2)

    def test_spotify_first_when_installed(self):
        self.mock_registry.launch.return_value = (True, "Launched Spotify")
        self.mock_registry.open_url.return_value = (True, "Opened URI")

        self.music._last_song_query = ""
        self.music._last_play_time = 0.0

        with patch.object(self.music, "is_spotify_installed", return_value=True):
            ok, msg = self.music.play_song("Shape of You")
            self.assertTrue(ok)
            self.assertIn("Spotify", msg)
            self.assertTrue(self.mock_registry.launch.called)
            self.mock_registry.launch.assert_called_with("spotify")

    def test_truthful_failure_when_browser_fails(self):
        self.mock_registry.find_brave_browser.return_value = None
        self.mock_registry.open_url.return_value = (False, "Process failed to spawn")

        self.music._last_song_query = ""
        self.music._last_play_time = 0.0

        with patch.object(self.music, "_resolve_youtube_video", return_value=("vid12345678", "Some Track")):
            with patch.object(self.music, "is_spotify_installed", return_value=False):
                ok, msg = self.music.play_song("Some Track", force_service="youtube")
                self.assertFalse(ok)
                self.assertIn("Could not open browser", msg)


if __name__ == "__main__":
    unittest.main()
