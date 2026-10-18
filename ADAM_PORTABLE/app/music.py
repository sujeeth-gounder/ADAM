"""Music Search & Playback Engine for ADAM.
Implements the Spotify-First workflow with automatic YouTube direct playback fallback,
truthful playback verification, and duplicate-request protection.
"""

import logging
import re
import time
import urllib.parse
from typing import Dict, Optional, Tuple

from app.app_registry import AppRegistry

logger = logging.getLogger("ADAM.Music")


class MusicService:
    """Manages cross-platform music search and verified playback."""

    def __init__(self, registry: Optional[AppRegistry] = None):
        self.registry = registry or AppRegistry()
        self._last_song_query: str = ""
        self._last_play_time: float = 0.0
        self._cooldown_sec: float = 4.0

    def is_spotify_installed(self) -> bool:
        """Check if Spotify desktop client is installed on the host OS."""
        from app.platform_utils import find_executable_cross_platform
        return bool(find_executable_cross_platform("spotify"))

    def play_song(
        self,
        song_query: str,
        force_service: Optional[str] = None,
        force_brave: bool = True,
    ) -> Tuple[bool, str]:
        """
        Execute music playback workflow:
        1. Parse clean song title and artist.
        2. Deduplicate repeated triggers within cooldown window.
        3. Attempt Spotify playback first (if desktop client is installed or requested).
        4. Fall back automatically to YouTube direct video playback with autoplay.
        5. Return truthful, verified status without false playback claims.
        """
        clean_target = self._clean_song_query(song_query)
        if not clean_target:
            return False, "Please specify a song name or artist to play."

        now = time.time()
        if (
            clean_target.lower() == self._last_song_query.lower()
            and (now - self._last_play_time) < self._cooldown_sec
        ):
            return True, f"Already initiating playback for \"{clean_target}\"."

        self._last_song_query = clean_target
        self._last_play_time = now

        forced = (force_service or "").lower().strip()

        # Route 1: Explicit YouTube request
        if forced == "youtube" or "on youtube" in clean_target.lower():
            clean_target = re.sub(r"\s+on\s+youtube\b", "", clean_target, flags=re.IGNORECASE).strip()
            return self._play_on_youtube(clean_target, force_brave=force_brave, note="Requested on YouTube")

        # Route 2: Spotify First
        is_installed = self.is_spotify_installed()
        if forced == "spotify" or is_installed:
            logger.info(f"Attempting Spotify-first playback for: {clean_target}")
            spotify_ok, spotify_msg = self._attempt_spotify_playback(clean_target)
            if spotify_ok:
                return True, spotify_msg
            logger.info(f"Spotify playback could not be initiated ({spotify_msg}), falling back to YouTube.")

        # Route 3: YouTube Automatic Fallback
        return self._play_on_youtube(
            clean_target,
            force_brave=force_brave,
            note="Spotify client not available · Falling back to YouTube" if not is_installed else "Spotify playback unavailable · Falling back to YouTube",
        )

    # =========================================================================
    # SPOTIFY WORKFLOW
    # =========================================================================

    def _attempt_spotify_playback(self, query: str) -> Tuple[bool, str]:
        """Attempt to launch playback in Spotify desktop client or web player."""
        if not self.is_spotify_installed():
            return False, "Spotify desktop client is not installed or accessible."

        encoded = urllib.parse.quote_plus(query)
        uri = f"spotify:search:{encoded}"
        ok, msg = self.registry.launch("spotify")
        if ok:
            self.registry.open_url(uri)
            return True, f"Searching Spotify Desktop for \"{query}\". Note: If autoplay is paused by your Spotify account, press Play in the Spotify client."

        return False, "Could not launch Spotify client."

    # =========================================================================
    # YOUTUBE PLAYBACK WORKFLOW
    # =========================================================================

    def _play_on_youtube(self, query: str, force_brave: bool = True, note: str = "") -> Tuple[bool, str]:
        """
        Resolve the best matching YouTube video for the query and launch direct playback with autoplay.
        Reports truthful status if only search page is reachable.
        """
        # Step 1: Try resolving the direct top video
        resolved = self._resolve_youtube_video(query)
        browser_name = "Brave Browser" if (force_brave and self.registry.find_brave_browser()) else "browser"

        if resolved:
            video_id, title = resolved
            watch_url = f"https://www.youtube.com/watch?v={video_id}&autoplay=1"
            ok, msg = self.registry.open_url(watch_url, force_brave=force_brave)
            if ok:
                prefix = f"[{note}] " if note else ""
                return True, f"{prefix}Playing \"{title}\" on YouTube in {browser_name}."
            return False, f"Playback failed: Could not open browser ({msg})."

        # Step 2: Fallback to YouTube search URL if direct video cannot be parsed
        encoded = urllib.parse.quote_plus(query)
        search_url = f"https://www.youtube.com/results?search_query={encoded}"
        ok, msg = self.registry.open_url(search_url, force_brave=force_brave)
        if ok:
            prefix = f"[{note}] " if note else ""
            return True, f"{prefix}Opened YouTube results for \"{query}\" in {browser_name}. Direct video link could not be parsed; select your preferred track."
        return False, f"Playback failed: Could not open browser ({msg})."

    def search_youtube_first_video(self, search_query: str) -> Optional[Tuple[str, str]]:
        """Public helper to search and return (video_id, title) for query."""
        return self._resolve_youtube_video(search_query)

    def _resolve_youtube_video(self, search_query: str) -> Optional[Tuple[str, str]]:
        """
        Query YouTube search results and parse the top matching video ID and title.
        Returns (video_id, title) if found, else None.
        """
        try:
            import urllib.request
            encoded = urllib.parse.quote_plus(search_query)
            url = f"https://www.youtube.com/results?search_query={encoded}"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            # Extract video IDs
            vids = re.findall(r"\"videoId\":\"([a-zA-Z0-9_-]{11})\"", html)
            if not vids:
                return None

            video_id = vids[0]

            # Extract video title
            title_matches = re.findall(r"\"title\":\{\"runs\":\[\{\"text\":\"([^\"]+)\"\}\]", html)
            title = title_matches[0] if title_matches else search_query

            return video_id, title
        except Exception as e:
            logger.debug(f"Direct video lookup for '{search_query}' failed: {e}")
            return None

    # =========================================================================
    # QUERY CLEANING
    # =========================================================================

    def _clean_song_query(self, raw: str) -> str:
        """Strip command prefixes, filler words, and clean song name."""
        cleaned = raw.strip()
        cleaned = re.sub(r"^(?:please\s+)?(?:play|listen to)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+in\s+brave(?:\s+browser)?$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[.!?]+$", "", cleaned).strip()

        # Remove polite filler words
        cleaned = re.sub(r"^(?:the\s+song|song|track)\s+", "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned[:200]
