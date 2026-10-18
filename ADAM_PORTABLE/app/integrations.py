"""Integrations Module for ADAM.
Handles Ollama local LLM inference, system commands, time/date, app & website launching,
YouTube search, 3D generation routing, and safe file operations with explicit user confirmation.
"""

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import webbrowser
import zoneinfo
from datetime import datetime, tzinfo
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple
import threading

from app.app_registry import AppRegistry, CommandParser
from app.config import (
    ADAM_SYSTEM_PROMPT,
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_HOST,
    RECOMMENDED_MODELS,
)
from app.email_service import GmailClient
from app.memory import MemoryManager
from app.three_d import generate_3d_object, get_available_shapes

logger = logging.getLogger("ADAM.Integrations")


@dataclass
class PendingAction:
    """Represents a dangerous/destructive action requiring explicit user confirmation."""
    action_id: str
    action_type: str        # e.g., 'DELETE_FILE', 'DELETE_DIR', 'KILL_PROCESS'
    target: str             # e.g., '/path/to/file'
    description: str        # Warning text displayed to user
    callback: Callable[[], Tuple[bool, str]]  # Executed upon user confirmation


class OllamaClient:
    """Wrapper for local Ollama API inference with auto-reconnection and model fallback."""

    def __init__(self, host: str = OLLAMA_HOST, model: str = DEFAULT_OLLAMA_MODEL):
        self.host = host.rstrip("/")
        self.model = model
        self.is_connected = False
        self.available_models: List[str] = []
        self._load_saved_model()
        self.check_connection()

    def _load_saved_model(self):
        config_file = Path(__file__).resolve().parent.parent / "data" / "model_settings.json"
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    saved = json.load(f).get("default_model")
                    if saved:
                        self.model = saved
            except Exception:
                pass

    def save_default_model(self, model_name: str):
        self.model = model_name
        config_dir = Path(__file__).resolve().parent.parent / "data"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "model_settings.json"
        try:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump({"default_model": model_name}, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist model setting: {e}")

    def set_model(self, model_name: str) -> bool:
        """Switch active model for inference."""
        self.model = model_name
        self.save_default_model(model_name)
        return True

    def pull_model_stream(self, model_name: str) -> Generator[Dict[str, Any], None, None]:
        """
        Pull / download an Ollama model with real-time status and progress updates.
        Yields status dicts: {'status': str, 'completed': int, 'total': int, 'percent': float}
        """
        import urllib.request
        import urllib.error
        url = f"{self.host}/api/pull"
        payload = json.dumps({"name": model_name, "stream": True}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for line in resp:
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        status = chunk.get("status", "")
                        total = chunk.get("total", 0)
                        completed = chunk.get("completed", 0)
                        percent = (completed / total * 100.0) if total > 0 else 0.0
                        yield {
                            "status": status,
                            "completed": completed,
                            "total": total,
                            "percent": percent,
                            "done": status == "success"
                        }
            # Refresh available models list
            self.check_connection()
        except urllib.error.HTTPError as he:
            err = he.read().decode("utf-8", errors="ignore")
            yield {"status": f"HTTP Error {he.code}: {err}", "error": True, "done": True}
        except Exception as e:
            yield {"status": f"Download connection error: {e}", "error": True, "done": True}

    def check_connection(self) -> bool:
        """Check if Ollama daemon is reachable and list installed models."""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    self.available_models = [m.get("name", "") for m in data.get("models", [])]
                    self.is_connected = True
                    # If current model not installed, pick first available if present
                    if self.available_models and not any(self.model in m for m in self.available_models):
                        self.model = self.available_models[0]
                    return True
        except Exception:
            self.is_connected = False
            self.available_models = []
        return False

    def get_status_info(self) -> str:
        """Return status string for UI badge."""
        if not self.is_connected:
            return "Ollama: Offline (run 'ollama serve')"
        return f"Ollama: Online ({self.model})"

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
        timeout: float = 60.0,
        max_chars: int = 12000,
    ) -> Generator[str, None, None]:
        """Generate a bounded, cancellable Ollama response; fall back only when chat is unsupported."""
        deadline = time.monotonic() + timeout
        emitted_chars = 0

        def should_stop() -> bool:
            return (
                (cancel_event is not None and cancel_event.is_set())
                or time.monotonic() >= deadline
                or emitted_chars >= max_chars
            )

        def bounded(piece: str) -> str:
            nonlocal emitted_chars
            remaining = max_chars - emitted_chars
            piece = piece[:max(0, remaining)]
            emitted_chars += len(piece)
            return piece

        if not self.check_connection():
            yield "Ollama is currently offline. Please launch the Ollama service using: `ollama serve` and pull a model with: `ollama pull qwen2.5:1.5b`."
            return

        sys_msg = system_prompt or ADAM_SYSTEM_PROMPT
        if hasattr(self, "memory_manager") and self.memory_manager:
            mem_block = self.memory_manager.get_memory_context_prompt()
            if mem_block:
                sys_msg = f"{sys_msg}\n\n{mem_block}"
        
        # Primary: /api/chat
        payload_chat = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt}
            ],
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 256,
            }
        }

        import urllib.request
        import urllib.error

        # Attempt /api/chat first
        try:
            data = json.dumps(payload_chat).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/chat",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=min(10.0, timeout)) as resp:
                for line in resp:
                    if should_stop():
                        return
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        text_piece = chunk.get("message", {}).get("content", "")
                        if text_piece:
                            yield bounded(text_piece)
                        if chunk.get("done", False):
                            return
        except urllib.error.HTTPError as he:
            err_text = ""
            try:
                raw_body = he.read().decode("utf-8", errors="replace")
                err_json = json.loads(raw_body)
                err_text = err_json.get("error", raw_body)
            except Exception:
                err_text = str(he)

            # If error is not a 404 on /api/chat, yield the real server diagnostic
            if he.code != 404:
                logger.error(f"Ollama server returned HTTP {he.code}: {err_text}")
                yield f"\n[Ollama Server Error (HTTP {he.code}): {err_text}]"
                return
        except Exception as e:
            logger.error("Ollama /api/chat request failed: %s", e)
            yield f"Ollama request failed: {e}"
            return

        if should_stop():
            return

        # Fallback: /api/generate
        payload_generate = {
            "model": self.model,
            "prompt": prompt,
            "system": sys_msg,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 256,
            }
        }

        try:
            data = json.dumps(payload_generate).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=min(10.0, timeout)) as resp:
                for line in resp:
                    if should_stop():
                        return
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        text_piece = chunk.get("response", "")
                        if text_piece:
                            yield bounded(text_piece)
                        if chunk.get("done", False):
                            break
        except urllib.error.HTTPError as he:
            err_text = ""
            try:
                raw_body = he.read().decode("utf-8", errors="replace")
                err_json = json.loads(raw_body)
                err_text = err_json.get("error", raw_body)
            except Exception:
                err_text = str(he)
            logger.error(f"Ollama /api/generate HTTP {he.code}: {err_text}")
            yield f"\n[Ollama Error (HTTP {he.code}): {err_text}]"
        except Exception as e:
            logger.error(f"Ollama streaming exception: {e}")
            yield f"\n[Ollama Connection Error: {e}]"




# =============================================================================
# TIMEZONE & TIME/DATE HANDLING (PYTHON DATETIME & ZONEINFO)
# =============================================================================

TIMEZONE_MAP: Dict[str, Tuple[str, str]] = {
    # Key -> (IANA timezone string, display location label)
    # India / IST (IANA: Asia/Kolkata, UTC+05:30)
    "india": ("Asia/Kolkata", "India (IST)"),
    "indian": ("Asia/Kolkata", "Indian time"),
    "indian time": ("Asia/Kolkata", "Indian time"),
    "ist": ("Asia/Kolkata", "IST"),
    "isd": ("Asia/Kolkata", "IST"),  # Common speech recognition variant
    "indian standard time": ("Asia/Kolkata", "Indian Standard Time"),
    "kolkata": ("Asia/Kolkata", "Kolkata"),
    "calcutta": ("Asia/Kolkata", "Calcutta"),
    "delhi": ("Asia/Kolkata", "Delhi"),
    "new delhi": ("Asia/Kolkata", "New Delhi"),
    "mumbai": ("Asia/Kolkata", "Mumbai"),
    "bombay": ("Asia/Kolkata", "Mumbai"),
    "chennai": ("Asia/Kolkata", "Chennai"),
    "madras": ("Asia/Kolkata", "Chennai"),
    "bangalore": ("Asia/Kolkata", "Bangalore"),
    "bengaluru": ("Asia/Kolkata", "Bengaluru"),
    "hyderabad": ("Asia/Kolkata", "Hyderabad"),

    # UTC / GMT
    "utc": ("UTC", "UTC"),
    "gmt": ("UTC", "GMT"),
    "zulu": ("UTC", "UTC"),
    "coordinated universal time": ("UTC", "Coordinated Universal Time"),
    "greenwich mean time": ("UTC", "Greenwich Mean Time"),

    # Americas
    "est": ("America/New_York", "Eastern Time"),
    "edt": ("America/New_York", "Eastern Daylight Time"),
    "eastern": ("America/New_York", "Eastern Time"),
    "eastern time": ("America/New_York", "Eastern Time"),
    "new york": ("America/New_York", "New York"),
    "washington": ("America/New_York", "Washington, D.C."),
    "miami": ("America/New_York", "Miami"),
    "boston": ("America/New_York", "Boston"),
    "toronto": ("America/Toronto", "Toronto"),

    "cst": ("America/Chicago", "Central Time"),
    "cdt": ("America/Chicago", "Central Daylight Time"),
    "central": ("America/Chicago", "Central Time"),
    "central time": ("America/Chicago", "Central Time"),
    "chicago": ("America/Chicago", "Chicago"),
    "dallas": ("America/Chicago", "Dallas"),
    "houston": ("America/Chicago", "Houston"),

    "mst": ("America/Denver", "Mountain Time"),
    "mdt": ("America/Denver", "Mountain Daylight Time"),
    "mountain": ("America/Denver", "Mountain Time"),
    "mountain time": ("America/Denver", "Mountain Time"),
    "denver": ("America/Denver", "Denver"),
    "phoenix": ("America/Phoenix", "Phoenix"),

    "pst": ("America/Los_Angeles", "Pacific Time"),
    "pdt": ("America/Los_Angeles", "Pacific Daylight Time"),
    "pacific": ("America/Los_Angeles", "Pacific Time"),
    "pacific time": ("America/Los_Angeles", "Pacific Time"),
    "los angeles": ("America/Los_Angeles", "Los Angeles"),
    "san francisco": ("America/Los_Angeles", "San Francisco"),
    "california": ("America/Los_Angeles", "California"),
    "seattle": ("America/Los_Angeles", "Seattle"),
    "vancouver": ("America/Vancouver", "Vancouver"),

    "alaska": ("America/Anchorage", "Alaska"),
    "akst": ("America/Anchorage", "Alaska Time"),
    "hawaii": ("Pacific/Honolulu", "Hawaii"),
    "hst": ("Pacific/Honolulu", "Hawaii Time"),

    # Europe
    "london": ("Europe/London", "London"),
    "uk": ("Europe/London", "UK"),
    "united kingdom": ("Europe/London", "United Kingdom"),
    "britain": ("Europe/London", "Britain"),
    "england": ("Europe/London", "England"),
    "bst": ("Europe/London", "British Summer Time"),

    "paris": ("Europe/Paris", "Paris"),
    "france": ("Europe/Paris", "France"),
    "berlin": ("Europe/Berlin", "Berlin"),
    "germany": ("Europe/Berlin", "Germany"),
    "rome": ("Europe/Rome", "Rome"),
    "italy": ("Europe/Rome", "Italy"),
    "madrid": ("Europe/Madrid", "Madrid"),
    "spain": ("Europe/Madrid", "Spain"),
    "amsterdam": ("Europe/Amsterdam", "Amsterdam"),
    "netherlands": ("Europe/Amsterdam", "Netherlands"),
    "cet": ("Europe/Paris", "Central European Time"),
    "cest": ("Europe/Paris", "Central European Summer Time"),

    # Asia & Middle East
    "tokyo": ("Asia/Tokyo", "Tokyo"),
    "japan": ("Asia/Tokyo", "Japan"),
    "jst": ("Asia/Tokyo", "Japan Standard Time"),
    "seoul": ("Asia/Seoul", "Seoul"),
    "korea": ("Asia/Seoul", "Korea"),
    "south korea": ("Asia/Seoul", "South Korea"),
    "kst": ("Asia/Seoul", "Korea Standard Time"),
    "singapore": ("Asia/Singapore", "Singapore"),
    "sgt": ("Asia/Singapore", "Singapore Time"),
    "hong kong": ("Asia/Hong_Kong", "Hong Kong"),
    "beijing": ("Asia/Shanghai", "Beijing"),
    "shanghai": ("Asia/Shanghai", "Shanghai"),
    "china": ("Asia/Shanghai", "China"),
    "dubai": ("Asia/Dubai", "Dubai"),
    "uae": ("Asia/Dubai", "UAE"),
    "abu dhabi": ("Asia/Dubai", "Abu Dhabi"),

    # Australia & Pacific
    "sydney": ("Australia/Sydney", "Sydney"),
    "melbourne": ("Australia/Melbourne", "Melbourne"),
    "australia": ("Australia/Sydney", "Australia"),
    "aest": ("Australia/Sydney", "Australian Eastern Standard Time"),
    "aedt": ("Australia/Sydney", "Australian Eastern Daylight Time"),
    "auckland": ("Pacific/Auckland", "Auckland"),
    "new zealand": ("Pacific/Auckland", "New Zealand"),
    "nzst": ("Pacific/Auckland", "New Zealand Standard Time"),
    "nzdt": ("Pacific/Auckland", "New Zealand Daylight Time"),
}


def format_tz_designation(now: datetime) -> str:
    """Format timezone and UTC offset: e.g. 'IST (UTC+05:30)' or 'EDT (UTC-04:00)'."""
    tz_abbr = now.strftime("%Z")
    offset = now.strftime("%z")
    if offset and len(offset) == 5:
        utc_str = f"UTC{offset[:3]}:{offset[3:]}"
    else:
        utc_str = "UTC"
    if tz_abbr and tz_abbr not in (offset, utc_str):
        return f"{tz_abbr} ({utc_str})"
    return utc_str


def resolve_requested_timezone(query_tz: Optional[str] = None) -> Tuple[tzinfo, str, bool]:
    """
    Resolve requested timezone or fall back to system local timezone.
    For local system and Indian queries, strictly uses zoneinfo.ZoneInfo("Asia/Kolkata").
    Returns: (tzinfo_object, display_label, is_explicit)
    """
    if not query_tz or not str(query_tz).strip():
        # Default directly to Asia/Kolkata for ADAM on local system clock
        try:
            return ZoneInfo("Asia/Kolkata"), "IST", False
        except Exception:
            local_dt = datetime.now().astimezone()
            return local_dt.tzinfo, "", False

    raw_clean = re.sub(r"[^\w\s\/\-\_]", "", str(query_tz).strip().lower())

    # 1. Check known lookup dictionary
    if raw_clean in TIMEZONE_MAP:
        iana_key, label = TIMEZONE_MAP[raw_clean]
        return ZoneInfo(iana_key), label, True

    # 2. Check if clean key without 'time' is in map (e.g. 'tokyo time' -> 'tokyo')
    sub_clean = re.sub(r"\b(time|standard|daylight|current)\b", "", raw_clean).strip()
    if sub_clean in TIMEZONE_MAP:
        iana_key, label = TIMEZONE_MAP[sub_clean]
        return ZoneInfo(iana_key), label, True

    # 3. Direct IANA timezone lookup (e.g. 'Asia/Kolkata', 'Europe/Paris')
    try:
        iana_tz = ZoneInfo(query_tz.strip())
        return iana_tz, query_tz.strip(), True
    except Exception:
        pass

    # 4. Search available IANA timezones
    try:
        available = zoneinfo.available_timezones()
        lower_map = {tz.lower(): tz for tz in available}
        if raw_clean in lower_map:
            match_key = lower_map[raw_clean]
            return ZoneInfo(match_key), match_key.split("/")[-1].replace("_", " "), True
        for low_z, orig_z in lower_map.items():
            if low_z.endswith("/" + raw_clean):
                return ZoneInfo(orig_z), orig_z.split("/")[-1].replace("_", " "), True
    except Exception:
        pass

    # Fallback to Asia/Kolkata
    try:
        return ZoneInfo("Asia/Kolkata"), "IST", False
    except Exception:
        local_dt = datetime.now().astimezone()
        return local_dt.tzinfo, "", False


def format_current_time(now: datetime) -> str:
    """Format time string with accurate timezone & UTC offset: e.g. '11:15 AM IST (UTC+05:30)'."""
    designation = format_tz_designation(now)
    return f"{now.strftime('%I:%M %p')} {designation}"


def format_current_date(now: datetime) -> str:
    """Format date string as 'Saturday, 17 October 2026'."""
    return f"{now.strftime('%A, %d %B %Y')}"


def get_current_time_string(tz: tzinfo) -> str:
    """Get and format current time in the requested timezone."""
    return format_current_time(datetime.now(tz))


def get_current_date_string(tz: tzinfo) -> str:
    """Get and format the current date in the requested timezone."""
    return format_current_date(datetime.now(tz))


def format_clock_command(intent: str, timezone_query: str = "") -> str:
    """Build a fresh deterministic clock response and log its exact source timestamp."""
    if timezone_query:
        tz, label, is_explicit = resolve_requested_timezone(timezone_query)
    else:
        tz, label, is_explicit = ZoneInfo("Asia/Kolkata"), "IST", False

    if is_explicit:
        now = datetime.now(tz)
    else:
        tz = ZoneInfo("Asia/Kolkata")
        label = "IST"
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
    date_str = format_current_date(now)
    time_str = format_current_time(now)

    if intent == "GET_TIME":
        if is_explicit:
            if label.lower() in ("indian time", "india (ist)", "ist", "india"):
                response = f"The current Indian time is {time_str} on {date_str}."
            else:
                response = f"In {label}, the current time is {time_str} on {date_str}."
        else:
            response = f"The current time is {time_str} on {date_str}."
    elif intent == "GET_DATE":
        designation = format_tz_designation(now)
        if is_explicit:
            response = f"Today's date in {label} is {date_str} ({designation})."
        else:
            response = f"Today's date is {date_str} ({designation})."
    elif intent == "GET_DAY":
        response = f"Today is {now.strftime('%A')}."
    elif intent == "GET_TIME_AND_DATE":
        if is_explicit:
            response = f"In {label}, the current time is {time_str} on {date_str}."
        else:
            response = f"The current time is {time_str} on {date_str}."
    else:
        raise ValueError(f"Unsupported clock intent: {intent}")

    logger.info(
        "Clock request intent=%s timestamp=%s timezone=%s response=%r",
        intent,
        now.isoformat(timespec="seconds"),
        now.tzinfo,
        response,
    )
    return response


class SystemIntegrations:
    """Handles desktop automations, system diagnostics, and safe file operations."""

    def __init__(
        self,
        confirmation_request_handler: Optional[Callable[[PendingAction], None]] = None,
        vision_service: Optional[Any] = None,
        three_d_callback: Optional[Callable[[Any], None]] = None,
        three_d_viewport_command_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        confirmation_request_handler: callback called when a destructive operation is proposed.
        It passes a PendingAction object to the UI, which displays [Confirm] / [Cancel] buttons.
        """
        self.confirmation_request_handler = confirmation_request_handler
        self.vision_service = vision_service
        self.three_d_callback = three_d_callback
        self.three_d_viewport_command_callback = three_d_viewport_command_callback
        self.pending_actions: Dict[str, PendingAction] = {}
        self.ollama = OllamaClient()
        self.memory = MemoryManager()
        self.ollama.memory_manager = self.memory
        self.app_registry = AppRegistry()
        self.command_parser = CommandParser(self.app_registry)
        self.gmail_client = GmailClient()

    # =========================================================================
    # INTENT PARSER & DISPATCHER
    # =========================================================================

    def try_handle_command(self, query: str) -> Tuple[bool, Optional[str], Optional[PendingAction]]:
        """
        Check if query is a direct system command (time, memory, app, website, youtube, 3D, diagnostics, file ops).
        Returns:
            (handled: bool, response_text: Optional[str], pending_action: Optional[PendingAction])
        """
        q = query.strip()
        q_lower = q.lower()

        # Step -1: Personal Memory & Identity Commands (deterministic, before Ollama)
        if hasattr(self, "memory") and self.memory:
            handled, mem_resp = self.memory.try_handle_memory_command(q)
            if handled:
                return True, mem_resp, None

        # Step 0: Direct Email / Gmail Status Queries
        if any(kw == q_lower for kw in ["check emails", "check gmail", "gmail status", "read emails", "unread emails"]):
            status = self.gmail_client.get_auth_status()
            if not status["authorized"]:
                return True, "📧 **Gmail Integration**: Not connected. To access your emails securely, authenticate with Google OAuth. (No tokens or credentials stored in source code).", None
            else:
                try:
                    msgs, _, total = self.gmail_client.list_messages(max_results=5)
                    return True, f"📧 **Gmail Connected** ({status['user_email']}): Found approximately {total} messages in your mailbox. 5 latest messages ready for preview.", None
                except Exception as e:
                    return True, f"📧 **Gmail Connected**: Error retrieving messages: {e}", None

        # Step 1: 3D Viewport Controls
        if any(kw in q_lower for kw in ["rotate model", "rotate the model", "rotate 3d", "turn model", "spin model"]):
            if self.three_d_viewport_command_callback:
                self.three_d_viewport_command_callback("rotate")
            return True, "Rotated the 3D model in the viewport.", None

        if any(kw in q_lower for kw in ["zoom in on model", "zoom in model", "zoom in the model", "zoom in 3d", "zoom in"]):
            if self.three_d_viewport_command_callback:
                self.three_d_viewport_command_callback("zoom_in")
            return True, "Zoomed in on the 3D model.", None

        if any(kw in q_lower for kw in ["zoom out of model", "zoom out model", "zoom out the model", "zoom out 3d", "zoom out"]):
            if self.three_d_viewport_command_callback:
                self.three_d_viewport_command_callback("zoom_out")
            return True, "Zoomed out of the 3D model.", None

        if any(kw in q_lower for kw in ["reset view", "reset 3d view", "reset camera view", "reset the view"]):
            if self.three_d_viewport_command_callback:
                self.three_d_viewport_command_callback("reset")
            return True, "Reset the 3D viewport camera.", None

        if any(kw in q_lower for kw in ["open in blender", "edit in blender", "launch blender"]):
            if self.three_d_viewport_command_callback:
                self.three_d_viewport_command_callback("open_blender")
            return True, "Opening model in Blender...", None

        if any(kw in q_lower for kw in ["save 3d model", "export 3d model", "save this model", "export model"]):
            if self.three_d_viewport_command_callback:
                self.three_d_viewport_command_callback("export")
            return True, "Opening 3D model export dialog...", None

        # Step 2: 3D Model Generation
        is_3d_intent = any(w in q_lower for w in ["3d", "blender", "model", "create", "generate", "make", "build", "scorpio", "backpack", "bottle", "chair", "mug", "car"])
        if is_3d_intent:
            three_d_create = re.search(
                r"(?:(?:create|generate|make|build|export|model)\s+)?(?:it\s+)?(?:into\s+)?(?:me\s+)?(?:a\s+)?(?:3d\s+)?(?:model\s+(?:of\s+)?)?(?:a\s+)?"
                r"(?:(low[- ]?poly|realistic|scifi|simple|stylized|standard)\s+)?"
                r"(?:(red|blue|green|yellow|orange|cyan|purple|pink|black|white|silver|gold|gray|grey)\s+)?"
                r"(mahindra[- ]?scorpio(?:[- ]?(?:car|suv))?|scorpio(?:[- ]?(?:car|suv))?|suv|water[- ]?bottle|bottle|backpack|bag|rucksack|chair|coffee[- ]?mug|mug|cup|sports[- ]?car|car|vehicle|robot|bot|table|desk|cube|box|sphere|ball|cylinder|torus|donut|pyramid|scifi[- ]?drone|drone|scifi[- ]?pedestal|pedestal)\b",
                q_lower
            )
            if three_d_create:
                raw_shape = three_d_create.group(3) or three_d_create.group(0)
                style_m = three_d_create.group(1)
                color_m = three_d_create.group(2)

                # Check if color is anywhere in query
                color = color_m or "silver"
                if not color_m:
                    for c in ["red", "blue", "green", "yellow", "orange", "cyan", "purple", "pink", "black", "white", "silver", "gold", "gray", "grey"]:
                        if re.search(rf"\b{c}\b", q_lower):
                            color = c
                            break

                # Check if style is anywhere in query
                style = "standard"
                if style_m:
                    style = style_m.replace("-", "_").replace(" ", "_")
                else:
                    for s in ["low_poly", "low-poly", "low poly", "realistic", "scifi", "sci-fi", "simple", "stylized"]:
                        if s in q_lower:
                            style = s.replace("-", "_").replace(" ", "_")
                            break

                return self._handle_3d_generation(raw_shape, color=color, style=style)

        # Step 3A: Person in Camera Queries
        if any(kw in q_lower for kw in [
            "who is in the camera", "who's in the camera", "who is in front of the camera",
            "who do you see", "who can you see", "who is that", "who is here",
            "is there a person in front of me", "is there a person in front of the camera",
            "is there a person", "is anyone in front of me", "is anyone there", "can you see me", "do you see me"
        ]):
            if self.vision_service:
                return True, self.vision_service.describe_person(), None
            return True, "The camera is currently stopped. Click 'Start Camera' or ask me to turn on the camera to see who is in view.", None

        # Step 3B: General Camera Scene Queries
        if any(kw in q_lower for kw in [
            "what is in the camera", "what's in the camera", "what is in front of the camera",
            "what can you see", "what do you see", "what's in front of me", "what is in front of me",
            "describe what you see", "describe the camera view", "tell me what you see"
        ]):
            if self.vision_service:
                return True, self.vision_service.describe_latest(), None
            return True, "The camera is currently stopped. Click 'Start Camera' or ask me to turn on the camera.", None

        # Step 3C: Object & Hand Inspection Queries
        if any(kw in q_lower for kw in [
            "identify the object", "identify this object", "inspect the object", "inspect this object",
            "what object am i holding", "what am i holding", "what is in my hand", "what am i showing you",
            "tell me about this phone", "what phone is this", "inspect the phone", "inspect this phone"
        ]):
            if self.vision_service:
                target = "cell phone" if "phone" in q_lower else None
                insp = self.vision_service.inspect_object(target)
                return True, insp["response"], None
            return True, "The camera is currently stopped. Start the camera to inspect objects.", None

        # Step 4: Universal Command & App Registry evaluation
        parsed = self.command_parser.parse(q)
        if parsed["handled"]:
            intent = parsed["intent"]
            if intent == "STOP_VOICE":
                return True, "Voice playback stopped.", None
            elif intent == "GET_TIME":
                tz_query = parsed["details"].get("timezone", "")
                return True, format_clock_command(intent, tz_query), None
            elif intent == "GET_DATE":
                tz_query = parsed["details"].get("timezone", "")
                return True, format_clock_command(intent, tz_query), None
            elif intent == "GET_DAY":
                tz_query = parsed["details"].get("timezone", "")
                return True, format_clock_command(intent, tz_query), None
            elif intent == "GET_TIME_AND_DATE":
                tz_query = parsed["details"].get("timezone", "")
                return True, format_clock_command(intent, tz_query), None
            elif intent == "PLAY_MUSIC":
                force_service = parsed["details"].get("force_service", None)
                ok, msg = self.app_registry.play_music(
                    parsed["details"]["song_query"],
                    in_brave=parsed["details"].get("in_brave", False),
                    force_service=force_service
                )
                return True, msg, None
            elif intent == "YOUTUBE_SEARCH":
                ok, msg = self.app_registry.search_youtube(
                    parsed["details"]["query"],
                    in_brave=parsed["details"].get("in_brave", False)
                )
                return True, msg, None
            elif intent == "BRAVE_SEARCH":
                ok, msg = self.app_registry.search_brave(parsed["details"]["query"])
                return True, msg, None
            elif intent == "WEB_SEARCH":
                ok, msg = self.app_registry.search_web(parsed["details"]["query"])
                return True, msg, None
            elif intent == "OPEN_SETTINGS":
                ok, msg = self.app_registry.open_settings(parsed["details"].get("subpage"))
                return True, msg, None
            elif intent == "OPEN_APP":
                target = parsed["target"]
                force_web = parsed["details"].get("force_web", False)
                ok, msg = self.app_registry.launch(target, force_web=force_web)
                return True, msg, None

        # Step 3: Basic System Diagnostics
        if any(kw in q_lower for kw in ["system info", "system stats", "battery", "memory usage", "disk space", "ram usage", "system diagnostic"]):
            return True, self.get_system_diagnostics(), None

        # Step 4: File Operations: List files
        list_match = re.search(r"^(?:list\s+files|show\s+files|dir|ls)(?:\s+in\s+|\s+of\s+|\s+)?(.*)$", q, re.IGNORECASE)
        if list_match:
            target_dir = list_match.group(1).strip() or "."
            return True, self._list_files(target_dir), None

        # Step 5: File Operations: Read file
        read_match = re.search(r"^(?:read\s+file|view\s+file|cat)\s+(.+)$", q, re.IGNORECASE)
        if read_match:
            target_file = read_match.group(1).strip()
            return True, self._read_file(target_file), None

        # Step 6: Destructive File Operations: Delete file or directory (REQUIRES CONFIRMATION)
        del_match = re.search(r"^(?:delete\s+file|remove\s+file|unlink|rm)\s+(.+)$", q, re.IGNORECASE)
        if del_match:
            target = del_match.group(1).strip()
            action = self._create_delete_file_action(target)
            return True, f"⚠️ WARNING: Deletion requested for '{target}'. Explicit confirmation is required before proceeding.", action

        del_dir_match = re.search(r"^(?:delete\s+folder|delete\s+directory|remove\s+folder|rmdir)\s+(.+)$", q, re.IGNORECASE)
        if del_dir_match:
            target = del_dir_match.group(1).strip()
            action = self._create_delete_folder_action(target)
            return True, f"⚠️ CAUTION: Recursive folder deletion requested for '{target}'. Explicit confirmation is required before proceeding.", action

        # Not a direct system command -> Route to Ollama LLM
        return False, None, None

    # =========================================================================
    # TOOL IMPLEMENTATIONS
    # =========================================================================

    def _handle_youtube_search(self, query: str) -> Tuple[bool, str, Optional[PendingAction]]:
        ok, msg = self.app_registry.search_youtube(query, in_brave=True)
        return True, msg, None

    def _handle_open_website(self, url: str) -> Tuple[bool, str, Optional[PendingAction]]:
        ok, msg = self.app_registry.open_url(url, force_brave=True)
        return True, msg, None

    def _handle_open_app(self, app_name: str) -> Tuple[bool, str]:
        """Launch application using AppRegistry with truthful status reporting."""
        return self.app_registry.launch(app_name)

    def _handle_3d_generation(
        self, shape: str, color: str = "silver", style: str = "standard"
    ) -> Tuple[bool, str, Optional[PendingAction]]:
        alias_map = {
            "scorpio": "scorpio_suv",
            "scorpio_car": "scorpio_suv",
            "scorpio car": "scorpio_suv",
            "scorpio_suv": "scorpio_suv",
            "scorpio suv": "scorpio_suv",
            "mahindra_scorpio": "scorpio_suv",
            "mahindra scorpio": "scorpio_suv",
            "mahindra_scorpio_suv": "scorpio_suv",
            "mahindra scorpio suv": "scorpio_suv",
            "mahindra_scorpio_car": "scorpio_suv",
            "mahindra scorpio car": "scorpio_suv",
            "suv": "scorpio_suv",
            "bottle": "water_bottle",
            "water-bottle": "water_bottle",
            "water bottle": "water_bottle",
            "bag": "backpack",
            "backpack": "backpack",
            "rucksack": "backpack",
            "mug": "coffee_mug",
            "cup": "coffee_mug",
            "coffee-mug": "coffee_mug",
            "coffee mug": "coffee_mug",
            "car": "sports_car",
            "sports-car": "sports_car",
            "sports car": "sports_car",
            "vehicle": "sports_car",
            "desk": "table",
            "box": "cube",
            "ball": "sphere",
            "donut": "torus",
            "bot": "robot",
            "drone": "scifi_drone",
            "pedestal": "scifi_pedestal",
        }
        clean_shape = alias_map.get(shape.lower().replace("-", "_").replace(" ", "_"), shape.lower().replace("-", "_").replace(" ", "_"))
        ok, asset, msg = generate_3d_object(shape_type=clean_shape, color=color, style=style)
        if ok and asset:
            if self.three_d_callback:
                try:
                    self.three_d_callback(asset)
                except Exception as e:
                    logger.warning("three_d_callback error: %s", e)

            card = (
                f"🎨 **3D Studio Creation Complete!**\n"
                f"• **Model**: {asset.name} ({asset.shape.replace('_', ' ').title()})\n"
                f"• **Color & Style**: {asset.color.title()} · {asset.style.title()}\n"
                f"• **Mesh Statistics**: {asset.vertex_count:,} vertices · {asset.polygon_count:,} polygons\n"
                f"• **Output Directory**: `{asset.directory}`\n"
                f"• **Files Generated**:\n"
                f"  - Blender Project: `{asset.blend_path.name}`\n"
                f"  - Web3D Binary: `{asset.glb_path.name}`\n"
                f"  - 3D Printing / CAD: `{asset.stl_path.name}`\n"
                f"  - Wavefront Mesh: `{asset.obj_path.name}`\n"
                f"  - Preview Thumbnail: `{asset.thumb_path.name if asset.thumb_path else 'None'}`\n\n"
                f"Interactive 360° model loaded into the 3D Studio viewport. Drag to rotate 360°, scroll to zoom, or open in Blender."
            )
            return True, card, None
        else:
            return True, f"Failed to generate 3D model: {msg}", None

    def get_system_diagnostics(self) -> str:
        """Collect real-time system metrics (RAM, OS, CPU, disk)."""
        metrics = []
        metrics.append(f"• OS: {platform.system()} {platform.release()} ({platform.machine()})")

        # Memory via /proc/meminfo or free or psutil
        try:
            import psutil
            mem = psutil.virtual_memory()
            metrics.append(f"• RAM: {mem.used / (1024**3):.1f} GB used / {mem.total / (1024**3):.1f} GB total ({mem.percent}% used)")
            cpu = psutil.cpu_percent(interval=0.1)
            metrics.append(f"• CPU Load: {cpu}%")
            disk = psutil.disk_usage(os.getcwd())
            metrics.append(f"• Workspace Storage: {disk.free / (1024**3):.1f} GB free of {disk.total / (1024**3):.1f} GB")
        except ImportError:
            # Fallback to standard linux parsing
            try:
                with open("/proc/meminfo") as f:
                    lines = f.readlines()
                    mem_dict = {l.split(":")[0]: int(l.split(":")[1].split()[0]) for l in lines[:5]}
                    total = mem_dict.get("MemTotal", 0) / (1024 * 1024)
                    avail = mem_dict.get("MemAvailable", 0) / (1024 * 1024)
                    metrics.append(f"• RAM: {total - avail:.1f} GB used / {total:.1f} GB total ({avail:.1f} GB free)")
            except Exception:
                pass

        # Uptime
        try:
            with open("/proc/uptime") as f:
                uptime_sec = float(f.readline().split()[0])
                hours = int(uptime_sec // 3600)
                minutes = int((uptime_sec % 3600) // 60)
                metrics.append(f"• System Uptime: {hours}h {minutes}m")
        except Exception:
            pass

        return "🖥️ **System Diagnostics**:\n" + "\n".join(metrics)

    def _list_files(self, target_dir: str) -> str:
        """Safely list files in given directory without modifying anything."""
        p = Path(target_dir).expanduser().resolve()
        if not p.exists():
            return f"Directory does not exist: {p}"
        if not p.is_dir():
            return f"Path is not a directory: {p}"

        try:
            items = list(p.iterdir())
            items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
            if not items:
                return f"Directory {p} is empty."

            listing = []
            for item in items[:25]:  # limit to 25 items
                if item.is_dir():
                    listing.append(f"📁 {item.name}/")
                else:
                    size_kb = item.stat().st_size / 1024
                    listing.append(f"📄 {item.name} ({size_kb:.1f} KB)")

            total_count = len(items)
            res = f"Contents of `{p}` ({min(25, total_count)} of {total_count} items):\n" + "\n".join(listing)
            return res
        except PermissionError:
            return f"Permission denied accessing directory: {p}"
        except Exception as e:
            return f"Error reading directory: {e}"

    def _read_file(self, target_file: str) -> str:
        """Safely read content of text file."""
        p = Path(target_file).expanduser().resolve()
        if not p.exists():
            return f"File not found: {p}"
        if p.is_dir():
            return f"Path is a directory, not a file: {p}"

        try:
            # Check size (< 500 KB to avoid flooding UI)
            if p.stat().st_size > 500 * 1024:
                return f"File is too large ({p.stat().st_size / 1024:.1f} KB) for inline display."

            with open(p, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(4000)
                if len(content) == 4000:
                    content += "\n... [truncated for display]"
            return f"📄 Content of `{p.name}`:\n```\n{content}\n```"
        except PermissionError:
            return f"Permission denied reading file: {p}"
        except Exception as e:
            return f"Error reading file: {e}"

    # =========================================================================
    # DESTRUCTIVE ACTIONS & SAFETY CONFIRMATIONS
    # =========================================================================

    def _create_delete_file_action(self, target: str) -> PendingAction:
        p = Path(target).expanduser().resolve()
        action_id = f"act_{int(datetime.now().timestamp() * 1000)}"

        def execute() -> Tuple[bool, str]:
            # Critical Safety Rules
            # Never delete root or system critical paths
            blocked_prefixes = ["/etc", "/boot", "/sys", "/proc", "/dev", "/usr", "/bin", "/lib"]
            for prefix in blocked_prefixes:
                if str(p).startswith(prefix):
                    return False, f"CRITICAL: Deletion of system directory '{p}' is prohibited."

            if not p.exists():
                return False, f"File does not exist: {p}"
            if p.is_dir():
                return False, f"Target '{p}' is a directory, not a file."

            try:
                p.unlink()
                return True, f"Successfully deleted file: {p}"
            except Exception as e:
                return False, f"Failed to delete file {p}: {e}"

        action = PendingAction(
            action_id=action_id,
            action_type="DELETE_FILE",
            target=str(p),
            description=f"Permanently remove file: {p}",
            callback=execute
        )
        self.pending_actions[action_id] = action
        return action

    def _create_delete_folder_action(self, target: str) -> PendingAction:
        p = Path(target).expanduser().resolve()
        action_id = f"act_{int(datetime.now().timestamp() * 1000)}"

        def execute() -> Tuple[bool, str]:
            blocked = ["/", "/home", "/etc", "/boot", "/sys", "/proc", "/dev", "/usr", "/bin", "/lib", "/var"]
            if str(p) in blocked or any(str(p) == b for b in blocked):
                return False, f"CRITICAL: Deletion of root/system path '{p}' is prohibited."

            if not p.exists():
                return False, f"Folder does not exist: {p}"
            if not p.is_dir():
                return False, f"Target '{p}' is not a directory."

            try:
                shutil.rmtree(p)
                return True, f"Successfully removed directory and its contents: {p}"
            except Exception as e:
                return False, f"Failed to delete folder {p}: {e}"

        action = PendingAction(
            action_id=action_id,
            action_type="DELETE_DIR",
            target=str(p),
            description=f"Permanently delete directory and all contents: {p}",
            callback=execute
        )
        self.pending_actions[action_id] = action
        return action

    def execute_pending_action(self, action_id: str) -> Tuple[bool, str]:
        """Execute an action that was previously approved by user confirmation."""
        action = self.pending_actions.pop(action_id, None)
        if not action:
            return False, "Action expired or already processed."
        return action.callback()

    def cancel_pending_action(self, action_id: str) -> str:
        """Cancel a pending action."""
        action = self.pending_actions.pop(action_id, None)
        if action:
            return f"Action '{action.action_type}' for {action.target} was cancelled."
        return "Action cancelled."
