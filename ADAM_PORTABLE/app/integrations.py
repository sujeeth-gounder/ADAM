"""Integrations Module for ADAM.
Handles Ollama local LLM inference, system commands, time/date, app & website launching,
YouTube search, 3D generation routing, and safe file operations with explicit user confirmation.
"""

import datetime
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple

from app.app_registry import AppRegistry, CommandParser
from app.config import (
    ADAM_SYSTEM_PROMPT,
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_HOST,
    RECOMMENDED_MODELS,
)
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
        self.check_connection()

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

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        """Stream response chunks from Ollama via /api/chat with /api/generate fallback and detailed error reporting."""
        if not self.check_connection():
            yield "Ollama is currently offline. Please launch the Ollama service using: `ollama serve` and pull a model with: `ollama pull qwen2.5:1.5b`."
            return

        sys_msg = system_prompt or ADAM_SYSTEM_PROMPT
        
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
            with urllib.request.urlopen(req, timeout=30) as resp:
                for line in resp:
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        text_piece = chunk.get("message", {}).get("content", "")
                        if text_piece:
                            yield text_piece
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
            logger.debug(f"/api/chat attempt failed ({e}), trying /api/generate fallback...")

        # Fallback: /api/generate
        payload_generate = {
            "model": self.model,
            "prompt": prompt,
            "system": sys_msg,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }

        try:
            data = json.dumps(payload_generate).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                for line in resp:
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        text_piece = chunk.get("response", "")
                        if text_piece:
                            yield text_piece
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



class SystemIntegrations:
    """Handles desktop automations, system diagnostics, and safe file operations."""

    def __init__(self, confirmation_request_handler: Optional[Callable[[PendingAction], None]] = None):
        """
        confirmation_request_handler: callback called when a destructive operation is proposed.
        It passes a PendingAction object to the UI, which displays [Confirm] / [Cancel] buttons.
        """
        self.confirmation_request_handler = confirmation_request_handler
        self.pending_actions: Dict[str, PendingAction] = {}
        self.ollama = OllamaClient()
        self.app_registry = AppRegistry()
        self.command_parser = CommandParser(self.app_registry)

    # =========================================================================
    # INTENT PARSER & DISPATCHER
    # =========================================================================

    def try_handle_command(self, query: str) -> Tuple[bool, Optional[str], Optional[PendingAction]]:
        """
        Check if query is a direct system command (time, app, website, youtube, 3D, diagnostics, file ops).
        Returns:
            (handled: bool, response_text: Optional[str], pending_action: Optional[PendingAction])
        """
        q = query.strip()
        q_lower = q.lower()

        # Step 1: Universal Command & App Registry evaluation
        parsed = self.command_parser.parse(q)
        if parsed["handled"]:
            intent = parsed["intent"]
            if intent == "STOP_VOICE":
                return True, "Voice playback stopped.", None
            elif intent == "GET_TIME":
                now = datetime.datetime.now().strftime("%I:%M %p")
                return True, f"The current time is {now}.", None
            elif intent == "GET_DATE":
                now = datetime.datetime.now().strftime("%A, %B %d, %Y")
                return True, f"Today is {now}.", None
            elif intent == "PLAY_MUSIC":
                ok, msg = self.app_registry.play_music(
                    parsed["details"]["song_query"],
                    in_brave=parsed["details"].get("in_brave", False)
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

        # Step 2: 3D Model Generation
        three_d_match = re.search(r"(?:create|generate|make|build|export)\s+(?:a\s+)?(?:3d\s+)?(cube|sphere|cylinder|torus|scifi_drone|scifi_pedestal|pyramid|drone|pedestal|robot)(?:\s+(?:in|as|to)\s+(stl|obj))?", q_lower)
        if three_d_match:
            shape = three_d_match.group(1).strip()
            fmt = three_d_match.group(2) or "stl"
            return self._handle_3d_generation(shape, fmt)

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

    def _handle_3d_generation(self, shape: str, fmt: str) -> Tuple[bool, str, Optional[PendingAction]]:
        # Map common aliases
        alias_map = {
            "drone": "scifi_drone",
            "robot": "scifi_drone",
            "pedestal": "scifi_pedestal",
        }
        shape_clean = alias_map.get(shape, shape)
        ok, path, msg = generate_3d_object(shape_type=shape_clean, export_format=fmt)
        if ok:
            return True, f"✨ 3D Object Created Successfully!\n• Object: {shape_clean}\n• Format: {fmt.upper()}\n• Location: {path.resolve()}\n• Details: {msg}", None
        else:
            return True, f"Failed to generate 3D object: {msg}", None

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
        action_id = f"act_{int(datetime.datetime.now().timestamp() * 1000)}"

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
        action_id = f"act_{int(datetime.datetime.now().timestamp() * 1000)}"

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
