"""Application Registry and Universal Command Execution System for ADAM.
Handles cross-platform desktop application launching, browser routing (Brave priority),
website fallbacks, YouTube music & video searching, OS settings subpages, and natural language command parsing.
"""

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ADAM.AppRegistry")


@dataclass
class AppSpec:
    """Defines an application or service in the registry."""
    app_id: str
    display_name: str
    aliases: List[str]
    exec_candidates: Dict[str, List[str]]  # Keyed by 'linux', 'windows', 'darwin'
    website_fallback: Optional[str] = None
    force_browser: Optional[str] = None    # e.g., 'brave'
    category: str = "app"                  # 'app', 'web', 'settings', 'utility', 'media'
    description: str = ""
    custom_launcher: Optional[Callable[..., Tuple[bool, str]]] = None


class AppRegistry:
    """
    Central registry for application resolution, browser dispatching, and system settings.
    Supports Linux (Fedora/GNOME/KDE/Flatpak), Windows 10/11, and macOS.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AppRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, custom_config_path: Optional[Path] = None):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.os_type = self._detect_os()
        self.custom_config_path = custom_config_path
        self._apps: Dict[str, AppSpec] = {}
        self._alias_map: Dict[str, str] = {}
        self._desktop_cache: Optional[Dict[str, Dict]] = None

        self._register_default_specs()
        self._load_custom_config()

    @staticmethod
    def _detect_os() -> str:
        sys_name = platform.system().lower()
        if "linux" in sys_name:
            return "linux"
        elif "windows" in sys_name:
            return "windows"
        elif "darwin" in sys_name or "mac" in sys_name:
            return "darwin"
        return "linux"

    # =========================================================================
    # BRAVE & BROWSER DETECTION
    # =========================================================================

    def find_brave_browser(self) -> Optional[str]:
        """Detect the Brave browser executable on the current system."""
        # 1. PATH lookup
        for candidate in ["brave-browser", "brave-browser-stable", "brave"]:
            p = shutil.which(candidate)
            if p:
                return p

        # 2. Linux direct paths & Flatpak
        if self.os_type == "linux":
            linux_paths = [
                "/usr/bin/brave-browser",
                "/usr/bin/brave-browser-stable",
                "/usr/bin/brave",
                "/opt/brave.com/brave/brave-browser",
                "/opt/brave.com/brave/brave",
                "/var/lib/flatpak/exports/bin/com.brave.Browser",
                os.path.expanduser("~/.local/share/flatpak/exports/bin/com.brave.Browser"),
            ]
            for lp in linux_paths:
                if os.path.exists(lp) and os.access(lp, os.X_OK):
                    return lp

            # Check flatpak CLI
            if shutil.which("flatpak"):
                try:
                    res = subprocess.run(["flatpak", "info", "com.brave.Browser"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if res.returncode == 0:
                        return "flatpak:com.brave.Browser"
                except Exception:
                    pass

        # 3. Windows paths
        elif self.os_type == "windows":
            prog_files = os.environ.get("ProgramFiles", r"C:\Program Files")
            prog_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
            local_appdata = os.environ.get("LOCALAPPDATA", "")
            win_paths = [
                os.path.join(prog_files, r"BraveSoftware\Brave-Browser\Application\brave.exe"),
                os.path.join(prog_files_x86, r"BraveSoftware\Brave-Browser\Application\brave.exe"),
                os.path.join(local_appdata, r"BraveSoftware\Brave-Browser\Application\brave.exe"),
            ]
            for wp in win_paths:
                if wp and os.path.exists(wp):
                    return wp

        # 4. macOS paths
        elif self.os_type == "darwin":
            mac_paths = [
                "/Applications/Brave Browser.app",
                os.path.expanduser("~/Applications/Brave Browser.app"),
            ]
            for mp in mac_paths:
                if os.path.exists(mp):
                    return mp

        return None

    def open_url(self, url: str, force_brave: bool = False) -> Tuple[bool, str]:
        """
        Open a URL in a browser.
        If force_brave is True or Brave is requested, attempts to launch Brave directly.
        Falls back smoothly to system default browser and returns a truthful status.
        """
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"https://{url}"

        brave_bin = self.find_brave_browser()

        # If user explicitly wants Brave or default preference
        if (force_brave or brave_bin) and brave_bin:
            try:
                if brave_bin.startswith("flatpak:"):
                    app_id = brave_bin.split(":", 1)[1]
                    subprocess.Popen(["flatpak", "run", app_id, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                elif self.os_type == "darwin" and brave_bin.endswith(".app"):
                    subprocess.Popen(["open", "-a", brave_bin, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    subprocess.Popen([brave_bin, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                return True, f"Opened {url} in Brave Browser."
            except Exception as e:
                logger.warning(f"Failed to open via Brave binary ({e}), falling back to default browser.")

        # Fallback to standard system browser
        try:
            webbrowser.open(url)
            browser_name = "default browser"
            if force_brave and not brave_bin:
                return True, f"Brave not detected. Opened {url} in {browser_name}."
            return True, f"Opened {url} in {browser_name}."
        except Exception as e:
            return False, f"Could not open browser for '{url}': {e}"

    # =========================================================================
    # APPLICATION REGISTRY SPECS
    # =========================================================================

    def _register_default_specs(self):
        """Populate the standard multi-platform registry."""
        specs = [
            # 1. WhatsApp
            AppSpec(
                app_id="whatsapp",
                display_name="WhatsApp",
                aliases=["whatsapp", "whats app", "what's app", "wa"],
                exec_candidates={
                    "linux": [
                        "whatsapp-for-linux",
                        "whatsapp",
                        "flatpak:com.github.eneshecan.WhatsAppForLinux",
                        "flatpak:io.github.mimbrero.WhatsAppDesktop",
                        "whatsapp-nativefier"
                    ],
                    "windows": ["WhatsApp.exe", "whatsapp"],
                    "darwin": ["/Applications/WhatsApp.app"]
                },
                website_fallback="https://web.whatsapp.com",
                force_browser="brave",
                category="media",
                description="WhatsApp Desktop or Web messaging"
            ),

            # 2. WhatsApp Web (Direct Web)
            AppSpec(
                app_id="whatsapp_web",
                display_name="WhatsApp Web",
                aliases=["whatsapp web", "whats app web", "web whatsapp", "wa web"],
                exec_candidates={},
                website_fallback="https://web.whatsapp.com",
                force_browser="brave",
                category="web",
                description="WhatsApp Web interface in browser"
            ),

            # 3. Instagram
            AppSpec(
                app_id="instagram",
                display_name="Instagram",
                aliases=["instagram", "insta gram", "insta", "ig"],
                exec_candidates={
                    "linux": ["instagram", "flatpak:com.instagram.Instagram"],
                    "windows": ["Instagram.exe"],
                    "darwin": ["/Applications/Instagram.app"]
                },
                website_fallback="https://www.instagram.com",
                force_browser="brave",
                category="web",
                description="Instagram web and media"
            ),

            # 4. Instagram Web
            AppSpec(
                app_id="instagram_web",
                display_name="Instagram Web",
                aliases=["instagram web", "insta web", "ig web"],
                exec_candidates={},
                website_fallback="https://www.instagram.com",
                force_browser="brave",
                category="web",
                description="Instagram Web in browser"
            ),

            # 5. YouTube
            AppSpec(
                app_id="youtube",
                display_name="YouTube",
                aliases=["youtube", "you tube", "yt"],
                exec_candidates={
                    "linux": ["freetube", "flatpak:io.freetubeapp.FreeTube"],
                    "windows": ["FreeTube.exe"],
                    "darwin": ["/Applications/FreeTube.app"]
                },
                website_fallback="https://www.youtube.com",
                force_browser="brave",
                category="web",
                description="YouTube video and music streaming"
            ),

            # 6. Brave Browser
            AppSpec(
                app_id="brave",
                display_name="Brave Browser",
                aliases=["brave", "brave browser", "brave-browser"],
                exec_candidates={
                    "linux": [
                        "brave-browser",
                        "brave-browser-stable",
                        "brave",
                        "/opt/brave.com/brave/brave-browser",
                        "flatpak:com.brave.Browser"
                    ],
                    "windows": ["brave.exe", "brave"],
                    "darwin": ["/Applications/Brave Browser.app"]
                },
                website_fallback="https://brave.com",
                category="app",
                description="Brave privacy web browser"
            ),

            # 7. Google
            AppSpec(
                app_id="google",
                display_name="Google",
                aliases=["google", "google search"],
                exec_candidates={},
                website_fallback="https://www.google.com",
                force_browser="brave",
                category="web",
                description="Google Search portal"
            ),

            # 8. Gmail
            AppSpec(
                app_id="gmail",
                display_name="Gmail",
                aliases=["gmail", "google mail", "mail"],
                exec_candidates={
                    "linux": ["thunderbird", "geary", "kmail"],
                    "windows": ["outlook", "thunderbird"],
                    "darwin": ["/System/Applications/Mail.app"]
                },
                website_fallback="https://mail.google.com",
                force_browser="brave",
                category="web",
                description="Gmail Web or system email client"
            ),

            # 9. GitHub
            AppSpec(
                app_id="github",
                display_name="GitHub",
                aliases=["github", "git hub"],
                exec_candidates={
                    "linux": ["github-desktop", "flatpak:io.github.shifteight.GitHubDesktop"],
                    "windows": ["GitHubDesktop.exe"],
                    "darwin": ["/Applications/GitHub Desktop.app"]
                },
                website_fallback="https://github.com",
                force_browser="brave",
                category="web",
                description="GitHub code hosting platform"
            ),

            # 10. ChatGPT
            AppSpec(
                app_id="chatgpt",
                display_name="ChatGPT",
                aliases=["chatgpt", "chat gpt", "openai chat"],
                exec_candidates={},
                website_fallback="https://chatgpt.com",
                force_browser="brave",
                category="web",
                description="OpenAI ChatGPT Web Portal"
            ),

            # 11. Google Drive
            AppSpec(
                app_id="google_drive",
                display_name="Google Drive",
                aliases=["google drive", "drive", "gdrive"],
                exec_candidates={},
                website_fallback="https://drive.google.com",
                force_browser="brave",
                category="web",
                description="Google Drive cloud storage"
            ),

            # 12. Spotify
            AppSpec(
                app_id="spotify",
                display_name="Spotify",
                aliases=["spotify", "music app"],
                exec_candidates={
                    "linux": ["spotify", "flatpak:com.spotify.Client", "spotify-client"],
                    "windows": ["Spotify.exe", "spotify"],
                    "darwin": ["/Applications/Spotify.app"]
                },
                website_fallback="https://open.spotify.com",
                force_browser="brave",
                category="media",
                description="Spotify music client or Web Player"
            ),

            # 13. System Settings
            AppSpec(
                app_id="settings",
                display_name="Settings",
                aliases=["settings", "system settings", "control panel", "preferences"],
                exec_candidates={
                    "linux": ["gnome-control-center", "systemsettings", "xfce4-settings-manager"],
                    "windows": ["control", "start ms-settings:"],
                    "darwin": ["/System/Applications/System Settings.app"]
                },
                category="settings",
                description="Operating system configuration center",
                custom_launcher=lambda sub=None: self.open_settings(sub)
            ),

            # 14. Bluetooth Settings
            AppSpec(
                app_id="bluetooth_settings",
                display_name="Bluetooth Settings",
                aliases=["bluetooth settings", "bluetooth", "blue tooth settings", "blue tooth"],
                exec_candidates={},
                category="settings",
                description="Bluetooth device pairing and settings",
                custom_launcher=lambda: self.open_settings("bluetooth")
            ),

            # 15. Wi-Fi / Network Settings
            AppSpec(
                app_id="wifi_settings",
                display_name="Wi-Fi Settings",
                aliases=["wifi settings", "wi-fi settings", "wi fi settings", "network settings", "wifi", "wi fi"],
                exec_candidates={},
                category="settings",
                description="Wi-Fi and Network connections",
                custom_launcher=lambda: self.open_settings("wifi")
            ),

            # 16. Terminal
            AppSpec(
                app_id="terminal",
                display_name="Terminal",
                aliases=["terminal", "console", "command prompt", "bash", "ptyxis"],
                exec_candidates={
                    "linux": ["ptyxis", "gnome-terminal", "konsole", "alacritty", "kitty", "xfce4-terminal", "xterm"],
                    "windows": ["wt.exe", "powershell.exe", "cmd.exe"],
                    "darwin": ["/System/Applications/Utilities/Terminal.app"]
                },
                category="utility",
                description="System command terminal"
            ),

            # 17. File Manager
            AppSpec(
                app_id="file_manager",
                display_name="File Manager",
                aliases=["file manager", "files", "explorer", "finder", "nautilus"],
                exec_candidates={
                    "linux": ["nautilus", "dolphin", "thunar", "nemo", "pcmanfm"],
                    "windows": ["explorer.exe"],
                    "darwin": ["/System/Library/CoreServices/Finder.app"]
                },
                category="utility",
                description="System graphical file browser"
            ),

            # 18. Downloads Folder
            AppSpec(
                app_id="downloads",
                display_name="Downloads",
                aliases=["downloads", "download folder", "downloads folder", "my downloads"],
                exec_candidates={},
                category="utility",
                description="User Downloads directory",
                custom_launcher=lambda: self.open_downloads_folder()
            ),

            # 19. Calculator
            AppSpec(
                app_id="calculator",
                display_name="Calculator",
                aliases=["calculator", "calc"],
                exec_candidates={
                    "linux": ["gnome-calculator", "kcalc", "galculator", "speedcrunch"],
                    "windows": ["calc.exe"],
                    "darwin": ["/System/Applications/Calculator.app"]
                },
                category="utility",
                description="Desktop calculator utility"
            ),

            # 20. Text Editor
            AppSpec(
                app_id="text_editor",
                display_name="Text Editor",
                aliases=["text editor", "editor", "notepad", "gedit", "kate"],
                exec_candidates={
                    "linux": ["gnome-text-editor", "gedit", "kate", "kwrite", "mousepad", "code"],
                    "windows": ["notepad.exe"],
                    "darwin": ["/System/Applications/TextEdit.app"]
                },
                category="utility",
                description="Text and code document editor"
            ),

            # 21. Visual Studio Code
            AppSpec(
                app_id="code",
                display_name="Visual Studio Code",
                aliases=["vs code", "vscode", "code"],
                exec_candidates={
                    "linux": ["code", "codium", "flatpak:com.visualstudio.code"],
                    "windows": ["code.cmd", "code.exe"],
                    "darwin": ["/Applications/Visual Studio Code.app"]
                },
                website_fallback="https://code.visualstudio.com",
                category="app",
                description="Visual Studio Code IDE"
            ),

            # 22. Blender 3D
            AppSpec(
                app_id="blender",
                display_name="Blender 3D",
                aliases=["blender", "blender 3d"],
                exec_candidates={
                    "linux": ["blender", "flatpak:org.blender.Blender"],
                    "windows": ["blender.exe"],
                    "darwin": ["/Applications/Blender.app"]
                },
                website_fallback="https://www.blender.org",
                category="app",
                description="Blender 3D Creation Suite"
            ),
        ]

        for spec in specs:
            self.register_app(spec)

    def register_app(self, spec: AppSpec):
        """Add or update an application in the registry."""
        self._apps[spec.app_id] = spec
        for alias in spec.aliases:
            self._alias_map[alias.lower().strip()] = spec.app_id

    def _load_custom_config(self):
        """Load external user-defined application mappings from JSON if present."""
        search_paths = []
        if self.custom_config_path:
            search_paths.append(self.custom_config_path)

        base_dir = Path(__file__).resolve().parent.parent
        search_paths.extend([
            base_dir / "data" / "apps.json",
            base_dir / "config" / "apps.json",
            base_dir / "ADAM_PORTABLE" / "config" / "apps.json"
        ])

        for p in search_paths:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for item in data.get("apps", []):
                        app_id = item.get("id")
                        if app_id:
                            spec = AppSpec(
                                app_id=app_id,
                                display_name=item.get("display_name", app_id.title()),
                                aliases=item.get("aliases", [app_id]),
                                exec_candidates=item.get("exec_candidates", {}),
                                website_fallback=item.get("website_fallback"),
                                force_browser=item.get("force_browser"),
                                category=item.get("category", "app"),
                                description=item.get("description", "")
                            )
                            self.register_app(spec)
                    logger.info(f"Loaded custom app registry from {p}")
                    break
                except Exception as e:
                    logger.warning(f"Could not load custom apps from {p}: {e}")

    # =========================================================================
    # SYSTEM DESKTOP ENTRIES (LINUX)
    # =========================================================================

    def _get_desktop_entries(self) -> Dict[str, Dict]:
        """Index installed .desktop files on Linux systems."""
        if self._desktop_cache is not None:
            return self._desktop_cache

        entries = {}
        if self.os_type != "linux":
            self._desktop_cache = entries
            return entries

        dirs = [
            "/usr/share/applications",
            "/usr/local/share/applications",
            os.path.expanduser("~/.local/share/applications"),
            "/var/lib/flatpak/exports/share/applications",
            os.path.expanduser("~/.local/share/flatpak/exports/share/applications")
        ]

        for d in dirs:
            if not os.path.exists(d):
                continue
            try:
                for fname in os.listdir(d):
                    if fname.endswith(".desktop"):
                        fpath = os.path.join(d, fname)
                        name, exec_cmd = self._parse_desktop_file(fpath)
                        if name and exec_cmd:
                            clean_name = name.lower().strip()
                            entries[clean_name] = {
                                "file": fpath,
                                "name": name,
                                "exec": exec_cmd
                            }
            except Exception:
                pass

        self._desktop_cache = entries
        return entries

    @staticmethod
    def _parse_desktop_file(path: str) -> Tuple[Optional[str], Optional[str]]:
        name = None
        exec_cmd = None
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                in_desktop_entry = False
                for line in f:
                    line = line.strip()
                    if line == "[Desktop Entry]":
                        in_desktop_entry = True
                        continue
                    if line.startswith("[") and in_desktop_entry:
                        break
                    if in_desktop_entry:
                        if line.startswith("Name=") and not name:
                            name = line.split("=", 1)[1].strip()
                        elif line.startswith("Exec=") and not exec_cmd:
                            raw_exec = line.split("=", 1)[1].strip()
                            # Clean field codes like %u, %F, %U
                            cleaned = re.sub(r"%[a-zA-Z]", "", raw_exec).strip()
                            exec_cmd = cleaned
        except Exception:
            pass
        return name, exec_cmd

    # =========================================================================
    # LAUNCH METHODS
    # =========================================================================

    def launch(self, query_target: str, force_web: bool = False) -> Tuple[bool, str]:
        """
        Main entry point for launching applications or websites.
        Returns: (success: bool, user_message: str)
        """
        normalized = query_target.lower().strip()

        # Check explicit registered alias
        spec_id = self._alias_map.get(normalized)
        if not spec_id:
            # Substring match in aliases
            for alias, sid in self._alias_map.items():
                if normalized == alias or (len(alias) > 3 and alias in normalized):
                    spec_id = sid
                    break

        if spec_id and spec_id in self._apps:
            spec = self._apps[spec_id]
            # Custom launcher (e.g., Settings, Downloads)
            if spec.custom_launcher and not force_web:
                try:
                    return spec.custom_launcher()
                except Exception as e:
                    return False, f"Failed executing launcher for {spec.display_name}: {e}"

            # If force web requested or no executable candidates defined
            if force_web and spec.website_fallback:
                use_brave = (spec.force_browser == "brave")
                return self.open_url(spec.website_fallback, force_brave=use_brave)

            # Attempt local execution
            candidates = spec.exec_candidates.get(self.os_type, [])
            for cand in candidates:
                success, msg = self._try_spawn_candidate(cand)
                if success:
                    return True, f"Launched {spec.display_name} successfully."

            # If local binary not found, fallback to official website if defined
            if spec.website_fallback:
                use_brave = (spec.force_browser == "brave")
                ok, msg = self.open_url(spec.website_fallback, force_brave=use_brave)
                if ok:
                    browser_used = "Brave Browser" if (use_brave and self.find_brave_browser()) else "browser"
                    return True, f"{spec.display_name} is not installed locally. Opened official {spec.display_name} in {browser_used}."
                return False, f"Could not launch {spec.display_name} or open its website: {msg}"

            return False, f"Application '{spec.display_name}' is not installed on this {platform.system()} system."

        # Check Linux desktop entries
        desktop_entries = self._get_desktop_entries()
        for dname, data in desktop_entries.items():
            if normalized == dname or normalized in dname:
                exec_cmd = data["exec"]
                try:
                    subprocess.Popen(exec_cmd, shell=True, start_new_session=True)
                    return True, f"Launched {data['name']} successfully."
                except Exception as e:
                    return False, f"Found desktop entry for {data['name']} but launch failed: {e}"

        # Check if query_target is an executable directly in PATH
        bin_path = shutil.which(normalized)
        if bin_path:
            try:
                subprocess.Popen([bin_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                return True, f"Launched {normalized.title()} successfully."
            except Exception as e:
                return False, f"Failed to launch binary '{bin_path}': {e}"

        # If it looks like a domain / web address
        if "." in normalized and not normalized.endswith(".exe"):
            return self.open_url(f"https://{normalized}", force_brave=True)

        return False, f"Application '{query_target}' was not found installed on this system."

    def _find_windows_app(self, candidate: str) -> Optional[str]:
        r"""
        Locate executable on Windows using:
        1. Explicit path existence
        2. PATH lookup (shutil.which)
        3. App Paths Registry (HKLM & HKCU\Software\Microsoft\Windows\CurrentVersion\App Paths)
        4. Standard Windows Install Directories (Program Files, LocalAppData, Start Menu)
        """
        if self.os_type != "windows":
            return None

        # 1. Direct or PATH
        if os.path.isabs(candidate) and os.path.exists(candidate):
            return candidate
        which_path = shutil.which(candidate)
        if which_path:
            return which_path

        # 2. Windows Registry lookup
        try:
            import winreg
            base_keys = [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]
            subkeys = [
                rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{candidate}",
                rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{candidate}.exe"
            ]
            for bk in base_keys:
                for sk in subkeys:
                    try:
                        with winreg.OpenKey(bk, sk) as key:
                            val, _ = winreg.QueryValueEx(key, "")
                            if val and os.path.exists(val):
                                return val
                    except Exception:
                        pass
        except ImportError:
            pass

        # 3. Known Common Install Locations
        prog_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        prog_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        roaming_appdata = os.environ.get("APPDATA", "")

        common_search = [
            # WhatsApp
            os.path.join(local_appdata, r"WhatsApp\WhatsApp.exe"),
            os.path.join(prog_files, r"WhatsApp\WhatsApp.exe"),
            # Chrome / Brave
            os.path.join(prog_files, r"Google\Chrome\Application\chrome.exe"),
            os.path.join(prog_files_x86, r"Google\Chrome\Application\chrome.exe"),
            os.path.join(prog_files, r"BraveSoftware\Brave-Browser\Application\brave.exe"),
            os.path.join(prog_files_x86, r"BraveSoftware\Brave-Browser\Application\brave.exe"),
            os.path.join(local_appdata, r"BraveSoftware\Brave-Browser\Application\brave.exe"),
            # Spotify
            os.path.join(roaming_appdata, r"Spotify\Spotify.exe"),
            # VS Code
            os.path.join(local_appdata, r"Programs\Microsoft VS Code\Code.exe"),
            os.path.join(prog_files, r"Microsoft VS Code\Code.exe"),
            # Blender
            os.path.join(prog_files, r"Blender Foundation\Blender\blender.exe"),
        ]

        cand_lower = candidate.lower()
        for p in common_search:
            if p and os.path.exists(p) and (cand_lower in p.lower() or Path(p).name.lower() == cand_lower):
                return p

        return None

    def _try_spawn_candidate(self, candidate: str) -> Tuple[bool, str]:
        """Spawn a binary, flatpak, UWP URI, or absolute path without unsafe shell string execution."""
        try:
            # 1. Windows UWP / Protocol scheme (e.g. 'start ms-settings:', 'calc.exe', 'explorer.exe')
            if self.os_type == "windows":
                # Check for ms-windows-store or protocol
                if candidate.startswith("ms-") or ":" in candidate and not os.path.exists(candidate):
                    # Clean and run via os.startfile for Windows protocol handlers
                    try:
                        clean_proto = candidate.replace("start ", "").strip()
                        os.startfile(clean_proto)
                        return True, clean_proto
                    except Exception as ex:
                        logger.debug(f"Protocol launch '{candidate}' failed: {ex}")

                # Windows native utilities
                win_system_tools = {
                    "explorer.exe": ["explorer.exe"],
                    "calc.exe": ["calc.exe"],
                    "notepad.exe": ["notepad.exe"],
                    "wt.exe": ["wt.exe"],
                    "powershell.exe": ["powershell.exe"],
                    "cmd.exe": ["cmd.exe"],
                    "control": ["control"],
                }
                if candidate in win_system_tools:
                    subprocess.Popen(win_system_tools[candidate], shell=False)
                    return True, candidate

                # Windows binary search
                win_exe = self._find_windows_app(candidate)
                if win_exe:
                    subprocess.Popen([win_exe], shell=False)
                    return True, win_exe

            # 2. Linux Flatpak runner
            if candidate.startswith("flatpak:"):
                app_id = candidate.split(":", 1)[1]
                if shutil.which("flatpak"):
                    chk = subprocess.run(["flatpak", "info", app_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if chk.returncode == 0:
                        subprocess.Popen(["flatpak", "run", app_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                        return True, f"flatpak run {app_id}"
                return False, "Flatpak app not installed"

            # 3. macOS App Bundle or Path
            if self.os_type == "darwin":
                if candidate.endswith(".app") and os.path.exists(candidate):
                    subprocess.Popen(["open", candidate], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return True, candidate

            # 4. Absolute or explicit path
            if "/" in candidate or "\\" in candidate:
                if os.path.exists(candidate):
                    if os.access(candidate, os.X_OK) or self.os_type == "windows":
                        subprocess.Popen([candidate], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=(self.os_type != "windows"))
                        return True, candidate

            # 5. Standard PATH binary
            bin_path = shutil.which(candidate)
            if bin_path:
                subprocess.Popen([bin_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=(self.os_type != "windows"))
                return True, bin_path

        except Exception as e:
            logger.debug(f"Candidate '{candidate}' spawn failed: {e}")
            return False, str(e)

        return False, "Not found"

    # =========================================================================
    # SPECIALIZED LAUNCHERS
    # =========================================================================

    def open_settings(self, subpage: Optional[str] = None) -> Tuple[bool, str]:
        """Cross-platform system settings opener with subpage navigation."""
        sub = (subpage or "").lower().strip()

        # Linux (Fedora / GNOME / KDE)
        if self.os_type == "linux":
            # GNOME Control Center
            if shutil.which("gnome-control-center"):
                cmd = ["gnome-control-center"]
                label = "System Settings"
                if "bluetooth" in sub:
                    cmd.append("bluetooth")
                    label = "Bluetooth Settings"
                elif "wifi" in sub or "wi-fi" in sub or "network" in sub:
                    cmd.append("wifi")
                    label = "Wi-Fi & Network Settings"
                elif "sound" in sub or "audio" in sub:
                    cmd.append("sound")
                    label = "Sound Settings"
                elif "display" in sub or "screen" in sub:
                    cmd.append("display")
                    label = "Display Settings"
                
                try:
                    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                    return True, f"Opened {label} on Fedora."
                except Exception as e:
                    return False, f"Failed to open GNOME Control Center: {e}"

            # KDE System Settings
            if shutil.which("systemsettings"):
                try:
                    subprocess.Popen(["systemsettings"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                    return True, "Opened KDE System Settings."
                except Exception as e:
                    return False, f"Failed to open systemsettings: {e}"

            # Fallback settings manager
            for sm in ["xfce4-settings-manager", "mate-control-center"]:
                if shutil.which(sm):
                    subprocess.Popen([sm], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                    return True, f"Opened {sm}."

        # Windows
        elif self.os_type == "windows":
            win_uri = "ms-settings:"
            label = "Windows Settings"
            if "bluetooth" in sub:
                win_uri = "ms-settings:bluetooth"
                label = "Bluetooth Settings"
            elif "wifi" in sub or "network" in sub:
                win_uri = "ms-settings:network-wifi"
                label = "Wi-Fi & Network Settings"

            try:
                subprocess.Popen(f"start {win_uri}", shell=True)
                return True, f"Opened {label}."
            except Exception as e:
                return False, f"Failed opening Windows Settings: {e}"

        # macOS
        elif self.os_type == "darwin":
            try:
                if sub and "bluetooth" in sub:
                    subprocess.Popen(["open", "/System/Library/PreferencePanes/Bluetooth.prefPane"])
                    return True, "Opened Bluetooth Settings on macOS."
                elif sub and ("wifi" in sub or "network" in sub):
                    subprocess.Popen(["open", "/System/Library/PreferencePanes/Network.prefPane"])
                    return True, "Opened Network Settings on macOS."
                else:
                    subprocess.Popen(["open", "/System/Applications/System Settings.app"])
                    return True, "Opened System Settings on macOS."
            except Exception as e:
                return False, f"Failed opening macOS System Settings: {e}"

        return False, "Could not locate a settings application for this system."

    def open_downloads_folder(self) -> Tuple[bool, str]:
        """Open the user's Downloads folder in the native file manager."""
        dl_path = Path.home() / "Downloads"
        if not dl_path.exists():
            dl_path = Path.home()

        return self.open_path(dl_path)

    def open_path(self, path: Path) -> Tuple[bool, str]:
        """Open a directory or file using native OS file viewer."""
        try:
            resolved = str(path.resolve())
            if self.os_type == "windows":
                os.startfile(resolved)
            elif self.os_type == "darwin":
                subprocess.Popen(["open", resolved])
            else:
                subprocess.Popen(["xdg-open", resolved], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            return True, f"Opened folder: {path.name}"
        except Exception as e:
            return False, f"Could not open '{path}': {e}"

    # =========================================================================
    # SEARCH & MUSIC DISPATCHERS
    # =========================================================================

    def search_youtube(self, query: str, in_brave: bool = False) -> Tuple[bool, str]:
        """Search YouTube for a topic or video."""
        clean = query.strip()
        encoded = urllib.parse.quote_plus(clean)
        url = f"https://www.youtube.com/results?search_query={encoded}"
        ok, msg = self.open_url(url, force_brave=in_brave)
        browser = "Brave Browser" if (in_brave and self.find_brave_browser()) else "browser"
        if ok:
            return True, f"Searching YouTube for \"{clean}\" in {browser}."
        return False, msg

    def search_brave(self, query: str) -> Tuple[bool, str]:
        """Perform a web search using Brave Search, preferentially in Brave Browser."""
        clean = query.strip()
        encoded = urllib.parse.quote_plus(clean)
        url = f"https://search.brave.com/search?q={encoded}"
        ok, msg = self.open_url(url, force_brave=True)
        if ok:
            return True, f"Searching Brave for \"{clean}\"."
        return False, msg

    def search_web(self, query: str) -> Tuple[bool, str]:
        """Perform general web search."""
        clean = query.strip()
        encoded = urllib.parse.quote_plus(clean)
        url = f"https://www.google.com/search?q={encoded}"
        return self.open_url(url, force_brave=True)

    def resolve_first_youtube_video(self, search_query: str) -> Optional[Tuple[str, str]]:
        """
        Attempt to scrape YouTube search results page for the first video ID and title.
        Returns (video_id, title) if found, else None.
        """
        try:
            import urllib.request
            encoded = urllib.parse.quote_plus(search_query)
            url = f"https://www.youtube.com/results?search_query={encoded}"
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
                
            # Search for /watch?v=VIDEO_ID in the rendered JSON / HTML
            matches = re.findall(r'\"videoId\":\"([a-zA-Z0-9_-]{11})\"', html)
            if matches:
                vid = matches[0]
                return vid, search_query
        except Exception as e:
            logger.debug(f"Direct video lookup failed ({e}), falling back to search query.")
        return None

    def play_music(self, song_query: str, in_brave: bool = False, force_service: Optional[str] = None) -> Tuple[bool, str]:
        """
        Play requested song using Spotify-First workflow with automatic YouTube direct playback fallback.
        """
        from app.music import MusicService
        if not hasattr(self, "_music_service") or self._music_service is None:
            self._music_service = MusicService(registry=self)
        return self._music_service.play_song(song_query, force_service=force_service, force_brave=in_brave)


# =============================================================================
# NATURAL LANGUAGE COMMAND PARSER
# =============================================================================

class CommandParser:
    """
    Parses spoken or typed user input into direct execution intents.
    Handles capitalization, punctuation, speech-to-text anomalies, and phrase variations.
    """

    def __init__(self, registry: Optional[AppRegistry] = None):
        self.registry = registry or AppRegistry()

    def parse(self, query: str) -> Dict[str, Any]:
        """
        Inspect query and return parsed intent dictionary:
        {
            "handled": bool,
            "intent": str,           # 'OPEN_APP', 'OPEN_WEB', 'MUSIC', 'YT_SEARCH', 'BRAVE_SEARCH', 'SETTINGS', etc.
            "target": str,
            "force_brave": bool,
            "details": Dict
        }
        """
        raw = query.strip()
        # Clean leading/trailing punctuation and polite filler words
        cleaned = re.sub(r"^[\s,\.!\?]+|[\s,\.!\?]+$", "", raw)
        cleaned = cleaned.replace("’", "'")
        cleaned = re.sub(r"^(?:please|could you please|can you please|could you|can you|hey adam|adam)\s+", "", cleaned, flags=re.IGNORECASE).strip()
        q_lower = cleaned.lower()

        # 1. Stop Speech / Voice Commands
        if any(q_lower == p for p in ["stop voice", "stop speaking", "be quiet", "shut up", "mute voice", "stop audio", "stop"]):
            return {
                "handled": True,
                "intent": "STOP_VOICE",
                "target": "",
                "force_brave": False,
                "details": {}
            }

        # 2. Time & Date Queries (Python system clock & zoneinfo, not Ollama)
        # Check combined date & time
        if "date and time" in q_lower or "time and date" in q_lower:
            tz_m = re.search(r"(?:date\s+and\s+time|time\s+and\s+date)(?:\s+(?:in|for|at)\s+(.+))?$", q_lower)
            tz_req = tz_m.group(1).strip() if tz_m and tz_m.group(1) else ""
            tz_req = re.sub(r"\b(please|now|right now|though)\b", "", tz_req).strip()
            return {
                "handled": True,
                "intent": "GET_TIME_AND_DATE",
                "target": tz_req,
                "force_brave": False,
                "details": {"timezone": tz_req}
            }

        # Explicit weekday questions return the local weekday, not a model-generated answer.
        day_patterns = [
            r"^what\s+day\s+is\s+(?:it\s+today|today|it)$",
            r"^(?:tell\s+(?:me\s+)?|show\s+(?:me\s+)?)(?:the\s+)?day(?:\s+of\s+the\s+week)?(?:\s+today)?$",
            r"^what\s+day\s+of\s+the\s+week\s+is\s+it(?:\s+today)?$",
        ]
        for pattern in day_patterns:
            if re.search(pattern, q_lower):
                return {
                    "handled": True,
                    "intent": "GET_DAY",
                    "target": "",
                    "force_brave": False,
                    "details": {"timezone": ""},
                }

        # Check Date Queries
        date_patterns = [
            r"^(?:tell\s+(?:me\s+)?|show\s+(?:me\s+)?|what(?:'s|\s+is|\s+s)?\s+)?(?:the\s+)?(?:today(?:'s|\s+s)?\s+date|date\s+today|date)(?:\s+(?:in|for|at)\s+(.+))?$",
            r"^what\s+date\s+is\s+it(?:\s+(?:in|for|at)\s+(.+))?$",
            r"^what\s+day\s+is\s+(?:it|today)(?:\s+(?:in|for|at)\s+(.+))?$",
            r"^(?:current\s+date)(?:\s+(?:in|for|at)\s+(.+))?$",
            r"^date\s+please$",
            r"^(?:tell\s+(?:me\s+)?|show\s+(?:me\s+)?)date$",
            r"^today(?:'s)?\s+date$",
        ]
        for dp in date_patterns:
            d_match = re.search(dp, q_lower)
            if d_match:
                tz_req = d_match.group(1).strip() if d_match.lastindex and d_match.group(1) else ""
                tz_req = re.sub(r"\b(please|now|right now|though)\b", "", tz_req).strip()
                return {
                    "handled": True,
                    "intent": "GET_DATE",
                    "target": tz_req,
                    "force_brave": False,
                    "details": {"timezone": tz_req}
                }

        # Check Time Queries (e.g. 'Show the current Indian time', 'Indian time', 'tell me time in IST')
        # 1. Adjective before time: 'indian time', 'ist time', 'show current indian time', 'tell me indian time'
        adj_match = re.search(r"^(?:(?:tell|show|give)\s+(?:me\s+)?|(?:what(?:'s|\s+is)?\s+(?:the\s+)?))?(?:the\s+)?(?:current\s+)?([a-z\s]+?)\s+time(?:\s+(?:please|now|right now))?$", q_lower)
        if adj_match:
            cand = adj_match.group(1).strip()
            cand_clean = re.sub(r"\b(current|local|the)\b", "", cand).strip()
            if cand_clean in ("india", "indian", "ist", "isd", "utc", "gmt", "london", "tokyo", "new york", "california", "pacific", "eastern", "central"):
                return {
                    "handled": True,
                    "intent": "GET_TIME",
                    "target": cand_clean,
                    "force_brave": False,
                    "details": {"timezone": cand_clean}
                }

        # 2. General Time queries
        time_patterns = [
            r"^what(?:'s|\s+is)?\s+(?:the\s+)?time(?:\s+now)?$",
            r"^(?:tell\s+(?:me\s+)?|show\s+(?:me\s+)?|give\s+(?:me\s+)?|what(?:'s|\s+is|\s+s)?\s+)?(?:the\s+)?(?:current\s+)?time(?:\s+(?:in|for|at)\s+(.+))?$",
            r"^what\s+time\s+is\s+it(?:\s+(?:in|for|at)\s+(.+))?$",
            r"^current\s+time(?:\s+(?:in|for|at)\s+(.+))?$",
            r"^time\s+(?:in|for|at)\s+(.+)$",
            r"^time\s+please$",
            r"^what\s+is\s+the\s+time$",
            r"^what\s+is\s+time$",
            r"^time\s+now$",
            r"^tell\s+(?:me\s+)?time$",
            r"^show\s+(?:me\s+)?time$",
        ]
        for tp in time_patterns:
            t_match = re.search(tp, q_lower)
            if t_match:
                tz_req = t_match.group(1).strip() if t_match.lastindex and t_match.group(1) else ""
                tz_req = re.sub(r"\b(please|now|right now|though)\b", "", tz_req).strip()
                return {
                    "handled": True,
                    "intent": "GET_TIME",
                    "target": tz_req,
                    "force_brave": False,
                    "details": {"timezone": tz_req}
                }

        # 3. Play Songs in Brave / Play songs
        # e.g., "play Tamil songs in Brave", "play songs in Brave", "play English songs", "play A.R. Rahman songs"
        play_brave_match = re.search(r"^(?:play|listen to)\s+(.+?)\s+in\s+brave(?:\s+browser)?$", cleaned, re.IGNORECASE)
        if play_brave_match:
            song_spec = play_brave_match.group(1).strip()
            return {
                "handled": True,
                "intent": "PLAY_MUSIC",
                "target": song_spec,
                "force_brave": True,
                "details": {"song_query": song_spec, "in_brave": True}
            }

        # e.g., "play songs in brave"
        if q_lower in ["play songs in brave", "play song in brave", "play music in brave"]:
            return {
                "handled": True,
                "intent": "PLAY_MUSIC",
                "target": "songs",
                "force_brave": True,
                "details": {"song_query": "songs", "in_brave": True}
            }

        # e.g., "play Tamil songs", "play English songs", "play Vaadi Pulla Vaadi"
        play_gen_match = re.search(r"^(?:play|listen to)\s+(.+)$", cleaned, re.IGNORECASE)
        if play_gen_match:
            song_spec = play_gen_match.group(1).strip()
            force_service = None
            if re.search(r"\s+on\s+spotify\b", song_spec, re.IGNORECASE):
                force_service = "spotify"
                song_spec = re.sub(r"\s+on\s+spotify\b", "", song_spec, flags=re.IGNORECASE).strip()
            elif re.search(r"\s+on\s+youtube\b", song_spec, re.IGNORECASE) or q_lower.startswith("play on youtube"):
                force_service = "youtube"
                song_spec = re.sub(r"\s+on\s+youtube\b", "", song_spec, flags=re.IGNORECASE).strip()

            # If specifies "in brave" at end
            in_brave = bool(re.search(r"\s+in\s+brave.*$", song_spec, re.IGNORECASE))
            if in_brave:
                song_spec = re.sub(r"\s+in\s+brave.*$", "", song_spec, flags=re.IGNORECASE).strip()

            return {
                "handled": True,
                "intent": "PLAY_MUSIC",
                "target": song_spec,
                "force_brave": in_brave,
                "details": {"song_query": song_spec, "in_brave": in_brave, "force_service": force_service}
            }

        # 4. Search Brave for ...
        # e.g., "search Brave for machine learning"
        brave_search_match = re.search(r"^(?:search\s+brave\s+for|search\s+in\s+brave\s+for|brave\s+search\s+for)\s+(.+)$", cleaned, re.IGNORECASE)
        if brave_search_match:
            search_query = brave_search_match.group(1).strip()
            return {
                "handled": True,
                "intent": "BRAVE_SEARCH",
                "target": search_query,
                "force_brave": True,
                "details": {"query": search_query}
            }

        # 5. Search YouTube for ...
        # e.g., "search YouTube for Python tutorials", "search on youtube for ..."
        yt_search_match = re.search(r"^(?:search\s+youtube\s+for|search\s+on\s+youtube\s+for|youtube\s+search\s+for|play\s+on\s+youtube)\s+(.+)$", cleaned, re.IGNORECASE)
        if yt_search_match:
            search_query = yt_search_match.group(1).strip()
            in_brave = bool(re.search(r"\s+in\s+brave.*$", search_query, re.IGNORECASE))
            if in_brave:
                search_query = re.sub(r"\s+in\s+brave.*$", "", search_query, flags=re.IGNORECASE).strip()
            return {
                "handled": True,
                "intent": "YOUTUBE_SEARCH",
                "target": search_query,
                "force_brave": in_brave,
                "details": {"query": search_query, "in_brave": in_brave}
            }

        # 6. Search the web for ...
        web_search_match = re.search(r"^(?:search\s+the\s+web\s+for|search\s+web\s+for|google\s+for|google)\s+(.+)$", q_lower)
        if web_search_match:
            search_query = web_search_match.group(1).strip()
            return {
                "handled": True,
                "intent": "WEB_SEARCH",
                "target": search_query,
                "force_brave": True,
                "details": {"query": search_query}
            }

        # 7. Open WhatsApp Web / Instagram Web (Explicit Web variants)
        if q_lower in ["open whatsapp web", "open whats app web", "launch whatsapp web", "whatsapp web"]:
            return {
                "handled": True,
                "intent": "OPEN_APP",
                "target": "whatsapp_web",
                "force_brave": True,
                "details": {"force_web": True}
            }

        if q_lower in ["open instagram web", "open insta web", "launch instagram web", "instagram web"]:
            return {
                "handled": True,
                "intent": "OPEN_APP",
                "target": "instagram_web",
                "force_brave": True,
                "details": {"force_web": True}
            }

        # 8. Open Specific Settings Subpages
        if any(q_lower == p for p in ["open bluetooth settings", "open bluetooth", "bluetooth settings"]):
            return {
                "handled": True,
                "intent": "OPEN_SETTINGS",
                "target": "bluetooth",
                "force_brave": False,
                "details": {"subpage": "bluetooth"}
            }

        if any(q_lower == p for p in ["open wifi settings", "open wi-fi settings", "open wi fi settings", "open network settings", "wifi settings"]):
            return {
                "handled": True,
                "intent": "OPEN_SETTINGS",
                "target": "wifi",
                "force_brave": False,
                "details": {"subpage": "wifi"}
            }

        # 9. Open Applications / Websites / Utilities
        # e.g., "open whatsapp", "open brave", "open downloads", "open settings"
        open_match = re.search(r"^(?:open|launch|start|run|go\s+to)(?:\s+the)?\s+([a-zA-Z0-9_\-\.\s]+)$", q_lower)
        if open_match:
            target = open_match.group(1).strip()
            # Exclude false positives that are actually 3D shapes or file operations
            if target not in ["time", "date", "cube", "sphere", "cylinder", "file", "folder", "directory"]:
                return {
                    "handled": True,
                    "intent": "OPEN_APP",
                    "target": target,
                    "force_brave": False,
                    "details": {}
                }

        # Direct app name spoken without "open" (e.g. user just says "whatsapp", "brave", "youtube")
        normalized_aliases = self.registry._alias_map
        if q_lower in normalized_aliases:
            return {
                "handled": True,
                "intent": "OPEN_APP",
                "target": q_lower,
                "force_brave": False,
                "details": {}
            }

        return {
            "handled": False,
            "intent": "UNKNOWN",
            "target": "",
            "force_brave": False,
            "details": {}
        }
