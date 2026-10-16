"""Cross-Platform OS Adapter for ADAM (Advanced Desktop AI Mate).
Handles operating system detection, portable relative paths, platform-specific audio playback,
application launching, Blender detection, and system diagnostics for Linux, Windows, and macOS.
"""

import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ADAM.OSAdapter")


class OSAdapter:
    """Provides unified cross-platform abstractions across Linux, Windows, and macOS."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OSAdapter, cls).__new__(cls)
            cls._instance._init_adapter()
        return cls._instance

    def _init_adapter(self):
        self.os_type = self._detect_os()
        # Portable base directory: ADAM_PORTABLE root
        self.root_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.root_dir / "data"
        self.config_dir = self.root_dir / "config"
        self.models_dir = self.root_dir / "models" / "piper"
        self.exports_dir = self.data_dir / "exports"

        # Ensure core runtime directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _detect_os() -> str:
        """Return standardized OS identifier: 'linux', 'windows', or 'darwin'."""
        system = platform.system().lower()
        if "linux" in system:
            return "linux"
        elif "windows" in system:
            return "windows"
        elif "darwin" in system or "mac" in system:
            return "darwin"
        return "linux"

    def get_os(self) -> str:
        return self.os_type

    def get_os_display_name(self) -> str:
        """Return user-friendly OS description."""
        if self.os_type == "linux":
            try:
                if Path("/etc/fedora-release").exists():
                    return f"Fedora Linux ({platform.release()})"
                elif Path("/etc/os-release").exists():
                    with open("/etc/os-release") as f:
                        for line in f:
                            if line.startswith("PRETTY_NAME="):
                                return line.split("=")[1].strip('"\n')
            except Exception:
                pass
            return f"Linux ({platform.release()})"
        elif self.os_type == "windows":
            ver = platform.version()
            return f"Windows {platform.release()} ({ver})"
        elif self.os_type == "darwin":
            return f"macOS {platform.mac_ver()[0]}"
        return platform.system()

    # =========================================================================
    # AUDIO PLAYBACK (WAV)
    # =========================================================================

    def play_wav(self, wav_path: Path, stop_checker=None) -> bool:
        """
        Play a WAV file using the most reliable native OS sound tool.
        Returns True if played successfully.
        """
        if not wav_path.exists():
            return False

        path_str = str(wav_path.resolve())

        # 1. Windows Native: winsound (zero extra dependencies, rock solid)
        if self.os_type == "windows":
            try:
                import winsound
                # SND_FILENAME = 0x00020000, SND_NODEFAULT = 0x00000002
                winsound.PlaySound(path_str, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
                return True
            except Exception as e:
                logger.debug(f"winsound failed: {e}")
                # Fallback to PowerShell sound player
                try:
                    ps_cmd = f'(New-Object Media.SoundPlayer "{path_str}").PlaySync();'
                    subprocess.run(["powershell", "-c", ps_cmd], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return True
                except Exception:
                    pass

        # 2. macOS Native: afplay (built-in on every Mac OS since 10.5)
        if self.os_type == "darwin":
            if shutil.which("afplay"):
                try:
                    proc = subprocess.Popen(["afplay", path_str], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    while proc.poll() is None:
                        if stop_checker and stop_checker():
                            proc.terminate()
                            return True
                        proc.wait(timeout=0.1)
                    return proc.returncode == 0
                except Exception as e:
                    logger.debug(f"afplay failed: {e}")

        # 3. Linux Native: paplay / aplay / pw-play
        if self.os_type == "linux":
            player = None
            if shutil.which("paplay"):
                player = ["paplay", path_str]
            elif shutil.which("pw-play"):
                player = ["pw-play", path_str]
            elif shutil.which("aplay"):
                player = ["aplay", "-q", path_str]
            elif shutil.which("play"):
                player = ["play", "-q", path_str]

            if player:
                try:
                    proc = subprocess.Popen(player, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    while proc.poll() is None:
                        if stop_checker and stop_checker():
                            proc.terminate()
                            return True
                        proc.wait(timeout=0.1)
                    return proc.returncode == 0
                except Exception as e:
                    logger.debug(f"Linux audio player failed: {e}")

        # 4. Universal Fallback: sounddevice / soundfile in Python
        try:
            import sounddevice as sd
            import soundfile as sf
            data, fs = sf.read(path_str)
            sd.play(data, fs)
            sd.wait()
            return True
        except Exception as e:
            logger.debug(f"sounddevice fallback playback failed: {e}")

        return False

    # =========================================================================
    # SYSTEM TTS FALLBACK (When Piper models are not downloaded)
    # =========================================================================

    def speak_system_fallback(self, text: str) -> bool:
        """Use OS native TTS if Piper neural voice is unavailable."""
        clean = text.replace('"', '\\"').replace("'", "\\'")

        # macOS: Built-in 'say' command
        if self.os_type == "darwin" and shutil.which("say"):
            try:
                subprocess.run(["say", text], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                pass

        # Windows: Built-in SAPI.SpVoice via PowerShell
        if self.os_type == "windows":
            try:
                ps = f'Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak("{clean}");'
                subprocess.run(["powershell", "-c", ps], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                pass

        # Linux: espeak-ng / spd-say
        if self.os_type == "linux":
            if shutil.which("spd-say"):
                try:
                    subprocess.run(["spd-say", "-t", "female1", text], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return True
                except Exception:
                    pass
            if shutil.which("espeak-ng"):
                try:
                    subprocess.run(["espeak-ng", "-v", "en-us+f3", text], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return True
                except Exception:
                    pass

        # Universal pyttsx3 fallback
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.say(text)
            eng.runAndWait()
            return True
        except Exception:
            pass

        return False

    # =========================================================================
    # APPLICATION & WEB LAUNCHING
    # =========================================================================

    def open_url_or_path(self, target: str) -> Tuple[bool, str]:
        """Launch web URL or open local path in file manager / viewer."""
        try:
            import webbrowser
            if target.startswith("http://") or target.startswith("https://"):
                webbrowser.open(target)
                return True, f"Navigated to {target}"

            # Local path
            p = Path(target).resolve()
            if self.os_type == "windows":
                os.startfile(str(p))
                return True, f"Opened {p.name}"
            elif self.os_type == "darwin":
                subprocess.Popen(["open", str(p)])
                return True, f"Opened {p.name}"
            else:
                subprocess.Popen(["xdg-open", str(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True, f"Opened {p.name}"
        except Exception as e:
            return False, f"Could not open '{target}': {e}"

    def launch_app(self, app_name: str) -> Tuple[bool, str]:
        """Cross-platform launcher for popular applications."""
        target = app_name.lower().strip()

        # OS-specific application mapping
        if self.os_type == "windows":
            win_apps = {
                "browser": ["start", "msedge"],
                "chrome": ["start", "chrome"],
                "firefox": ["start", "firefox"],
                "terminal": ["start", "wt", "powershell"],
                "powershell": ["start", "powershell"],
                "files": ["explorer"],
                "file manager": ["explorer"],
                "calculator": ["calc"],
                "text editor": ["notepad"],
                "notepad": ["notepad"],
                "blender": ["blender"],
            }
            if target in win_apps:
                try:
                    cmd = win_apps[target]
                    subprocess.Popen(cmd, shell=True)
                    return True, f"Launched {target.title()} on Windows."
                except Exception as e:
                    return False, f"Failed to launch {target}: {e}"

        elif self.os_type == "darwin":
            mac_apps = {
                "browser": "Safari",
                "safari": "Safari",
                "chrome": "Google Chrome",
                "firefox": "Firefox",
                "terminal": "Terminal",
                "files": "Finder",
                "file manager": "Finder",
                "calculator": "Calculator",
                "text editor": "TextEdit",
                "blender": "Blender",
            }
            app_id = mac_apps.get(target, target.title())
            try:
                subprocess.Popen(["open", "-a", app_id])
                return True, f"Launched {app_id} on macOS."
            except Exception as e:
                return False, f"Failed to launch {target}: {e}"

        else:
            # Linux (Fedora / GNOME / KDE)
            linux_apps = {
                "browser": ["firefox", "google-chrome", "chromium"],
                "firefox": ["firefox"],
                "chrome": ["google-chrome", "chromium"],
                "terminal": ["gnome-terminal", "konsole", "alacritty", "xterm"],
                "files": ["nautilus", "dolphin", "thunar"],
                "file manager": ["nautilus", "dolphin"],
                "calculator": ["gnome-calculator", "kcalc"],
                "text editor": ["gnome-text-editor", "gedit", "kate"],
                "blender": ["blender"],
            }
            candidates = linux_apps.get(target, [target])
            for bin_name in candidates:
                if shutil.which(bin_name):
                    try:
                        subprocess.Popen([bin_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                        return True, f"Launched {bin_name.title()} on Linux."
                    except Exception as e:
                        return False, f"Error launching {bin_name}: {e}"

        # General xdg-open fallback
        if shutil.which("xdg-open"):
            try:
                subprocess.Popen(["xdg-open", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True, f"Sent open request for '{target}'."
            except Exception:
                pass

        return False, f"Application '{app_name}' not found on this system."

    # =========================================================================
    # BLENDER AUTO-DETECTION
    # =========================================================================

    def detect_blender(self, custom_path: Optional[str] = None) -> Optional[str]:
        """Detect existing Blender executable across Linux, Windows, and macOS."""
        # 1. Custom user-specified path
        if custom_path and Path(custom_path).exists():
            return str(Path(custom_path).resolve())

        # 2. Check PATH
        path_in_env = shutil.which("blender")
        if path_in_env:
            return path_in_env

        # 3. Linux known locations
        if self.os_type == "linux":
            linux_candidates = [
                "/usr/bin/blender",
                "/usr/local/bin/blender",
                "/opt/blender/blender",
                "/var/lib/flatpak/exports/bin/org.blender.Blender",
                str(Path.home() / ".local/share/flatpak/exports/bin/org.blender.Blender"),
                "/snap/bin/blender",
            ]
            for cand in linux_candidates:
                if Path(cand).exists():
                    return cand

        # 4. Windows known locations
        elif self.os_type == "windows":
            program_files = [
                os.environ.get("ProgramFiles", r"C:\Program Files"),
                os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                os.environ.get("LOCALAPPDATA", "")
            ]
            for pf in program_files:
                if not pf:
                    continue
                bf_dir = Path(pf) / "Blender Foundation"
                if bf_dir.exists():
                    for sub in bf_dir.glob("Blender*"):
                        exe = sub / "blender.exe"
                        if exe.exists():
                            return str(exe)

        # 5. macOS known locations
        elif self.os_type == "darwin":
            mac_candidates = [
                "/Applications/Blender.app/Contents/MacOS/Blender",
                str(Path.home() / "Applications/Blender.app/Contents/MacOS/Blender"),
            ]
            for cand in mac_candidates:
                if Path(cand).exists():
                    return cand

        return None

    # =========================================================================
    # SYSTEM DIAGNOSTICS & STORAGE CHECK
    # =========================================================================

    def get_storage_info(self, path: Optional[Path] = None) -> Dict[str, float]:
        """Return total, used, free storage in Gigabytes for the volume containing path."""
        target_path = path or self.root_dir
        try:
            stat = shutil.disk_usage(str(target_path))
            gb = 1024 ** 3
            return {
                "total_gb": round(stat.total / gb, 2),
                "used_gb": round(stat.used / gb, 2),
                "free_gb": round(stat.free / gb, 2),
                "percent_used": round((stat.used / stat.total) * 100, 1)
            }
        except Exception:
            return {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0, "percent_used": 0.0}

    def get_system_diagnostics(self) -> Dict[str, str]:
        """Collect platform, CPU, RAM, and disk metrics."""
        diag = {
            "os": self.get_os_display_name(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        }

        # Memory
        try:
            import psutil
            vm = psutil.virtual_memory()
            diag["ram_total"] = f"{vm.total / (1024**3):.1f} GB"
            diag["ram_avail"] = f"{vm.available / (1024**3):.1f} GB"
            diag["ram_used_pct"] = f"{vm.percent}%"
            diag["cpu_load"] = f"{psutil.cpu_percent(interval=0.05)}%"
        except ImportError:
            # Fallback for Linux /proc/meminfo
            if self.os_type == "linux" and Path("/proc/meminfo").exists():
                try:
                    with open("/proc/meminfo") as f:
                        lines = f.readlines()
                        m = {l.split(":")[0]: int(l.split(":")[1].split()[0]) for l in lines[:5]}
                        tot = m.get("MemTotal", 0) / (1024 * 1024)
                        avl = m.get("MemAvailable", 0) / (1024 * 1024)
                        diag["ram_total"] = f"{tot:.1f} GB"
                        diag["ram_avail"] = f"{avl:.1f} GB"
                        diag["ram_used_pct"] = f"{((tot - avl) / tot) * 100:.1f}%"
                except Exception:
                    pass

        # Storage
        st = self.get_storage_info()
        diag["disk_free"] = f"{st['free_gb']} GB free of {st['total_gb']} GB"

        # Blender
        blender_path = self.detect_blender()
        diag["blender"] = blender_path if blender_path else "Not detected"

        return diag
