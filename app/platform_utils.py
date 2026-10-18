"""Cross-Platform Abstraction Layer for ADAM.
Provides unified OS detection, process execution, application resolution,
browser launching, and file opening across Linux, Windows, and macOS.
"""

import logging
import os
import platform
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("ADAM.Platform")


def get_os_type() -> str:
    """Return normalized OS string: 'windows', 'linux', or 'darwin'."""
    s = platform.system().lower()
    if s.startswith("win"):
        return "windows"
    if s.startswith("darwin") or s.startswith("mac"):
        return "darwin"
    return "linux"


def is_windows() -> bool:
    return get_os_type() == "windows"


def is_linux() -> bool:
    return get_os_type() == "linux"


def is_macos() -> bool:
    return get_os_type() == "darwin"


def get_user_data_dir(app_name: str = "ADAM") -> Path:
    """
    Return appropriate cross-platform directory for user data, databases, and logs.
    Windows: %APPDATA%/ADAM
    Linux:   ~/.local/share/ADAM or project data directory
    macOS:   ~/Library/Application Support/ADAM
    """
    os_name = get_os_type()
    if os_name == "windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif os_name == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path.home() / ".local" / "share"

    data_dir = base / app_name
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return data_dir


def open_url_in_browser(url: str, force_brave: bool = False) -> Tuple[bool, str]:
    """
    Cross-platform browser URL opener.
    Tries Brave first when available or requested; falls back cleanly to system browser.
    """
    clean_url = url.strip()
    lowered = clean_url.lower()
    if lowered.startswith("javascript:") or lowered.startswith("data:") or lowered.startswith("file:"):
        return False, f"Refusing to open unsafe URL scheme: {clean_url}"

    if not clean_url.startswith("http://") and not clean_url.startswith("https://") and not clean_url.startswith("spotify:"):
        clean_url = f"https://{clean_url}"

    brave_bin = find_brave_browser()

    if (force_brave or brave_bin) and brave_bin and not clean_url.startswith("spotify:"):
        try:
            if brave_bin.startswith("flatpak:"):
                app_id = brave_bin.split(":", 1)[1]
                subprocess.Popen(["flatpak", "run", app_id, clean_url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif is_macos() and brave_bin.endswith(".app"):
                subprocess.Popen(["open", "-a", brave_bin, clean_url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen([brave_bin, clean_url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True, f"Opened {clean_url} in Brave Browser."
        except Exception as e:
            logger.debug(f"Direct Brave launch failed ({e}), falling back to default browser.")

    # System default browser fallback
    try:
        if is_windows():
            try:
                os.startfile(clean_url)
                return True, f"Opened {clean_url} in default browser."
            except Exception:
                pass
        webbrowser.open(clean_url)
        return True, f"Opened {clean_url} in default browser."
    except Exception as e:
        return False, f"Could not open browser for '{clean_url}': {e}"


def open_path_in_file_manager(path: Path) -> Tuple[bool, str]:
    """Open a folder or file in the native file manager (Explorer, Nautilus/Dolphin, Finder)."""
    resolved = path.resolve()
    if not resolved.exists():
        return False, f"Path does not exist: {resolved}"

    try:
        os_name = get_os_type()
        if os_name == "windows":
            os.startfile(str(resolved))
        elif os_name == "darwin":
            subprocess.Popen(["open", str(resolved)])
        else:
            subprocess.Popen(["xdg-open", str(resolved)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, f"Opened: {resolved.name}"
    except Exception as e:
        return False, f"Could not open '{resolved}': {e}"


def find_executable_cross_platform(name: str) -> Optional[str]:
    """Find binary executable across Linux, Windows (%PATH% & standard paths), and macOS."""
    clean_name = name.strip()
    try:
        which_res = shutil.which(clean_name)
        if which_res:
            return which_res
    except Exception:
        pass

    os_name = get_os_type()
    if os_name == "windows":
        if not clean_name.lower().endswith(".exe"):
            try:
                which_exe = shutil.which(f"{clean_name}.exe")
                if which_exe:
                    return which_exe
            except Exception:
                pass

        # Search standard Windows install locations
        prog_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        prog_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        local_app = os.environ.get("LOCALAPPDATA", "")
        roaming_app = os.environ.get("APPDATA", "")

        known_paths = [
            # Brave
            os.path.join(prog_files, r"BraveSoftware\Brave-Browser\Application\brave.exe"),
            os.path.join(local_app, r"BraveSoftware\Brave-Browser\Application\brave.exe"),
            # Spotify
            os.path.join(roaming_app, r"Spotify\Spotify.exe"),
            # Chrome
            os.path.join(prog_files, r"Google\Chrome\Application\chrome.exe"),
            os.path.join(prog_files_x86, r"Google\Chrome\Application\chrome.exe"),
            # VS Code
            os.path.join(local_app, r"Programs\Microsoft VS Code\Code.exe"),
            # Blender
            os.path.join(prog_files, r"Blender Foundation\Blender\blender.exe"),
        ]
        name_lower = clean_name.lower()
        for p in known_paths:
            if p and os.path.isfile(p) and (name_lower in p.lower() or Path(p).name.lower().startswith(name_lower)):
                return p

    return None


def find_brave_browser() -> Optional[str]:
    """Find installed Brave Browser binary across platforms."""
    os_name = get_os_type()
    if os_name == "windows":
        prog_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        prog_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        for p in [
            os.path.join(prog_files, r"BraveSoftware\Brave-Browser\Application\brave.exe"),
            os.path.join(prog_files_x86, r"BraveSoftware\Brave-Browser\Application\brave.exe"),
            os.path.join(local_appdata, r"BraveSoftware\Brave-Browser\Application\brave.exe"),
        ]:
            if os.path.isfile(p):
                return p
        return shutil.which("brave.exe") or shutil.which("brave")

    elif os_name == "linux":
        for b in ["brave-browser", "brave", "brave-bin"]:
            found = shutil.which(b)
            if found:
                return found
        if shutil.which("flatpak"):
            try:
                res = subprocess.run(["flatpak", "info", "com.brave.Browser"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if res.returncode == 0:
                    return "flatpak:com.brave.Browser"
            except Exception:
                pass

    elif os_name == "darwin":
        for mp in ["/Applications/Brave Browser.app", str(Path.home() / "Applications" / "Brave Browser.app")]:
            if os.path.exists(mp):
                return mp

    return None


def find_blender_binary() -> Optional[str]:
    """Locate Blender executable across Linux, Windows, and macOS."""
    bin_in_path = shutil.which("blender") or shutil.which("blender.exe")
    if bin_in_path:
        return bin_in_path

    os_name = get_os_type()
    if os_name == "windows":
        prog_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        blender_dir = Path(prog_files) / "Blender Foundation"
        if blender_dir.is_dir():
            for exe in blender_dir.glob("**/blender.exe"):
                if exe.is_file():
                    return str(exe)
    elif os_name == "linux":
        for p in ["/usr/bin/blender", "/usr/local/bin/blender", "/opt/blender/blender"]:
            if os.path.isfile(p):
                return p
    elif os_name == "darwin":
        mac_blender = "/Applications/Blender.app/Contents/MacOS/Blender"
        if os.path.isfile(mac_blender):
            return mac_blender

    return None
