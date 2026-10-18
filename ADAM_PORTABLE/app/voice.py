"""Voice Synthesis (TTS) Module for ADAM.
Supports multiple Piper TTS voices (Amy, Lessac, Ryan, Alba, etc.), pyttsx3,
and system speech engines with dynamic voice switching, speed/volume control,
voice previews, and persistence.
"""

import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Ensure local venv site-packages is accessible
_ROOT_DIR = Path(__file__).resolve().parent.parent
_VENV_CANDIDATES = [
    _ROOT_DIR / "venv",
    _ROOT_DIR.parent / "venv",
]
for venv_root in _VENV_CANDIDATES:
    if venv_root.exists():
        for sp in venv_root.glob("lib*/python*/site-packages"):
            if sp.exists() and str(sp) not in sys.path:
                sys.path.insert(0, str(sp))

from app.config import (
    PIPER_CONFIG_PATH,
    PIPER_MODEL_NAME,
    PIPER_ONNX_PATH,
)

logger = logging.getLogger("ADAM.Voice")


def clean_text_for_speech(text: str) -> str:
    """
    Sanitize raw responses (removing URLs, markdown, emojis, asterisks)
    so neural Piper TTS synthesizes immediately without multi-minute hangs or phonetic stalling.
    """
    if not text:
        return ""
    t = text.strip()

    # 1. Remove URLs (e.g. https://youtu.be/... or http://...)
    t = re.sub(r"https?://\S+", "", t)

    # 2. Markdown links: [Link text](url) -> Link text
    t = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", t)

    # 3. Markdown styling (*, **, `, #)
    t = re.sub(r"\*\*([^\*]+)\*\*", r"\1", t)
    t = re.sub(r"\*([^\*]+)\*", r"\1", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"#+\s*", "", t)

    # 4. Remove emojis and unusual symbols (range: \U00010000 to \U0010ffff)
    t = re.sub(r"[\U00010000-\U0010ffff]", "", t)
    t = re.sub(r"[^\w\s\.,!\?:;'\"]", " ", t)

    # 5. Remove empty parentheses / brackets leftover from URL removal
    t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r"\[\s*\]", "", t)

    # 6. Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # 7. For lengthy responses (> 350 chars), speak the primary 2-3 sentences for fast conversational response
    if len(t) > 350:
        sentences = re.split(r"(?<=[.!?])\s+", t)
        shortened = ""
        for s in sentences:
            if len(shortened) + len(s) < 320:
                shortened += (" " if shortened else "") + s
            else:
                break
        t = shortened or sentences[0]

    return t.strip()


@dataclass
class VoiceProfile:
    """Represents a TTS voice available for selection."""
    id: str
    name: str
    provider: str         # 'piper', 'pyttsx3', 'system'
    onnx_path: Optional[Path] = None
    config_path: Optional[Path] = None
    pyttsx3_id: Optional[str] = None
    language: str = "en"
    gender: str = "female"
    description: str = ""


class PiperTTS:
    """
    Advanced TTS Engine supporting Piper multi-voice models, speed/volume adjustments,
    audio previews, and graceful system fallbacks.
    """

    def __init__(
        self,
        onnx_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        on_start: Optional[Callable[[], None]] = None,
        on_finish: Optional[Callable[[], None]] = None,
    ):
        self.on_start = on_start
        self.on_finish = on_finish

        self._queue: queue.Queue = queue.Queue()
        self._is_speaking = False
        self._stop_requested = False
        self._current_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

        # Config directory & persistence
        self.data_dir = _ROOT_DIR / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.voice_settings_file = self.data_dir / "voice_settings.json"

        # Speech rate & volume settings
        self.speed: float = 1.0       # 0.5 to 2.0 (Piper length_scale: inverse)
        self.volume: float = 1.0      # 0.1 to 2.0

        # Discovered voice profiles
        self.available_voices: Dict[str, VoiceProfile] = {}
        self.active_voice_id: str = "piper:en_US-amy-medium"

        # Find Piper executable
        self.piper_bin = self._find_piper_binary()

        # Discover all voices and load saved preferences
        self.refresh_voices(default_onnx=onnx_path, default_config=config_path)
        self._load_saved_settings()

        # Engine mode: 'piper_bin', 'piper_py', 'pyttsx3', 'system_speech', or 'dummy'
        self.engine_type = "dummy"
        self._evaluate_engine()

        # Start playback worker thread
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _find_piper_binary(self) -> Optional[str]:
        """Locate piper CLI in system PATH or ADAM virtual environment."""
        # 1. System PATH
        which_path = shutil.which("piper")
        if which_path:
            return which_path

        # 2. Local venv
        venv_candidates = [
            _ROOT_DIR / "venv" / "bin" / "piper",
            _ROOT_DIR / "venv" / "Scripts" / "piper.exe",
            _ROOT_DIR.parent / "venv" / "bin" / "piper",
        ]
        for vc in venv_candidates:
            if vc.exists() and os.access(vc, os.X_OK):
                return str(vc)

        return None

    def refresh_voices(self, default_onnx: Optional[Path] = None, default_config: Optional[Path] = None) -> List[VoiceProfile]:
        """Scan all known voice directories for Piper .onnx models and pyttsx3 voices."""
        self.available_voices.clear()

        # Known search paths for Piper voices
        search_dirs = [
            _ROOT_DIR / "voices",
            _ROOT_DIR / "models" / "piper",
            _ROOT_DIR / "models",
            Path.home() / "ADAM" / "voices",
            Path.home() / ".local" / "share" / "piper" / "voices",
        ]
        if default_onnx and default_onnx.parent not in search_dirs:
            search_dirs.insert(0, default_onnx.parent)

        # 1. Scan Piper ONNX models
        for sdir in search_dirs:
            if sdir.exists():
                for onnx in sdir.glob("*.onnx"):
                    cfg = onnx.with_suffix(".onnx.json")
                    if not cfg.exists():
                        cfg = onnx.parent / f"{onnx.name}.json"

                    voice_id = f"piper:{onnx.stem}"
                    display_name = onnx.stem.replace("_", " ").replace("-", " ").title()
                    
                    profile = VoiceProfile(
                        id=voice_id,
                        name=f"Piper: {display_name}",
                        provider="piper",
                        onnx_path=onnx,
                        config_path=cfg if cfg.exists() else None,
                        language=onnx.stem.split("-")[0] if "-" in onnx.stem else "en",
                        description=f"Neural ONNX model at {onnx.name}"
                    )
                    self.available_voices[voice_id] = profile

        # Ensure Amy voice is explicitly registered if present
        amy_onnx = default_onnx or PIPER_ONNX_PATH
        amy_cfg = default_config or PIPER_CONFIG_PATH
        if amy_onnx.exists() and "piper:en_US-amy-medium" not in self.available_voices:
            self.available_voices["piper:en_US-amy-medium"] = VoiceProfile(
                id="piper:en_US-amy-medium",
                name="Piper: Amy (English Natural)",
                provider="piper",
                onnx_path=amy_onnx,
                config_path=amy_cfg if amy_cfg.exists() else None,
                language="en_US",
                gender="female",
                description="Clear, natural female English voice"
            )

        # 2. Scan pyttsx3 system voices
        try:
            import pyttsx3
            engine = pyttsx3.init()
            sys_voices = engine.getProperty("voices") or []
            for v in sys_voices:
                # Include English / major voices
                v_name = v.name or v.id
                v_id = f"pyttsx3:{v.id}"
                self.available_voices[v_id] = VoiceProfile(
                    id=v_id,
                    name=f"System: {v_name}",
                    provider="pyttsx3",
                    pyttsx3_id=v.id,
                    description=f"System TTS voice ({v.id})"
                )
        except Exception as e:
            logger.debug(f"pyttsx3 voice scanning notice: {e}")

        # If no active voice selected, pick Amy or first available
        if self.active_voice_id not in self.available_voices and self.available_voices:
            if "piper:en_US-amy-medium" in self.available_voices:
                self.active_voice_id = "piper:en_US-amy-medium"
            else:
                self.active_voice_id = list(self.available_voices.keys())[0]

        return list(self.available_voices.values())

    def _load_saved_settings(self):
        """Load user-selected voice and speech parameters."""
        if not self.voice_settings_file.exists():
            return
        try:
            with open(self.voice_settings_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            saved_id = data.get("active_voice_id")
            if saved_id and saved_id in self.available_voices:
                self.active_voice_id = saved_id
            self.speed = float(data.get("speed", 1.0))
            self.volume = float(data.get("volume", 1.0))
        except Exception as e:
            logger.debug(f"Could not load voice settings: {e}")

    def save_settings(self):
        """Persist user-selected voice and speech parameters to disk."""
        try:
            data = {
                "active_voice_id": self.active_voice_id,
                "speed": self.speed,
                "volume": self.volume
            }
            with open(self.voice_settings_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save voice settings: {e}")

    def set_voice(self, voice_id: str) -> bool:
        """Switch active voice profile."""
        if voice_id in self.available_voices:
            self.active_voice_id = voice_id
            self.save_settings()
            self._evaluate_engine()
            logger.info(f"Switched active voice to: {voice_id}")
            return True
        return False

    def set_speed(self, speed: float):
        """Set voice speed factor (0.5 = slow, 1.0 = normal, 1.5 = fast)."""
        self.speed = max(0.5, min(2.0, speed))
        self.save_settings()

    def set_volume(self, volume: float):
        """Set voice volume factor (0.0 to 1.5)."""
        self.volume = max(0.0, min(1.5, volume))
        self.save_settings()

    def _evaluate_engine(self):
        """Determine active engine type for current profile."""
        prof = self.available_voices.get(self.active_voice_id)
        if prof and prof.provider == "piper" and prof.onnx_path and prof.onnx_path.exists():
            if self.piper_bin:
                self.engine_type = "piper_bin"
                return
            try:
                import piper  # noqa: F401
                self.engine_type = "piper_py"
                return
            except ImportError:
                pass

        if prof and prof.provider == "pyttsx3":
            self.engine_type = "pyttsx3"
            return

        # Fallbacks
        try:
            import pyttsx3
            self.engine_type = "pyttsx3"
            return
        except Exception:
            pass

        if shutil.which("spd-say") or shutil.which("espeak-ng") or shutil.which("espeak"):
            self.engine_type = "system_speech"
            return

        self.engine_type = "dummy"

    def get_status_info(self) -> str:
        """Return human-readable status for UI badge."""
        prof = self.available_voices.get(self.active_voice_id)
        if prof:
            return f"Voice: {prof.name} [{self.engine_type}]"
        return f"Voice: [{self.engine_type}]"

    def is_speaking(self) -> bool:
        return self._is_speaking

    def speak(self, text: str, interrupt: bool = False):
        """Queue text to be spoken. If interrupt is True, clear existing speech."""
        clean_text = text.strip()
        if not clean_text:
            return

        if interrupt:
            self.stop()

        self._queue.put(clean_text)

    def preview_voice(self, text: str = "Hello! I am ADAM, your desktop AI companion."):
        """Play immediate voice preview."""
        self.speak(text, interrupt=True)

    def stop(self):
        """Immediately stop speaking and clear remaining queue."""
        self._stop_requested = True
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    break

            if self._current_process and self._current_process.poll() is None:
                try:
                    self._current_process.terminate()
                except Exception:
                    pass
                self._current_process = None

        self._is_speaking = False
        if self.on_finish:
            try:
                self.on_finish()
            except Exception:
                pass

    def _worker_loop(self):
        """Continuously process speech queue in background thread."""
        while True:
            text = self._queue.get()
            self._stop_requested = False
            self._is_speaking = True

            if self.on_start:
                try:
                    self.on_start()
                except Exception as e:
                    logger.debug(f"on_start callback error: {e}")

            try:
                self._synthesize_and_play(text)
            except Exception as e:
                logger.error(f"Error during TTS synthesis/playback: {e}")
            finally:
                self._is_speaking = False
                self._queue.task_done()
                if self.on_finish:
                    try:
                        self.on_finish()
                    except Exception as e:
                        logger.debug(f"on_finish callback error: {e}")

    def _synthesize_and_play(self, text: str):
        """Synthesize and play audio with active profile."""
        prof = self.available_voices.get(self.active_voice_id)

        # 1. Try Piper binary if piper voice
        if prof and prof.provider == "piper" and prof.onnx_path and prof.onnx_path.exists():
            if self.piper_bin:
                if self._speak_with_piper_bin(text, prof):
                    return

        # 2. Try Piper Python API
        if prof and prof.provider == "piper" and prof.onnx_path and prof.onnx_path.exists():
            if self._speak_with_piper_py(text, prof):
                return

        # 3. Try pyttsx3
        if self._speak_with_pyttsx3(text, prof):
            return

        # 4. Fallback system speech
        if self._speak_with_system_speech(text):
            return

        logger.info(f"[ADAM Spoke]: {text}")

    def _speak_with_piper_bin(self, text: str, prof: VoiceProfile) -> bool:
        """Synthesize using piper CLI with length-scale (speed) and volume."""
        speech_text = clean_text_for_speech(text)
        if not speech_text:
            return True

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = Path(tmp_wav.name)

        try:
            # Piper speed uses length-scale: higher = slower, lower = faster
            length_scale = 1.0 / max(0.5, self.speed)
            
            cmd = [
                self.piper_bin or "piper",
                "--model", str(prof.onnx_path),
                "--output_file", str(tmp_wav_path),
                "--length-scale", f"{length_scale:.2f}",
                "--volume", f"{self.volume:.2f}"
            ]
            if prof.config_path and prof.config_path.exists():
                cmd.extend(["--config", str(prof.config_path)])

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            _, err = proc.communicate(input=speech_text, timeout=45)
            if proc.returncode != 0 or not tmp_wav_path.exists() or tmp_wav_path.stat().st_size == 0:
                logger.error(f"Piper binary error: {err}")
                return False

            if self._stop_requested:
                return True

            return self._play_wav_file(tmp_wav_path)
        except Exception as e:
            logger.error(f"Piper execution error: {e}")
            return False
        finally:
            if tmp_wav_path.exists():
                try:
                    tmp_wav_path.unlink()
                except Exception:
                    pass

    def _speak_with_piper_py(self, text: str, prof: VoiceProfile) -> bool:
        """Synthesize using piper python library."""
        speech_text = clean_text_for_speech(text)
        if not speech_text:
            return True

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = Path(tmp_wav.name)

        try:
            import wave
            from piper import PiperVoice

            cfg_arg = str(prof.config_path) if prof.config_path and prof.config_path.exists() else None
            voice = PiperVoice.load(str(prof.onnx_path), config_path=cfg_arg)
            with wave.open(str(tmp_wav_path), "wb") as wav_file:
                voice.synthesize_wav(speech_text, wav_file)

            if self._stop_requested:
                return True

            return self._play_wav_file(tmp_wav_path)
        except Exception as e:
            logger.error(f"Piper Python library error: {e}")
            return False
        finally:
            if tmp_wav_path.exists():
                try:
                    tmp_wav_path.unlink()
                except Exception:
                    pass

    def _play_wav_file(self, wav_path: Path) -> bool:
        """Play a WAV file using winsound (Windows), pw-play, paplay, aplay, play, or sounddevice."""
        if sys.platform == "win32":
            try:
                import winsound
                with self._lock:
                    if self._stop_requested:
                        return True
                winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
                return True
            except Exception as e:
                logger.debug(f"winsound playback failed: {e}")

        player_cmd = None
        if shutil.which("pw-play"):
            player_cmd = ["pw-play", str(wav_path)]
        elif shutil.which("paplay"):
            player_cmd = ["paplay", str(wav_path)]
        elif shutil.which("aplay"):
            player_cmd = ["aplay", "-q", str(wav_path)]
        elif shutil.which("play"):
            player_cmd = ["play", "-q", str(wav_path)]

        if player_cmd:
            with self._lock:
                if self._stop_requested:
                    return True
                self._current_process = subprocess.Popen(
                    player_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            self._current_process.wait()
            with self._lock:
                self._current_process = None
            return True

        # Fallback to sounddevice
        try:
            import sounddevice as sd
            import soundfile as sf
            data, fs = sf.read(str(wav_path))
            sd.play(data * self.volume, fs)
            sd.wait()
            return True
        except Exception as e:
            logger.debug(f"sounddevice playback failed: {e}")
            return False

    def _speak_with_pyttsx3(self, text: str, prof: Optional[VoiceProfile] = None) -> bool:
        """Fallback to pyttsx3."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            # Speed & volume
            rate = int(175 * self.speed)
            engine.setProperty("rate", rate)
            engine.setProperty("volume", min(1.0, self.volume))

            if prof and prof.pyttsx3_id:
                engine.setProperty("voice", prof.pyttsx3_id)
            else:
                voices = engine.getProperty("voices") or []
                for v in voices:
                    if "en" in (v.id or "").lower() or "english" in (v.name or "").lower():
                        engine.setProperty("voice", v.id)
                        break

            engine.say(text)
            engine.runAndWait()
            return True
        except Exception as e:
            logger.debug(f"pyttsx3 synthesis error: {e}")
            return False

    def _speak_with_system_speech(self, text: str) -> bool:
        """Fallback to system spd-say / espeak / Windows SAPI."""
        if sys.platform == "win32":
            try:
                escaped = text.replace("'", "''")
                ps_script = f"Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Speak('{escaped}')"
                cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script]
                with self._lock:
                    if self._stop_requested:
                        return True
                    self._current_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                self._current_process.wait()
                with self._lock:
                    self._current_process = None
                return True
            except Exception:
                return False

        if shutil.which("spd-say"):
            cmd = ["spd-say", "-t", "female1", "-r", str(int(self.speed * 10)), text]
        elif shutil.which("espeak-ng"):
            cmd = ["espeak-ng", "-v", "en-us+f3", "-s", str(int(160 * self.speed)), text]
        elif shutil.which("espeak"):
            cmd = ["espeak", "-v", "en+f3", "-s", str(int(160 * self.speed)), text]
        else:
            return False

        try:
            with self._lock:
                if self._stop_requested:
                    return True
                self._current_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            self._current_process.wait()
            with self._lock:
                self._current_process = None
            return True
        except Exception:
            return False
