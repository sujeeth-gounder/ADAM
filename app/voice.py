"""Voice Synthesis (TTS) Module for ADAM.
Integrates Piper TTS using the Amy English voice (en_US-amy-medium).
Provides threaded queue, non-blocking audio playback, visualizer hooks, and graceful fallbacks.
"""

import logging
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from app.config import (
    PIPER_CONFIG_PATH,
    PIPER_MODEL_NAME,
    PIPER_ONNX_PATH,
)

logger = logging.getLogger("ADAM.Voice")


class PiperTTS:
    """Piper TTS Engine targeting en_US-amy-medium voice."""

    def __init__(
        self,
        onnx_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        on_start: Optional[Callable[[], None]] = None,
        on_finish: Optional[Callable[[], None]] = None,
    ):
        self.onnx_path = onnx_path or PIPER_ONNX_PATH
        self.config_path = config_path or PIPER_CONFIG_PATH
        self.on_start = on_start
        self.on_finish = on_finish

        self._queue: queue.Queue = queue.Queue()
        self._is_speaking = False
        self._stop_requested = False
        self._current_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

        # Engine mode: 'piper_bin', 'piper_py', 'pyttsx3', 'espeak', or 'dummy'
        self.engine_type = "dummy"
        self._init_engine()

        # Start playback worker thread
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _init_engine(self):
        """Determine the best available TTS backend."""
        # 1. Check for piper CLI binary
        if shutil.which("piper") and self.onnx_path.exists():
            self.engine_type = "piper_bin"
            logger.info("Initialized Piper TTS via CLI binary.")
            return

        # 2. Check for python piper package
        try:
            import piper  # noqa: F401
            if self.onnx_path.exists():
                self.engine_type = "piper_py"
                logger.info("Initialized Piper TTS via python library.")
                return
        except ImportError:
            pass

        # 3. Check for pyttsx3 fallback
        try:
            import pyttsx3
            # Test engine init
            test_engine = pyttsx3.init()
            test_engine.stop()
            self.engine_type = "pyttsx3"
            logger.info("Fallback: Initialized pyttsx3 TTS.")
            return
        except Exception:
            pass

        # 4. Check for system espeak / espeak-ng / spd-say
        if shutil.which("espeak-ng") or shutil.which("espeak") or shutil.which("spd-say"):
            self.engine_type = "system_speech"
            logger.info("Fallback: Initialized system speech (espeak/spd-say).")
            return

        # 5. Fallback dummy
        self.engine_type = "dummy"
        logger.warning("No audio TTS engine available. Speech will log to terminal.")

    def has_amy_voice_model(self) -> bool:
        """Check if Amy ONNX model files are downloaded locally."""
        return self.onnx_path.exists() and self.config_path.exists()

    def get_status_info(self) -> str:
        """Return human-readable status for UI badges."""
        if self.has_amy_voice_model():
            return f"Piper TTS (Amy Voice) [{self.engine_type}]"
        elif self.engine_type == "pyttsx3":
            return "TTS (pyttsx3 fallback - Download Amy voice)"
        elif self.engine_type == "system_speech":
            return "TTS (System Speech fallback)"
        else:
            return "TTS (Silent / Voice model pending)"

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

    def stop(self):
        """Immediately stop speaking and clear remaining queue."""
        self._stop_requested = True
        with self._lock:
            # Drain queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    break

            # Terminate active process if playing
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
        """Synthesize and play audio based on active engine."""
        # Check if Amy model has been downloaded dynamically
        if self.engine_type in ["dummy", "pyttsx3", "system_speech"] and self.has_amy_voice_model():
            if shutil.which("piper"):
                self.engine_type = "piper_bin"
            else:
                try:
                    import piper  # noqa: F401
                    self.engine_type = "piper_py"
                except ImportError:
                    pass

        # 1. Try Piper binary if model exists
        if self.engine_type == "piper_bin" and self.onnx_path.exists():
            if self._speak_with_piper_bin(text):
                return

        # 2. Try Piper Python API
        if self.engine_type == "piper_py" and self.onnx_path.exists():
            if self._speak_with_piper_py(text):
                return

        # 3. Fallback: pyttsx3
        if self.engine_type == "pyttsx3":
            if self._speak_with_pyttsx3(text):
                return

        # 4. Fallback: system speech
        if self.engine_type == "system_speech":
            if self._speak_with_system_speech(text):
                return

        # 5. Last resort: log to stdout
        logger.info(f"[ADAM Spoke]: {text}")

    def _speak_with_piper_bin(self, text: str) -> bool:
        """Synthesize with piper binary and play with paplay/aplay."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = Path(tmp_wav.name)

        try:
            piper_cmd = [
                "piper",
                "--model", str(self.onnx_path),
                "--config", str(self.config_path),
                "--output_file", str(tmp_wav_path)
            ]

            proc = subprocess.Popen(
                piper_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            _, err = proc.communicate(input=text, timeout=20)
            if proc.returncode != 0 or not tmp_wav_path.exists() or tmp_wav_path.stat().st_size == 0:
                logger.error(f"Piper binary error: {err}")
                return False

            if self._stop_requested:
                return True

            # Play WAV
            return self._play_wav_file(tmp_wav_path)
        except Exception as e:
            logger.error(f"Piper binary execution error: {e}")
            return False
        finally:
            if tmp_wav_path.exists():
                try:
                    tmp_wav_path.unlink()
                except Exception:
                    pass

    def _speak_with_piper_py(self, text: str) -> bool:
        """Synthesize using Python piper library."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = Path(tmp_wav.name)

        try:
            import wave
            from piper import PiperVoice

            voice = PiperVoice.load(str(self.onnx_path), config_path=str(self.config_path))
            with wave.open(str(tmp_wav_path), "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)

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
        """Play a WAV file using system sound tools (paplay, aplay, sounddevice)."""
        # Determine player
        player_cmd = None
        if shutil.which("paplay"):
            player_cmd = ["paplay", str(wav_path)]
        elif shutil.which("aplay"):
            player_cmd = ["aplay", "-q", str(wav_path)]
        elif shutil.which("pw-play"):
            player_cmd = ["pw-play", str(wav_path)]
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

        # Fallback to sounddevice / soundfile in python if installed
        try:
            import sounddevice as sd
            import soundfile as sf
            data, fs = sf.read(str(wav_path))
            sd.play(data, fs)
            sd.wait()
            return True
        except Exception as e:
            logger.debug(f"sounddevice playback failed: {e}")
            return False

    def _speak_with_pyttsx3(self, text: str) -> bool:
        """Fallback to pyttsx3."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            # Set speech rate & volume
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 0.95)
            # Find female English voice if available
            voices = engine.getProperty("voices")
            for v in voices:
                if "english" in v.name.lower() or "en" in v.id.lower():
                    if "female" in v.name.lower() or "zira" in v.name.lower() or "amy" in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break
            engine.say(text)
            engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")
            return False

    def _speak_with_system_speech(self, text: str) -> bool:
        """Fallback to system espeak or spd-say."""
        if shutil.which("spd-say"):
            cmd = ["spd-say", "-t", "female1", "-r", "10", text]
        elif shutil.which("espeak-ng"):
            cmd = ["espeak-ng", "-v", "en-us+f3", "-s", "160", text]
        elif shutil.which("espeak"):
            cmd = ["espeak", "-v", "en+f3", "-s", "160", text]
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
