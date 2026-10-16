"""Speech Recognition and Wake Word Module for ADAM.
Fixes Fedora/Linux/Windows/macOS microphone detection issues by filtering valid input devices,
suppressing ALSA error spam, testing live audio input, and background 'Hii' wake-word listening.
"""

import contextlib
import ctypes
import logging
import math
import os
import re
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Ensure local venv site-packages is accessible even if launched with system python
_ROOT_DIR = Path(__file__).resolve().parent.parent
_VENV_PACKAGES = [
    _ROOT_DIR / "venv" / "lib" / "python3.14" / "site-packages",
    _ROOT_DIR / "venv" / "lib64" / "python3.14" / "site-packages",
]
for p in _VENV_PACKAGES:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.config import PRIMARY_WAKE_WORD, WAKE_WORDS

logger = logging.getLogger("ADAM.Speech")


@contextlib.contextmanager
def suppress_c_stderr():
    """Silences low-level ALSA / JACK errors printed directly to stderr file descriptor."""
    if sys.platform.startswith("linux"):
        try:
            asound = ctypes.cdll.LoadLibrary("libasound.so.2")
            c_error_handler = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
            def py_error_handler(filename, line, function, err, fmt):
                pass
            c_handler = c_error_handler(py_error_handler)
            asound.snd_lib_error_set_handler(c_handler)
        except Exception:
            pass
    try:
        yield
    finally:
        pass


def list_input_microphones() -> List[Dict]:
    """
    Safely enumerate audio devices and return ONLY devices with maxInputChannels > 0.
    Filters out HDMI and other output-only devices that cause PyAudio/ALSA to crash.
    """
    valid_devices = []
    with suppress_c_stderr():
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            try:
                count = p.get_device_count()
                for i in range(count):
                    try:
                        info = p.get_device_info_by_index(i)
                        in_channels = int(info.get("maxInputChannels", 0))
                        if in_channels > 0:
                            valid_devices.append({
                                "index": i,
                                "name": info.get("name", f"Microphone #{i}"),
                                "channels": in_channels,
                                "sample_rate": int(info.get("defaultSampleRate", 16000)),
                                "host_api": info.get("hostApi", 0)
                            })
                    except Exception:
                        pass
            finally:
                p.terminate()
        except ImportError:
            logger.warning("PyAudio not available.")
        except Exception as e:
            logger.error(f"Error enumerating audio devices: {e}")

    return valid_devices


def test_microphone_level(device_index: Optional[int], duration_sec: float = 2.0) -> Tuple[bool, float, str]:
    """
    Record for duration_sec and compute normalized audio peak RMS energy (0.0 to 1.0).
    Returns: (success: bool, peak_energy: float, message: str)
    """
    with suppress_c_stderr():
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            chunk_size = 1024
            format_type = pyaudio.paInt16

            dev_info = {}
            if device_index is not None:
                try:
                    dev_info = p.get_device_info_by_index(device_index)
                except Exception:
                    pass
            else:
                try:
                    dev_info = p.get_default_input_device_info()
                except Exception:
                    pass

            native_rate = int(dev_info.get("defaultSampleRate", 16000))
            max_in = int(dev_info.get("maxInputChannels", 1))
            ch_options = [1] if max_in < 2 else [1, 2]
            rate_options = [native_rate, 48000, 44100, 16000]
            # Remove duplicates preserving order
            rate_options = list(dict.fromkeys(rate_options))

            stream = None
            rate = 16000
            last_err = None

            for ch in ch_options:
                for r in rate_options:
                    try:
                        stream = p.open(
                            format=format_type,
                            channels=ch,
                            rate=r,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=chunk_size
                        )
                        rate = r
                        break
                    except Exception as err:
                        last_err = err
                if stream is not None:
                    break

            if stream is None:
                p.terminate()
                return False, 0.0, f"Microphone stream configuration failed: {last_err}"

            max_rms = 0.0
            num_chunks = int(rate / chunk_size * duration_sec)

            for _ in range(max(1, num_chunks)):
                raw_data = stream.read(chunk_size, exception_on_overflow=False)
                count = len(raw_data) // 2
                shorts = struct.unpack(f"<{count}h", raw_data)
                sum_squares = sum(s * s for s in shorts)
                rms = math.sqrt(sum_squares / count) if count > 0 else 0.0
                if rms > max_rms:
                    max_rms = rms

            stream.stop_stream()
            stream.close()
            p.terminate()

            normalized_peak = min(1.0, max_rms / 12000.0)
            if normalized_peak > 0.03:
                return True, normalized_peak, f"Audio signal verified! (Peak: {normalized_peak*100:.1f}%)"
            else:
                return True, normalized_peak, f"Audio detected ({normalized_peak*100:.1f}%). Speak closer to the microphone."

        except Exception as e:
            return False, 0.0, f"Microphone test failed: {e}"


class SpeechListener:
    """Handles microphone capture, SpeechRecognition, and wake-word parsing."""

    def __init__(
        self,
        on_wake_detected: Optional[Callable[[str], None]] = None,
        on_listening_state_change: Optional[Callable[[bool, str], None]] = None,
        on_audio_energy: Optional[Callable[[float], None]] = None,
    ):
        self.on_wake_detected = on_wake_detected
        self.on_listening_state_change = on_listening_state_change
        self.on_audio_energy = on_audio_energy

        self.mic_available = False
        self.is_listening = False
        self.is_active_prompt = False
        self._stop_requested = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        self._wake_session_until: float = 0.0

        self.device_index: Optional[int] = None
        self.device_name: str = "Default"
        self._recognizer = None
        self._microphone = None

        self._init_speech_recognition()

    def _init_speech_recognition(self, preferred_index: Optional[int] = None):
        """Safely initialize SpeechRecognition and verify microphone hardware."""
        try:
            import speech_recognition as sr
            self._sr = sr
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = 300
            self._recognizer.dynamic_energy_threshold = True
            self._recognizer.pause_threshold = 0.8
            self._recognizer.non_speaking_duration = 0.5

            # Find valid input devices
            devices = list_input_microphones()
            if not devices:
                self.mic_available = False
                self.device_name = "No Input Device"
                logger.warning("No microphone input devices detected.")
                return

            # Choose candidate devices with priority: default/pipewire/pulse, then others
            sorted_candidates = []
            if preferred_index is not None:
                for d in devices:
                    if d["index"] == preferred_index:
                        sorted_candidates.append(d)
                        break

            # Add system mixers first
            for d in devices:
                d_name = d["name"].lower()
                if any(k in d_name for k in ["default", "pipewire", "pulse"]) and d not in sorted_candidates:
                    sorted_candidates.append(d)

            # Add remaining devices
            for d in devices:
                if d not in sorted_candidates:
                    sorted_candidates.append(d)

            # Attempt to initialize microphone on candidate devices
            initialized = False
            for candidate in sorted_candidates:
                candidate_idx = candidate["index"]
                try:
                    with suppress_c_stderr():
                        mic = sr.Microphone(device_index=candidate_idx)
                        with mic as source:
                            self._recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    self._microphone = mic
                    self.device_index = candidate_idx
                    self.device_name = candidate["name"]
                    self.mic_available = True
                    initialized = True
                    logger.info(f"Microphone initialized successfully on device [{self.device_index}] {self.device_name}")
                    break
                except Exception as ex:
                    logger.debug(f"Device [{candidate_idx}] {candidate['name']} failed init: {ex}")
                    continue

            if not initialized:
                self.mic_available = False
                self.device_name = "Unavailable"
                logger.warning("Could not initialize any detected microphone.")

        except Exception as e:
            self.mic_available = False
            self.device_name = "Unavailable"
            logger.warning(f"Speech recognition initialization error: {e}")

    def switch_microphone(self, new_index: int) -> bool:
        """Switch active microphone device dynamically."""
        was_running = self.is_listening and not self._paused
        self.stop()
        self._init_speech_recognition(preferred_index=new_index)
        if was_running and self.mic_available:
            self.start_wake_word_loop()
        return self.mic_available

    def get_status_info(self) -> str:
        """Return status string for UI badge."""
        if not self.mic_available:
            return "Microphone: Not Detected (Text mode active)"
        if self._paused:
            return f"Microphone: Muted [{self.device_name[:16]}]"
        if self.is_listening:
            return f"Microphone: 'Hii' [{self.device_name[:16]}]"
        return f"Microphone: Ready [{self.device_name[:16]}]"

    def start_wake_word_loop(self):
        """Start background daemon thread to listen for wake words."""
        if not self.mic_available or not self._microphone:
            return

        if self._thread and self._thread.is_alive():
            return

        self._stop_requested = False
        self._paused = False
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_requested = True
        self.is_listening = False

    def pause(self):
        self._paused = True
        if self.on_listening_state_change:
            self.on_listening_state_change(False, f"Muted [{self.device_name[:16]}]")

    def resume(self):
        self._paused = False
        if self.on_listening_state_change:
            self.on_listening_state_change(True, f"Listening for 'Hii' [{self.device_name[:16]}]")

    def toggle_pause(self) -> bool:
        if self._paused:
            self.resume()
            return True
        else:
            self.pause()
            return False

    def listen_once(self, timeout: float = 6.0, phrase_time_limit: float = 10.0) -> Optional[str]:
        """Record and transcribe a single phrase on demand (Push-to-Talk)."""
        if not self.mic_available or not self._microphone:
            logger.warning("Push-to-talk unavailable: No microphone.")
            return None

        prev_paused = self._paused
        self._paused = True

        if self.on_listening_state_change:
            self.on_listening_state_change(True, "Listening...")

        result_text = None
        try:
            with suppress_c_stderr(), self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = self._recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )

            result_text = self._recognizer.recognize_google(audio, language="en-US")
            logger.info(f"Push-to-talk transcribed: {result_text}")
        except Exception as e:
            logger.debug(f"Push-to-talk event: {e}")
        finally:
            self._paused = prev_paused
            if self.on_listening_state_change:
                status_msg = f"Muted [{self.device_name[:16]}]" if self._paused else f"Listening for 'Hii' [{self.device_name[:16]}]"
                self.on_listening_state_change(not self._paused, status_msg)

        return result_text

    def _listen_loop(self):
        """Continuously capture ambient audio chunks and test for wake words."""
        logger.info(f"Wake word daemon started on microphone '{self.device_name}'")
        self.is_listening = True

        try:
            with suppress_c_stderr(), self._microphone as source:
                logger.info("Calibrating microphone for ambient room noise...")
                self._recognizer.adjust_for_ambient_noise(source, duration=0.8)

            while not self._stop_requested:
                if self._paused:
                    time.sleep(0.3)
                    continue

                if self.on_listening_state_change:
                    self.on_listening_state_change(True, f"Listening for 'Hii' [{self.device_name[:16]}]")

                try:
                    with suppress_c_stderr(), self._microphone as source:
                        audio = self._recognizer.listen(
                            source,
                            timeout=3.5,
                            phrase_time_limit=8.0
                        )

                    # Trigger energy hook for visualizer animation
                    if self.on_audio_energy:
                        try:
                            raw_len = len(audio.get_raw_data())
                            self.on_audio_energy(min(1.0, raw_len / 35000.0))
                        except Exception:
                            pass

                    # Recognize phrase in English
                    try:
                        phrase = self._recognizer.recognize_google(audio, language="en-US").strip()
                        logger.info(f"Audio heard: '{phrase}'")
                        self._process_recognized_phrase(phrase)
                    except self._sr.UnknownValueError:
                        pass
                    except self._sr.RequestError as e:
                        logger.debug(f"Speech recognition service notice: {e}")
                        time.sleep(1.0)

                except self._sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    logger.debug(f"Mic loop encounter: {e}")
                    time.sleep(0.5)

        except Exception as e:
            logger.error(f"Fatal error in microphone listen loop: {e}")
        finally:
            self.is_listening = False

    def _normalize_speech_variants(self, text: str) -> str:
        """Fix common speech-recognition misinterpretations for commands."""
        t = text.lower().strip()
        replacements = [
            (r"\bwhats\s+app\b", "whatsapp"),
            (r"\bwhat's\s+app\b", "whatsapp"),
            (r"\binsta\s+gram\b", "instagram"),
            (r"\byou\s+tube\b", "youtube"),
            (r"\bwi\s+fi\b", "wi-fi"),
            (r"\bvs\s+code\b", "vscode"),
            (r"\bchat\s+gpt\b", "chatgpt"),
            (r"\bopen\s+a\s+i\b", "openai"),
        ]
        for pattern, repl in replacements:
            t = re.sub(pattern, repl, t)
        return t

    def _process_recognized_phrase(self, phrase: str):
        """Parse phrase for wake-word 'Hii' or active session commands."""
        phrase_clean = self._normalize_speech_variants(phrase)
        logger.info(f"Audio heard & normalized: '{phrase_clean}'")

        wake_patterns = [
            r"^(hii|hi|hey\s+adam|hi\s+adam|adam)\b[\s,\.!\?]*(.*)",
            r"\b(hii|hey\s+adam|hi\s+adam)\b[\s,\.!\?]*(.*)",
        ]

        detected = False
        remaining_prompt = ""

        for pattern in wake_patterns:
            match = re.match(pattern, phrase_clean)
            if match:
                detected = True
                remaining_prompt = match.group(2).strip()
                break

        now = time.time()
        # Case 1: Wake word explicitly detected
        if detected:
            if remaining_prompt:
                # Prompt provided directly with wake word (e.g. "Hii open whatsapp")
                self._wake_session_until = 0.0
                if self.on_wake_detected:
                    logger.info(f"Wake word detected with prompt: '{remaining_prompt}'")
                    self.on_wake_detected(remaining_prompt)
            else:
                # User just said "Hii" -> Open 15-second active command window
                self._wake_session_until = now + 15.0
                if self.on_wake_detected:
                    logger.info("Wake word heard alone. Opening 15s session window.")
                    self.on_wake_detected("")
            return

        # Case 2: In active command session window (after user said "Hii")
        if now < self._wake_session_until:
            # Check for dismissal words
            if any(w in phrase_clean for w in ["cancel", "never mind", "dismiss", "stop listening"]):
                logger.info("Command session dismissed by user.")
                self._wake_session_until = 0.0
                return

            self._wake_session_until = 0.0
            if self.on_wake_detected:
                logger.info(f"Command heard within active window: '{phrase_clean}'")
                self.on_wake_detected(phrase_clean)
