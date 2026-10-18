"""Speech Recognition and Wake Word Module for ADAM.
Integrates Faster-Whisper (1.2.1, base & small models on CPU int8) with local cache reuse,
Google Speech fallback, live microphone testing, PipeWire diagnostics, and wake-word 'Hii' detection.
"""

import contextlib
import ctypes
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure user site-packages (where Faster-Whisper is installed on Fedora) and venv site-packages are accessible
_ROOT_DIR = Path(__file__).resolve().parent.parent

def _inject_user_site_packages():
    """Ensure user site-packages (Fedora ~/.local/lib/pythonX.Y/site-packages) are on sys.path."""
    user_candidates = [
        Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        Path.home() / ".local" / "lib64" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        _ROOT_DIR / "venv",
        _ROOT_DIR.parent / "venv",
    ]
    try:
        import site
        usp = site.getusersitepackages()
        if usp:
            user_candidates.append(Path(usp))
    except Exception:
        pass

    for cand in user_candidates:
        if cand.is_dir():
            if cand.name == "site-packages":
                if str(cand) not in sys.path:
                    sys.path.insert(0, str(cand))
            else:
                for sp in cand.glob("lib*/python*/site-packages"):
                    if sp.exists() and str(sp) not in sys.path:
                        sys.path.insert(0, str(sp))

_inject_user_site_packages()

from app.config import (
    PRIMARY_WAKE_WORD,
    WAKE_WORDS,
    DEFAULT_WAKE_WORDS,
    DEFAULT_WAKE_TIMEOUT_SEC,
    DEFAULT_WAKE_COOLDOWN_SEC,
)

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


_GLOBAL_PYAUDIO = None

def get_global_pyaudio():
    """Retrieve process-wide PyAudio instance to avoid multiple conflicting PortAudio initializations."""
    global _GLOBAL_PYAUDIO
    if _GLOBAL_PYAUDIO is None:
        try:
            with suppress_c_stderr():
                import pyaudio
                _GLOBAL_PYAUDIO = pyaudio.PyAudio()
        except Exception as e:
            logger.debug(f"PyAudio initialization note: {e}")
            _GLOBAL_PYAUDIO = None
    return _GLOBAL_PYAUDIO


def _cleanup_global_pyaudio():
    global _GLOBAL_PYAUDIO
    if _GLOBAL_PYAUDIO is not None:
        try:
            with suppress_c_stderr():
                _GLOBAL_PYAUDIO.terminate()
        except Exception:
            pass
        _GLOBAL_PYAUDIO = None

import atexit
atexit.register(_cleanup_global_pyaudio)


def list_input_microphones() -> List[Dict]:
    """
    Safely enumerate audio devices using PyAudio or sounddevice.
    Returns ONLY devices with maxInputChannels > 0, filtering out output-only / HDMI devices.
    """
    valid_devices = []
    seen_names = set()

    # 1. Primary: PyAudio
    with suppress_c_stderr():
        try:
            p = get_global_pyaudio()
            if p is not None:
                count = p.get_device_count()
                for i in range(count):
                    try:
                        info = p.get_device_info_by_index(i)
                        in_channels = int(info.get("maxInputChannels", 0))
                        name = info.get("name", f"Microphone #{i}")
                        normalized_name = name.lower()
                        if (
                            in_channels > 0
                            and name not in seen_names
                            and not any(token in normalized_name for token in ("monitor", "loopback", "stereo mix", "what u hear", "speaker", "sink", "output"))
                        ):
                            seen_names.add(name)
                            valid_devices.append({
                                "index": i,
                                "name": name,
                                "channels": in_channels,
                                "sample_rate": int(info.get("defaultSampleRate", 16000)),
                                "host_api": info.get("hostApi", 0),
                                "backend": "pyaudio"
                            })
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"PyAudio enumeration notice: {e}")

    # 2. Secondary: sounddevice fallback if pyaudio returned empty
    if not valid_devices:
        try:
            import sounddevice as sd
            devs = sd.query_devices()
            for idx, d in enumerate(devs):
                in_channels = int(d.get("max_input_channels", 0))
                name = d.get("name", f"Device #{idx}")
                normalized_name = name.lower()
                if (
                    in_channels > 0
                    and name not in seen_names
                    and not any(token in normalized_name for token in ("monitor", "loopback", "stereo mix", "what u hear", "speaker", "sink", "output"))
                ):
                    seen_names.add(name)
                    valid_devices.append({
                        "index": idx,
                        "name": name,
                        "channels": in_channels,
                        "sample_rate": int(d.get("default_samplerate", 16000)),
                        "host_api": d.get("hostapi", 0),
                        "backend": "sounddevice"
                    })
        except Exception as e:
            logger.debug(f"sounddevice enumeration notice: {e}")

    # 3. Sort devices by priority so system default / digital mic comes first
    def device_priority(dev: Dict) -> int:
        n = dev["name"].lower()
        if n == "default":
            return 0
        if "digital microphone" in n or "dmic" in n or "internal microphone" in n:
            return 1
        if n in ("pipewire", "pulse"):
            return 2
        if "usb" in n or "headset" in n or "bluetooth" in n:
            return 3
        if "hw:" in n or "speaker" in n or "analog" in n:
            return 10
        return 5

    valid_devices.sort(key=device_priority)
    return valid_devices



def diagnose_microphone_system() -> Dict[str, Any]:
    """
    Diagnose actual audio subsystem status on Fedora/Linux/Windows/macOS.
    Checks PipeWire, PulseAudio, ALSA permissions, and audio devices.
    """
    diag = {
        "pyaudio_available": False,
        "sounddevice_available": False,
        "pipewire_active": False,
        "pulseaudio_active": False,
        "device_count": 0,
        "devices": [],
        "diagnosis_message": ""
    }

    try:
        import pyaudio
        diag["pyaudio_available"] = True
    except ImportError:
        pass

    try:
        import sounddevice as _sd
        diag["sounddevice_available"] = True
        try:
            import atexit
            atexit.unregister(_sd._exit_handler)
        except Exception:
            pass
    except ImportError:
        pass

    import shutil
    import subprocess
    if shutil.which("pactl"):
        try:
            res = subprocess.run(["pactl", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=0.5)
            if "PipeWire" in res.stdout:
                diag["pipewire_active"] = True
            elif "PulseAudio" in res.stdout:
                diag["pulseaudio_active"] = True
        except Exception:
            pass

    devices = list_input_microphones()
    diag["device_count"] = len(devices)
    diag["devices"] = devices

    if diag["device_count"] > 0:
        active_names = ", ".join([d["name"][:20] for d in devices[:2]])
        diag["diagnosis_message"] = f"Microphone system healthy. Detected {len(devices)} input device(s): {active_names}."
    else:
        if not diag["pyaudio_available"] and not diag["sounddevice_available"]:
            diag["diagnosis_message"] = "Audio driver libraries missing: Neither PyAudio nor sounddevice is accessible in the Python environment."
        elif not (diag["pipewire_active"] or diag["pulseaudio_active"]):
            diag["diagnosis_message"] = "No input capture devices detected. Verify audio subsystem (PipeWire/PulseAudio) or plug in a microphone/headset."
        else:
            diag["diagnosis_message"] = "Audio server active, but no input capture channels are available or permission was denied."

    return diag


def test_microphone_level(device_index: Optional[int], duration_sec: float = 2.0) -> Tuple[bool, float, str]:
    """
    Record for duration_sec and compute normalized audio peak AC energy (0.0 to 1.0) with DC offset cancellation.
    Returns: (success: bool, peak_energy: float, message: str)
    """
    with suppress_c_stderr():
        try:
            p = get_global_pyaudio()
            if p is None:
                return False, 0.0, "PyAudio audio driver not initialized."
            chunk_size = 1024
            format_type = p.get_format_from_width(2)

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
            rate_options = [16000, native_rate, 48000, 44100]
            rate_options = list(dict.fromkeys(rate_options))

            stream = None
            rate = 16000
            last_err = None

            for ch in ch_options:
                for r in rate_options:
                    candidate_stream = None
                    try:
                        candidate_stream = p.open(
                            format=format_type,
                            channels=ch,
                            rate=r,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=chunk_size
                        )
                        sample_data = candidate_stream.read(chunk_size, exception_on_overflow=False)
                        if len(sample_data) != chunk_size * ch * 2:
                            candidate_stream.stop_stream()
                            candidate_stream.close()
                            continue
                        stream = candidate_stream
                        rate = r
                        break
                    except Exception as err:
                        last_err = err
                        if candidate_stream is not None:
                            try:
                                candidate_stream.stop_stream()
                                candidate_stream.close()
                            except Exception:
                                pass
                if stream is not None:
                    break

            if stream is None:
                return False, 0.0, f"Microphone stream configuration failed: {last_err}"

            max_ac_rms = 0.0
            max_ac_peak = 0.0
            num_chunks = int(rate / chunk_size * duration_sec)

            for _ in range(max(1, num_chunks)):
                raw_data = stream.read(chunk_size, exception_on_overflow=False)
                if len(raw_data) != chunk_size * ch * 2:
                    stream.stop_stream()
                    stream.close()
                    return False, 0.0, "Microphone returned audio in an unexpected sample format."
                count = len(raw_data) // 2
                if count == 0:
                    continue
                shorts = struct.unpack(f"<{count}h", raw_data)
                dc_offset = sum(shorts) / count
                ac_sum_sq = sum((s - dc_offset) ** 2 for s in shorts)
                ac_rms = math.sqrt(ac_sum_sq / count)
                ac_peak = max(abs(s - dc_offset) for s in shorts)
                if ac_rms > max_ac_rms:
                    max_ac_rms = ac_rms
                if ac_peak > max_ac_peak:
                    max_ac_peak = ac_peak

            stream.stop_stream()
            stream.close()

            # Voice energy normalizer: speaking voice AC RMS is typically 300 - 3500
            normalized_peak = min(1.0, max_ac_rms / 3500.0)
            if max_ac_rms > 350:
                return True, normalized_peak, f"Audio signal verified! (Energy: {normalized_peak*100:.1f}%, Peak: {int(max_ac_peak)})"
            elif max_ac_rms > 100:
                return True, normalized_peak, f"Audio detected ({normalized_peak*100:.1f}%). Speak louder or closer to mic."
            else:
                return True, normalized_peak, f"Microphone is silent (RMS: {int(max_ac_rms)}). Speak into the microphone."

        except Exception as e:
            return False, 0.0, f"Microphone test failed: {e}"


test_microphone_level.__test__ = False


def record_diagnostic_sample(
    device_index: Optional[int] = None,
    duration_sec: float = 3.0,
    save_wav_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Record an explicit diagnostic sample to verify physical microphone, permissions,
    sample rate, RMS, peak amplitude, and DC bias.
    Optionally saves clean 16-bit PCM WAV file.
    """
    res = {
        "success": False,
        "device_index": device_index,
        "device_name": "Unknown",
        "sample_rate": 16000,
        "channels": 1,
        "input_channels": 1,
        "duration_sec": duration_sec,
        "samples_count": 0,
        "dc_offset": 0.0,
        "ac_rms": 0.0,
        "ac_peak": 0.0,
        "saved_path": None,
        "raw_pcm": b"",
        "is_silent": True,
        "message": ""
    }
    with suppress_c_stderr():
        try:
            p = get_global_pyaudio()
            if p is None:
                res["message"] = "PyAudio driver unavailable."
                return res

            if device_index is not None:
                try:
                    dev_info = p.get_device_info_by_index(device_index)
                except Exception:
                    dev_info = {}
            else:
                try:
                    dev_info = p.get_default_input_device_info()
                except Exception:
                    dev_info = {}

            res["device_name"] = dev_info.get("name", "Default Microphone")
            native_rate = int(dev_info.get("defaultSampleRate", 16000))
            max_channels = int(dev_info.get("maxInputChannels", 1))

            stream = None
            rate = 16000
            channels = 1
            initial_frame = None
            for ch in ([1, 2] if max_channels >= 2 else [1]):
                for r in dict.fromkeys([16000, native_rate, 48000, 44100]):
                    candidate_stream = None
                    try:
                        candidate_stream = p.open(
                            format=p.get_format_from_width(2),
                            channels=ch,
                            rate=r,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=1024
                        )
                        candidate_frame = candidate_stream.read(1024, exception_on_overflow=False)
                        if len(candidate_frame) != 1024 * ch * 2:
                            candidate_stream.stop_stream()
                            candidate_stream.close()
                            continue
                        stream = candidate_stream
                        initial_frame = candidate_frame
                        rate = r
                        channels = ch
                        break
                    except Exception:
                        if candidate_stream is not None:
                            try:
                                candidate_stream.stop_stream()
                                candidate_stream.close()
                            except Exception:
                                pass
                if stream is not None:
                    break

            if stream is None:
                res["message"] = "Could not open audio stream on selected microphone."
                return res

            res["sample_rate"] = rate
            res["input_channels"] = channels
            num_chunks = int(rate / 1024 * duration_sec)
            frames = [initial_frame]

            for _ in range(max(1, num_chunks) - 1):
                data = stream.read(1024, exception_on_overflow=False)
                if len(data) != 1024 * channels * 2:
                    stream.stop_stream()
                    stream.close()
                    res["message"] = "Microphone returned audio in an unexpected sample format."
                    return res
                frames.append(data)

            stream.stop_stream()
            stream.close()

            raw_bytes = b"".join(frames)
            import numpy as np
            captured = np.frombuffer(raw_bytes, dtype=np.int16).reshape(-1, channels).astype(np.float32)
            arr = captured.mean(axis=1)

            res["samples_count"] = len(arr)
            res["duration_sec"] = round(len(arr) / rate, 2)

            dc_bias = float(np.mean(arr))
            clean_arr = arr - dc_bias
            ac_rms = float(np.sqrt(np.mean(clean_arr ** 2)))
            ac_peak = float(np.max(np.abs(clean_arr)))

            res["dc_offset"] = round(dc_bias, 1)
            res["ac_rms"] = round(ac_rms, 1)
            res["ac_peak"] = round(ac_peak, 1)
            res["is_silent"] = bool(ac_rms < 150.0)
            res["success"] = True

            if save_wav_path:
                clean_pcm = np.clip(clean_arr, -32768, 32767).astype(np.int16).tobytes()
                res["raw_pcm"] = clean_pcm
                import wave
                wav_p = Path(save_wav_path).resolve()
                wav_p.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(wav_p), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(rate)
                    wf.writeframes(clean_pcm)
                res["saved_path"] = str(wav_p)
            else:
                res["raw_pcm"] = np.clip(clean_arr, -32768, 32767).astype(np.int16).tobytes()

            if ac_rms > 350:
                res["message"] = f"Input signal level is active (AC RMS: {ac_rms:.1f}, Peak: {int(ac_peak)}); this level check does not identify speech."
            elif ac_rms > 120:
                res["message"] = f"Input signal is low (AC RMS: {ac_rms:.1f}); check the selected device and input level."
            else:
                res["message"] = f"No measurable input signal (AC RMS: {ac_rms:.1f}). Check the selected device, mute switch, or PipeWire permissions."

            return res

        except Exception as e:
            res["message"] = f"Diagnostic recording error: {e}"
            return res


# =============================================================================
# WAKE WORD NORMALIZATION & DETECTION UTILITIES
# =============================================================================

def normalize_speech_variants(text: str) -> str:
    """Normalize common speech-recognition misinterpretations for commands and wake words."""
    if not text:
        return ""
    t = text.lower().strip()
    # Normalize punctuation marks to spaces
    t = re.sub(r"[\.,!\?:;\"\'`~]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # Collapse elongated greeting vowels (e.g. 'heyyy' -> 'hey', 'hiii' -> 'hi', 'oiii' -> 'oi')
    t = re.sub(r"\bh+e+y+\b", "hey", t)
    t = re.sub(r"\bh+a+y+\b", "hey", t)
    t = re.sub(r"\bh+[i|y]+\b", "hi", t)
    t = re.sub(r"\bo+[i|y]+\b", "oi", t)

    replacements = [
        (r"\bwhats\s+app\b", "whatsapp"),
        (r"\bwhat's\s+app\b", "whatsapp"),
        (r"\binsta\s+gram\b", "instagram"),
        (r"\byou\s+tube\b", "youtube"),
        (r"\bwi\s+fi\b", "wi-fi"),
        (r"\bvs\s+code\b", "vscode"),
        (r"\bchat\s+gpt\b", "chatgpt"),
        (r"\bopen\s+a\s+i\b", "openai"),
        (r"\b(high)\b", "hi"),
        (r"\b(atom|atem)\b", "adam"),
        (r"\b(isd)\b", "ist"),
    ]
    for pattern, repl in replacements:
        t = re.sub(pattern, repl, t)
    return t


def _update_speech_onset(
    onset_chunks: list,
    pre_buffer: list,
    frame: bytes,
    active: bool,
    pre_buffer_limit: int,
) -> bool:
    """Require two consecutive active frames while retaining interrupted onset in preroll."""
    if active:
        onset_chunks.append(frame)
        return len(onset_chunks) >= 2
    if onset_chunks:
        pre_buffer.extend(onset_chunks)
        del pre_buffer[:-pre_buffer_limit]
        onset_chunks.clear()
    return False


class DoubleClapDetector:
    """Recognize two validated sharp transients without repeating on cooldown."""

    def __init__(self, min_interval: float = 0.18, max_interval: float = 0.8, cooldown: float = 4.0):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.cooldown = cooldown
        self._first_clap_at = 0.0
        self._cooldown_until = 0.0

    def reset(self):
        self._first_clap_at = 0.0

    def update(self, is_clap_pulse: bool, now: float) -> bool:
        if now < self._cooldown_until:
            return False
        if not is_clap_pulse:
            if self._first_clap_at and now - self._first_clap_at > self.max_interval:
                self._first_clap_at = 0.0
            return False
        interval = now - self._first_clap_at
        if self._first_clap_at and self.min_interval <= interval <= self.max_interval:
            self._first_clap_at = 0.0
            self._cooldown_until = now + self.cooldown
            return True
        self._first_clap_at = now
        return False


def _resample_mono_float(audio, in_rate: int, out_rate: int = 16000):
    """Use PyAV or numpy/scipy interpolation to produce correctly timed mono audio."""
    import numpy as np

    if in_rate <= 0 or out_rate <= 0:
        raise ValueError("Sample rates must be positive.")
    if in_rate == out_rate or not len(audio):
        return np.asarray(audio, dtype=np.float32)

    try:
        import av
        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(audio, dtype=np.float32).reshape(1, -1),
            format="fltp",
            layout="mono",
        )
        frame.sample_rate = in_rate
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=out_rate)
        frames = resampler.resample(frame)
        frames.extend(resampler.resample(None))
        if not frames:
            return np.empty(0, dtype=np.float32)
        return np.concatenate([item.to_ndarray().reshape(-1) for item in frames]).astype(np.float32)
    except ImportError:
        duration = len(audio) / float(in_rate)
        num_target_samples = int(round(duration * out_rate))
        orig_times = np.linspace(0.0, duration, len(audio), endpoint=False)
        target_times = np.linspace(0.0, duration, num_target_samples, endpoint=False)
        return np.interp(target_times, orig_times, audio).astype(np.float32)


def resample_to_16k_mono(pcm_bytes: bytes, in_rate: int = 16000, in_channels: int = 1) -> bytes:
    """Downmix signed 16-bit PCM and correctly resample it to 16 kHz mono."""
    if not pcm_bytes:
        return b""
    import numpy as np

    if in_channels <= 0 or in_rate <= 0 or len(pcm_bytes) % (2 * in_channels):
        raise ValueError("PCM data does not match its sample rate and channel count.")
    audio = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
    if in_channels > 1:
        audio = audio.reshape(-1, in_channels).mean(axis=1)
    audio -= float(np.mean(audio))
    audio = _resample_mono_float(audio, in_rate)
    return np.clip(audio * 32768.0, -32768, 32767).astype("<i2").tobytes()


def match_wake_word(transcript: str, wake_words: List[str]) -> Tuple[bool, str, str]:
    """
    Detect whether transcript starts with any supported activation phrase.
    Tolerates capitalization, punctuation, and minor transcription variations.
    Returns: (detected: bool, matched_phrase: str, remaining_command: str)
    """
    if not transcript:
        return False, "", ""

    clean_text = normalize_speech_variants(transcript)
    if not clean_text:
        return False, "", ""

    # Deduplicate and sort phrases by descending word count, then canonical match preference
    clean_words = list(dict.fromkeys([w.lower().strip() for w in wake_words if w.strip()]))
    sorted_words = sorted(
        clean_words,
        key=lambda s: (len(s.split()), -len(s) if s == normalize_speech_variants(s) else len(s)),
        reverse=True
    )

    seen_norms = set()
    for phrase in sorted_words:
        phrase_norm = normalize_speech_variants(phrase)
        if not phrase_norm or phrase_norm in seen_norms:
            continue
        seen_norms.add(phrase_norm)
        pattern = r"^" + re.escape(phrase_norm) + r"(?:\s+|$)(.*)"
        match = re.match(pattern, clean_text)
        if match:
            remaining = match.group(1).strip()
            # Return canonical phrase from wake_words or phrase_norm
            matched_return = phrase_norm if phrase_norm in clean_words else phrase
            return True, matched_return, remaining

    return False, "", clean_text


# =============================================================================
# FASTER-WHISPER TRANSCRIBER & MANAGER
# =============================================================================

class FasterWhisperTranscriber:
    """
    Manages local Faster-Whisper model loading on CPU with int8 quantization.
    Reuses existing local HuggingFace cache and converts PyAudio/WAV frames to numpy float32.
    """

    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8", language: Optional[str] = "en"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.model = None
        self.is_loaded = False
        self.last_error = ""
        self._load_lock = threading.Lock()
        self._transcribe_lock = threading.Lock()
        self._request_slot = threading.Lock()

    def load_model(self) -> bool:
        """Load or download model into local HuggingFace cache."""
        if self.is_loaded and self.model is not None:
            return True
        with self._load_lock:
            if self.is_loaded and self.model is not None:
                return True
            _inject_user_site_packages()
            try:
                from faster_whisper import WhisperModel
                logger.info(f"Loading Faster-Whisper '{self.model_size}' model on {self.device} ({self.compute_type})...")
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=4
                )
                self.is_loaded = True
                self.last_error = ""
                logger.info(f"Faster-Whisper '{self.model_size}' model loaded successfully.")
                return True
            except Exception as e:
                self.is_loaded = False
                self.last_error = str(e)
                logger.error(f"Failed to load Faster-Whisper model ({self.model_size}): {e}")
                return False

    def transcribe_raw_pcm(
        self,
        raw_bytes: bytes,
        in_rate: int = 16000,
        in_channels: int = 1,
        beam_size: int = 5,
        vad_filter: bool = True
    ) -> Optional[str]:
        """Transcribe 16-bit PCM bytes directly into float32 array using Faster-Whisper."""
        if not self.is_loaded:
            if not self.load_model():
                return None

        self.last_error = ""
        with self._transcribe_lock:
            try:
                import numpy as np

                if in_channels <= 0 or in_rate <= 0 or len(raw_bytes) % (2 * in_channels):
                    raise ValueError("PCM data does not match its sample rate and channel count.")
                audio_np = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32) / 32768.0
                if in_channels > 1:
                    audio_np = audio_np.reshape(-1, in_channels).mean(axis=1)
                if not len(audio_np):
                    return None
                audio_np -= float(np.mean(audio_np))
                audio_np = _resample_mono_float(audio_np, in_rate)

                # Ignore audio that is too short for meaningful speech (< 200ms = 3200 samples)
                if len(audio_np) < 3200:
                    logger.debug(f"Audio chunk too short ({len(audio_np)} samples), ignoring.")
                    return None

                # Compute clean AC RMS energy of normalized audio to reject pure silence
                rms = float(np.sqrt(np.mean(audio_np ** 2)))
                if rms < 0.005:
                    logger.debug(f"Audio RMS energy ({rms:.4f}) below speech floor (0.005), rejecting as silence/noise.")
                    return None

                logger.info(f"Faster-Whisper transcribing: {len(audio_np)/16000:.2f}s audio, RMS={rms:.4f}, model={self.model_size}")

                segments, info = self.model.transcribe(
                    audio_np,
                    language=self.language,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=300, threshold=0.4),
                    condition_on_previous_text=False,
                    temperature=0.0
                )

                text_parts = []
                for segment in segments:
                    no_speech = getattr(segment, "no_speech_prob", 0.0)
                    avg_logprob = getattr(segment, "avg_logprob", 0.0)
                    logger.debug(f"Whisper segment: '{segment.text}' (no_speech_prob={no_speech:.2f}, avg_logprob={avg_logprob:.2f})")
                    if no_speech < 0.60 and avg_logprob > -1.5:
                        seg_text = segment.text.strip()
                        if seg_text:
                            text_parts.append(seg_text)

                full_text = " ".join(text_parts).strip()
                if not full_text:
                    return None

                # Discard common Whisper hallucinations on low-energy background audio
                clean_test = full_text.lower().strip(".,!?:;\"' \t\n")
                whisper_hallucinations = {
                    "thank you", "thank you.", "thank you very much", "thank you for watching",
                    "thanks for watching", "thanks for watching!", "subtitles by", "amara.org",
                    "bye", "bye bye", "you", "so", "yeah", "okay", "silence", "applause",
                    "start the engine", "i am going to start the engine", "air conditioner",
                    "the engine", "engine", "air conditioning", "ac", "subscribe"
                }
                if clean_test in whisper_hallucinations:
                    logger.info(f"Filtered Whisper hallucination: '{full_text}'")
                    return None

                logger.info(f"Faster-Whisper transcription result: '{full_text}'")
                return full_text

            except Exception as e:
                self.last_error = str(e)
                logger.error(f"Faster-Whisper raw PCM transcription exception: {e}")
                return None

    def transcribe_with_timeout(
        self,
        raw_bytes: bytes,
        in_rate: int = 16000,
        in_channels: int = 1,
        beam_size: int = 5,
        vad_filter: bool = True,
        timeout: float = 7.0
    ) -> Optional[str]:
        """Transcribe PCM bytes with a strict execution timeout to prevent hanging."""
        if not self._request_slot.acquire(blocking=False):
            self.last_error = "A previous transcription is still in progress."
            return None

        res_container: List[Optional[str]] = [None]
        err_container: List[Optional[Exception]] = [None]

        def worker():
            try:
                res_container[0] = self.transcribe_raw_pcm(
                    raw_bytes,
                    in_rate=in_rate,
                    in_channels=in_channels,
                    beam_size=beam_size,
                    vad_filter=vad_filter
                )
            except Exception as e:
                err_container[0] = e
            finally:
                self._request_slot.release()

        try:
            t = threading.Thread(target=worker, daemon=True)
            t.start()
        except Exception:
            self._request_slot.release()
            raise
        t.join(timeout=timeout)
        if t.is_alive():
            self.last_error = f"Transcription timed out after {timeout:.1f}s"
            logger.warning(self.last_error)
            return None
        if err_container[0]:
            self.last_error = str(err_container[0])
            logger.error(f"Transcription worker error: {err_container[0]}")
            return None
        return res_container[0]

    def transcribe_audio_data(self, audio_data, beam_size: int = 5, vad_filter: bool = True) -> Optional[str]:
        """Transcribe SpeechRecognition AudioData or raw PCM bytes using faster-whisper."""
        if hasattr(audio_data, "get_raw_data"):
            raw_bytes = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
            return self.transcribe_raw_pcm(raw_bytes, in_rate=16000, in_channels=1, beam_size=beam_size, vad_filter=vad_filter)
        elif isinstance(audio_data, (bytes, bytearray)):
            return self.transcribe_raw_pcm(audio_data, in_rate=16000, in_channels=1, beam_size=beam_size, vad_filter=vad_filter)
        return None



# =============================================================================
# MAIN SPEECH LISTENER
# =============================================================================

class SpeechListener:
    """
    Handles continuous microphone capture, Voice Activity Detection (VAD),
    Faster-Whisper transcription, and activation phrase matching.
    """

    def __init__(
        self,
        on_wake_detected: Optional[Callable[[str], None]] = None,
        on_listening_state_change: Optional[Callable[[bool, str], None]] = None,
        on_audio_energy: Optional[Callable[[float], None]] = None,
        on_flow_state_change: Optional[Callable[[str, str], None]] = None,
        on_command_recognized: Optional[Callable[[str], None]] = None,
        on_command_timeout: Optional[Callable[[str], None]] = None,
        on_command_failed: Optional[Callable[[str], None]] = None,
    ):
        self.on_wake_detected = on_wake_detected
        self.on_listening_state_change = on_listening_state_change
        self.on_audio_energy = on_audio_energy
        self.on_flow_state_change = on_flow_state_change
        self.on_command_recognized = on_command_recognized
        self.on_command_timeout = on_command_timeout
        self.on_command_failed = on_command_failed

        self.mic_available = False
        self.is_listening = False
        self._stream_active = False
        self._stop_requested = False
        self._paused = False
        self._capture_requested = False
        self._thread: Optional[threading.Thread] = None

        self.device_index: Optional[int] = None
        self.device_name: str = "Default"
        self._recognizer = None
        self._microphone = None

        # STT Engine Selection: 'faster_whisper'
        self.stt_engine = "faster_whisper"
        self.whisper_model_name = "small"  # Default: 'small' (int8 on CPU)
        self.stt_language: Optional[str] = "en"
        self.whisper_transcriber: Optional[FasterWhisperTranscriber] = None

        # Wake Word & Flow State Settings
        self.wake_words: List[str] = list(WAKE_WORDS)
        self.wake_timeout: float = DEFAULT_WAKE_TIMEOUT_SEC
        self.wake_cooldown: float = DEFAULT_WAKE_COOLDOWN_SEC
        self.min_energy_threshold: int = 280
        self.flow_state: str = "MIC_OFF"
        self._wake_session_until: float = 0.0
        self.last_cooldown_until: float = 0.0
        self.is_tts_speaking: bool = False
        # Deduplication & Utterance Tracking
        self._processed_utterance_ids = set()
        self._last_dispatched_command: str = ""
        self._last_dispatched_time: float = 0.0
        self._drain_hardware_buffer: bool = False
        self._active_wake_utt_id: str = ""
        self.is_busy_processing = False
        self._reader_active = threading.Event()
        self._listen_once_lock = threading.Lock()
        self._utterance_lock = threading.Lock()
        self.last_error = ""

        # Live Diagnostic Telemetry
        self.telemetry: Dict[str, Any] = {
            "selected_device": "Default",
            "device_index": None,
            "sample_rate": 16000,
            "channels": 1,
            "live_rms": 0.0,
            "live_peak": 0.0,
            "is_silent": True,
            "wake_detector_active": False,
            "last_wake_detection_time": 0.0,
            "last_wake_result": "None",
            "command_recording_started": False,
            "command_recording_ended": False,
            "command_duration_sec": 0.0,
            "transcription_in_progress": False,
            "last_transcription_duration_sec": 0.0,
            "last_transcription_result": "",
            "current_utterance_id": "",
            "last_processed_utterance_id": "",
            "processed_utterance_count": 0,
        }

        self._load_stt_settings()
        self.stt_language = "en"
        self.save_stt_settings()
        self._init_transcriber()
        self._init_speech_recognition()

    def get_diagnostics_snapshot(self) -> Dict[str, Any]:
        """Return real-time diagnostic telemetry dictionary."""
        snap = dict(self.telemetry)
        snap["mic_available"] = self.mic_available
        snap["is_listening"] = self.is_listening
        snap["is_paused"] = self._paused
        snap["flow_state"] = self.flow_state
        snap["wake_words_count"] = len(self.wake_words)
        snap["whisper_model"] = self.whisper_model_name
        snap["stt_engine"] = self.stt_engine
        snap["device_name"] = self.device_name
        snap["device_index"] = self.device_index
        return snap

    def set_flow_state(self, state: str, details: str = ""):
        """Update active listening flow state."""
        self.flow_state = state
        if self.on_flow_state_change:
            try:
                self.on_flow_state_change(state, details)
            except Exception:
                pass

    def set_tts_speaking(self, is_speaking: bool):
        """Pause capture while TTS is speaking to prevent self-triggering."""
        self.is_tts_speaking = is_speaking
        if is_speaking:
            self.last_cooldown_until = time.time() + 999999.0
            self.set_flow_state("SPEAKING", "ADAM is speaking...")
        else:
            # Acoustic muting release: 350ms cooldown and purge soundcard hardware buffer
            self.last_cooldown_until = time.time() + 0.35
            self._drain_hardware_buffer = True
            if self.is_listening and not self._paused:
                self.set_flow_state("LISTENING_WAKE", 'Listening for "Hey Adam"...')
            else:
                self.set_flow_state("MIC_OFF", "Audio capture is stopped.")

    def _init_transcriber(self):
        """Initialize FasterWhisperTranscriber instance and preload in background."""
        try:
            self.whisper_transcriber = FasterWhisperTranscriber(
                model_size=self.whisper_model_name,
                device="cpu",
                compute_type="int8",
                language=self.stt_language,
            )
            def preload():
                try:
                    self.whisper_transcriber.load_model()
                except Exception:
                    logger.exception("Faster-Whisper preload failed.")
            threading.Thread(target=preload, daemon=True).start()
        except Exception as e:
            logger.warning(f"Could not initialize FasterWhisper: {e}")
            self.whisper_transcriber = None

    def _load_stt_settings(self):
        """Load saved STT engine and wake-word preferences."""
        self.wake_words = list(DEFAULT_WAKE_WORDS)
        self.wake_timeout = DEFAULT_WAKE_TIMEOUT_SEC
        self.wake_cooldown = DEFAULT_WAKE_COOLDOWN_SEC
        self.min_energy_threshold = 280
        self.clap_activation_enabled = False
        self.clap_sensitivity = 0.7

        cfg_file = _ROOT_DIR / "data" / "stt_settings.json"
        if cfg_file.exists():
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.stt_engine = data.get("stt_engine", "faster_whisper")
                self.whisper_model_name = data.get("whisper_model", "small")
                self.stt_language = "en"
                saved_ww = data.get("wake_words")
                if saved_ww and isinstance(saved_ww, list):
                    clean_saved = [str(w).lower().strip() for w in saved_ww if str(w).strip()]
                    if clean_saved:
                        self.wake_words = clean_saved
                self.wake_timeout = float(data.get("wake_timeout_sec", DEFAULT_WAKE_TIMEOUT_SEC))
                self.wake_cooldown = float(data.get("wake_cooldown_sec", DEFAULT_WAKE_COOLDOWN_SEC))
                self.min_energy_threshold = int(data.get("min_energy_threshold", 280))
                self.clap_activation_enabled = bool(data.get("clap_activation_enabled", False))
                self.clap_sensitivity = min(1.0, max(0.0, float(data.get("clap_sensitivity", 0.7))))
            except Exception:
                pass

    def save_stt_settings(self) -> bool:
        """Persist STT engine and wake-word preferences to disk."""
        data_dir = _ROOT_DIR / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        cfg_file = data_dir / "stt_settings.json"
        try:
            with open(cfg_file, "w", encoding="utf-8") as f:
                json.dump({
                    "stt_engine": self.stt_engine,
                    "whisper_model": self.whisper_model_name,
                    "language": self.stt_language or "auto",
                    "wake_words": self.wake_words,
                    "wake_timeout_sec": self.wake_timeout,
                    "wake_cooldown_sec": self.wake_cooldown,
                    "min_energy_threshold": self.min_energy_threshold,
                    "clap_activation_enabled": self.clap_activation_enabled,
                    "clap_sensitivity": self.clap_sensitivity,
                }, f, indent=2)
        except Exception:
            logger.exception("Could not save speech recognition settings.")
            return False
        return True

    @staticmethod
    def is_clap_pulse(rms: float, peak: float, noise_floor: float, sensitivity: float) -> bool:
        """Validate a sharp transient against both absolute and ambient energy."""
        sensitivity = min(1.0, max(0.0, sensitivity))
        peak_threshold = 5200.0 - 2800.0 * sensitivity
        rms_threshold = max(450.0, noise_floor * 3.0, peak_threshold / 5.0)
        crest_threshold = 3.2 + (1.0 - sensitivity) * 2.0
        return peak >= peak_threshold and rms >= rms_threshold and peak / max(rms, 1.0) >= crest_threshold

    def get_wake_words(self) -> List[str]:
        """Return currently configured activation phrases."""
        return list(self.wake_words)

    def set_wake_words(self, words: List[str]):
        """Replace the list of activation phrases and persist."""
        clean = [str(w).lower().strip() for w in words if str(w).strip()]
        if clean:
            self.wake_words = list(dict.fromkeys(clean))
            self.save_stt_settings()

    def add_wake_word(self, word: str) -> bool:
        """Add a single activation phrase."""
        w = word.lower().strip()
        if w and w not in self.wake_words:
            self.wake_words.append(w)
            self.save_stt_settings()
            return True
        return False

    def remove_wake_word(self, word: str) -> bool:
        """Remove an activation phrase."""
        w = word.lower().strip()
        if w in self.wake_words and len(self.wake_words) > 1:
            self.wake_words.remove(w)
            self.save_stt_settings()
            return True
        return False

    def reset_wake_words_to_default(self):
        """Reset wake words to project default list."""
        self.wake_words = list(DEFAULT_WAKE_WORDS)
        self.save_stt_settings()


    def set_stt_model(self, model_name: str) -> bool:
        """Switch Faster-Whisper model ('base' or 'small') and reload."""
        if model_name not in ["base", "small"]:
            return False
        self.whisper_model_name = model_name
        self.stt_engine = "faster_whisper"
        self.save_stt_settings()
        self.whisper_transcriber = FasterWhisperTranscriber(
            model_size=self.whisper_model_name,
            device="cpu",
            compute_type="int8",
            language=self.stt_language,
        )
        return self.whisper_transcriber.load_model()

    def set_stt_language(self, language: Optional[str]) -> bool:
        """Keep recognition explicitly configured for the user's English input."""
        if language != "en":
            return False
        self.stt_language = "en"
        self.save_stt_settings()
        if self.whisper_transcriber is not None:
            self.whisper_transcriber.language = "en"
        return True

    def test_transcription_live(self, duration_sec: float = 3.0, device_index: Optional[int] = None) -> Tuple[bool, str]:
        """Record real audio from microphone and transcribe with Faster-Whisper."""
        target_idx = device_index if device_index is not None else self.device_index
        prev_paused = self._paused
        self._paused = True
        try:
            diag = record_diagnostic_sample(
                device_index=target_idx,
                duration_sec=duration_sec,
            )
            if not diag.get("success"):
                return False, diag.get("message", "Could not capture audio from selected microphone.")

            if diag.get("is_silent"):
                return False, f"Audio captured, but signal was silent (AC RMS: {diag.get('ac_rms', 0):.1f}). Please speak closer and louder."

            if not self.whisper_transcriber:
                return False, "Faster-Whisper engine is not initialized."

            res = self.whisper_transcriber.transcribe_raw_pcm(
                diag["raw_pcm"],
                in_rate=diag["sample_rate"],
                in_channels=1,
                beam_size=5,
                vad_filter=True,
            )
            if res:
                return True, res
            elif self.whisper_transcriber.last_error:
                return False, f"Faster-Whisper error: {self.whisper_transcriber.last_error}"
            else:
                return False, "Recording contained audio, but no words were recognized. Check the selected input device and try again."
        except Exception as e:
            return False, f"Speech recognition test error: {e}"
        finally:
            self._paused = prev_paused

    def test_wake_word_live(self, duration_sec: float = 4.0, device_index: Optional[int] = None) -> Tuple[bool, str, str, str]:
        """
        Record a real audio sample and test against configured wake phrases.
        Returns: (detected: bool, matched_phrase: str, remaining_command: str, full_transcript: str)
        """
        ok, transcript = self.test_transcription_live(duration_sec=duration_sec, device_index=device_index)
        if not ok or not transcript:
            return False, "", "", transcript or "No speech detected"

        detected, matched_word, remaining_cmd = match_wake_word(transcript, self.wake_words)
        return detected, matched_word, remaining_cmd, transcript

    def _init_speech_recognition(self, preferred_index: Optional[int] = None):
        """Select an input and verify that it can actually open and deliver frames."""
        try:
            sr = None
            try:
                import speech_recognition as sr_module
                sr = sr_module
                self._sr = sr
                self._recognizer = sr.Recognizer()
                self._recognizer.energy_threshold = 300
                self._recognizer.dynamic_energy_threshold = False
                self._recognizer.pause_threshold = 0.8
                self._recognizer.non_speaking_duration = 0.5
            except ImportError:
                self._sr = None
                self._recognizer = None

            devices = list_input_microphones()
            if not devices:
                self.mic_available = False
                self.device_name = "No Input Device"
                logger.warning("No microphone input devices detected.")
                return

            sorted_candidates = []
            if preferred_index is not None:
                for d in devices:
                    if d["index"] == preferred_index:
                        sorted_candidates.append(d)
                        break

            for d in devices:
                d_name = d["name"].lower()
                if any(k in d_name for k in ["default", "pipewire", "pulse"]) and d not in sorted_candidates:
                    sorted_candidates.append(d)

            for d in devices:
                if d not in sorted_candidates:
                    sorted_candidates.append(d)

            initialized = False
            p = get_global_pyaudio()
            for candidate in sorted_candidates:
                candidate_idx = candidate["index"]
                candidate_name = candidate["name"].lower()
                if candidate_name.endswith(".monitor") or any(
                    token in candidate_name for token in ("loopback", "stereo mix", "what u hear")
                ):
                    continue
                try:
                    stream = None
                    if p is not None:
                        dev_info = p.get_device_info_by_index(candidate_idx)
                        if int(dev_info.get("maxInputChannels", 0)) <= 0:
                            continue
                        last_error = None
                        for rate in (16000, int(dev_info.get("defaultSampleRate", 16000)), 48000, 44100):
                            try:
                                stream = p.open(
                                    format=p.get_format_from_width(2),
                                    channels=1,
                                    rate=rate,
                                    input=True,
                                    input_device_index=candidate_idx,
                                    frames_per_buffer=1024,
                                )
                                frames = stream.read(1024, exception_on_overflow=False)
                                if len(frames) == 1024 * 2:
                                    break
                                stream.close()
                                stream = None
                            except Exception as exc:
                                last_error = exc
                                if stream is not None:
                                    try:
                                        stream.close()
                                    except Exception:
                                        pass
                                    stream = None
                        if stream is None:
                            logger.debug("Input [%s] failed live capture check: %s", candidate_idx, last_error)
                            continue
                        stream.stop_stream()
                        stream.close()

                    mic = None
                    if sr is not None:
                        mic = sr.Microphone(device_index=candidate_idx)
                        with suppress_c_stderr(), mic as source:
                            self._recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    if p is None and mic is None:
                        continue

                    self._microphone = mic if mic is not None else candidate
                    self.device_index = candidate_idx
                    self.device_name = candidate["name"]
                    self.mic_available = True
                    initialized = True
                    logger.info(f"Microphone initialized on device [{self.device_index}] {self.device_name}")
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

    def get_state(self) -> str:
        """Return state: 'UNAVAILABLE', 'MUTED', 'LISTENING', 'ACTIVE'"""
        if not self.mic_available or self._microphone is None:
            return "UNAVAILABLE"
        if self._paused or self._stop_requested or not self.is_listening:
            return "MUTED"
        if self._wake_session_until > time.time():
            return "ACTIVE"
        return "LISTENING"

    def get_status_info(self) -> str:
        """Return status string for UI badge."""
        stt_tag = f"Whisper {self.whisper_model_name}" if self.stt_engine == "faster_whisper" else "Google STT"
        if not self.mic_available:
            return "Microphone: Not Detected (Text mode active)"
        if self.flow_state == "MIC_STARTING":
            return f"Microphone Activating ({stt_tag})"
        if self.flow_state == "MIC_ERROR":
            return f"Microphone Error ({stt_tag})"
        if self._paused or self._stop_requested or not self._capture_requested or self.flow_state == "MIC_OFF":
            return f"Microphone: OFF ({stt_tag})"
        if self.flow_state == "WAKE_DETECTED":
            return f"⚡ Wake Word Detected! ({stt_tag})"
        if self.flow_state in ("LISTENING_COMMAND", "LISTENING_WAKE"):
            return f"Listening... ({stt_tag})"
        if self.flow_state == "SPEECH_DETECTED":
            return f"Speech Detected ({stt_tag})"
        if self.flow_state == "RECORDING":
            return f"Recording... ({stt_tag})"
        if self.flow_state == "TRANSCRIBING":
            return f"Query is Transcribing... ({stt_tag})"
        if self.flow_state == "QUERY_READY":
            return f"Transcript Ready ({stt_tag})"
        if self.flow_state == "PROCESSING":
            return f"⚙️ Thinking... ({stt_tag})"
        if self.flow_state == "SPEAKING":
            return f"🗣️ Speaking... ({stt_tag})"
        if self.flow_state == "READY":
            return f"✔ Ready ({stt_tag})"
        return f"Microphone: ON - Wake Active ({stt_tag}) [{self.device_name[:12]}]"

    def start_wake_word_loop(self):
        """Start background daemon thread to listen for wake words."""
        if not self.mic_available:
            self._capture_requested = False
            self.set_flow_state("MIC_ERROR", "No microphone detected on this system.")
            return

        self._capture_requested = True
        self.set_flow_state("MIC_STARTING", "Initializing audio device...")

        if self._thread and self._thread.is_alive():
            self._stop_requested = False
            self._paused = False
            if self._stream_active:
                self._stream_active = True
                self.is_listening = True
                self.set_flow_state("LISTENING_WAKE", f"ADAM is listening on {self.device_name[:14]}")
                if self.on_listening_state_change:
                    self.on_listening_state_change(True, f"MIC ON [{self.device_name[:16]}]")
            else:
                self.set_flow_state("MIC_STARTING", "Microphone Activating")
            return

        self._stop_requested = False
        self._paused = False
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop listening and request background thread exit."""
        self._stop_requested = True
        self._paused = True
        self._capture_requested = False
        self.is_listening = False
        self._wake_session_until = 0.0
        self.set_flow_state("MIC_OFF", "Audio capture is stopped.")
        if self.on_listening_state_change:
            self.on_listening_state_change(False, f"MIC OFF [{self.device_name[:16]}]")

    def pause(self):
        """Mute / turn off microphone capture without killing thread completely."""
        self._paused = True
        self._capture_requested = False
        self.is_listening = False
        self._wake_session_until = 0.0
        self.set_flow_state("MIC_OFF", "Audio capture is stopped.")
        if self.on_listening_state_change:
            self.on_listening_state_change(False, f"MIC OFF [{self.device_name[:16]}]")

    def resume(self):
        """Turn on microphone capture and wake-word detection."""
        if not self.mic_available:
            self.set_flow_state("MIC_ERROR", "No microphone detected.")
            return
        self._paused = False
        self._stop_requested = False
        self._capture_requested = True
        self.set_flow_state("MIC_STARTING", "Initializing audio device...")
        if not (self._thread and self._thread.is_alive()):
            self.start_wake_word_loop()
        elif self._stream_active:
            self.is_listening = True
            self.set_flow_state("LISTENING_WAKE", "ADAM is listening. Say 'Hey Adam'.")
            if self.on_listening_state_change:
                self.on_listening_state_change(True, f"MIC ON [{self.device_name[:16]}]")

    def toggle_listening(self) -> bool:
        """Toggle microphone listening state. Returns True if now ON, False if now OFF."""
        if not self.mic_available:
            return False
        if self._paused or self._stop_requested or not self._capture_requested:
            self.resume()
            return True
        else:
            self.pause()
            return False

    def toggle_pause(self) -> bool:
        return self.toggle_listening()

    def _transcribe_audio(self, audio) -> Optional[str]:
        """Dispatch audio to Faster-Whisper."""
        if self.whisper_transcriber:
            text = self.whisper_transcriber.transcribe_audio_data(audio, beam_size=5, vad_filter=True)
            if text:
                return text.strip()
        return None

    def listen_once(self, timeout: float = 6.0, phrase_time_limit: float = 10.0) -> Optional[str]:
        """
        Record and transcribe a single phrase on demand (Push-to-Talk) with real VAD.
        Does NOT require any wake word.
        """
        if not self._listen_once_lock.acquire(blocking=False):
            message = "A microphone recording is already in progress. Please wait and try again."
            if self.on_command_failed:
                self.on_command_failed(message)
            return None
        try:
            return self._listen_once_capture(timeout, phrase_time_limit)
        finally:
            self._listen_once_lock.release()

    def _listen_once_capture(self, timeout: float, phrase_time_limit: float) -> Optional[str]:
        """Capture one utterance after pausing the wake-word stream."""
        if not self.mic_available:
            message = "Microphone Error: no available input device."
            logger.warning("Push-to-talk unavailable: No microphone.")
            self.set_flow_state("MIC_ERROR", message)
            if self.on_command_failed:
                self.on_command_failed(message)
            return None

        prev_paused = self._paused
        self._paused = True
        reader_deadline = time.monotonic() + 2.0
        while self._reader_active.is_set() and time.monotonic() < reader_deadline:
            time.sleep(0.01)
        if self._reader_active.is_set():
            self._paused = prev_paused
            message = "Microphone Error: the wake-word stream did not release in time."
            self.set_flow_state("MIC_ERROR", message)
            if self.on_command_failed:
                self.on_command_failed(message)
            return None

        p = get_global_pyaudio()
        if p is None:
            self.set_flow_state("MIC_ERROR", "Audio system unavailable.")
            self._paused = prev_paused
            if self.on_command_failed:
                self.on_command_failed("Audio capture is unavailable. Install PortAudio and PyAudio.")
            return None

        target_idx = self.device_index
        stream = None
        actual_rate = 16000
        actual_channels = 1
        chunk_size = 1024
        self.set_flow_state("MIC_STARTING", "Microphone Activating")

        try:
            device_info = p.get_device_info_by_index(target_idx) if target_idx is not None else p.get_default_input_device_info()
            native_rate = int(device_info.get("defaultSampleRate", 16000))
        except Exception:
            native_rate = 16000

        for test_rate in dict.fromkeys((native_rate, 16000, 48000, 44100)):
            candidate_stream = None
            try:
                candidate_stream = p.open(
                    format=p.get_format_from_width(2),
                    channels=1,
                    rate=test_rate,
                    input=True,
                    input_device_index=target_idx,
                    frames_per_buffer=chunk_size
                )
                first_frame = candidate_stream.read(chunk_size, exception_on_overflow=False)
                if len(first_frame) != chunk_size * actual_channels * 2:
                    candidate_stream.stop_stream()
                    candidate_stream.close()
                    continue
                stream = candidate_stream
                actual_rate = test_rate
                break
            except Exception:
                if candidate_stream is not None:
                    try:
                        candidate_stream.stop_stream()
                        candidate_stream.close()
                    except Exception:
                        pass
                continue

        if stream is None:
            message = "Microphone Error: could not open the selected input stream."
            self.last_error = message
            self.set_flow_state("MIC_ERROR", message)
            self._paused = prev_paused
            if self.on_command_failed:
                self.on_command_failed(message)
            return None

        self.set_flow_state("LISTENING_COMMAND", "Listening...")
        if self.on_listening_state_change:
            self.on_listening_state_change(True, "Listening...")

        audio_chunks = []
        speech_started = False
        silence_count = 0
        noise_floor = 100.0
        start_wait_time = time.time()
        max_chunks = int((phrase_time_limit * actual_rate) / chunk_size)
        pre_speech_chunks = []
        onset_chunks = []

        capture_error = None
        try:
            while True:
                try:
                    data = stream.read(chunk_size, exception_on_overflow=False)
                except Exception as exc:
                    capture_error = exc
                    break
                if not data:
                    continue
                if len(data) != chunk_size * actual_channels * 2:
                    capture_error = ValueError("unexpected audio frame size")
                    break

                count = len(data) // 2
                if count == 0:
                    continue
                shorts = struct.unpack(f"<{count}h", data)
                dc_offset = sum(shorts) / count
                ac_sum_sq = sum((s - dc_offset) ** 2 for s in shorts)
                rms = math.sqrt(ac_sum_sq / count)
                ac_peak = max(abs(s - dc_offset) for s in shorts)

                clean_shorts = [int(max(-32768, min(32767, s - dc_offset))) for s in shorts]
                clean_data = struct.pack(f"<{count}h", *clean_shorts)

                speech_thresh = max(180.0, noise_floor * 1.6)
                norm_peak = min(1.0, ac_peak / 8000.0)

                if self.on_audio_energy:
                    self.on_audio_energy(norm_peak)

                now = time.time()

                if not speech_started:
                    # Waiting for user to begin speaking
                    onset_confirmed = _update_speech_onset(
                        onset_chunks, pre_speech_chunks, clean_data, rms > speech_thresh, 4
                    )
                    if onset_confirmed:
                        speech_started = True
                        self.set_flow_state("SPEECH_DETECTED", "Speech Detected")
                        self.set_flow_state("RECORDING", "Recording...")
                        audio_chunks = list(pre_speech_chunks) + onset_chunks
                        onset_chunks = []
                        silence_count = 0
                    elif rms <= speech_thresh:
                        noise_floor = 0.95 * noise_floor + 0.05 * rms
                        pre_speech_chunks.append(clean_data)
                        if len(pre_speech_chunks) > 4:  # ~250ms pre-speech audio
                            pre_speech_chunks.pop(0)

                        if (now - start_wait_time) > timeout:
                            logger.info(f"Listen Once timed out after {timeout:.1f}s waiting for speech.")
                            break
                else:
                    # User is speaking
                    audio_chunks.append(clean_data)
                    if rms < speech_thresh:
                        silence_count += 1
                        # ~500ms of silence signals end of utterance
                        silence_chunks_needed = int(0.5 * actual_rate / chunk_size)
                        if silence_count >= max(6, silence_chunks_needed):
                            logger.info(f"End of utterance detected ({len(audio_chunks)} chunks).")
                            break
                    else:
                        silence_count = 0

                    if len(audio_chunks) >= max_chunks:
                        logger.info("Utterance reached maximum time limit.")
                        break

        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception as exc:
                if capture_error is None:
                    capture_error = exc
            self._drain_hardware_buffer = True
            if speech_started and audio_chunks:
                self.set_flow_state("TRANSCRIBING", "Query is Transcribing...")
            self._paused = prev_paused
            capture_resumed = not prev_paused and self._stream_active
            if self.on_listening_state_change:
                self.on_listening_state_change(
                    capture_resumed,
                    "Listening..." if capture_resumed else f"MIC OFF [{self.device_name[:16]}]",
                )

        if capture_error is not None:
            message = f"Microphone Error: audio capture failed ({capture_error})."
            self.last_error = message
            self.set_flow_state("MIC_ERROR", message)
            if self.on_command_failed:
                self.on_command_failed(message)
            return None

        if not speech_started or not audio_chunks:
            message = "I couldn't hear speech. Please try again."
            logger.info("Listen Once finished without capturing speech: %s", message)
            if self.on_command_failed:
                self.on_command_failed(message)
            if not self._paused and self.is_listening:
                self.set_flow_state("LISTENING_WAKE", 'Listening for "Hey Adam"...')
            else:
                self.set_flow_state("MIC_OFF", "Audio capture is stopped.")
            return None

        # State 6: TRANSCRIBING
        self.set_flow_state("TRANSCRIBING", "Query is Transcribing...")
        raw_pcm = b"".join(audio_chunks)

        transcript = None
        if self.whisper_transcriber:
            transcript = self.whisper_transcriber.transcribe_with_timeout(
                raw_pcm,
                in_rate=actual_rate,
                in_channels=1,
                beam_size=5,
                vad_filter=True,
                timeout=10.0,
            )

        if transcript and transcript.strip():
            clean_cmd = normalize_speech_variants(transcript).strip()
            logger.info(f"Listen Once transcribed: '{clean_cmd}'")
            self.telemetry["last_transcription_result"] = clean_cmd
            self.set_flow_state("QUERY_READY", f"Transcript Ready: {clean_cmd}")
            return clean_cmd
        else:
            error = self.whisper_transcriber.last_error if self.whisper_transcriber else ""
            message = f"Transcription Error: {error}" if error else "I couldn't understand that command. Please try again."
            logger.info("Listen Once transcription failed: %s", message)
            if self.on_command_failed:
                self.on_command_failed(message)
            if not self._paused and self.is_listening:
                self.set_flow_state("LISTENING_WAKE", 'Listening for "Hey Adam"...')
            else:
                self.set_flow_state("MIC_OFF", "Audio capture is stopped.")
            return None

    def _listen_loop(self):
        """
        Continuous, low-latency audio capture loop implementing the 8-state voice interaction pipeline:
        State 1: MIC OFF
        State 2: MIC STARTING (device open & frame arrival verified)
        State 3: LISTENING FOR WAKE WORD (wake-word buffer)
        State 4: WAKE WORD DETECTED
        State 5: LISTENING FOR COMMAND (fresh command buffer, silence & max duration timeout)
        State 6: RECORDING STOPPED / TRANSCRIBING (Faster-Whisper on worker thread with 7s timeout)
        State 7: PROCESSING / THINKING
        State 8: SPEAKING / READY
        """
        logger.info(f"Wake-word daemon started on device [{self.device_index}] {self.device_name} with {self.stt_engine}")
        self.set_flow_state("MIC_STARTING", "Starting microphone...")

        p = get_global_pyaudio()
        if p is None:
            logger.warning("PyAudio unavailable, using fallback recognition loop.")
            self._listen_loop_fallback()
            return

        stream = None
        actual_rate = 16000
        actual_channels = 1
        chunk_size = 1024

        dev_info = {}
        if self.device_index is not None:
            try:
                dev_info = p.get_device_info_by_index(self.device_index)
            except Exception:
                pass
        else:
            try:
                dev_info = p.get_default_input_device_info()
            except Exception:
                pass

        native_rate = int(dev_info.get("defaultSampleRate", 16000))
        max_channels = int(dev_info.get("maxInputChannels", 1))

        rate_candidates = [16000, native_rate, 48000, 44100]
        rate_candidates = list(dict.fromkeys([r for r in rate_candidates if r > 0]))
        ch_candidates = [1] if max_channels < 2 else [1, 2]

        last_stream_err = None
        for test_rate in rate_candidates:
            for test_ch in ch_candidates:
                candidate_stream = None
                try:
                    test_chunk = int(test_rate * 0.064)
                    candidate_stream = p.open(
                        format=p.get_format_from_width(2),
                        channels=test_ch,
                        rate=test_rate,
                        input=True,
                        input_device_index=self.device_index,
                        frames_per_buffer=test_chunk
                    )
                    # Verify audio frames are actually arriving
                    test_data = candidate_stream.read(test_chunk, exception_on_overflow=False)
                    if len(test_data) != test_chunk * test_ch * 2:
                        candidate_stream.stop_stream()
                        candidate_stream.close()
                        continue
                    stream = candidate_stream
                    actual_rate = test_rate
                    actual_channels = test_ch
                    chunk_size = test_chunk
                    logger.info(f"PyAudio capture stream verified at {actual_rate}Hz, {actual_channels}ch, chunk={chunk_size}")
                    break
                except Exception as ex:
                    last_stream_err = ex
                    logger.debug(f"Stream open attempt ({test_rate}Hz, {test_ch}ch) failed: {ex}")
                    if candidate_stream is not None:
                        try:
                            candidate_stream.stop_stream()
                            candidate_stream.close()
                        except Exception:
                            pass
                    continue
            if stream is not None:
                break

        if stream is None:
            err_msg = f"Audio device initialization failed: {last_stream_err or 'No frames received'}"
            self.last_error = err_msg
            logger.warning(err_msg)
            self.set_flow_state("MIC_ERROR", err_msg)
            self.is_listening = False
            self._capture_requested = False
            return

        self.is_listening = True
        self._stream_active = True
        self.set_flow_state("LISTENING_WAKE", 'Listening for "Hey Adam"...')
        if self.on_listening_state_change:
            self.on_listening_state_change(True, f"MIC ON [{self.device_name[:16]}]")

        pre_buffer_size = 6  # ~384ms pre-speech audio
        pre_buffer = []
        in_speech = False
        wake_chunks = []
        wake_onset_chunks = []
        silence_chunks_count = 0
        noise_floor = 180.0

        # Command mode tracking
        cmd_chunks = []
        cmd_onset_chunks = []
        cmd_in_speech = False
        cmd_silence_count = 0
        cmd_start_time = 0.0
        current_cmd_utt_id = ""
        clap_detector = DoubleClapDetector()
        # Background non-blocking audio stream reader thread
        import queue
        audio_q: queue.Queue = queue.Queue(maxsize=400)
        reader_stop_event = threading.Event()

        def audio_reader_worker():
            while not reader_stop_event.is_set():
                if self._paused or self._stop_requested:
                    time.sleep(0.04)
                    continue
                try:
                    self._reader_active.set()
                    raw = stream.read(chunk_size, exception_on_overflow=False)
                    if raw:
                        if audio_q.full():
                            try:
                                audio_q.get_nowait()
                            except queue.Empty:
                                pass
                        audio_q.put(raw)
                except Exception:
                    time.sleep(0.02)
                finally:
                    self._reader_active.clear()

        reader_thread = threading.Thread(target=audio_reader_worker, daemon=True)
        reader_thread.start()

        try:
            while not self._stop_requested:
                if self._paused:
                    time.sleep(0.06)
                    while not audio_q.empty():
                        try:
                            audio_q.get_nowait()
                        except queue.Empty:
                            break
                    pre_buffer.clear()
                    wake_chunks.clear()
                    cmd_chunks.clear()
                    wake_onset_chunks.clear()
                    cmd_onset_chunks.clear()
                    in_speech = False
                    cmd_in_speech = False
                    continue

                if self.is_tts_speaking or getattr(self, "is_busy_processing", False):
                    time.sleep(0.04)
                    while not audio_q.empty():
                        try:
                            audio_q.get_nowait()
                        except queue.Empty:
                            break
                    pre_buffer.clear()
                    wake_chunks.clear()
                    cmd_chunks.clear()
                    wake_onset_chunks.clear()
                    cmd_onset_chunks.clear()
                    clap_detector.reset()
                    in_speech = False
                    cmd_in_speech = False
                    continue

                # State Isolation: Do NOT accumulate audio while transcribing, thinking, or speaking
                if self.flow_state in ("TRANSCRIBING", "QUERY_READY", "PROCESSING", "SPEAKING"):
                    time.sleep(0.04)
                    pre_buffer.clear()
                    wake_chunks.clear()
                    cmd_chunks.clear()
                    wake_onset_chunks.clear()
                    cmd_onset_chunks.clear()
                    clap_detector.reset()
                    in_speech = False
                    cmd_in_speech = False
                    continue

                # Acoustic Muting Release & Soundcard Hardware Buffer Purge
                if getattr(self, "_drain_hardware_buffer", False):
                    while not audio_q.empty():
                        try:
                            audio_q.get_nowait()
                        except queue.Empty:
                            break
                    self._drain_hardware_buffer = False
                    pre_buffer.clear()
                    wake_chunks.clear()
                    cmd_chunks.clear()
                    wake_onset_chunks.clear()
                    cmd_onset_chunks.clear()
                    in_speech = False
                    cmd_in_speech = False

                now = time.time()
                if now < self.last_cooldown_until:
                    time.sleep(0.03)
                    pre_buffer.clear()
                    wake_chunks.clear()
                    cmd_chunks.clear()
                    wake_onset_chunks.clear()
                    cmd_onset_chunks.clear()
                    clap_detector.reset()
                    in_speech = False
                    cmd_in_speech = False
                    continue

                try:
                    data = audio_q.get(timeout=0.08)
                except queue.Empty:
                    continue

                # Compute AC RMS energy with DC offset cancellation
                count = len(data) // 2
                if count == 0:
                    continue
                shorts = struct.unpack(f"<{count}h", data)
                dc_offset = sum(shorts) / count
                ac_sum_sq = sum((s - dc_offset) ** 2 for s in shorts)
                rms = math.sqrt(ac_sum_sq / count)
                ac_peak = max(abs(s - dc_offset) for s in shorts)

                clean_shorts = [int(max(-32768, min(32767, s - dc_offset))) for s in shorts]
                clean_data = struct.pack(f"<{count}h", *clean_shorts)

                speech_threshold = max(180.0, noise_floor * 1.6)
                norm_peak = min(1.0, ac_peak / 8000.0)

                # Update live telemetry for Section 4 diagnostic requirements
                self.telemetry["live_rms"] = round(rms, 1)
                self.telemetry["live_peak"] = round(norm_peak, 3)
                self.telemetry["is_silent"] = bool(rms <= speech_threshold)

                # Report live energy for UI visualizer and VU meter
                if self.on_audio_energy and self.flow_state in ("LISTENING_WAKE", "LISTENING_COMMAND", "WAKE_DETECTED"):
                    self.on_audio_energy(norm_peak)

                if (
                    self.clap_activation_enabled
                    and self.flow_state == "LISTENING_WAKE"
                ):
                    clap_detected = clap_detector.update(
                        self.is_clap_pulse(rms, ac_peak, noise_floor, self.clap_sensitivity),
                        time.monotonic(),
                    )
                    if clap_detected:
                        cmd_start_time = now
                        cmd_in_speech = False
                        cmd_chunks.clear()
                        cmd_onset_chunks.clear()
                        pre_buffer.clear()
                        current_cmd_utt_id = f"clap_{uuid.uuid4().hex}"
                        self.telemetry["current_utterance_id"] = current_cmd_utt_id
                        self.set_flow_state("LISTENING_COMMAND", "Clap detected. Listening for a command...")
                        continue
                else:
                    clap_detector.reset()

                # ==============================================================
                # STATE 5: LISTENING FOR COMMAND
                # ==============================================================
                if self.flow_state in ("LISTENING_COMMAND", "RECORDING"):
                    # Check silence timeout if speech has not started yet
                    if not cmd_in_speech:
                        if (now - cmd_start_time) > 6.0:
                            logger.info("Command listening session timed out (6.0s no speech).")
                            self.telemetry["command_recording_ended"] = True
                            self.set_flow_state("LISTENING_WAKE", 'Listening for "Hey Adam"...')
                            if self.on_command_timeout:
                                self.on_command_timeout("Listening timed out. No command detected.")
                            cmd_chunks.clear()
                            cmd_onset_chunks.clear()
                            continue

                    if rms > speech_threshold:
                        if not cmd_in_speech:
                            if not _update_speech_onset(
                                cmd_onset_chunks, pre_buffer, clean_data, True, pre_buffer_size
                            ):
                                continue
                            cmd_in_speech = True
                            self.telemetry["command_recording_started"] = True
                            self.telemetry["command_recording_ended"] = False
                            cmd_chunks = list(pre_buffer) + cmd_onset_chunks
                            cmd_onset_chunks.clear()
                            cmd_silence_count = 0
                            self.set_flow_state("SPEECH_DETECTED", "Speech Detected")
                            self.set_flow_state("RECORDING", "Recording...")
                        else:
                            cmd_chunks.append(clean_data)
                        cmd_silence_count = 0
                        # Max command length: ~10s
                        if len(cmd_chunks) > 160:
                            cmd_in_speech = False
                            self.telemetry["command_recording_ended"] = True
                            self.telemetry["command_duration_sec"] = round(len(cmd_chunks) * (chunk_size / actual_rate), 2)
                            raw_cmd = b"".join(cmd_chunks)
                            cmd_chunks.clear()
                            cmd_onset_chunks.clear()
                            pre_buffer.clear()
                            self._launch_command_transcription(raw_cmd, in_rate=actual_rate, in_channels=actual_channels, utterance_id=current_cmd_utt_id)
                    else:
                        if not cmd_in_speech:
                            _update_speech_onset(
                                cmd_onset_chunks, pre_buffer, clean_data, False, pre_buffer_size
                            )
                            noise_floor = 0.95 * noise_floor + 0.05 * rms
                            pre_buffer.append(clean_data)
                            if len(pre_buffer) > pre_buffer_size:
                                pre_buffer.pop(0)
                        else:
                            cmd_chunks.append(clean_data)
                            cmd_silence_count += 1
                            # ~500ms silence signals end of command speech
                            if cmd_silence_count >= 8:
                                total_cmd_speech = len(cmd_chunks) - cmd_silence_count
                                raw_cmd = b"".join(cmd_chunks)
                                cmd_in_speech = False
                                self.telemetry["command_recording_ended"] = True
                                self.telemetry["command_duration_sec"] = round(len(cmd_chunks) * (chunk_size / actual_rate), 2)
                                cmd_chunks.clear()
                                cmd_onset_chunks.clear()
                                pre_buffer.clear()
                                cmd_silence_count = 0
                                if total_cmd_speech >= 3:
                                    self._launch_command_transcription(raw_cmd, in_rate=actual_rate, in_channels=actual_channels, utterance_id=current_cmd_utt_id)
                                else:
                                    self.set_flow_state("LISTENING_WAKE", 'Listening for "Hey Adam"...')
                    continue

                # ==============================================================
                # STATE 3: LISTENING FOR WAKE WORD
                # ==============================================================
                if rms > speech_threshold:
                    if not in_speech:
                        if not _update_speech_onset(
                            wake_onset_chunks, pre_buffer, clean_data, True, pre_buffer_size
                        ):
                            continue
                        in_speech = True
                        wake_chunks = list(pre_buffer) + wake_onset_chunks
                        wake_onset_chunks.clear()
                        silence_chunks_count = 0
                    else:
                        wake_chunks.append(clean_data)
                    silence_chunks_count = 0
                    if len(wake_chunks) > 55:  # ~3.5s max for wake phrase
                        in_speech = False
                        raw_wake = b"".join(wake_chunks)
                        wake_chunks.clear()
                        pre_buffer.clear()
                        cmd_started = self._check_wake_audio(raw_wake, in_rate=actual_rate, in_channels=actual_channels)
                        if cmd_started:
                            cmd_start_time = time.time()
                            cmd_in_speech = False
                            cmd_chunks.clear()
                            cmd_onset_chunks.clear()
                            pre_buffer.clear()
                            cmd_silence_count = 0
                            current_cmd_utt_id = getattr(self, "_active_wake_utt_id", "") or f"utt_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
                            self.telemetry["current_utterance_id"] = current_cmd_utt_id
                else:
                    if not in_speech:
                        _update_speech_onset(
                            wake_onset_chunks, pre_buffer, clean_data, False, pre_buffer_size
                        )
                        noise_floor = 0.95 * noise_floor + 0.05 * rms
                        pre_buffer.append(clean_data)
                        if len(pre_buffer) > pre_buffer_size:
                            pre_buffer.pop(0)
                    else:
                        wake_chunks.append(clean_data)
                        silence_chunks_count += 1
                        # ~450ms silence signals end of wake phrase
                        if silence_chunks_count >= 7:
                            total_wake_speech = len(wake_chunks) - silence_chunks_count
                            raw_wake = b"".join(wake_chunks)
                            in_speech = False
                            wake_chunks.clear()
                            silence_chunks_count = 0
                            if total_wake_speech >= 3:
                                cmd_started = self._check_wake_audio(raw_wake, in_rate=actual_rate, in_channels=actual_channels)
                                if cmd_started:
                                    cmd_start_time = time.time()
                                    cmd_in_speech = False
                                    cmd_chunks.clear()
                                    cmd_onset_chunks.clear()
                                    pre_buffer.clear()
                                    cmd_silence_count = 0
                                    current_cmd_utt_id = getattr(self, "_active_wake_utt_id", "") or f"utt_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
                                    self.telemetry["current_utterance_id"] = current_cmd_utt_id

        finally:
            reader_stop_event.set()
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            self.is_listening = False
            self._stream_active = False
            if self._stop_requested or self.flow_state == "MIC_ERROR":
                self._capture_requested = False

    def _process_recognized_phrase(self, transcript: str) -> None:
        """Process one transcript in the fallback listener's wake/command state."""
        if (
            not transcript
            or time.time() < self.last_cooldown_until
            or self.is_tts_speaking
            or self.is_busy_processing
        ):
            return

        phrase = normalize_speech_variants(transcript).strip()
        if not phrase:
            return
        detected, matched, command = match_wake_word(phrase, self.wake_words)
        if detected:
            self.set_flow_state("WAKE_DETECTED", "Wake word detected.")
            self._wake_session_until = time.time() + self.wake_timeout
            if command:
                self._wake_session_until = 0.0
                self.set_flow_state("QUERY_READY", f"Transcript Ready: {command}")
                self.set_flow_state("PROCESSING", "Thinking... processing your request")
                self._notify_command_recognized(command)
            else:
                self.set_flow_state("LISTENING_COMMAND", "Listening...")
                if self.on_wake_detected:
                    self.on_wake_detected("")
            return

        if self._wake_session_until > time.time():
            self._wake_session_until = 0.0
            self.set_flow_state("PROCESSING", "Thinking... processing your request")
            self._notify_command_recognized(phrase)

    def _notify_command_recognized(self, command: str, utterance_id: str = "") -> None:
        """Dispatch one recognized command with its stable utterance identity."""
        request_id = utterance_id or f"utt_{uuid.uuid4().hex}"
        self.telemetry["last_processed_utterance_id"] = request_id
        callback = self.on_command_recognized or self.on_wake_detected
        if callback:
            callback(command)

    def _check_wake_audio(self, raw_bytes: bytes, in_rate: int = 16000, in_channels: int = 1) -> bool:
        """
        Check if audio chunk contains an activation phrase.
        Returns True if sole wake word detected and command listening should begin.
        """
        if self.is_tts_speaking or getattr(self, "is_busy_processing", False) or self.flow_state in ("TRANSCRIBING", "PROCESSING", "SPEAKING"):
            return False

        phrase = None
        if self.whisper_transcriber:
            phrase = self.whisper_transcriber.transcribe_with_timeout(
                raw_bytes, in_rate=in_rate, in_channels=in_channels, timeout=4.0
            )

        if not phrase:
            self.telemetry["last_wake_result"] = "No speech detected"
            return False

        clean_text = normalize_speech_variants(phrase).strip()
        detected, matched_phrase, remaining_cmd = match_wake_word(clean_text, self.wake_words)

        if detected:
            utt_id = f"utt_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
            self._active_wake_utt_id = utt_id
            self.telemetry["wake_detector_active"] = True
            self.telemetry["last_wake_detection_time"] = time.time()
            self.telemetry["last_wake_result"] = f"Detected: '{matched_phrase}'"
            self.telemetry["current_utterance_id"] = utt_id

            logger.info(f"Wake word detected [ID {utt_id}]: '{matched_phrase}' | Remaining: '{remaining_cmd}'")
            # State 4: WAKE WORD DETECTED
            self.set_flow_state("WAKE_DETECTED", "Wake word detected! Start speaking your command.")

            # Check if this was a combined wake + command phrase (e.g. "Hey Adam, what time is it?", "Drake, what time is it?")
            is_assistant_addressed = any(a in matched_phrase.lower() for a in ["adam", "assistant", "mate", "bot", "hii", "drake", "hey adam", "hi adam", "hello adam"]) or (
                remaining_cmd and (
                    remaining_cmd.lower().startswith("adam ") or remaining_cmd.lower().startswith("drake ")
                )
            )

            if remaining_cmd:
                if remaining_cmd.lower().startswith("adam ") or remaining_cmd.lower().startswith("drake "):
                    remaining_cmd = re.sub(r"^(?:adam|drake)\s+", "", remaining_cmd, flags=re.IGNORECASE).strip()

                if is_assistant_addressed:
                    # Combined phrase explicitly directed to ADAM!
                    self.telemetry["last_transcription_result"] = remaining_cmd
                    self.telemetry["last_processed_utterance_id"] = utt_id
                    self.telemetry["processed_utterance_count"] += 1
                    self.set_flow_state("QUERY_READY", f"Transcript Ready: {remaining_cmd}")
                    self.set_flow_state("PROCESSING", "Thinking... processing your request")
                    self._notify_command_recognized(remaining_cmd, utt_id)
                    return False
                else:
                    # Trailing phrase without addressing Adam -> avoid treating room chatter as command.
                    # Prompt user for command and transition to LISTENING_COMMAND.
                    if self.on_wake_detected:
                        self.on_wake_detected("")
                    self.set_flow_state("LISTENING_COMMAND", "Listening to your command...")
                    return True
            else:
                # Sole wake phrase: e.g. "Hey Adam", "Hi", "Hey", "Oi", "Hello Adam"
                if self.on_wake_detected:
                    self.on_wake_detected("")
                # Transition smoothly to State 5: LISTENING_COMMAND!
                self.set_flow_state("LISTENING_COMMAND", "Listening to your command...")
                return True

        self.telemetry["last_wake_result"] = f"Speech unrecognized as wake phrase ('{clean_text[:20]}')"
        return False

    def _launch_command_transcription(self, raw_bytes: bytes, in_rate: int = 16000, in_channels: int = 1, utterance_id: str = ""):
        """Transcribe command audio in a worker thread with timeout and dispatch result."""
        if not utterance_id:
            utterance_id = f"utt_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"

        with self._utterance_lock:
            if utterance_id in self._processed_utterance_ids:
                logger.warning(f"Utterance [{utterance_id}] already processed. Dropping duplicate.")
                return

            self._processed_utterance_ids.add(utterance_id)
            if len(self._processed_utterance_ids) > 200:
                self._processed_utterance_ids.pop()

        self.telemetry["current_utterance_id"] = utterance_id
        self.telemetry["transcription_in_progress"] = True

        # State 7: RECORDING COMPLETE — TRANSCRIBING
        self.set_flow_state("TRANSCRIBING", "Recording complete — transcribing...")

        def worker():
            t0 = time.time()
            text = None
            if self.whisper_transcriber:
                text = self.whisper_transcriber.transcribe_with_timeout(
                    raw_bytes, in_rate=in_rate, in_channels=in_channels, timeout=7.0
                )

            duration = time.time() - t0
            self.telemetry["last_transcription_duration_sec"] = round(duration, 2)
            self.telemetry["transcription_in_progress"] = False

            if text and text.strip():
                clean_cmd = normalize_speech_variants(text).strip()
                if any(w in clean_cmd for w in ["cancel", "never mind", "dismiss", "stop listening"]):
                    logger.info("Command session dismissed by user.")
                    self.set_flow_state("LISTENING_WAKE", 'Listening for "Hey Adam"...')
                    return

                # Suppress immediate duplicate command
                now = time.time()
                if clean_cmd == self._last_dispatched_command and (now - self._last_dispatched_time) < 2.5:
                    logger.info(f"Duplicate command [{utterance_id}] suppressed: '{clean_cmd}'")
                    self.set_flow_state("LISTENING_WAKE", 'Listening for "Hey Adam"...')
                    return

                self._last_dispatched_command = clean_cmd
                self._last_dispatched_time = now
                self.telemetry["last_transcription_result"] = clean_cmd
                self.telemetry["last_processed_utterance_id"] = utterance_id
                self.telemetry["processed_utterance_count"] += 1

                logger.info(f"Utterance [{utterance_id}] transcribed in {duration:.2f}s: '{clean_cmd}'")
                self.set_flow_state("QUERY_READY", f"Transcript Ready: {clean_cmd}")
                # State 10: Thinking... processing your request
                self.set_flow_state("PROCESSING", "Thinking... processing your request")
                self._notify_command_recognized(clean_cmd, utterance_id)
            else:
                self.telemetry["last_transcription_result"] = "[No speech / unrecognized]"
                logger.info(f"Utterance [{utterance_id}] transcription yielded no speech or timed out ({duration:.2f}s).")
                if self.is_listening and not self._paused:
                    self.set_flow_state("LISTENING_WAKE", 'Listening for "Hey Adam"...')
                else:
                    self.set_flow_state("MIC_OFF", "Audio capture is stopped.")
                if self.on_command_failed:
                    self.on_command_failed("I couldn't understand that command. Please try again.")

        threading.Thread(target=worker, daemon=True).start()

    def _dispatch_utterance(self, raw_bytes: bytes, in_rate: int = 16000, in_channels: int = 1):
        """Compatibility dispatcher for manual audio utterances or unit tests."""
        if time.time() < self.last_cooldown_until or self.is_tts_speaking or getattr(self, "is_busy_processing", False):
            return
        self._launch_command_transcription(raw_bytes, in_rate, in_channels)

    def _listen_loop_fallback(self):
        """Fallback recognition loop using SpeechRecognition if direct stream fails."""
        logger.info("Starting fallback SpeechRecognition wake-word loop...")
        self.is_listening = True

        while not self._stop_requested:
            if self._paused or self.is_tts_speaking:
                time.sleep(0.3)
                continue

            now = time.time()
            if now < self.last_cooldown_until:
                time.sleep(0.2)
                continue

            try:
                if not self._microphone:
                    break
                with suppress_c_stderr(), self._microphone as source:
                    audio = self._recognizer.listen(source, timeout=3.0, phrase_time_limit=8.0)

                phrase = self._transcribe_audio(audio)
                if phrase:
                    self._process_recognized_phrase(phrase)
            except Exception as exc:
                logger.warning("Fallback microphone capture/transcription failed: %s", exc)
                time.sleep(0.3)

        self.is_listening = False
