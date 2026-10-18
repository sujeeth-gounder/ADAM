#!/usr/bin/env python3
"""ADAM Physical Microphone & Speech-to-Text Diagnostic Tool.
Provides independent verification of physical microphone input, PipeWire/ALSA status,
sample rate, hardware DC bias cancellation, True AC RMS audio energy, WAV export,
playback, and Faster-Whisper transcription.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from app.speech import (
    list_input_microphones,
    record_diagnostic_sample,
    test_microphone_level,
    FasterWhisperTranscriber,
)


def print_header(title: str):
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)


def run_diagnostics(device_index=None, duration=3.0, save_wav=False, transcribe=True, play_audio=False):
    print_header("ADAM AUDIO & MICROPHONE HARDWARE DIAGNOSTICS")

    devices = list_input_microphones()
    if not devices:
        print("❌ ERROR: No input microphones detected by audio subsystem.")
        print("Check if PipeWire or PulseAudio is running and mic is connected.")
        return False

    print("Detected Input Microphones:")
    for d in devices:
        idx = d["index"]
        name = d["name"]
        rate = d["sample_rate"]
        ch = d["channels"]
        backend = d.get("backend", "pyaudio")
        marker = " [TARGET]" if (device_index is not None and idx == device_index) else ""
        print(f"  [{idx:2d}] {name} ({ch} ch, {rate} Hz, {backend}){marker}")

    chosen_idx = device_index
    chosen_name = None
    if chosen_idx is None:
        # Prioritize system default, internal digital microphone, or pipewire
        for target_kw in ["default", "digital microphone", "pipewire", "pulse"]:
            for d in devices:
                if target_kw in d["name"].lower():
                    chosen_idx = d["index"]
                    chosen_name = d["name"]
                    break
            if chosen_idx is not None:
                break

        if chosen_idx is None:
            chosen_idx = devices[0]["index"]
            chosen_name = devices[0]["name"]
        print(f"\n✔ Auto-selected active system microphone: [{chosen_idx}] {chosen_name}")
    else:
        for d in devices:
            if d["index"] == chosen_idx:
                chosen_name = d["name"]
                break
        print(f"\nTarget device chosen: [{chosen_idx}] {chosen_name or 'Custom Device'}")

    # Advisory if user explicitly chose an analog ALSA device
    if chosen_name and any(kw in chosen_name.lower() for kw in ["alc245 analog", "hw:"]):
        print("  ⚠️ NOTE: This is a direct ALSA 3.5mm jack device. If nothing is plugged in,")
        print("          it will capture floating electromagnetic static (like FM radio white noise).")
        print("          For the laptop's built-in mic, use device 'default' or 'Digital Microphone'.")

    print(f"\nRecording {duration:.1f} seconds of live microphone audio from device [{chosen_idx}]...")
    print(">>> SPEAK INTO THE MICROPHONE NOW (e.g. 'What are the things you can do?') <<<")

    wav_dest = str(ROOT_DIR / "exports" / "diagnostic_mic_test.wav") if save_wav else None
    diag = record_diagnostic_sample(
        device_index=chosen_idx,
        duration_sec=duration,
        save_wav_path=wav_dest
    )

    if not diag.get("success"):
        print(f"\n❌ Recording Failed: {diag.get('message')}")
        return False

    print("\n" + "-" * 40)
    print("DIAGNOSTIC METRICS:")
    print(f"  • Device Name:       {diag['device_name']} (Index {chosen_idx})")
    print(f"  • Sample Rate:       {diag['sample_rate']} Hz (Mono)")
    print(f"  • Captured Duration: {diag['duration_sec']}s ({diag['samples_count']} samples)")
    print(f"  • Hardware DC Bias:  {diag['dc_offset']} (Offset removed)")
    print(f"  • True Sound Energy: AC RMS = {diag['ac_rms']:.1f}")
    print(f"  • Peak AC Amplitude: {diag['ac_peak']:.1f}")
    print(f"  • Signal Status:     {'SILENT / LOW LEVEL' if diag['is_silent'] else 'SIGNAL PRESENT (not speech-verified)'}")
    print(f"  • Status Message:    {diag['message']}")
    print("-" * 40)

    if diag.get("saved_path"):
        saved_file = diag["saved_path"]
        print(f"\n💾 Saved Diagnostic WAV: {saved_file}")
        print("To play back the recording:")
        if shutil.which("pw-play"):
            print(f"   pw-play \"{saved_file}\"")
        elif shutil.which("paplay"):
            print(f"   paplay \"{saved_file}\"")
        elif shutil.which("aplay"):
            print(f"   aplay \"{saved_file}\"")
        else:
            print(f"   Play using your preferred audio player: {saved_file}")

        # Playback
        if play_audio or (sys.stdin.isatty() and not play_audio):
            if play_audio:
                do_play = True
            else:
                try:
                    ans = input("\nWould you like to play back this recording right now? [y/N]: ").strip().lower()
                    do_play = (ans == "y")
                except (KeyboardInterrupt, EOFError):
                    do_play = False

            if do_play:
                if shutil.which("pw-play"):
                    subprocess.run(["pw-play", saved_file])
                elif shutil.which("paplay"):
                    subprocess.run(["paplay", saved_file])
                elif shutil.which("aplay"):
                    subprocess.run(["aplay", saved_file])

    if transcribe and not diag["is_silent"]:
        print_header("FASTER-WHISPER LOCAL STT VERIFICATION")
        print("Transcribing recorded sample with Faster-Whisper ('small' model on CPU int8)...")
        transcriber = FasterWhisperTranscriber(model_size="small", language=None)
        if transcriber.load_model():
            result = transcriber.transcribe_raw_pcm(
                diag["raw_pcm"],
                in_rate=diag["sample_rate"],
                in_channels=1,
            )
            print(f"\n✔ TRANSCRIPTION RESULT: \"{result or '[No speech recognized]'}\"")
        else:
            print(f"❌ Faster-Whisper failed to load: {transcriber.last_error}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAM Physical Microphone Diagnostics")
    parser.add_argument("--device", type=int, default=None, help="Device index to test")
    parser.add_argument("--duration", type=float, default=3.0, help="Recording duration in seconds")
    parser.add_argument("--play", action="store_true", help="Automatically play back recorded audio")
    parser.add_argument("--save-wav", action="store_true", help="Explicitly save raw diagnostic audio as a WAV file")
    parser.add_argument("--no-save", dest="save_wav", action="store_false", help="Do not save a WAV file (default)")
    parser.set_defaults(save_wav=False)
    parser.add_argument("--no-transcribe", action="store_true", help="Do not run Whisper transcription")

    args = parser.parse_args()
    run_diagnostics(
        device_index=args.device,
        duration=args.duration,
        save_wav=args.save_wav,
        transcribe=not args.no_transcribe,
        play_audio=args.play
    )
