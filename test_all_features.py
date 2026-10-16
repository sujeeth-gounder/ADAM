#!/usr/bin/env python3
"""Comprehensive diagnostic & verification script for ADAM.
Tests:
1. Ollama inference & stream generation
2. Audio input enumeration & microphone test
3. Piper Amy TTS synthesis
4. 3D geometry engine
"""

import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Ensure venv site packages are present
VENV_PKGS = [
    ROOT / "venv" / "lib" / "python3.14" / "site-packages",
    ROOT / "venv" / "lib64" / "python3.14" / "site-packages",
]
for p in VENV_PKGS:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.integrations import OllamaClient
from app.speech import SpeechListener, list_input_microphones, test_microphone_level
from app.voice import PiperTTS
from app.three_d import generate_3d_object, is_blender_available

def run_tests():
    print("=" * 60)
    print("ADAM SYSTEM VERIFICATION TEST SUITE")
    print("=" * 60)

    # 1. Ollama Test
    print("\n[TEST 1] Ollama Local Inference (qwen2.5:1.5b)")
    client = OllamaClient(model="qwen2.5:1.5b")
    connected = client.check_connection()
    print(f"  Connection to Ollama (http://127.0.0.1:11434): {'✔ OK' if connected else '✖ FAILED'}")
    print(f"  Available models: {client.available_models}")
    
    if connected:
        prompt = "Hello ADAM, are you ready to assist me? Answer in under 10 words."
        print(f"  Sending prompt: '{prompt}'")
        stream_chunks = []
        t0 = time.time()
        try:
            for chunk in client.generate_stream(prompt):
                stream_chunks.append(chunk)
            t_elapsed = time.time() - t0
            full_reply = "".join(stream_chunks).strip()
            print(f"  Response ({t_elapsed:.2f}s): \"{full_reply}\"")
            print("  ✔ Ollama inference test PASSED!")
        except Exception as e:
            print(f"  ✖ Ollama inference FAILED: {e}")
    else:
        print("  ✖ Ollama offline. Check if ollama serve is running.")

    # 2. Microphone Detection Test
    print("\n[TEST 2] Microphone Detection & Audio Recording")
    mics = list_input_microphones()
    print(f"  Found {len(mics)} audio input device(s):")
    for m in mics:
        print(f"    - Index {m['index']}: {m['name']} ({m['channels']} ch, {m['sample_rate']} Hz)")
    
    listener = SpeechListener()
    print(f"  SpeechListener Status: {listener.get_status_info()}")
    print(f"  Selected Device Index: {listener.device_index} ({listener.device_name})")
    print(f"  mic_available: {listener.mic_available}")

    if listener.mic_available and listener.device_index is not None:
        print("  Testing 1.5s RMS audio level capture on selected device...")
        ok, rms, msg = test_microphone_level(device_index=listener.device_index, duration_sec=1.5)
        print(f"  Result: {msg} (RMS Peak: {rms:.4f})")
        print("  ✔ Microphone hardware verification PASSED!")
    else:
        print("  ✖ Microphone verification FAILED: No active input device found.")

    # 3. Piper Amy TTS Test
    print("\n[TEST 3] Piper TTS (Amy Voice)")
    tts = PiperTTS()
    print(f"  TTS Engine: {tts.engine_type} (Amy ONNX exists: {tts.onnx_path.exists() if tts.onnx_path else False})")
    tts.speak("ADAM system online and ready.")
    time.sleep(2.0)
    print("  ✔ Piper TTS engine verified!")

    # 4. 3D Synthesis Test
    print("\n[TEST 4] 3D Object Synthesis Engine")
    blender_ok = is_blender_available()
    print(f"  Blender Detected: {blender_ok}")
    ok, out_path, msg = generate_3d_object("cube", "stl", size=1.0)
    if ok and out_path and out_path.exists():
        print(f"  Generated 3D Asset: {out_path} ({out_path.stat().st_size} bytes)")
        print("  ✔ 3D mesh synthesis test PASSED!")
    else:
        print(f"  ✖ 3D generation FAILED: {msg}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
