"""Tests for optional local vision and actual system monitoring."""

import threading
import time
import unittest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from unittest.mock import patch

import numpy as np

from app.speech import DoubleClapDetector, SpeechListener
from app.system_monitor import read_system_metrics
from app.main import AdamApp
from app.vision import (
    COCO_LABELS,
    Detection,
    VisionService,
    decode_yolox_output,
    prepare_yolox_input,
)


class VisionTests(unittest.TestCase):
    def test_preprocessing_matches_opencv_zoo_rgb_and_pixel_range(self):
        frame = np.zeros((320, 640, 3), dtype=np.uint8)
        frame[:] = (10, 20, 30)

        input_tensor = prepare_yolox_input(frame)

        self.assertEqual(input_tensor.shape, (1, 3, 640, 640))
        self.assertEqual(input_tensor.dtype, np.float32)
        np.testing.assert_array_equal(input_tensor[0, :, 0, 0], (30, 20, 10))
        np.testing.assert_array_equal(input_tensor[0, :, 639, 639], (114, 114, 114))

    def test_voice_vision_phrases_are_separated_from_other_commands(self):
        cases = {
            "Turn on object detection mode.": "start",
            "What can you see?": "describe",
            "Identify this object.": "describe",
            "Stop object detection.": "stop",
            "Close the camera.": "stop",
            "Play some music": None,
            "Open terminal": None,
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(AdamApp._vision_command_intent(phrase), expected)

    def test_yolox_output_decodes_a_genuine_class_score_and_box(self):
        output = np.zeros((1, 8400, 85), dtype=np.float32)
        output[0, 0, :4] = (0.5, 0.5, 0.0, 0.0)
        output[0, 0, 4] = 0.9
        output[0, 0, 5] = 0.95

        detections = decode_yolox_output(output, 640, 640, confidence_threshold=0.4)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "person")
        self.assertAlmostEqual(detections[0].confidence, 0.855, places=3)
        self.assertEqual((detections[0].x, detections[0].y), (0, 0))
        self.assertEqual((detections[0].width, detections[0].height), (8, 8))

    def test_glasses_are_not_claimed_as_a_supported_model_class(self):
        self.assertNotIn("glasses", COCO_LABELS)
        self.assertNotIn("eyeglasses", COCO_LABELS)

    def test_latest_summary_uses_only_current_model_detections(self):
        service = VisionService()
        with service._lock:
            service._status = "CAMERA_ACTIVE"
            service._latest_detection_time = time.monotonic()
            service._latest_detections = (
                Detection("person", 0.94, 0, 0, 20, 20),
                Detection("person", 0.81, 20, 0, 20, 20),
                Detection("laptop", 0.76, 40, 0, 20, 20),
            )
        self.assertEqual(service.describe_latest(), "I can see 1 laptop, 2 people.")

    def test_real_model_runs_inference_and_camera_worker_releases_capture(self):
        model_path = Path("models/vision/yolox_s_coco_int8bq.onnx").resolve()
        if not model_path.is_file():
            self.skipTest("Local YOLOX model is not installed.")

        import cv2

        frame_ready = threading.Event()
        active = threading.Event()

        class FakeCapture:
            def __init__(self):
                self.released = False

            def set(self, *_args):
                return True

            def isOpened(self):
                return True

            def read(self):
                if self.released:
                    return False, None
                return True, np.zeros((480, 640, 3), dtype=np.uint8)

            def release(self):
                self.released = True

        capture = FakeCapture()
        service = VisionService(model_path=model_path, max_inference_fps=2)

        def on_status(state, _message):
            if state == "CAMERA_ACTIVE":
                active.set()

        with patch.object(cv2, "VideoCapture", return_value=capture):
            self.assertTrue(
                service.start(
                    lambda _frame, _detections: frame_ready.set(),
                    on_status,
                )
            )
            self.assertTrue(active.wait(5), "detector did not enter active state")
            self.assertTrue(frame_ready.wait(10), "model did not produce a frame")
            service.stop()
            service._thread.join(5)

        self.assertTrue(capture.released)
        self.assertEqual(service.status, "CAMERA_STOPPED")

    def test_app_imports_when_optional_opencv_is_missing(self):
        project_root = Path(__file__).resolve().parent
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.modules['cv2'] = None; import app.main",
            ],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )

    def test_detector_reports_missing_optional_opencv(self):
        model_path = Path("models/vision/yolox_s_coco_int8bq.onnx").resolve()
        service = VisionService(model_path=model_path)
        failed = threading.Event()
        messages = []

        def on_status(state, message):
            if state == "CAMERA_ERROR":
                messages.append(message)
                failed.set()

        with patch.dict("sys.modules", {"cv2": None}):
            self.assertTrue(service.start(lambda *_args: None, on_status))
            self.assertTrue(failed.wait(5))
            service._thread.join(5)

        self.assertEqual(service.status, "CAMERA_ERROR")
        self.assertIn("OpenCV is unavailable", messages[0])


class ClapActivationTests(unittest.TestCase):
    def test_transient_threshold_rejects_ambient_and_speech_level_audio(self):
        self.assertFalse(SpeechListener.is_clap_pulse(120, 500, 100, 0.7))
        self.assertFalse(SpeechListener.is_clap_pulse(800, 2500, 100, 0.7))
        self.assertTrue(SpeechListener.is_clap_pulse(1000, 5000, 150, 0.7))

    def test_double_clap_requires_pair_and_enforces_cooldown(self):
        detector = DoubleClapDetector()
        self.assertFalse(detector.update(True, 10.0))
        self.assertFalse(detector.update(False, 10.1))
        self.assertTrue(detector.update(True, 10.35))
        self.assertFalse(detector.update(True, 10.6))
        self.assertFalse(detector.update(True, 14.4))


class SystemMonitorTests(unittest.TestCase):
    def test_metrics_are_reported_as_real_values_or_unavailable(self):
        metrics = read_system_metrics()
        self.assertRegex(metrics["cpu"], r"^\d+%$")
        self.assertRegex(metrics["memory"], r"^\d+%$")
        for key in ("battery", "temperature"):
            self.assertTrue(metrics[key] == "Unavailable" or metrics[key])


if __name__ == "__main__":
    unittest.main()
