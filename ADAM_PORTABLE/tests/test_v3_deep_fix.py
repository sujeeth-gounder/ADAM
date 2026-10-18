"""Unit and Integration Tests for ADAM AI Deep Fix:
- Scorpio SUV and Backpack 3D procedural generation with asset validation
- Bounding box normalization and geometry verification
- 3D intent routing ("Make it 3D Scorpio car", "Make me a 3D bag")
- Camera and vision intent routing ("Who is in the camera?", person detection & identity disclaimer)
- Multi-wake-word activation ("Hey Adam", "Drake", "Hi", "Oi") and word boundary safety
"""

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest

from app.config import DEFAULT_WAKE_WORDS
from app.integrations import SystemIntegrations
from app.speech import match_wake_word
from app.three_d import (
    MeshData,
    build_backpack_fallback,
    build_scorpio_suv_fallback,
    generate_3d_object,
    load_obj_mesh,
)
from app.vision import Detection, VisionService


class Test3DGeometryAndNormalization(unittest.TestCase):
    def test_pure_python_backpack_geometry(self):
        mesh = build_backpack_fallback(size=1.0)
        valid, msg = mesh.validate_geometry()
        self.assertTrue(valid, f"Backpack fallback failed validation: {msg}")
        (min_c, max_c) = mesh.get_bounding_box()
        # Max extent should equal 1.5
        extent = max(max_c[0] - min_c[0], max_c[1] - min_c[1], max_c[2] - min_c[2])
        self.assertAlmostEqual(extent, 1.5, places=3)
        self.assertGreater(len(mesh.vertices), 50)
        self.assertGreater(len(mesh.faces), 50)

    def test_pure_python_scorpio_suv_geometry(self):
        mesh = build_scorpio_suv_fallback(size=1.0)
        valid, msg = mesh.validate_geometry()
        self.assertTrue(valid, f"Scorpio fallback failed validation: {msg}")
        (min_c, max_c) = mesh.get_bounding_box()
        extent = max(max_c[0] - min_c[0], max_c[1] - min_c[1], max_c[2] - min_c[2])
        self.assertAlmostEqual(extent, 1.5, places=3)
        self.assertGreater(len(mesh.vertices), 100)
        self.assertGreater(len(mesh.faces), 100)

    def test_blender_generation_scorpio_and_backpack(self):
        # Test Scorpio SUV generation
        ok, asset, msg = generate_3d_object("scorpio_suv", "standard", "silver")
        self.assertTrue(ok, f"Failed Scorpio generation: {msg}")
        self.assertIsNotNone(asset)
        self.assertTrue(asset.blend_path.exists() and asset.blend_path.stat().st_size > 50)
        self.assertTrue(asset.glb_path.exists() and asset.glb_path.stat().st_size > 50)
        self.assertTrue(asset.stl_path.exists() and asset.stl_path.stat().st_size > 50)
        self.assertTrue(asset.obj_path.exists() and asset.obj_path.stat().st_size > 50)
        self.assertGreater(asset.vertex_count, 1000)
        self.assertGreater(asset.polygon_count, 1000)

        # Test OBJ loading and automatic normalization
        loaded_mesh = load_obj_mesh(asset.obj_path, normalize=True)
        valid, val_msg = loaded_mesh.validate_geometry()
        self.assertTrue(valid, f"Scorpio OBJ validation failed: {val_msg}")
        (min_c, max_c) = loaded_mesh.get_bounding_box()
        self.assertAlmostEqual(max_c[2] - min_c[2], 1.5, places=1)

        # Test Backpack generation
        ok2, asset2, msg2 = generate_3d_object("backpack", "standard", "blue")
        self.assertTrue(ok2, f"Failed Backpack generation: {msg2}")
        self.assertIsNotNone(asset2)
        self.assertTrue(asset2.glb_path.exists() and asset2.glb_path.stat().st_size > 50)
        self.assertGreater(asset2.vertex_count, 1000)


class TestIntentRouting(unittest.TestCase):
    def setUp(self):
        self.vision = VisionService()
        self.integrations = SystemIntegrations(vision_service=self.vision)

    def test_3d_scorpio_intent_routing(self):
        queries = [
            "Make it 3D Scorpio car",
            "make a 3d scorpio suv",
            "Create a 3D Scorpio car",
            "build a 3d mahindra scorpio suv",
            "Make me a 3D bag",
            "create a backpack",
        ]
        for q in queries:
            handled, resp, act = self.integrations.try_handle_command(q)
            self.assertTrue(handled, f"Query '{q}' was not handled as 3D creation intent!")
            self.assertIn("3D Studio Creation Complete", resp)

    def test_camera_who_queries_routed(self):
        who_queries = [
            "Who is in the camera?",
            "who is in front of the camera",
            "who do you see",
            "who can you see",
            "is there a person in front of me",
            "is anyone there",
            "can you see me",
        ]
        for q in who_queries:
            handled, resp, act = self.integrations.try_handle_command(q)
            self.assertTrue(handled, f"Query '{q}' was not intercepted by deterministic vision route!")
            self.assertNotIn("rear-facing", resp.lower())
            self.assertNotIn("front-facing", resp.lower())

    def test_hand_object_inspection_routing(self):
        queries = [
            "What object am I holding?",
            "what am i holding",
            "what is in my hand",
            "inspect this phone",
        ]
        for q in queries:
            handled, resp, act = self.integrations.try_handle_command(q)
            self.assertTrue(handled, f"Query '{q}' was not routed to vision inspection!")


class TestVisionTruthfulness(unittest.TestCase):
    def setUp(self):
        self.vision = VisionService()

    def test_stopped_camera_truthfulness(self):
        desc = self.vision.describe_person()
        self.assertIn("stopped", desc.lower())

    def test_person_detection_identity_disclaimer(self):
        # Simulate active camera with a detected person
        self.vision._status = "CAMERA_ACTIVE"
        self.vision._latest_detection_time = 1e12  # Far in future to avoid stale check
        import time
        self.vision._latest_detection_time = time.time()
        self.vision._latest_detections = (
            Detection(
                label="person",
                confidence=0.92,
                x=100,
                y=80,
                width=240,
                height=320,
                area_pct=25.0,
                dominant_color="black",
                timestamp=time.time(),
            ),
        )
        desc = self.vision.describe_person()
        self.assertIn("person", desc.lower())
        self.assertIn("facial recognition", desc.lower())
        self.assertIn("cannot verify personal identity", desc.lower())


class TestMultiWakeWord(unittest.TestCase):
    def test_supported_wake_words(self):
        words = DEFAULT_WAKE_WORDS
        self.assertIn("hey adam", words)
        self.assertIn("drake", words)
        self.assertIn("hi", words)
        self.assertIn("oi", words)

        # Test "hey adam"
        det, phrase, rem = match_wake_word("hey adam what time is it", words)
        self.assertTrue(det)
        self.assertEqual(phrase, "hey adam")
        self.assertEqual(rem, "what time is it")

        # Test "drake"
        det, phrase, rem = match_wake_word("drake make a 3d car", words)
        self.assertTrue(det)
        self.assertEqual(phrase, "drake")
        self.assertEqual(rem, "make a 3d car")

        # Test "hi"
        det, phrase, rem = match_wake_word("hi tell me the time", words)
        self.assertTrue(det)
        self.assertEqual(phrase, "hi")

        # Test "oi"
        det, phrase, rem = match_wake_word("oi open youtube", words)
        self.assertTrue(det)
        self.assertEqual(phrase, "oi")

        # Test boundary protection (ensure "higher" and "oil" do not trigger)
        det, _, _ = match_wake_word("higher ground", words)
        self.assertFalse(det)
        det, _, _ = match_wake_word("oil changes near me", words)
        self.assertFalse(det)


if __name__ == "__main__":
    unittest.main()
