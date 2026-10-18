"""Automated tests for ADAM AI 3.0 Vision and 3D Studio Engine."""

import os
import sys
import unittest
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.integrations import SystemIntegrations
from app.vision import VisionService, list_available_cameras, Detection
from app.three_d import generate_3d_object, is_blender_available, get_available_shapes


class TestADAM3VisionAndThreeD(unittest.TestCase):

    def setUp(self):
        self.vision = VisionService()
        self.generated_assets = []
        self.viewport_actions = []

        def _on_asset(asset):
            self.generated_assets.append(asset)

        def _on_action(action):
            self.viewport_actions.append(action)

        self.integrations = SystemIntegrations(
            vision_service=self.vision,
            three_d_callback=_on_asset,
            three_d_viewport_command_callback=_on_action,
        )

    def test_list_available_cameras(self):
        cams = list_available_cameras(max_probe=2)
        self.assertIsInstance(cams, list)
        for c in cams:
            self.assertIn("index", c)
            self.assertIn("name", c)

    def test_vision_describe_when_stopped(self):
        resp = self.vision.describe_latest()
        self.assertIn("camera is currently stopped", resp.lower())

    def test_vision_inspect_phone_when_stopped(self):
        insp = self.vision.inspect_object("cell phone")
        self.assertIn("camera is stopped", insp["response"].lower())

    def test_vision_inspect_phone_honest_disclaimer(self):
        """Simulate a detected phone and verify honest identification disclaimer."""
        self.vision._status = "CAMERA_ACTIVE"
        import time
        self.vision._latest_detection_time = time.time()
        self.vision._latest_detections = (
            Detection(
                label="cell phone",
                confidence=0.92,
                x=100,
                y=100,
                width=200,
                height=350,
                area_pct=22.8,
                dominant_color="dark / black",
                timestamp=time.time(),
            ),
        )
        insp = self.vision.inspect_object("cell phone")
        self.assertTrue(insp["found"])
        self.assertEqual(insp["label"], "cell phone")
        self.assertIn("generic smartphone", insp["response"].lower())
        self.assertIn("cannot be confirmed with certainty", insp["response"].lower())
        self.assertIn("black", insp["response"].lower())

    def test_3d_generation_intents(self):
        """Verify natural language 3D creation queries are parsed and executed."""
        test_queries = [
            ("Create a water bottle", "water_bottle"),
            ("Make a chair", "chair"),
            ("Create a 3D model of a coffee mug", "coffee_mug"),
            ("Make a table", "table"),
            ("Create a low-poly sports car", "sports_car"),
            ("Create a simple robot", "robot"),
            ("Make a red backpack with two pockets", "backpack"),
            ("Create a cube", "cube"),
            ("Generate a sphere", "sphere"),
        ]

        for query, expected_shape in test_queries:
            with self.subTest(query=query):
                handled, resp, pending = self.integrations.try_handle_command(query)
                self.assertTrue(handled, f"Failed to handle: {query}")
                self.assertIsNotNone(resp)
                self.assertIn("3D Studio Creation Complete", resp)
                self.assertIn(expected_shape.replace("_", " "), resp.lower())

    def test_3d_viewport_navigation_commands(self):
        """Verify 3D viewport control voice/typed commands trigger callbacks."""
        test_actions = [
            ("rotate the model", "rotate"),
            ("zoom in on the model", "zoom_in"),
            ("zoom out of model", "zoom_out"),
            ("reset view", "reset"),
            ("open in blender", "open_blender"),
            ("save 3d model", "export"),
        ]

        for query, expected_action in test_actions:
            with self.subTest(query=query):
                self.viewport_actions.clear()
                handled, resp, pending = self.integrations.try_handle_command(query)
                self.assertTrue(handled, f"Failed to handle: {query}")
                self.assertIn(expected_action, self.viewport_actions)

    def test_blender_exported_files_integrity(self):
        """Verify generated model contains all 5 required asset files."""
        ok, asset, msg = generate_3d_object("water_bottle", color="blue", style="standard")
        self.assertTrue(ok)
        self.assertIsNotNone(asset)
        self.assertTrue(asset.stl_path.exists(), "Missing STL")
        self.assertTrue(asset.obj_path.exists(), "Missing OBJ")
        self.assertTrue(asset.meta_path.exists(), "Missing JSON metadata")
        if is_blender_available():
            self.assertTrue(asset.blend_path.exists(), "Missing .blend")
            self.assertTrue(asset.glb_path.exists(), "Missing .glb")
            if asset.thumb_path:
                self.assertTrue(asset.thumb_path.exists(), "Missing thumbnail .png")


if __name__ == "__main__":
    unittest.main()
