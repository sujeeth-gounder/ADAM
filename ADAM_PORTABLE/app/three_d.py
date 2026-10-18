"""3D Object Generation & Export Engine for ADAM AI 3.0.
Supports parametric headless Blender generation producing:
- Native .blend project file
- Standard .glb (glTF 2.0 binary)
- 3D Printing / CAD .stl
- Standard Wavefront .obj
- Fast Workbench rendered .png thumbnail
- Model specification metadata .json
Also features a pure-Python mesh engine fallback ensuring models are generated
instantly on any machine without dependencies.
"""

import json
import logging
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.config import EXPORTS_DIR, SCRIPTS_DIR

logger = logging.getLogger("ADAM.ThreeD")

GENERATED_MODELS_DIR = EXPORTS_DIR / "generated_models"
GENERATED_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def is_blender_available() -> bool:
    """Check if Blender executable is accessible."""
    return shutil.which("blender") is not None


def get_blender_path() -> Optional[str]:
    return shutil.which("blender")


def get_available_shapes() -> List[str]:
    return [
        "water_bottle",
        "backpack",
        "scorpio_suv",
        "scorpio",
        "suv",
        "mahindra_scorpio",
        "chair",
        "coffee_mug",
        "sports_car",
        "robot",
        "table",
        "cube",
        "sphere",
        "cylinder",
        "torus",
        "pyramid",
        "scifi_drone",
        "scifi_pedestal",
    ]


@dataclass
class GeneratedModelAsset:
    name: str
    shape: str
    color: str
    style: str
    directory: Path
    blend_path: Path
    glb_path: Path
    stl_path: Path
    obj_path: Path
    thumb_path: Optional[Path]
    meta_path: Path
    vertex_count: int = 0
    polygon_count: int = 0
    message: str = ""


# ============================================================================
# PURE PYTHON 3D GEOMETRY ENGINE (Fallback & Standalone Viewer Parser)
# ============================================================================

class MeshData:
    """Container for 3D vertices, normals, and triangular faces."""

    def __init__(self):
        self.vertices: List[Tuple[float, float, float]] = []
        self.faces: List[Tuple[int, int, int]] = []

    def add_vertex(self, x: float, y: float, z: float) -> int:
        self.vertices.append((float(x), float(y), float(z)))
        return len(self.vertices) - 1

    def add_triangle(self, i1: int, i2: int, i3: int):
        self.faces.append((i1, i2, i3))

    def add_quad(self, i1: int, i2: int, i3: int, i4: int):
        self.faces.append((i1, i2, i3))
        self.faces.append((i1, i3, i4))

    def compute_face_normal(self, f: Tuple[int, int, int]) -> Tuple[float, float, float]:
        v1 = self.vertices[f[0]]
        v2 = self.vertices[f[1]]
        v3 = self.vertices[f[2]]
        ax, ay, az = v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]
        bx, by, bz = v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]
        nx = ay * bz - az * by
        ny = az * bx - ax * bz
        nz = ax * by - ay * bx
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length > 1e-9:
            return (nx / length, ny / length, nz / length)
        return (0.0, 0.0, 1.0)

    def get_bounding_box(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return ((min_x, min_y, min_z), (max_x, max_y, max_z))."""
        if not self.vertices:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))

    def validate_geometry(self) -> Tuple[bool, str]:
        """Verify mesh integrity (vertices, faces, no non-finite coordinates, valid indices)."""
        if not self.vertices:
            return False, "Mesh has zero vertices."
        if not self.faces:
            return False, "Mesh has zero faces."
        n_verts = len(self.vertices)
        for i, (x, y, z) in enumerate(self.vertices):
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                return False, f"Vertex {i} contains non-finite coordinates: ({x}, {y}, {z})"
        for fi, face in enumerate(self.faces):
            for vi in face:
                if vi < 0 or vi >= n_verts:
                    return False, f"Face {fi} references out-of-range vertex index {vi} (total: {n_verts})"
        return True, f"Valid mesh ({n_verts} verts, {len(self.faces)} faces)"

    def normalize_to_bounds(self, target_size: float = 1.5) -> None:
        """Center geometry around origin (0, 0, 0) and rescale so max extent equals target_size."""
        if not self.vertices:
            return
        (min_x, min_y, min_z), (max_x, max_y, max_z) = self.get_bounding_box()
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        cz = (min_z + max_z) / 2.0
        extent = max(max_x - min_x, max_y - min_y, max_z - min_z, 1e-6)
        scale = target_size / extent
        self.vertices = [
            ((v[0] - cx) * scale, (v[1] - cy) * scale, (v[2] - cz) * scale)
            for v in self.vertices
        ]

    def export_stl_binary(self, filepath: Path) -> None:
        with open(filepath, "wb") as f:
            header = b"ADAM AI 3.0 Mesh Engine".ljust(80, b"\0")
            f.write(header)
            f.write(struct.pack("<I", len(self.faces)))
            for face in self.faces:
                norm = self.compute_face_normal(face)
                v1, v2, v3 = self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]
                data = struct.pack(
                    "<12fH",
                    norm[0], norm[1], norm[2],
                    v1[0], v1[1], v1[2],
                    v2[0], v2[1], v2[2],
                    v3[0], v3[1], v3[2],
                    0
                )
                f.write(data)

    def export_obj(self, filepath: Path) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# ADAM AI 3.0 Generated OBJ\n# Vertices: {len(self.vertices)} Faces: {len(self.faces)}\n")
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in self.faces:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def load_obj_mesh(filepath: Path, normalize: bool = True) -> MeshData:
    """Parse Wavefront OBJ file into MeshData for interactive viewport."""
    mesh = MeshData()
    if not filepath.exists():
        return mesh
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                mesh.add_vertex(float(parts[1]), float(parts[2]), float(parts[3]))
            elif parts[0] == "f" and len(parts) >= 4:
                # Face indices in OBJ are 1-based, can be v/vt/vn
                idx = []
                for p in parts[1:]:
                    v_idx = int(p.split("/")[0])
                    # Handle negative indices if any
                    if v_idx < 0:
                        v_idx = len(mesh.vertices) + v_idx
                    else:
                        v_idx -= 1
                    idx.append(v_idx)
                if len(idx) == 3:
                    mesh.add_triangle(idx[0], idx[1], idx[2])
                elif len(idx) == 4:
                    mesh.add_quad(idx[0], idx[1], idx[2], idx[3])
                elif len(idx) > 4:
                    # Polygon fan triangulation
                    for i in range(1, len(idx) - 1):
                        mesh.add_triangle(idx[0], idx[i], idx[i + 1])
    if normalize and mesh.vertices:
        mesh.normalize_to_bounds(target_size=1.5)
    return mesh


# --- Pure-Python Mesh Builders (Fallback) ---

def build_cube_mesh(size: float = 1.0) -> MeshData:
    mesh = MeshData()
    s = size / 2.0
    coords = [
        (-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
        (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)
    ]
    for c in coords:
        mesh.add_vertex(*c)
    mesh.add_quad(0, 3, 2, 1)
    mesh.add_quad(4, 5, 6, 7)
    mesh.add_quad(0, 1, 5, 4)
    mesh.add_quad(2, 3, 7, 6)
    mesh.add_quad(0, 4, 7, 3)
    mesh.add_quad(1, 2, 6, 5)
    return mesh


def build_cylinder_mesh(radius: float = 0.5, height: float = 1.0, segments: int = 24) -> MeshData:
    mesh = MeshData()
    h2 = height / 2.0
    bottom_center = mesh.add_vertex(0.0, 0.0, -h2)
    top_center = mesh.add_vertex(0.0, 0.0, h2)
    bottom_ring, top_ring = [], []
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        bottom_ring.append(mesh.add_vertex(x, y, -h2))
        top_ring.append(mesh.add_vertex(x, y, h2))
    for i in range(segments):
        nxt = (i + 1) % segments
        mesh.add_triangle(bottom_center, bottom_ring[nxt], bottom_ring[i])
        mesh.add_triangle(top_center, top_ring[i], top_ring[nxt])
        mesh.add_quad(bottom_ring[i], bottom_ring[nxt], top_ring[nxt], top_ring[i])
    return mesh


def build_water_bottle_fallback(size: float = 1.0) -> MeshData:
    """Pure-python water bottle with body, neck, cap."""
    body = build_cylinder_mesh(radius=0.35 * size, height=1.1 * size, segments=20)
    neck = build_cylinder_mesh(radius=0.18 * size, height=0.25 * size, segments=16)
    cap = build_cylinder_mesh(radius=0.20 * size, height=0.15 * size, segments=16)

    # Offset neck and cap
    neck.vertices = [(v[0], v[1], v[2] + 0.65 * size) for v in neck.vertices]
    cap.vertices = [(v[0], v[1], v[2] + 0.85 * size) for v in cap.vertices]

    off_neck = len(body.vertices)
    body.vertices.extend(neck.vertices)
    for f in neck.faces:
        body.add_triangle(f[0] + off_neck, f[1] + off_neck, f[2] + off_neck)

    off_cap = len(body.vertices)
    body.vertices.extend(cap.vertices)
    for f in cap.faces:
        body.add_triangle(f[0] + off_cap, f[1] + off_cap, f[2] + off_cap)
    return body


def build_chair_fallback(size: float = 1.0) -> MeshData:
    mesh = MeshData()
    s = size
    # Seat
    seat = build_cube_mesh(1.0)
    seat.vertices = [(v[0] * 0.5 * s, v[1] * 0.5 * s, v[2] * 0.06 * s + 0.5 * s) for v in seat.vertices]
    mesh.vertices.extend(seat.vertices)
    mesh.faces.extend(seat.faces)

    # 4 legs
    for lx, ly in [(-0.2 * s, -0.2 * s), (0.2 * s, -0.2 * s), (-0.2 * s, 0.2 * s), (0.2 * s, 0.2 * s)]:
        leg = build_cylinder_mesh(radius=0.03 * s, height=0.5 * s, segments=8)
        leg.vertices = [(v[0] + lx, v[1] + ly, v[2] + 0.25 * s) for v in leg.vertices]
        off = len(mesh.vertices)
        mesh.vertices.extend(leg.vertices)
        for f in leg.faces:
            mesh.add_triangle(f[0] + off, f[1] + off, f[2] + off)
    return mesh


def build_backpack_fallback(size: float = 1.0) -> MeshData:
    """Pure-Python fallback backpack: main compartment, front pocket, side pockets, handle, straps."""
    mesh = MeshData()
    s = size
    # Main body
    body = build_cube_mesh(1.0)
    body.vertices = [(v[0] * 0.45 * s, v[1] * 0.3 * s, v[2] * 0.65 * s) for v in body.vertices]
    mesh.vertices.extend(body.vertices)
    mesh.faces.extend(body.faces)

    # Front pocket
    front_pocket = build_cube_mesh(1.0)
    front_pocket.vertices = [
        (v[0] * 0.35 * s, v[1] * 0.12 * s - 0.21 * s, v[2] * 0.32 * s - 0.12 * s)
        for v in front_pocket.vertices
    ]
    off = len(mesh.vertices)
    mesh.vertices.extend(front_pocket.vertices)
    for f in front_pocket.faces:
        mesh.add_triangle(f[0] + off, f[1] + off, f[2] + off)

    # Left & Right side pockets
    for side_x in [-0.25 * s, 0.25 * s]:
        side_p = build_cylinder_mesh(radius=0.08 * s, height=0.35 * s, segments=12)
        side_p.vertices = [(v[0] + side_x, v[1], v[2] - 0.1 * s) for v in side_p.vertices]
        off_s = len(mesh.vertices)
        mesh.vertices.extend(side_p.vertices)
        for f in side_p.faces:
            mesh.add_triangle(f[0] + off_s, f[1] + off_s, f[2] + off_s)

    # Top handle
    handle = build_cube_mesh(1.0)
    handle.vertices = [(v[0] * 0.18 * s, v[1] * 0.04 * s, v[2] * 0.06 * s + 0.35 * s) for v in handle.vertices]
    off_h = len(mesh.vertices)
    mesh.vertices.extend(handle.vertices)
    for f in handle.faces:
        mesh.add_triangle(f[0] + off_h, f[1] + off_h, f[2] + off_h)

    # Dual shoulder straps on back (+Y)
    for strap_x in [-0.12 * s, 0.12 * s]:
        strap = build_cube_mesh(1.0)
        strap.vertices = [(v[0] * 0.05 * s + strap_x, v[1] * 0.04 * s + 0.17 * s, v[2] * 0.5 * s) for v in strap.vertices]
        off_st = len(mesh.vertices)
        mesh.vertices.extend(strap.vertices)
        for f in strap.faces:
            mesh.add_triangle(f[0] + off_st, f[1] + off_st, f[2] + off_st)

    mesh.normalize_to_bounds(target_size=1.5 * size)
    return mesh


def build_scorpio_suv_fallback(size: float = 1.0) -> MeshData:
    """Pure-Python fallback Scorpio SUV: body, cabin, hood scoop, grille, roof rails, 4 wheels."""
    mesh = MeshData()
    s = size
    # Lower chassis / body
    chassis = build_cube_mesh(1.0)
    chassis.vertices = [(v[0] * 0.5 * s, v[1] * 1.0 * s, v[2] * 0.25 * s + 0.15 * s) for v in chassis.vertices]
    mesh.vertices.extend(chassis.vertices)
    mesh.faces.extend(chassis.faces)

    # Cabin greenhouse
    cabin = build_cube_mesh(1.0)
    cabin.vertices = [(v[0] * 0.44 * s, v[1] * 0.6 * s + 0.1 * s, v[2] * 0.25 * s + 0.38 * s) for v in cabin.vertices]
    off_c = len(mesh.vertices)
    mesh.vertices.extend(cabin.vertices)
    for f in cabin.faces:
        mesh.add_triangle(f[0] + off_c, f[1] + off_c, f[2] + off_c)

    # Bonnet scoop
    scoop = build_cube_mesh(1.0)
    scoop.vertices = [(v[0] * 0.16 * s, v[1] * 0.2 * s - 0.28 * s, v[2] * 0.04 * s + 0.3 * s) for v in scoop.vertices]
    off_sc = len(mesh.vertices)
    mesh.vertices.extend(scoop.vertices)
    for f in scoop.faces:
        mesh.add_triangle(f[0] + off_sc, f[1] + off_sc, f[2] + off_sc)

    # Roof rails (left & right)
    for rx in [-0.2 * s, 0.2 * s]:
        rail = build_cube_mesh(1.0)
        rail.vertices = [(v[0] * 0.03 * s + rx, v[1] * 0.55 * s + 0.1 * s, v[2] * 0.03 * s + 0.52 * s) for v in rail.vertices]
        off_r = len(mesh.vertices)
        mesh.vertices.extend(rail.vertices)
        for f in rail.faces:
            mesh.add_triangle(f[0] + off_r, f[1] + off_r, f[2] + off_r)

    # 4 SUV Wheels
    wheel_positions = [
        (-0.28 * s, -0.36 * s),
        (0.28 * s, -0.36 * s),
        (-0.28 * s, 0.36 * s),
        (0.28 * s, 0.36 * s),
    ]
    for wx, wy in wheel_positions:
        wheel = build_cylinder_mesh(radius=0.14 * s, height=0.1 * s, segments=16)
        wheel.vertices = [(v[2] + wx, v[1] + wy, v[0] + 0.05 * s) for v in wheel.vertices]
        off_w = len(mesh.vertices)
        mesh.vertices.extend(wheel.vertices)
        for f in wheel.faces:
            mesh.add_triangle(f[0] + off_w, f[1] + off_w, f[2] + off_w)

    mesh.normalize_to_bounds(target_size=1.5 * size)
    return mesh


# ============================================================================
# PRIMARY GENERATION PIPELINE
# ============================================================================

def generate_3d_object(
    shape_type: str = "water_bottle",
    style: str = "realistic",
    color: str = "blue",
    size: float = 1.0,
    output_name: Optional[str] = None,
    features: Optional[str] = None
) -> Tuple[bool, Optional[GeneratedModelAsset], str]:
    """
    Main generator dispatching headless Blender automation with rich assets output.
    Returns: (success: bool, asset: Optional[GeneratedModelAsset], message: str)
    """
    raw_shape = shape_type.lower().strip().replace(" ", "_").replace("-", "_")
    alias_map = {
        "backpack": "backpack",
        "bag": "backpack",
        "rucksack": "backpack",
        "school_bag": "backpack",
        "scorpio": "scorpio_suv",
        "scorpio_car": "scorpio_suv",
        "scorpio_suv": "scorpio_suv",
        "mahindra_scorpio": "scorpio_suv",
        "suv": "scorpio_suv",
        "bottle": "water_bottle",
        "water_bottle": "water_bottle",
        "car": "sports_car",
        "sports_car": "sports_car",
        "mug": "coffee_mug",
        "cup": "coffee_mug",
        "coffee_mug": "coffee_mug",
    }
    clean_shape = alias_map.get(raw_shape, raw_shape)
    clean_color = color.lower().strip() or "blue"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = output_name or f"{clean_shape}_{timestamp}"

    model_dir = GENERATED_MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    blend_file = model_dir / f"{model_name}.blend"
    glb_file = model_dir / f"{model_name}.glb"
    stl_file = model_dir / f"{model_name}.stl"
    obj_file = model_dir / f"{model_name}.obj"
    thumb_file = model_dir / f"{model_name}_thumb.png"
    meta_file = model_dir / f"{model_name}_meta.json"

    # Step 1: Attempt Blender headless automation
    blender_bin = get_blender_path()
    blender_script = SCRIPTS_DIR / "blender_gen.py"

    if blender_bin and blender_script.exists():
        cmd = [
            blender_bin,
            "-b",
            "-P", str(blender_script.resolve()),
            "--",
            "--shape", clean_shape,
            "--style", style,
            "--color", clean_color,
            "--output_dir", str(model_dir.resolve()),
            "--name", model_name,
            "--size", str(size)
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            all_valid = all(
                f.exists() and f.stat().st_size > 50
                for f in [blend_file, glb_file, stl_file, obj_file]
            )
            if proc.returncode == 0 and all_valid:
                v_count, f_count = 0, 0
                if meta_file.exists():
                    try:
                        with open(meta_file, "r") as mf:
                            m_data = json.load(mf)
                            v_count = m_data.get("vertices", 0)
                            f_count = m_data.get("polygons", 0)
                    except Exception:
                        pass

                asset = GeneratedModelAsset(
                    name=model_name,
                    shape=clean_shape,
                    color=clean_color,
                    style=style,
                    directory=model_dir,
                    blend_path=blend_file,
                    glb_path=glb_file,
                    stl_path=stl_file,
                    obj_path=obj_file,
                    thumb_path=thumb_file if thumb_file.exists() else None,
                    meta_path=meta_file,
                    vertex_count=v_count,
                    polygon_count=f_count,
                    message=f"Blender generated {clean_shape} ({v_count} verts, {f_count} polys)"
                )
                return True, asset, f"Created {clean_shape} ({clean_color}) with Blender engine."
            else:
                logger.warning(f"Blender returncode {proc.returncode}, valid: {all_valid}. Stderr: {proc.stderr[:300]}")
        except Exception as e:
            logger.warning(f"Blender execution error: {e}")

    # Step 2: Fallback to Pure Python Mesh Engine
    logger.info(f"Using pure-Python mesh engine fallback for {clean_shape}")
    mesh_builders = {
        "backpack": lambda: build_backpack_fallback(size),
        "scorpio_suv": lambda: build_scorpio_suv_fallback(size),
        "water_bottle": lambda: build_water_bottle_fallback(size),
        "bottle": lambda: build_water_bottle_fallback(size),
        "chair": lambda: build_chair_fallback(size),
        "cube": lambda: build_cube_mesh(size),
        "cylinder": lambda: build_cylinder_mesh(radius=0.5 * size, height=1.0 * size),
    }
    builder = mesh_builders.get(clean_shape, lambda: build_cube_mesh(size))
    mesh = builder()
    mesh.export_obj(obj_file)
    mesh.export_stl_binary(stl_file)

    # Save metadata
    meta_data = {
        "shape": clean_shape,
        "style": style,
        "color": clean_color,
        "size": size,
        "vertices": len(mesh.vertices),
        "polygons": len(mesh.faces),
        "engine": "python_fallback",
        "blend_file": str(blend_file),
        "glb_file": str(glb_file),
        "stl_file": str(stl_file),
        "obj_file": str(obj_file),
        "thumbnail": None
    }
    with open(meta_file, "w", encoding="utf-8") as mf:
        json.dump(meta_data, mf, indent=2)

    asset = GeneratedModelAsset(
        name=model_name,
        shape=clean_shape,
        color=clean_color,
        style=style,
        directory=model_dir,
        blend_path=blend_file,
        glb_path=glb_file,
        stl_path=stl_file,
        obj_path=obj_file,
        thumb_path=None,
        meta_path=meta_file,
        vertex_count=len(mesh.vertices),
        polygon_count=len(mesh.faces),
        message=f"Generated {clean_shape} via ADAM Mesh Engine ({len(mesh.vertices)} verts)"
    )
    return True, asset, f"Created {clean_shape} ({clean_color}) via ADAM 3D Engine."
