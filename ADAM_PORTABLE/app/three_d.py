"""3D Object Generation & Export Module for ADAM (Portable).
Supports Blender headless generation to OBJ/STL, plus an integrated pure-Python
STL/OBJ generator fallback so 3D objects can be generated immediately without external dependencies.
"""

import math
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.os_adapter import OSAdapter
from config.settings_manager import SettingsManager


def get_blender_path() -> Optional[str]:
    """Detect Blender executable using OSAdapter and user preferences."""
    settings = SettingsManager()
    custom_path = settings.get("blender", "custom_executable_path", None)
    adapter = OSAdapter()
    return adapter.detect_blender(custom_path)


def is_blender_available() -> bool:
    """Return True if Blender executable is detected."""
    return get_blender_path() is not None


def get_available_shapes() -> List[str]:
    """Return list of supported parametric 3D shapes."""
    return [
        "cube",
        "sphere",
        "cylinder",
        "torus",
        "scifi_drone",
        "scifi_pedestal",
        "pyramid",
    ]


# ============================================================================
# PURE PYTHON 3D GEOMETRY ENGINE (Fallback & Standalone)
# Generates valid Wavefront OBJ and Standard Binary STL files
# ============================================================================

class MeshData:
    """Container for 3D vertices, normals, and triangular faces."""

    def __init__(self):
        self.vertices: List[Tuple[float, float, float]] = []
        self.faces: List[Tuple[int, int, int]] = []  # 0-indexed triangle tuples

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

    def export_stl_binary(self, filepath: Path) -> None:
        """Export mesh to standard binary STL format."""
        with open(filepath, "wb") as f:
            header = b"ADAM Portable - Pure Python STL Exporter".ljust(80, b"\0")
            f.write(header)
            f.write(struct.pack("<I", len(self.faces)))

            for face in self.faces:
                norm = self.compute_face_normal(face)
                v1 = self.vertices[face[0]]
                v2 = self.vertices[face[1]]
                v3 = self.vertices[face[2]]
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
        """Export mesh to Wavefront OBJ format."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# ADAM Portable 3D Object Exporter\n")
            f.write(f"# Vertices: {len(self.vertices)} Faces: {len(self.faces)}\n")
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in self.faces:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


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
    bottom_ring = []
    top_ring = []
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        bottom_ring.append(mesh.add_vertex(x, y, -h2))
        top_ring.append(mesh.add_vertex(x, y, h2))

    for i in range(segments):
        nxt = (i + 1) % segments
        mesh.add_triangle(bottom_center, bottom_ring[nxt], bottom_ring[i])
        mesh.add_triangle(top_center, top_ring[i], top_ring[nxt])
        mesh.add_quad(bottom_ring[i], bottom_ring[nxt], top_ring[nxt], top_ring[i])
    return mesh


def build_sphere_mesh(radius: float = 0.5, rings: int = 16, sectors: int = 24) -> MeshData:
    mesh = MeshData()
    for r in range(rings + 1):
        theta = math.pi * r / rings
        for s in range(sectors):
            phi = 2.0 * math.pi * s / sectors
            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.sin(theta) * math.sin(phi)
            z = radius * math.cos(theta)
            mesh.add_vertex(x, y, z)

    for r in range(rings):
        for s in range(sectors):
            s_next = (s + 1) % sectors
            i1 = r * sectors + s
            i2 = r * sectors + s_next
            i3 = (r + 1) * sectors + s_next
            i4 = (r + 1) * sectors + s
            if r == 0:
                mesh.add_triangle(i1, i3, i4)
            elif r == rings - 1:
                mesh.add_triangle(i1, i2, i3)
            else:
                mesh.add_quad(i1, i2, i3, i4)
    return mesh


def build_torus_mesh(major_r: float = 0.6, minor_r: float = 0.2, num_major: int = 24, num_minor: int = 16) -> MeshData:
    mesh = MeshData()
    for i in range(num_major):
        u = 2.0 * math.pi * i / num_major
        cos_u = math.cos(u)
        sin_u = math.sin(u)
        for j in range(num_minor):
            v = 2.0 * math.pi * j / num_minor
            x = (major_r + minor_r * math.cos(v)) * cos_u
            y = (major_r + minor_r * math.cos(v)) * sin_u
            z = minor_r * math.sin(v)
            mesh.add_vertex(x, y, z)

    for i in range(num_major):
        i_next = (i + 1) % num_major
        for j in range(num_minor):
            j_next = (j + 1) % num_minor
            mesh.add_quad(
                i * num_minor + j,
                i * num_minor + j_next,
                i_next * num_minor + j_next,
                i_next * num_minor + j
            )
    return mesh


def build_pyramid_mesh(base_size: float = 1.0, height: float = 1.0) -> MeshData:
    mesh = MeshData()
    s = base_size / 2.0
    apex = mesh.add_vertex(0.0, 0.0, height / 2.0)
    b0 = mesh.add_vertex(-s, -s, -height / 2.0)
    b1 = mesh.add_vertex(s, -s, -height / 2.0)
    b2 = mesh.add_vertex(s, s, -height / 2.0)
    b3 = mesh.add_vertex(-s, s, -height / 2.0)
    mesh.add_quad(b0, b3, b2, b1)
    mesh.add_triangle(apex, b0, b1)
    mesh.add_triangle(apex, b1, b2)
    mesh.add_triangle(apex, b2, b3)
    mesh.add_triangle(apex, b3, b0)
    return mesh


def build_scifi_drone_mesh(scale: float = 1.0) -> MeshData:
    core = build_sphere_mesh(radius=0.4 * scale, rings=12, sectors=16)
    arm_dist = 0.65 * scale
    pod_size = 0.22 * scale
    for angle_deg in [45, 135, 225, 315]:
        rad = math.radians(angle_deg)
        cx = arm_dist * math.cos(rad)
        cy = arm_dist * math.sin(rad)
        s = pod_size / 2.0
        base_idx = len(core.vertices)
        for dx, dy, dz in [
            (-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
            (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)
        ]:
            core.add_vertex(cx + dx, cy + dy, dz)
        core.add_quad(base_idx + 0, base_idx + 3, base_idx + 2, base_idx + 1)
        core.add_quad(base_idx + 4, base_idx + 5, base_idx + 6, base_idx + 7)
        core.add_quad(base_idx + 0, base_idx + 1, base_idx + 5, base_idx + 4)
        core.add_quad(base_idx + 2, base_idx + 3, base_idx + 7, base_idx + 6)
        core.add_quad(base_idx + 0, base_idx + 4, base_idx + 7, base_idx + 3)
        core.add_quad(base_idx + 1, base_idx + 2, base_idx + 6, base_idx + 5)
    return core


def build_scifi_pedestal_mesh(scale: float = 1.0) -> MeshData:
    base = build_cylinder_mesh(radius=0.8 * scale, height=0.15 * scale, segments=24)
    base.vertices = [(v[0], v[1], v[2] - 0.4 * scale) for v in base.vertices]
    mid = build_cylinder_mesh(radius=0.35 * scale, height=0.5 * scale, segments=16)
    offset_idx = len(base.vertices)
    base.vertices.extend(mid.vertices)
    for f in mid.faces:
        base.add_triangle(f[0] + offset_idx, f[1] + offset_idx, f[2] + offset_idx)
    top = build_cylinder_mesh(radius=0.6 * scale, height=0.1 * scale, segments=24)
    top.vertices = [(v[0], v[1], v[2] + 0.35 * scale) for v in top.vertices]
    offset_top = len(base.vertices)
    base.vertices.extend(top.vertices)
    for f in top.faces:
        base.add_triangle(f[0] + offset_top, f[1] + offset_top, f[2] + offset_top)
    return base


def generate_with_fallback(
    shape: str,
    export_format: str,
    output_path: Path,
    size: float = 1.0
) -> Tuple[bool, str]:
    """Pure-Python geometry generation."""
    shape_lower = shape.lower().replace("-", "_").replace(" ", "_")
    builders = {
        "cube": lambda: build_cube_mesh(size),
        "sphere": lambda: build_sphere_mesh(radius=0.5 * size),
        "cylinder": lambda: build_cylinder_mesh(radius=0.5 * size, height=1.0 * size),
        "torus": lambda: build_torus_mesh(major_r=0.6 * size, minor_r=0.2 * size),
        "pyramid": lambda: build_pyramid_mesh(base_size=1.0 * size, height=1.0 * size),
        "scifi_drone": lambda: build_scifi_drone_mesh(scale=size),
        "scifi_pedestal": lambda: build_scifi_pedestal_mesh(scale=size),
    }

    builder = builders.get(shape_lower, lambda: build_cube_mesh(size))
    mesh = builder()

    if export_format.lower() == "stl":
        mesh.export_stl_binary(output_path)
    elif export_format.lower() == "obj":
        mesh.export_obj(output_path)
    else:
        output_path = output_path.with_suffix(".stl")
        mesh.export_stl_binary(output_path)

    return True, f"Generated {shape_lower} ({len(mesh.faces)} triangles) via Portable 3D Engine -> {output_path.name}"


def generate_with_blender(
    shape: str,
    export_format: str,
    output_path: Path,
    size: float = 1.0
) -> Tuple[bool, str]:
    """Execute detected Blender in headless background mode."""
    blender_bin = get_blender_path()
    if not blender_bin:
        return False, "Blender executable not found on this system."

    adapter = OSAdapter()
    script_path = adapter.root_dir / "scripts" / "blender_gen.py"
    if not script_path.exists():
        return False, f"Blender helper script missing at {script_path}"

    cmd = [
        blender_bin,
        "-b",
        "--python", str(script_path.resolve()),
        "--",
        "--shape", shape,
        "--format", export_format,
        "--output", str(output_path.resolve()),
        "--size", str(size)
    ]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=45)
        if proc.returncode == 0 and output_path.exists():
            return True, f"Blender ({blender_bin}) rendered {shape} -> {output_path.name}"
        else:
            return False, f"Blender exited with code {proc.returncode}: {proc.stderr[:160]}"
    except Exception as e:
        return False, f"Blender execution failed: {e}"


def generate_3d_object(
    shape_type: str = "cube",
    export_format: str = "stl",
    output_name: Optional[str] = None,
    size: float = 1.0
) -> Tuple[bool, Path, str]:
    """
    Main 3D generator entrypoint.
    Returns (success: bool, filepath: Path, details: str).
    """
    adapter = OSAdapter()
    export_format = export_format.lower().strip().replace(".", "")
    if export_format not in ["stl", "obj"]:
        export_format = "stl"

    clean_shape = shape_type.lower().strip().replace(" ", "_")
    if not output_name:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_name = f"{clean_shape}_{timestamp}.{export_format}"
    else:
        if not output_name.endswith(f".{export_format}"):
            output_name = f"{output_name}.{export_format}"

    output_path = adapter.exports_dir / output_name

    # Check Blender availability
    if is_blender_available():
        ok, msg = generate_with_blender(clean_shape, export_format, output_path, size)
        if ok:
            return True, output_path, f"[Blender Engine] {msg}"
        # Fallback if Blender command errored
        ok_fb, msg_fb = generate_with_fallback(clean_shape, export_format, output_path, size)
        return ok_fb, output_path, f"[Fallback 3D Engine] {msg_fb} (Blender notice: {msg})"
    else:
        ok_fb, msg_fb = generate_with_fallback(clean_shape, export_format, output_path, size)
        return ok_fb, output_path, f"[Portable 3D Engine] {msg_fb} (Note: Install Blender for smoothed rendering)"
