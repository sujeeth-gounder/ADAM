"""Interactive 360-degree 3D Viewport & Studio for ADAM AI 3.0.
Provides an embedded, hardware-independent Tkinter 3D canvas supporting:
- Full 360° mouse drag rotation (azimuth & elevation).
- Mouse wheel and button zoom in / zoom out.
- Right-click / Shift-drag panning.
- Double-click / Button camera reset.
- Flat shading with Lambertian directional lighting and Painter's depth sorting.
- Wireframe and Shaded rendering modes.
- Toggle between Interactive 3D Mesh and High-Res Blender Photo Render.
- Direct export and "Open in Blender" integration.
"""

import math
import os
import shutil
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional, Tuple

from app.config import THEME
from app.three_d import GeneratedModelAsset, MeshData, load_obj_mesh


class Interactive3DViewport(tk.Frame):
    """Interactive 360-degree 3D Studio & Viewport for ADAM."""

    def __init__(self, master, on_export: Optional[callable] = None, on_regenerate: Optional[callable] = None, **kwargs):
        super().__init__(master, bg=THEME["bg_card"], **kwargs)

        self.on_export_callback = on_export
        self.on_regenerate_callback = on_regenerate

        self.current_asset: Optional[GeneratedModelAsset] = None
        self.mesh: Optional[MeshData] = None

        # Camera & Transform state
        self.yaw = 35.0         # Azimuth in degrees
        self.pitch = 25.0       # Elevation in degrees
        self.zoom = 2.4         # Camera distance factor
        self.pan_x = 0.0
        self.pan_y = 0.0

        # Mouse drag state
        self._last_mouse_x = 0
        self._last_mouse_y = 0

        # Render settings
        self.shading_mode = "shaded"   # 'shaded', 'wireframe', 'both'
        self.lighting_mode = "cyber"   # 'cyber', 'studio', 'warm'
        self.view_mode = "3d"          # '3d' or 'photo'
        self._photo_image = None

        self._build_ui()

    def _build_ui(self):
        # 1. Top Studio Header Toolbar
        self.header_frame = tk.Frame(self, bg="#0B132B", padx=8, pady=6, highlightthickness=1, highlightbackground=THEME["border_glass"])
        self.header_frame.pack(side=tk.TOP, fill=tk.X)

        self.lbl_title = tk.Label(
            self.header_frame,
            text="📦 3D STUDIO — Interactive Viewport",
            font=(THEME["font_main"], 10, "bold"),
            fg=THEME["accent_cyan"],
            bg="#0B132B",
        )
        self.lbl_title.pack(side=tk.LEFT, padx=(4, 12))

        self.lbl_poly_count = tk.Label(
            self.header_frame,
            text="No model loaded",
            font=(THEME["font_mono"], 8),
            fg="#94A3B8",
            bg="#0B132B",
        )
        self.lbl_poly_count.pack(side=tk.LEFT)

        # Right-aligned view controls
        controls_frame = tk.Frame(self.header_frame, bg="#0B132B")
        controls_frame.pack(side=tk.RIGHT)

        self.btn_toggle_view = tk.Button(
            controls_frame,
            text="🖼 Render Photo",
            font=(THEME["font_main"], 8, "bold"),
            bg="#1E293B",
            fg=THEME["accent_cyan"],
            bd=0,
            padx=8,
            pady=3,
            cursor="hand2",
            command=self._toggle_view_mode,
        )
        self.btn_toggle_view.pack(side=tk.LEFT, padx=3)

        self.btn_reset = tk.Button(
            controls_frame,
            text="🔄 Reset View",
            font=(THEME["font_main"], 8),
            bg="#1E293B",
            fg="#CBD5E1",
            bd=0,
            padx=8,
            pady=3,
            cursor="hand2",
            command=self.reset_view,
        )
        self.btn_reset.pack(side=tk.LEFT, padx=3)

        self.btn_shading = tk.Button(
            controls_frame,
            text="🎨 Shaded",
            font=(THEME["font_main"], 8),
            bg="#1E293B",
            fg="#CBD5E1",
            bd=0,
            padx=8,
            pady=3,
            cursor="hand2",
            command=self._cycle_shading,
        )
        self.btn_shading.pack(side=tk.LEFT, padx=3)

        # Zoom buttons
        tk.Button(
            controls_frame,
            text="➕",
            font=(THEME["font_mono"], 8, "bold"),
            bg="#1E293B",
            fg="#CBD5E1",
            bd=0,
            padx=6,
            pady=3,
            cursor="hand2",
            command=self.zoom_in,
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            controls_frame,
            text="➖",
            font=(THEME["font_mono"], 8, "bold"),
            bg="#1E293B",
            fg="#CBD5E1",
            bd=0,
            padx=6,
            pady=3,
            cursor="hand2",
            command=self.zoom_out,
        ).pack(side=tk.LEFT, padx=2)

        # 2. Main 3D Canvas
        self.canvas_container = tk.Frame(self, bg="#020617")
        self.canvas_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            self.canvas_container,
            bg="#020617",
            highlightthickness=0,
            cursor="fleur",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind mouse interactions
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_down)
        self.canvas.bind("<B3-Motion>", self._on_pan_drag)
        self.canvas.bind("<Shift-B1-Motion>", self._on_pan_drag)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)  # Windows / macOS
        self.canvas.bind("<Button-4>", lambda e: self.zoom_in())   # Linux scroll up
        self.canvas.bind("<Button-5>", lambda e: self.zoom_out())  # Linux scroll down
        self.canvas.bind("<Double-Button-1>", lambda e: self.reset_view())

        # 3. Bottom Action & Export Bar
        self.footer_frame = tk.Frame(self, bg="#0B132B", padx=8, pady=6, highlightthickness=1, highlightbackground=THEME["border_glass"])
        self.footer_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_location = tk.Label(
            self.footer_frame,
            text="Ready to create 3D assets with Blender automation.",
            font=(THEME["font_mono"], 8),
            fg="#94A3B8",
            bg="#0B132B",
            anchor="w",
        )
        self.lbl_location.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.btn_open_blender = tk.Button(
            self.footer_frame,
            text="🚀 Open in Blender",
            font=(THEME["font_main"], 8, "bold"),
            bg="#EA580C",
            fg="#FFFFFF",
            bd=0,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._open_in_blender,
        )
        self.btn_open_blender.pack(side=tk.RIGHT, padx=4)

        self.btn_export = tk.Button(
            self.footer_frame,
            text="💾 Save / Export",
            font=(THEME["font_main"], 8, "bold"),
            bg="#0284C7",
            fg="#FFFFFF",
            bd=0,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._export_dialog,
        )
        self.btn_export.pack(side=tk.RIGHT, padx=4)

    def load_asset(self, asset: GeneratedModelAsset):
        """Load and display a generated model asset in both interactive 3D and thumbnail views."""
        self.current_asset = asset
        self.lbl_title.config(text=f"📦 {asset.shape.replace('_', ' ').title()} ({asset.color.title()})")
        self.lbl_poly_count.config(text=f"{asset.vertex_count:,} verts | {asset.polygon_count:,} polys | {asset.style.title()}")
        self.lbl_location.config(text=f"Asset: {asset.glb_path.name} ({asset.directory})")

        # Load OBJ mesh for interactive viewport
        if asset.obj_path.exists():
            self.mesh = load_obj_mesh(asset.obj_path)
        else:
            self.mesh = None

        self.reset_view()
        self.redraw()

    def reset_view(self):
        """Reset camera orientation, zoom factor, and pan offsets."""
        self.yaw = 35.0
        self.pitch = 25.0
        self.zoom = 2.4
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.redraw()

    def zoom_in(self):
        self.zoom = max(0.5, self.zoom * 0.85)
        self.redraw()

    def zoom_out(self):
        self.zoom = min(8.0, self.zoom * 1.18)
        self.redraw()

    def rotate_relative(self, delta_yaw: float = 15.0, delta_pitch: float = 0.0):
        """Rotate model programmatically (used by voice and chat commands)."""
        self.yaw = (self.yaw + delta_yaw) % 360.0
        self.pitch = max(-89.0, min(89.0, self.pitch + delta_pitch))
        self.redraw()

    def _on_mouse_down(self, event):
        self._last_mouse_x = event.x
        self._last_mouse_y = event.y

    def _on_mouse_drag(self, event):
        dx = event.x - self._last_mouse_x
        dy = event.y - self._last_mouse_y
        self._last_mouse_x = event.x
        self._last_mouse_y = event.y

        self.yaw = (self.yaw + dx * 0.8) % 360.0
        self.pitch = max(-88.0, min(88.0, self.pitch - dy * 0.8))
        self.redraw()

    def _on_pan_down(self, event):
        self._last_mouse_x = event.x
        self._last_mouse_y = event.y

    def _on_pan_drag(self, event):
        dx = event.x - self._last_mouse_x
        dy = event.y - self._last_mouse_y
        self._last_mouse_x = event.x
        self._last_mouse_y = event.y

        self.pan_x += dx * 0.005 * self.zoom
        self.pan_y -= dy * 0.005 * self.zoom
        self.redraw()

    def _on_mouse_wheel(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def _cycle_shading(self):
        modes = ["shaded", "wireframe", "both"]
        cur_idx = modes.index(self.shading_mode)
        self.shading_mode = modes[(cur_idx + 1) % len(modes)]
        btn_labels = {"shaded": "🎨 Shaded", "wireframe": "🕸 Wireframe", "both": "💎 Solid+Edges"}
        self.btn_shading.config(text=btn_labels[self.shading_mode])
        self.redraw()

    def _toggle_view_mode(self):
        if self.view_mode == "3d":
            if self.current_asset and self.current_asset.thumb_path and self.current_asset.thumb_path.exists():
                self.view_mode = "photo"
                self.btn_toggle_view.config(text="🔄 Interactive 3D")
            else:
                messagebox.showinfo("Render Photo", "Photo thumbnail is not available for this model yet.")
                return
        else:
            self.view_mode = "3d"
            self.btn_toggle_view.config(text="🖼 Render Photo")
        self.redraw()

    def redraw(self):
        """Redraw the 3D scene on the Tkinter canvas."""
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return

        # Photo render mode
        if self.view_mode == "photo" and self.current_asset and self.current_asset.thumb_path:
            try:
                import cv2
                img = cv2.imread(str(self.current_asset.thumb_path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ih, iw = img_rgb.shape[:2]
                    scale = min((w - 20) / iw, (h - 20) / ih, 1.5)
                    rw, rh = int(iw * scale), int(ih * scale)
                    resized = cv2.resize(img_rgb, (rw, rh), interpolation=cv2.INTER_AREA)
                    ppm = f"P6\n{rw} {rh}\n255\n".encode("ascii") + resized.tobytes()
                    self._photo_image = tk.PhotoImage(data=ppm.decode("latin-1"), format="PPM")
                    self.canvas.create_image(w // 2, h // 2, image=self._photo_image)
                    self.canvas.create_text(
                        12, 16,
                        text="[Blender Studio Render]",
                        font=(THEME["font_mono"], 9, "bold"),
                        fill=THEME["accent_cyan"],
                        anchor="nw"
                    )
                    return
            except Exception as e:
                pass

        # 3D interactive mode
        if not self.mesh or not self.mesh.vertices:
            self.canvas.create_text(
                w // 2, h // 2,
                text="No 3D Model Loaded\nSay 'Create a water bottle' or 'Make a chair' to generate geometry.",
                font=(THEME["font_main"], 11),
                fill="#475569",
                justify=tk.CENTER
            )
            return

        # Draw grid background
        self._draw_grid(w, h)

        # 3D Transformation Math
        rad_yaw = math.radians(self.yaw)
        rad_pitch = math.radians(self.pitch)
        cos_y, sin_y = math.cos(rad_yaw), math.sin(rad_yaw)
        cos_p, sin_p = math.cos(rad_pitch), math.sin(rad_pitch)

        # Light vector (directional key light from upper right)
        lx, ly, lz = 0.577, -0.577, 0.577

        # Project vertices
        proj_verts = []
        fov = min(w, h) * 0.95

        for v in self.mesh.vertices:
            x, y, z = v
            # Yaw
            x1 = x * cos_y - y * sin_y
            y1 = x * sin_y + y * cos_y
            z1 = z
            # Pitch
            x2 = x1
            y2 = y1 * cos_p - z1 * sin_p
            z2 = y1 * sin_p + z1 * cos_p

            z_cam = z2 + self.zoom
            if z_cam <= 0.05:
                z_cam = 0.05
            factor = 1.0 / z_cam
            sx = w / 2.0 + (x2 + self.pan_x) * factor * fov
            sy = h / 2.0 - (y2 + self.pan_y) * factor * fov
            proj_verts.append((sx, sy, z_cam))

        # Face depth sorting (Painter's Algorithm)
        face_render_list = []
        for f in self.mesh.faces:
            z_avg = (proj_verts[f[0]][2] + proj_verts[f[1]][2] + proj_verts[f[2]][2]) / 3.0
            face_render_list.append((z_avg, f))

        # Sort from furthest to nearest
        face_render_list.sort(key=lambda item: item[0], reverse=True)

        # Palette selection
        base_color = self._get_palette_color()

        # Render faces
        for _, f in face_render_list:
            p1, p2, p3 = proj_verts[f[0]], proj_verts[f[1]], proj_verts[f[2]]
            pts = [p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]]

            # Compute face normal in object space
            v1, v2, v3 = self.mesh.vertices[f[0]], self.mesh.vertices[f[1]], self.mesh.vertices[f[2]]
            ax, ay, az = v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]
            bx, by, bz = v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]
            nx = ay * bz - az * by
            ny = az * bx - ax * bz
            nz = ax * by - ay * bx
            n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
            if n_len > 1e-9:
                nx, ny, nz = nx / n_len, ny / n_len, nz / n_len
            else:
                nx, ny, nz = 0.0, 0.0, 1.0

            # Transform normal to view space via yaw & pitch
            nx1 = nx * cos_y - ny * sin_y
            ny1 = nx * sin_y + ny * cos_y
            nz1 = nz
            nx2 = nx1
            ny2 = ny1 * cos_p - nz1 * sin_p
            nz2 = ny1 * sin_p + nz1 * cos_p

            dot = abs(nx2 * lx + ny2 * ly + nz2 * lz)
            # Ambient + diffuse lighting
            intensity = 0.28 + 0.72 * dot
            face_color = self._shade_color(base_color, intensity)

            if self.shading_mode == "shaded":
                self.canvas.create_polygon(pts, fill=face_color, outline=face_color)
            elif self.shading_mode == "wireframe":
                self.canvas.create_polygon(pts, fill="", outline="#00E5FF", width=1)
            elif self.shading_mode == "both":
                self.canvas.create_polygon(pts, fill=face_color, outline="#0891B2", width=1)

        # Viewport HUD Info
        self.canvas.create_text(
            12, 14,
            text=f"Yaw: {int(self.yaw)}°  Pitch: {int(self.pitch)}°  Zoom: {self.zoom:.1f}x",
            font=(THEME["font_mono"], 8),
            fill="#64748B",
            anchor="nw"
        )
        self.canvas.create_text(
            12, h - 14,
            text="[Drag: 360° Rotate]  [Scroll: Zoom]  [Right-Drag: Pan]  [Double-Click: Reset]",
            font=(THEME["font_mono"], 8),
            fill="#475569",
            anchor="sw"
        )

    def _draw_grid(self, w, h):
        """Subtle 3D ground grid."""
        cx, cy = w // 2, h // 2 + 60
        for r in range(1, 5):
            rad = r * 35
            self.canvas.create_oval(cx - rad, cy - rad // 2, cx + rad, cy + rad // 2, outline="#0F172A", width=1)

    def _get_palette_color(self) -> Tuple[int, int, int]:
        color_name = self.current_asset.color if self.current_asset else "cyan"
        palettes = {
            "red": (210, 45, 45),
            "blue": (30, 120, 230),
            "green": (35, 185, 75),
            "yellow": (230, 195, 25),
            "orange": (230, 115, 25),
            "cyan": (15, 195, 215),
            "purple": (155, 45, 215),
            "silver": (180, 185, 195),
            "gray": (120, 128, 135),
            "gold": (220, 175, 40),
            "white": (225, 225, 230),
            "black": (45, 50, 60),
        }
        return palettes.get(color_name.lower().strip(), (15, 195, 215))

    def _shade_color(self, base_rgb: Tuple[int, int, int], intensity: float) -> str:
        r = int(min(255, max(0, base_rgb[0] * intensity)))
        g = int(min(255, max(0, base_rgb[1] * intensity)))
        b = int(min(255, max(0, base_rgb[2] * intensity)))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _open_in_blender(self):
        """Open the generated .blend project in Blender GUI."""
        if not self.current_asset or not self.current_asset.blend_path.exists():
            messagebox.showwarning("Open in Blender", "No .blend project file is currently available.")
            return

        blender_bin = shutil.which("blender")
        if not blender_bin:
            messagebox.showerror("Blender Missing", "Blender executable was not found in PATH.")
            return

        try:
            subprocess.Popen([blender_bin, str(self.current_asset.blend_path.resolve())])
        except Exception as e:
            messagebox.showerror("Launch Error", f"Could not launch Blender: {e}")

    def _export_dialog(self):
        """Save/Export model asset to user-selected location."""
        if not self.current_asset:
            messagebox.showinfo("Export", "No model is currently loaded to export.")
            return

        dest = filedialog.askdirectory(title="Select Destination Folder for 3D Asset Export")
        if not dest:
            return

        dest_dir = Path(dest)
        copied = []
        for p in [self.current_asset.blend_path, self.current_asset.glb_path, self.current_asset.stl_path, self.current_asset.obj_path, self.current_asset.thumb_path]:
            if p and p.exists():
                try:
                    target = dest_dir / p.name
                    shutil.copy2(p, target)
                    copied.append(target.name)
                except Exception:
                    pass

        messagebox.showinfo("Export Complete", f"Exported {len(copied)} asset files to:\n{dest_dir}\n\nFiles: {', '.join(copied)}")
