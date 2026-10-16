"""Headless Blender Script for ADAM 3D Object Generation.
Invoked via: blender -b -P scripts/blender_gen.py -- --shape <shape> --format <obj/stl> --output <path> --size <size>
"""

import sys
import argparse

def run_blender_gen():
    try:
        import bpy
    except ImportError:
        print("ERROR: This script must be run within Blender (blender -b -P ...)", file=sys.stderr)
        sys.exit(1)

    # Parse arguments after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Blender Parametric Generator")
    parser.add_argument("--shape", type=str, default="cube", help="Shape to generate")
    parser.add_argument("--format", type=str, default="stl", help="Export format: obj or stl")
    parser.add_argument("--output", type=str, required=True, help="Destination file path")
    parser.add_argument("--size", type=float, default=1.0, help="Scale / size")
    args = parser.parse_args(argv)

    # Clean the default scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    shape = args.shape.lower().replace("-", "_").replace(" ", "_")
    size = args.size

    # Build geometry
    if shape == "cube":
        bpy.ops.mesh.primitive_cube_add(size=size, location=(0, 0, 0))
    elif shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=size / 2.0, location=(0, 0, 0))
    elif shape == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(radius=size / 2.0, depth=size, location=(0, 0, 0))
    elif shape == "torus":
        bpy.ops.mesh.primitive_torus_add(
            major_radius=0.6 * size,
            minor_radius=0.2 * size,
            location=(0, 0, 0)
        )
    elif shape == "pyramid":
        bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=size / 1.414, depth=size, location=(0, 0, 0))
    elif shape == "scifi_drone":
        # Central core
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.4 * size, location=(0, 0, 0))
        core = bpy.context.active_object
        # 4 Sensor Thruster Pods
        arm_dist = 0.65 * size
        import math
        for deg in [45, 135, 225, 315]:
            rad = math.radians(deg)
            x = arm_dist * math.cos(rad)
            y = arm_dist * math.sin(rad)
            bpy.ops.mesh.primitive_cube_add(size=0.25 * size, location=(x, y, 0))
        # Select all and join
        bpy.ops.object.select_all(action='SELECT')
        bpy.context.view_layer.objects.active = core
        bpy.ops.object.join()
    elif shape == "scifi_pedestal":
        # Base
        bpy.ops.mesh.primitive_cylinder_add(radius=0.8 * size, depth=0.15 * size, location=(0, 0, -0.4 * size))
        base = bpy.context.active_object
        # Pillar
        bpy.ops.mesh.primitive_cylinder_add(radius=0.35 * size, depth=0.5 * size, location=(0, 0, 0))
        # Top Plate
        bpy.ops.mesh.primitive_cylinder_add(radius=0.6 * size, depth=0.1 * size, location=(0, 0, 0.35 * size))
        bpy.ops.object.select_all(action='SELECT')
        bpy.context.view_layer.objects.active = base
        bpy.ops.object.join()
    else:
        # Default primitive cube
        bpy.ops.mesh.primitive_cube_add(size=size, location=(0, 0, 0))

    # Add subtle bevel modifier for industrial finish
    active_obj = bpy.context.active_object
    if active_obj:
        mod = active_obj.modifiers.new(name="Bevel", type='BEVEL')
        mod.width = 0.02 * size
        mod.segments = 2
        # Apply modifier
        bpy.ops.object.modifier_apply(modifier=mod.name)

    # Export to requested format
    fmt = args.format.lower().replace(".", "")
    output_path = args.output

    if fmt == "stl":
        # Check Blender 4.x/3.x operator compatibility
        if hasattr(bpy.ops.wm, "stl_export"):
            bpy.ops.wm.stl_export(filepath=output_path)
        elif hasattr(bpy.ops.export_mesh, "stl"):
            bpy.ops.export_mesh.stl(filepath=output_path)
        else:
            print("ERROR: No STL exporter operator found in bpy.ops", file=sys.stderr)
            sys.exit(2)
    elif fmt == "obj":
        if hasattr(bpy.ops.wm, "obj_export"):
            bpy.ops.wm.obj_export(filepath=output_path)
        elif hasattr(bpy.ops.export_scene, "obj"):
            bpy.ops.export_scene.obj(filepath=output_path)
        else:
            print("ERROR: No OBJ exporter operator found in bpy.ops", file=sys.stderr)
            sys.exit(2)
    else:
        print(f"ERROR: Unsupported format '{fmt}'", file=sys.stderr)
        sys.exit(3)

    print(f"SUCCESS: Generated {shape} to {output_path}")

if __name__ == "__main__":
    run_blender_gen()
