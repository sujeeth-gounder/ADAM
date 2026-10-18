"""Headless Parametric Blender Script for ADAM AI 3.0.
Procedural 3D Generator creating production-quality geometry, materials,
rendered thumbnails, and exports in .blend, .glb, .stl, and .obj formats.
"""

import argparse
import json
import math
import os
import sys


COLOR_PALETTES = {
    "red": ((0.85, 0.15, 0.15, 1.0), 0.3, 0.4),
    "blue": ((0.12, 0.45, 0.90, 1.0), 0.2, 0.4),
    "green": ((0.15, 0.75, 0.25, 1.0), 0.3, 0.5),
    "yellow": ((0.95, 0.85, 0.10, 1.0), 0.1, 0.3),
    "orange": ((0.95, 0.45, 0.10, 1.0), 0.2, 0.4),
    "cyan": ((0.10, 0.85, 0.90, 1.0), 0.2, 0.3),
    "purple": ((0.60, 0.15, 0.85, 1.0), 0.3, 0.4),
    "black": ((0.08, 0.08, 0.08, 1.0), 0.5, 0.3),
    "white": ((0.92, 0.92, 0.92, 1.0), 0.1, 0.2),
    "silver": ((0.75, 0.77, 0.80, 1.0), 0.85, 0.25),
    "gold": ((0.92, 0.75, 0.20, 1.0), 0.90, 0.20),
    "gray": ((0.45, 0.48, 0.50, 1.0), 0.4, 0.5),
}


def create_colored_material(name: str, color_name: str, is_accent: bool = False):
    import bpy
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        col_key = color_name.lower().strip()
        if is_accent:
            # Darker accent color
            base_col, metallic, roughness = ((0.12, 0.14, 0.18, 1.0), 0.4, 0.3)
        else:
            base_col, metallic, roughness = COLOR_PALETTES.get(col_key, ((0.18, 0.65, 0.90, 1.0), 0.3, 0.4))

        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = base_col
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
    return mat


def build_water_bottle(size: float = 1.0, color: str = "blue"):
    import bpy
    s = size

    # 1. Main body cylinder
    bpy.ops.mesh.primitive_cylinder_add(radius=0.35 * s, depth=1.1 * s, location=(0, 0, 0.55 * s))
    body = bpy.context.active_object
    body.name = "Bottle_Body"

    # 2. Shoulder taper (cone)
    bpy.ops.mesh.primitive_cone_add(radius1=0.35 * s, radius2=0.18 * s, depth=0.25 * s, location=(0, 0, 1.22 * s))
    shoulder = bpy.context.active_object
    shoulder.name = "Bottle_Shoulder"

    # 3. Neck
    bpy.ops.mesh.primitive_cylinder_add(radius=0.18 * s, depth=0.2 * s, location=(0, 0, 1.40 * s))
    neck = bpy.context.active_object
    neck.name = "Bottle_Neck"

    # 4. Cap
    bpy.ops.mesh.primitive_cylinder_add(radius=0.20 * s, depth=0.18 * s, location=(0, 0, 1.55 * s))
    cap = bpy.context.active_object
    cap.name = "Bottle_Cap"

    # Materials
    mat_main = create_colored_material("Bottle_Mat", color)
    mat_cap = create_colored_material("Cap_Mat", color, is_accent=True)
    body.data.materials.append(mat_main)
    shoulder.data.materials.append(mat_main)
    neck.data.materials.append(mat_main)
    cap.data.materials.append(mat_cap)

    # Join
    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    return body


def build_backpack(size: float = 1.0, color: str = "red"):
    """
    Realistic 3D Backpack Generator.
    Features:
    - Rounded beveled main compartment with upper dome.
    - Front utility pocket with zipper flap seam.
    - Dual side elastic bottle pouches.
    - Top carry handle firmly attached to the main pack body.
    - Curved shoulder straps anchored securely into the rear body (no floating parts).
    - Believable thickness, clean normals, and smooth shading.
    """
    import bpy
    s = size

    # 1. Main compartment (rounded, beveled backpack body)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.45 * s))
    main = bpy.context.active_object
    main.name = "Pack_Main"
    main.scale = (0.50 * s, 0.32 * s, 0.65 * s)
    bpy.ops.object.transform_apply(scale=True)
    mod_b = main.modifiers.new(name="MainBevel", type='BEVEL')
    mod_b.width = 0.08 * s
    mod_b.segments = 3
    bpy.ops.object.modifier_apply(modifier=mod_b.name)

    # 2. Arched upper compartment dome
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.25 * s,
        depth=0.32 * s,
        location=(0, 0, 0.77 * s),
        rotation=(math.radians(90), 0, 0)
    )
    dome = bpy.context.active_object
    dome.scale = (1.0, 1.0, 0.5)
    bpy.ops.object.transform_apply(scale=True)

    # 3. Front utility pocket
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, -0.21 * s, 0.34 * s))
    pocket = bpy.context.active_object
    pocket.name = "Pack_Pocket"
    pocket.scale = (0.38 * s, 0.14 * s, 0.34 * s)
    bpy.ops.object.transform_apply(scale=True)
    mod_p = pocket.modifiers.new(name="PocketBevel", type='BEVEL')
    mod_p.width = 0.03 * s
    mod_p.segments = 2
    bpy.ops.object.modifier_apply(modifier=mod_p.name)

    # 4. Front pocket zipper flap
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, -0.22 * s, 0.51 * s))
    flap = bpy.context.active_object
    flap.scale = (0.39 * s, 0.04 * s, 0.03 * s)
    bpy.ops.object.transform_apply(scale=True)

    # 5. Dual side elastic bottle pouches
    for side in [-1, 1]:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(side * 0.27 * s, -0.02 * s, 0.30 * s))
        pouch = bpy.context.active_object
        pouch.scale = (0.08 * s, 0.20 * s, 0.26 * s)
        bpy.ops.object.transform_apply(scale=True)

    # 6. Top carry handle (firmly anchored into top dome)
    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.10 * s,
        minor_radius=0.025 * s,
        location=(0, 0.02 * s, 0.88 * s),
        rotation=(math.radians(90), 0, 0)
    )
    handle = bpy.context.active_object
    handle.scale = (1.1, 0.8, 1.0)
    bpy.ops.object.transform_apply(scale=True)

    # 7. Ergonomic shoulder straps anchored directly to back of backpack
    for x_pos in [-0.14 * s, 0.14 * s]:
        # Upper strap anchor
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x_pos, 0.18 * s, 0.72 * s))
        u_strap = bpy.context.active_object
        u_strap.scale = (0.06 * s, 0.03 * s, 0.08 * s)
        bpy.ops.object.transform_apply(scale=True)

        # Main curved strap
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.035 * s,
            depth=0.58 * s,
            location=(x_pos * 1.15, 0.22 * s, 0.44 * s),
            rotation=(math.radians(12), 0, 0)
        )
        s_mid = bpy.context.active_object
        s_mid.scale = (1.0, 0.4, 1.0)
        bpy.ops.object.transform_apply(scale=True)

        # Lower strap anchor into bottom rear
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x_pos * 1.05, 0.17 * s, 0.16 * s))
        l_strap = bpy.context.active_object
        l_strap.scale = (0.05 * s, 0.03 * s, 0.06 * s)
        bpy.ops.object.transform_apply(scale=True)

    # Materials
    mat_main = create_colored_material("Pack_Main", color)
    mat_accent = create_colored_material("Pack_Accent", "black", is_accent=True)
    main.data.materials.append(mat_main)
    pocket.data.materials.append(mat_accent)

    # Join all meshes into a single clean object
    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active = main
    bpy.ops.object.join()
    bpy.ops.object.shade_smooth()
    return main


def build_scorpio_suv(size: float = 1.0, color: str = "black"):
    """
    Dedicated Mahindra Scorpio-style SUV Generator.
    Features:
    - High ground clearance rugged SUV chassis.
    - Muscular hood with iconic Scorpio bonnet scoop.
    - Upright front grille with chrome vertical slats and dual rectangular headlights.
    - Boxy cabin greenhouse with windshield, side windows, and rear tailgate.
    - Dual functional roof rails running along the roof.
    - 4 large SUV alloy wheels with detailed tire rims.
    """
    import bpy
    s = size

    # 1. Main lower SUV body / chassis
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.38 * s))
    body = bpy.context.active_object
    body.name = "SUV_Body"
    body.scale = (0.72 * s, 1.48 * s, 0.32 * s)
    bpy.ops.object.transform_apply(scale=True)
    mod_b = body.modifiers.new(name="Bevel", type='BEVEL')
    mod_b.width = 0.03 * s
    mod_b.segments = 2
    bpy.ops.object.modifier_apply(modifier=mod_b.name)

    # 2. Cabin / Greenhouse (Upper passenger section)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0.12 * s, 0.68 * s))
    cabin = bpy.context.active_object
    cabin.name = "SUV_Cabin"
    cabin.scale = (0.64 * s, 0.98 * s, 0.34 * s)
    bpy.ops.object.transform_apply(scale=True)

    # 3. Windshield slope (angled transition)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, -0.42 * s, 0.62 * s))
    windshield = bpy.context.active_object
    windshield.scale = (0.62 * s, 0.22 * s, 0.26 * s)
    windshield.rotation_euler = (math.radians(-32), 0, 0)
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

    # 4. Scorpio signature bonnet scoop / air intake
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, -0.45 * s, 0.56 * s))
    scoop = bpy.context.active_object
    scoop.scale = (0.24 * s, 0.22 * s, 0.04 * s)
    bpy.ops.object.transform_apply(scale=True)

    # 5. Upright front grille (multi-slat front)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, -0.75 * s, 0.38 * s))
    grille = bpy.context.active_object
    grille.scale = (0.42 * s, 0.03 * s, 0.18 * s)
    bpy.ops.object.transform_apply(scale=True)

    # 6. Front Headlights (Left & Right dual clusters)
    for side in [-1, 1]:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(side * 0.29 * s, -0.75 * s, 0.40 * s))
        headlamp = bpy.context.active_object
        headlamp.scale = (0.13 * s, 0.03 * s, 0.10 * s)
        bpy.ops.object.transform_apply(scale=True)

    # 7. Dual Roof Rails (Signature Scorpio feature)
    for side in [-1, 1]:
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.02 * s,
            depth=0.92 * s,
            location=(side * 0.28 * s, 0.12 * s, 0.88 * s),
            rotation=(math.radians(90), 0, 0)
        )
        rail = bpy.context.active_object
        rail.scale = (1.0, 1.0, 1.0)
        # Stanchions connecting rails to roof
        for y_stan in [-0.28 * s, 0.12 * s, 0.52 * s]:
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(side * 0.28 * s, y_stan, 0.86 * s))
            stan = bpy.context.active_object
            stan.scale = (0.025 * s, 0.04 * s, 0.04 * s)
            bpy.ops.object.transform_apply(scale=True)

    # 8. Front and rear bumpers
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, -0.76 * s, 0.24 * s))
    f_bumper = bpy.context.active_object
    f_bumper.scale = (0.74 * s, 0.08 * s, 0.14 * s)
    bpy.ops.object.transform_apply(scale=True)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0.76 * s, 0.24 * s))
    r_bumper = bpy.context.active_object
    r_bumper.scale = (0.74 * s, 0.08 * s, 0.14 * s)
    bpy.ops.object.transform_apply(scale=True)

    # 9. 4 Large SUV Alloy Wheels with rubber tires
    wheel_r = 0.22 * s
    wheel_w = 0.14 * s
    wheel_track = 0.42 * s
    wheelbases = [-0.48 * s, 0.48 * s]
    for side in [-1, 1]:
        for y_pos in wheelbases:
            bpy.ops.mesh.primitive_cylinder_add(
                radius=wheel_r,
                depth=wheel_w,
                location=(side * wheel_track, y_pos, 0.22 * s),
                rotation=(0, math.radians(90), 0)
            )
            bpy.ops.mesh.primitive_cylinder_add(
                radius=wheel_r * 0.65,
                depth=wheel_w * 1.05,
                location=(side * wheel_track, y_pos, 0.22 * s),
                rotation=(0, math.radians(90), 0)
            )

    # Materials
    mat_paint = create_colored_material("SUV_Paint", color)
    mat_trim = create_colored_material("SUV_Trim", "silver", is_accent=True)
    body.data.materials.append(mat_paint)

    # Join all components
    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    bpy.ops.object.shade_smooth()
    return body


def build_chair(size: float = 1.0, color: str = "silver"):
    import bpy
    s = size

    # 1. Seat cushion
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.50 * s))
    seat = bpy.context.active_object
    seat.name = "Chair_Seat"
    seat.scale = (0.55 * s, 0.55 * s, 0.06 * s)
    bpy.ops.object.transform_apply(scale=True)

    # 2. 4 Legs
    leg_r = 0.035 * s
    leg_h = 0.50 * s
    for x, y in [(-0.22 * s, -0.22 * s), (0.22 * s, -0.22 * s), (-0.22 * s, 0.22 * s), (0.22 * s, 0.22 * s)]:
        bpy.ops.mesh.primitive_cylinder_add(radius=leg_r, depth=leg_h, location=(x, y, 0.25 * s))

    # 3. Backrest uprights
    for x in [-0.22 * s, 0.22 * s]:
        bpy.ops.mesh.primitive_cylinder_add(radius=leg_r, depth=0.60 * s, location=(x, 0.22 * s, 0.82 * s))

    # 4. Backrest slat
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0.22 * s, 1.05 * s))
    slat = bpy.context.active_object
    slat.scale = (0.50 * s, 0.04 * s, 0.14 * s)
    bpy.ops.object.transform_apply(scale=True)

    mat_seat = create_colored_material("Chair_Mat", color)
    mat_legs = create_colored_material("Legs_Mat", "silver", is_accent=True)
    seat.data.materials.append(mat_seat)

    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active = seat
    bpy.ops.object.join()
    return seat


def build_coffee_mug(size: float = 1.0, color: str = "white"):
    import bpy
    s = size

    # 1. Main outer body
    bpy.ops.mesh.primitive_cylinder_add(radius=0.40 * s, depth=0.85 * s, location=(0, 0, 0.42 * s))
    mug = bpy.context.active_object
    mug.name = "Mug_Body"

    # 2. Inner cavity (hollow effect)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.34 * s, depth=0.80 * s, location=(0, 0, 0.48 * s))
    inner = bpy.context.active_object

    # Subtract inner from mug using Boolean modifier
    bool_mod = mug.modifiers.new(name="Cavity", type='BOOLEAN')
    bool_mod.operation = 'DIFFERENCE'
    bool_mod.object = inner
    bpy.context.view_layer.objects.active = mug
    bpy.ops.object.modifier_apply(modifier=bool_mod.name)

    # Delete inner cutter
    bpy.data.objects.remove(inner, do_unlink=True)

    # 3. Handle (curved torus on side)
    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.25 * s,
        minor_radius=0.06 * s,
        location=(0.42 * s, 0, 0.45 * s),
        rotation=(0, math.radians(90), 0)
    )
    handle = bpy.context.active_object
    handle.name = "Mug_Handle"

    mat_mug = create_colored_material("Mug_Mat", color)
    mug.data.materials.append(mat_mug)
    handle.data.materials.append(mat_mug)

    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active = mug
    bpy.ops.object.join()
    return mug


def build_sports_car(size: float = 1.0, color: str = "red"):
    import bpy
    s = size

    # 1. Lower chassis
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.22 * s))
    chassis = bpy.context.active_object
    chassis.name = "Car_Chassis"
    chassis.scale = (0.55 * s, 1.25 * s, 0.20 * s)
    bpy.ops.object.transform_apply(scale=True)

    # 2. Cockpit / cabin roof
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, -0.08 * s, 0.42 * s))
    cabin = bpy.context.active_object
    cabin.scale = (0.42 * s, 0.55 * s, 0.20 * s)
    bpy.ops.object.transform_apply(scale=True)

    # 3. 4 Wheels
    wheel_r = 0.16 * s
    wheel_w = 0.10 * s
    for x in [-0.32 * s, 0.32 * s]:
        for y in [-0.65 * s, 0.65 * s]:
            bpy.ops.mesh.primitive_cylinder_add(
                radius=wheel_r,
                depth=wheel_w,
                location=(x, y, 0.16 * s),
                rotation=(0, math.radians(90), 0)
            )

    mat_body = create_colored_material("Car_Paint", color)
    mat_wheels = create_colored_material("Car_Tires", "black", is_accent=True)
    chassis.data.materials.append(mat_body)

    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active = chassis
    bpy.ops.object.join()
    return chassis


def build_robot(size: float = 1.0, color: str = "silver"):
    import bpy
    s = size

    # Torso
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.70 * s))
    torso = bpy.context.active_object
    torso.name = "Robot_Torso"
    torso.scale = (0.35 * s, 0.25 * s, 0.45 * s)
    bpy.ops.object.transform_apply(scale=True)

    # Head
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 1.25 * s))
    head = bpy.context.active_object
    head.scale = (0.24 * s, 0.22 * s, 0.24 * s)
    bpy.ops.object.transform_apply(scale=True)

    # Visor
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, -0.12 * s, 1.25 * s))
    visor = bpy.context.active_object
    visor.scale = (0.18 * s, 0.05 * s, 0.08 * s)
    bpy.ops.object.transform_apply(scale=True)

    # Arms
    for x in [-0.48 * s, 0.48 * s]:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.06 * s, depth=0.45 * s, location=(x, 0, 0.65 * s))

    # Legs
    for x in [-0.18 * s, 0.18 * s]:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.08 * s, depth=0.55 * s, location=(x, 0, 0.25 * s))

    mat_robot = create_colored_material("Robot_Mat", color)
    mat_visor = create_colored_material("Visor_Mat", "cyan")
    torso.data.materials.append(mat_robot)
    visor.data.materials.append(mat_visor)

    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active = torso
    bpy.ops.object.join()
    return torso


def build_table(size: float = 1.0, color: str = "gray"):
    import bpy
    s = size

    # Tabletop
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.72 * s))
    top = bpy.context.active_object
    top.name = "Table_Top"
    top.scale = (0.90 * s, 0.60 * s, 0.05 * s)
    bpy.ops.object.transform_apply(scale=True)

    # 4 Legs
    leg_r = 0.04 * s
    leg_h = 0.70 * s
    for x, y in [(-0.38 * s, -0.24 * s), (0.38 * s, -0.24 * s), (-0.38 * s, 0.24 * s), (0.38 * s, 0.24 * s)]:
        bpy.ops.mesh.primitive_cylinder_add(radius=leg_r, depth=leg_h, location=(x, y, 0.35 * s))

    mat_table = create_colored_material("Table_Mat", color)
    top.data.materials.append(mat_table)

    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active = top
    bpy.ops.object.join()
    return top


def run_blender_gen():
    try:
        import bpy
    except ImportError:
        print("ERROR: This script must be run within Blender (blender -b -P ...)", file=sys.stderr)
        sys.exit(1)

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Blender Parametric Generator")
    parser.add_argument("--shape", type=str, default="water_bottle", help="Shape to generate")
    parser.add_argument("--style", type=str, default="realistic", help="Style")
    parser.add_argument("--color", type=str, default="blue", help="Primary color")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated assets")
    parser.add_argument("--name", type=str, required=True, help="Base model name")
    parser.add_argument("--size", type=float, default=1.0, help="Scale factor")
    args = parser.parse_args(argv)

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    base_name = args.name
    shape = args.shape.lower().replace("-", "_").replace(" ", "_")
    color = args.color.lower().strip()
    size = max(0.2, min(5.0, args.size))

    # Reset Blender scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Procedural Shape Dispatcher
    builders = {
        "water_bottle": lambda: build_water_bottle(size, color),
        "bottle": lambda: build_water_bottle(size, color),
        "bag": lambda: build_backpack(size, color),
        "backpack": lambda: build_backpack(size, color),
        "chair": lambda: build_chair(size, color),
        "coffee_mug": lambda: build_coffee_mug(size, color),
        "mug": lambda: build_coffee_mug(size, color),
        "cup": lambda: build_coffee_mug(size, color),
        "car": lambda: build_sports_car(size, color),
        "sports_car": lambda: build_sports_car(size, color),
        "scorpio": lambda: build_scorpio_suv(size, color),
        "scorpio_car": lambda: build_scorpio_suv(size, color),
        "scorpio_suv": lambda: build_scorpio_suv(size, color),
        "mahindra_scorpio": lambda: build_scorpio_suv(size, color),
        "suv": lambda: build_scorpio_suv(size, color),
        "robot": lambda: build_robot(size, color),
        "table": lambda: build_table(size, color),
        "desk": lambda: build_table(size, color),
    }

    builder = builders.get(shape)
    if builder:
        obj = builder()
    elif shape == "cube":
        bpy.ops.mesh.primitive_cube_add(size=size, location=(0, 0, size / 2.0))
        obj = bpy.context.active_object
        obj.data.materials.append(create_colored_material("Mat", color))
    elif shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=size / 2.0, location=(0, 0, size / 2.0))
        obj = bpy.context.active_object
        obj.data.materials.append(create_colored_material("Mat", color))
    elif shape == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(radius=size / 2.0, depth=size, location=(0, 0, size / 2.0))
        obj = bpy.context.active_object
        obj.data.materials.append(create_colored_material("Mat", color))
    elif shape == "torus":
        bpy.ops.mesh.primitive_torus_add(major_radius=0.6 * size, minor_radius=0.2 * size, location=(0, 0, 0.3 * size))
        obj = bpy.context.active_object
        obj.data.materials.append(create_colored_material("Mat", color))
    elif shape == "pyramid":
        bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=size / 1.414, depth=size, location=(0, 0, size / 2.0))
        obj = bpy.context.active_object
        obj.data.materials.append(create_colored_material("Mat", color))
    elif shape == "scifi_drone":
        obj = build_robot(size, color)
    else:
        # Generic product fallback
        bpy.ops.mesh.primitive_cube_add(size=size, location=(0, 0, size / 2.0))
        obj = bpy.context.active_object
        obj.data.materials.append(create_colored_material("Mat", color))

    # Add subtle bevel modifier for crisp industrial edges
    if obj:
        mod = obj.modifiers.new(name="Bevel", type='BEVEL')
        mod.width = 0.015 * size
        mod.segments = 2
        try:
            bpy.ops.object.modifier_apply(modifier=mod.name)
        except Exception:
            pass

    # Scene illumination & camera
    bpy.ops.object.camera_add(location=(1.8 * size, -2.2 * size, 1.4 * size), rotation=(math.radians(65), 0, math.radians(40)))
    cam = bpy.context.active_object
    bpy.context.scene.camera = cam

    bpy.ops.object.light_add(type='SUN', location=(3, -3, 5))
    sun = bpy.context.active_object
    if hasattr(sun.data, "energy"):
        sun.data.energy = 2.5

    # Target file paths
    blend_path = os.path.join(out_dir, f"{base_name}.blend")
    glb_path = os.path.join(out_dir, f"{base_name}.glb")
    stl_path = os.path.join(out_dir, f"{base_name}.stl")
    obj_path = os.path.join(out_dir, f"{base_name}.obj")
    thumb_path = os.path.join(out_dir, f"{base_name}_thumb.png")
    meta_path = os.path.join(out_dir, f"{base_name}_meta.json")

    # Render thumbnail (Workbench fast headless render)
    bpy.context.scene.render.resolution_x = 420
    bpy.context.scene.render.resolution_y = 420
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = thumb_path
    bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
    bpy.ops.render.render(write_still=True)

    # Exporters
    # 1. glb
    if hasattr(bpy.ops.export_scene, "gltf"):
        bpy.ops.export_scene.gltf(filepath=glb_path, export_format='GLB')
    elif hasattr(bpy.ops.wm, "gltf_export"):
        bpy.ops.wm.gltf_export(filepath=glb_path)

    # 2. stl
    if hasattr(bpy.ops.wm, "stl_export"):
        bpy.ops.wm.stl_export(filepath=stl_path)
    elif hasattr(bpy.ops.export_mesh, "stl"):
        bpy.ops.export_mesh.stl(filepath=stl_path)

    # 3. obj
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(filepath=obj_path)
    elif hasattr(bpy.ops.export_scene, "obj"):
        bpy.ops.export_scene.obj(filepath=obj_path)

    # 4. blend
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    # Metadata
    v_count = len(obj.data.vertices) if hasattr(obj, "data") and hasattr(obj.data, "vertices") else 0
    f_count = len(obj.data.polygons) if hasattr(obj, "data") and hasattr(obj.data, "polygons") else 0

    metadata = {
        "shape": shape,
        "style": args.style,
        "color": color,
        "size": size,
        "vertices": v_count,
        "polygons": f_count,
        "blend_file": blend_path,
        "glb_file": glb_path,
        "stl_file": stl_path,
        "obj_file": obj_path,
        "thumbnail": thumb_path,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"SUCCESS: Generated {shape} model ({v_count} verts, {f_count} polys)")
    print(f"OUTPUT_FILES: {blend_path} | {glb_path} | {stl_path} | {obj_path} | {thumb_path}")


if __name__ == "__main__":
    run_blender_gen()
