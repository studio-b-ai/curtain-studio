"""
Generate synthetic window training data using Blender.

Run inside Blender:
    blender --background --python generate_synthetic.py -- --count 2000 --output data/synthetic/

Generates parametric rooms with window openings of known dimensions,
renders from varied camera positions and lighting, and outputs JPEG images
with a ground-truth labels.json manifest.

If Blender's `bpy` module is unavailable (e.g., running outside Blender for
testing), the script generates the labels manifest only without rendering.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Window type configurations: realistic dimension ranges in inches
# --------------------------------------------------------------------------- #
WINDOW_CONFIGS: dict[str, dict[str, Any]] = {
    "single_hung": {
        "width_range": (24, 48),
        "height_range": (36, 72),
        "description": "Single-hung window",
    },
    "double_hung": {
        "width_range": (24, 48),
        "height_range": (36, 72),
        "description": "Double-hung window",
    },
    "casement": {
        "width_range": (18, 36),
        "height_range": (36, 72),
        "description": "Casement window (hinged side)",
    },
    "picture": {
        "width_range": (36, 96),
        "height_range": (36, 72),
        "description": "Fixed picture window",
    },
    "sliding": {
        "width_range": (36, 84),
        "height_range": (24, 60),
        "description": "Horizontal sliding window",
    },
    "arched": {
        "width_range": (24, 60),
        "height_range": (36, 84),
        "description": "Arched/round-top window",
    },
    "divided_light": {
        "width_range": (24, 48),
        "height_range": (36, 60),
        "description": "Window with grille/muntin pattern",
    },
}

# Camera parameter ranges
CAMERA_DISTANCE_FT = (3.0, 8.0)
CAMERA_HORIZ_ANGLE_DEG = (-20.0, 20.0)
CAMERA_VERT_ANGLE_DEG = (-10.0, 10.0)
CAMERA_FOCAL_LENGTH_MM = (24.0, 52.0)

# Lighting presets
LIGHTING_PRESETS = {
    "morning": {
        "sun_elevation": 25.0,
        "sun_azimuth": 90.0,
        "color_temp": 4000,
        "intensity": 0.7,
    },
    "noon": {
        "sun_elevation": 70.0,
        "sun_azimuth": 180.0,
        "color_temp": 5500,
        "intensity": 1.0,
    },
    "evening": {
        "sun_elevation": 15.0,
        "sun_azimuth": 270.0,
        "color_temp": 3200,
        "intensity": 0.5,
    },
    "overcast": {
        "sun_elevation": 45.0,
        "sun_azimuth": 180.0,
        "color_temp": 6500,
        "intensity": 0.4,
    },
}

# Render resolution
RENDER_WIDTH = 1280
RENDER_HEIGHT = 960

# --------------------------------------------------------------------------- #
# Try importing Blender — graceful degradation if unavailable
# --------------------------------------------------------------------------- #
try:
    import bpy  # type: ignore[import-not-found]
    import bmesh  # type: ignore[import-not-found]
    from mathutils import Vector, Euler  # type: ignore[import-not-found]

    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False


def inches_to_meters(inches: float) -> float:
    """Convert inches to meters."""
    return inches * 0.0254


def random_window_params(window_type: str) -> dict[str, Any]:
    """Generate random window dimensions for a given type."""
    config = WINDOW_CONFIGS[window_type]
    width = random.uniform(*config["width_range"])
    height = random.uniform(*config["height_range"])
    return {
        "window_type": window_type,
        "width_inches": round(width, 1),
        "height_inches": round(height, 1),
        "width_m": round(inches_to_meters(width), 4),
        "height_m": round(inches_to_meters(height), 4),
    }


def random_camera_params() -> dict[str, float]:
    """Generate random camera parameters."""
    return {
        "distance_ft": round(random.uniform(*CAMERA_DISTANCE_FT), 2),
        "horiz_angle_deg": round(random.uniform(*CAMERA_HORIZ_ANGLE_DEG), 1),
        "vert_angle_deg": round(random.uniform(*CAMERA_VERT_ANGLE_DEG), 1),
        "focal_length_mm": round(random.uniform(*CAMERA_FOCAL_LENGTH_MM), 1),
    }


def random_lighting() -> tuple[str, dict[str, Any]]:
    """Pick a random lighting preset."""
    name = random.choice(list(LIGHTING_PRESETS.keys()))
    return name, LIGHTING_PRESETS[name]


# --------------------------------------------------------------------------- #
# Blender scene setup (only runs when bpy is available)
# --------------------------------------------------------------------------- #
def clear_scene() -> None:
    """Remove all objects from the Blender scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def create_room(
    room_width: float, room_depth: float, room_height: float
) -> "bpy.types.Object":
    """Create a simple box room (interior visible). Dimensions in meters."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, room_height / 2))
    room = bpy.context.active_object
    room.scale = (room_width, room_depth, room_height)
    room.name = "Room"

    # Flip normals inward so interior faces render
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(room.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bmesh.update_edit_mesh(room.data)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Basic wall material
    mat = bpy.data.materials.new("WallMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    # Random wall color (muted tones)
    r = random.uniform(0.6, 0.95)
    g = random.uniform(0.6, 0.95)
    b = random.uniform(0.6, 0.95)
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    room.data.materials.append(mat)

    return room


def create_window_opening(
    room: "bpy.types.Object",
    width_m: float,
    height_m: float,
    wall_y: float,
    sill_height: float,
) -> "bpy.types.Object":
    """Cut a window opening in the room wall using boolean modifier."""
    # Position the cutter on the far wall (positive Y)
    center_z = sill_height + height_m / 2
    bpy.ops.mesh.primitive_cube_add(
        size=1, location=(0, wall_y, center_z)
    )
    cutter = bpy.context.active_object
    cutter.scale = (width_m, 0.3, height_m)
    cutter.name = "WindowCutter"

    # Boolean difference
    bool_mod = room.modifiers.new(name="WindowBool", type="BOOLEAN")
    bool_mod.operation = "DIFFERENCE"
    bool_mod.object = cutter
    bpy.context.view_layer.objects.active = room
    bpy.ops.object.modifier_apply(modifier="WindowBool")

    # Hide cutter from render
    cutter.hide_render = True
    cutter.hide_viewport = True

    return cutter


def setup_camera(params: dict[str, float], window_center_z: float) -> None:
    """Position and configure the camera."""
    dist_m = params["distance_ft"] * 0.3048
    h_angle = math.radians(params["horiz_angle_deg"])
    v_angle = math.radians(params["vert_angle_deg"])

    # Camera position: facing the window wall (positive Y direction)
    x = dist_m * math.sin(h_angle)
    y = -dist_m * math.cos(h_angle)
    z = window_center_z + dist_m * math.sin(v_angle)

    bpy.ops.object.camera_add(location=(x, y, z))
    camera = bpy.context.active_object
    camera.name = "Camera"

    # Point camera at window center
    direction = Vector((0, 0, window_center_z)) - Vector((x, y, z))
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()

    # Set focal length
    camera.data.lens = params["focal_length_mm"]
    camera.data.sensor_width = 36.0  # Full-frame equivalent

    bpy.context.scene.camera = camera


def setup_lighting(preset: dict[str, Any]) -> None:
    """Add a sun light matching the preset."""
    elevation = math.radians(preset["sun_elevation"])
    azimuth = math.radians(preset["sun_azimuth"])

    bpy.ops.object.light_add(
        type="SUN",
        rotation=(math.pi / 2 - elevation, 0, azimuth),
        location=(0, 0, 5),
    )
    sun = bpy.context.active_object
    sun.data.energy = preset["intensity"]
    sun.name = "SunLight"

    # Add ambient fill light
    bpy.ops.object.light_add(type="AREA", location=(0, 0, 3))
    fill = bpy.context.active_object
    fill.data.energy = preset["intensity"] * 0.3
    fill.data.size = 4.0
    fill.name = "FillLight"


def configure_render(output_path: str) -> None:
    """Set render settings for JPEG output."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 90
    scene.render.filepath = output_path


def render_sample(
    index: int,
    output_dir: str,
    window_params: dict[str, Any],
    camera_params: dict[str, float],
    lighting_name: str,
    lighting_preset: dict[str, Any],
) -> dict[str, Any]:
    """Set up and render a single synthetic sample."""
    clear_scene()

    # Room dimensions (meters)
    room_w = random.uniform(3.0, 5.0)
    room_d = random.uniform(3.0, 5.0)
    room_h = random.uniform(2.4, 3.0)

    room = create_room(room_w, room_d, room_h)

    # Window on the far wall (+Y side)
    sill_height = random.uniform(0.6, 1.0)  # meters from floor
    wall_y = room_d / 2
    window_center_z = sill_height + window_params["height_m"] / 2

    create_window_opening(
        room,
        window_params["width_m"],
        window_params["height_m"],
        wall_y,
        sill_height,
    )

    setup_camera(camera_params, window_center_z)
    setup_lighting(lighting_preset)

    # Render
    filename = f"synth_{index:05d}.jpg"
    output_path = os.path.join(output_dir, filename)
    configure_render(output_path)
    bpy.ops.render.render(write_still=True)

    return {
        "index": index,
        "filename": filename,
        "window": window_params,
        "camera": camera_params,
        "lighting": lighting_name,
        "room": {
            "width_m": round(room_w, 2),
            "depth_m": round(room_d, 2),
            "height_m": round(room_h, 2),
        },
        "sill_height_m": round(sill_height, 3),
        "render": {
            "width": RENDER_WIDTH,
            "height": RENDER_HEIGHT,
        },
    }


# --------------------------------------------------------------------------- #
# Labels-only mode (no Blender)
# --------------------------------------------------------------------------- #
def generate_labels_only(
    index: int,
    window_params: dict[str, Any],
    camera_params: dict[str, float],
    lighting_name: str,
) -> dict[str, Any]:
    """Generate a label entry without rendering (for testing)."""
    return {
        "index": index,
        "filename": f"synth_{index:05d}.jpg",
        "window": window_params,
        "camera": camera_params,
        "lighting": lighting_name,
        "room": {
            "width_m": round(random.uniform(3.0, 5.0), 2),
            "depth_m": round(random.uniform(3.0, 5.0), 2),
            "height_m": round(random.uniform(2.4, 3.0), 2),
        },
        "sill_height_m": round(random.uniform(0.6, 1.0), 3),
        "render": {
            "width": RENDER_WIDTH,
            "height": RENDER_HEIGHT,
        },
    }


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments. Handles Blender's '--' arg separator."""
    # When run via `blender --background --python script.py -- --count 100`,
    # everything after '--' is in sys.argv for the script.
    if argv is None:
        argv = sys.argv
        if "--" in argv:
            argv = argv[argv.index("--") + 1 :]
        else:
            # Not launched through Blender's -- separator; use all args
            argv = argv[1:]

    parser = argparse.ArgumentParser(
        description="Generate synthetic window training data"
    )
    parser.add_argument(
        "--count", type=int, default=2000, help="Number of samples to generate (default: 2000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic/",
        help="Output directory (default: data/synthetic/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Generate labels.json without rendering (for testing)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    use_blender = HAS_BLENDER and not args.labels_only
    if not use_blender:
        if not HAS_BLENDER:
            print(
                "WARNING: Blender (bpy) not available. "
                "Generating labels.json only (no rendered images)."
            )
        else:
            print("Labels-only mode: generating labels.json without rendering.")

    window_types = list(WINDOW_CONFIGS.keys())
    labels: list[dict[str, Any]] = []

    for i in range(args.count):
        window_type = random.choice(window_types)
        window_params = random_window_params(window_type)
        camera_params = random_camera_params()
        lighting_name, lighting_preset = random_lighting()

        if use_blender:
            label = render_sample(
                i, output_dir, window_params, camera_params, lighting_name, lighting_preset
            )
        else:
            label = generate_labels_only(i, window_params, camera_params, lighting_name)

        labels.append(label)

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{args.count}] {window_type} "
                  f"{window_params['width_inches']}x{window_params['height_inches']}\" "
                  f"({'rendered' if use_blender else 'label only'})")

    # Write labels manifest
    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nDone. {len(labels)} samples generated.")
    print(f"Labels: {labels_path}")
    if use_blender:
        print(f"Images: {output_dir}/synth_*.jpg")

    # Print distribution summary
    type_counts: dict[str, int] = {}
    for label in labels:
        wt = label["window"]["window_type"]
        type_counts[wt] = type_counts.get(wt, 0) + 1

    print("\nWindow type distribution:")
    for wt, count in sorted(type_counts.items()):
        print(f"  {wt}: {count} ({100*count/len(labels):.1f}%)")


if __name__ == "__main__":
    main()
