"""
generate_dataset.py
-------------------
Stage 1 of the ControlNet dataset pipeline (Multi-Mesh Version).

Responsibility: Use Mitsuba 3 to render various cloth meshes with randomised
lighting, materials, and geometry. Each frame produces TWO image files:
  - dataset/renders/render_XXXX.png  → beauty (ControlNet target / output)
  - dataset/ao/ao_XXXX.png          → ambient occlusion pass (used by Stage 2)
"""

import mitsuba as mi
import numpy as np
import cv2
import os
import json
import random
import glob
import math

mi.set_variant('scalar_rgb')

# ---------------------------------------------------------------------------
# 1. PATH RESOLUTION & DIRECTORY SETUP
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
# NEW: Pointing to a folder full of diverse .obj files instead of a single scarf
MESHES_DIR    = os.path.join(BASE_DIR, "cloth_meshes") 
DATASET_DIR   = os.path.join(BASE_DIR, "dataset")
RENDERS_DIR   = os.path.join(DATASET_DIR, "renders")
AO_DIR        = os.path.join(DATASET_DIR, "ao")

for d in (RENDERS_DIR, AO_DIR, MESHES_DIR):
    os.makedirs(d, exist_ok=True)

# Discover all .obj files in the meshes directory
mesh_files = sorted(glob.glob(os.path.join(MESHES_DIR, "*.obj")))

print("--- PATH DIAGNOSTICS ---")
print(f"Meshes dir : {MESHES_DIR}")
print(f"  Found    : {len(mesh_files)} .obj file(s)")
print(f"Renders dir: {RENDERS_DIR}")
print(f"AO dir     : {AO_DIR}")
print("------------------------\n")

if not mesh_files:
    print(f"[ERROR] No .obj files found in {MESHES_DIR}.")
    print("Please add your cloth meshes to that folder and run again.")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# 2. RENDER LOOP
# ---------------------------------------------------------------------------
NUM_SAMPLES = 1000  # Full training dataset
metadata_records = []


existing = sum(
    1 for j in range(NUM_SAMPLES)
    if os.path.exists(os.path.join(RENDERS_DIR, f"render_{j:04d}.png"))
    and os.path.exists(os.path.join(AO_DIR,      f"ao_{j:04d}.png"))
)
print(f"Checkpoint: {existing}/{NUM_SAMPLES} frames already done, resuming from where we left off.\n")

for i in range(NUM_SAMPLES):
    frame_str   = f"{i:04d}"
    render_path = os.path.join(RENDERS_DIR, f"render_{frame_str}.png")
    ao_path     = os.path.join(AO_DIR,      f"ao_{frame_str}.png")

    # --- CHECKPOINT: skip frames that were already rendered ---
    if os.path.exists(render_path) and os.path.exists(ao_path):
        print(f"  [{i+1:>4}/{NUM_SAMPLES}] Skipping {frame_str} (already exists)")
        continue

    # --- Randomise Geometry ---
    current_mesh_path = random.choice(mesh_files)
    mesh_name = os.path.basename(current_mesh_path).replace(".obj", "")
    
    # Calculate bounding box for this specific mesh
    shape_dict = {'type': 'obj', 'filename': current_mesh_path}
    loaded_shape = mi.load_dict(shape_dict)
    bbox = loaded_shape.bbox()
    center = bbox.center()
    extents = bbox.extents()
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])

    # DYNAMIC FRAMING: Calculate camera distance based on the largest dimension of the mesh
    # This prevents the camera from clipping into large capes or being too far from small napkins
    max_extent = max(extents[0], extents[1], extents[2])
    # Base distance + random zoom factor — kept tight so cloth fills the frame
    zoom_factor = random.uniform(0.6, 0.9)
    cam_dist    = (max_extent * 1.0 + 1.0) * zoom_factor

    # --- Randomise camera orbit (position on a hemisphere around the object) ---
    cam_azimuth   = random.uniform(0.0, 360.0)      # horizontal orbit, full 360°
    cam_elevation = random.uniform(10.0, 70.0)       # 10° (side) to 70° (near top-down)
    fov           = random.uniform(25.0, 60.0)       # telephoto ↔ wide-angle


    az_rad = math.radians(cam_azimuth)
    el_rad = math.radians(cam_elevation)
    cam_origin = [
        cx + cam_dist * math.cos(el_rad) * math.sin(az_rad),
        cy + cam_dist * math.sin(el_rad),
        cz + cam_dist * math.cos(el_rad) * math.cos(az_rad),
    ]

    # --- Randomise lighting ---
    # Key light: randomised direction
    lx = random.uniform(-1.0, 1.0)
    ly = random.uniform(-0.2, 1.0)   # mostly above the cloth
    lz = random.uniform(-1.0, -0.1)

    # Light color temperature: warm / neutral / cool
    temp_choice = random.choice(['warm', 'neutral', 'cool'])
    if temp_choice == 'warm':    lt = [1.3, 1.0, 0.7]   # golden / sunset
    elif temp_choice == 'cool':  lt = [0.8, 0.9, 1.3]   # overcast / blue sky
    else:                        lt = [1.0, 1.0, 1.0]   # neutral white

    # Light intensity
    intensity = random.uniform(1.5, 6.0)
    key_irr   = [lt[0] * intensity, lt[1] * intensity, lt[2] * intensity]

    # Fill light: dimmer, opposite side (always neutral white)
    fill_intensity = intensity * random.uniform(0.1, 0.4)
    fill_irr = [fill_intensity, fill_intensity, fill_intensity]

    # --- Randomise mesh rotation ---
    yaw   = random.uniform(0.0,   360.0)  # full turntable spin around Y axis
    pitch = random.uniform(-20.0,  20.0)  # modest tilt so cloth stays visible

    # Build transform: rotate around the mesh's own center so it stays framed
    mesh_transform = (
        mi.ScalarTransform4f.translate([cx, cy, cz])
        @ mi.ScalarTransform4f.rotate(axis=[0, 1, 0], angle=yaw)
        @ mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=pitch)
        @ mi.ScalarTransform4f.translate([-cx, -cy, -cz])
    )

    # --- Randomise material ---
    roughness  = random.uniform(0.1, 0.9)   # 0.1 = shiny silk, 0.9 = matte wool
    r, g, b    = (random.uniform(0.1, 0.9) for _ in range(3))
    sheen      = random.uniform(0.0, 1.0)   # microfiber glow (velvet, towel, wool)
    sheen_tint = random.uniform(0.0, 1.0)   # 0=white sheen, 1=tinted toward base color
    anisotropic= random.uniform(0.0, 0.8)   # stretched highlights (woven fabric)
    specular   = random.uniform(0.0, 1.0)   # surface specular intensity

    material_desc = "shiny silk" if roughness < 0.4 else "matte wool"
    prompt = (
        f"a photorealistic 3D render of a {material_desc} cloth, "
        "physical rendering, detailed fabric folds"
    )

    # --- Build scene with AOV integrator ---
    scene = mi.load_dict({
        'type': 'scene',

        'integrator': {
            'type': 'aov',
            'aovs': 'ao_channel:albedo',
            'my_path': {
                'type': 'path',
                'max_depth': 6,
            },
        },

        'sensor': {
            'type': 'perspective',
            'fov': fov,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=cam_origin,
                target=[cx, cy, cz],
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': 512,
                'height': 512,
                'pixel_format': 'rgba',
            },
            'sampler': {
                'type': 'independent',
                'sample_count': 128,
            },
        },

        # Key light
        'key_light': {
            'type': 'directional',
            'direction': [lx, ly, lz],
            'irradiance': {'type': 'rgb', 'value': key_irr},
        },

        # Fill light (opposite horizontal direction, dimmer)
        'fill_light': {
            'type': 'directional',
            'direction': [-lx, ly, -lz],
            'irradiance': {'type': 'rgb', 'value': fill_irr},
        },

        'cloth_object': {
            'type': 'obj',
            'filename': current_mesh_path,
            'to_world': mesh_transform,
            'bsdf': {
                'type': 'principled',
                'base_color':   {'type': 'rgb', 'value': [r, g, b]},
                'roughness':    roughness,
                'sheen':        sheen,
                'sheen_tint':   sheen_tint,
                'anisotropic':  anisotropic,
                'specular':     specular,
            },
        },
    })

    # --- Render → multi-channel tensor (H, W, C) ---
    render_np = np.array(mi.render(scene))

    # ---- Save beauty render ----
    beauty_np    = np.clip(render_np[:, :, :3], 0.0, 1.0)
    beauty_uint8 = (beauty_np * 255).astype(np.uint8)
    cv2.imwrite(render_path, cv2.cvtColor(beauty_uint8, cv2.COLOR_RGB2BGR))

    # ---- Save AO pass ----
    if render_np.shape[2] >= 7:
        ao_rgb  = render_np[:, :, 4:7]
        ao_gray = np.mean(ao_rgb, axis=2)
    else:
        print(f"  [WARNING] AOV channels not found for frame {frame_str}; using luminance as AO proxy.")
        ao_gray = np.mean(render_np[:, :, :3], axis=2)

    ao_uint8 = (np.clip(ao_gray, 0.0, 1.0) * 255).astype(np.uint8)
    cv2.imwrite(ao_path, ao_uint8)

    # ---- Record metadata ----
    metadata_records.append({
        "file_name":          f"renders/render_{frame_str}.png",
        "conditioning_image": f"conditioning/conditioning_{frame_str}.png",
        "ao_image":           f"ao/ao_{frame_str}.png",
        "text":               prompt,
    })

    print(f"  [{i+1:>3}/{NUM_SAMPLES}] Saved {frame_str} | Mesh: {mesh_name[:15]} | {material_desc}")

# ---------------------------------------------------------------------------
# 3. WRITE METADATA
# ---------------------------------------------------------------------------
metadata_path = os.path.join(DATASET_DIR, "metadata.jsonl")
with open(metadata_path, 'w') as f:
    for record in metadata_records:
        f.write(json.dumps(record) + '\n')

print(f"\n✓ Done! {NUM_SAMPLES} render pairs saved.")
print("\nNext: run  python generate_sketches.py  to create the conditioning inputs.")