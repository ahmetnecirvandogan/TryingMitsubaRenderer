import mitsuba as mi
import os

# Set the variant to standard CPU rendering to bypass LLVM warnings during testing
mi.set_variant('scalar_rgb')

# --- 1. DYNAMIC PATH RESOLUTION ---
# This guarantees the script finds the .obj regardless of where the terminal is opened
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCARF_PATH = os.path.join(BASE_DIR, "WomensScarf", "10152_WomensScarf_v01_L3.obj")

print("--- PATH DIAGNOSTICS ---")
print(f"Looking for scarf at: {SCARF_PATH}")
print(f"Does the file actually exist here? {os.path.exists(SCARF_PATH)}")
print("------------------------\n")

print("Loading the scarf geometry to calculate framing...")

# --- 2. MESH DIAGNOSTICS ---
# Load JUST the shape to inspect its mathematical properties
scarf_shape = mi.load_dict({
    'type': 'obj',
    'filename': SCARF_PATH
})

# Calculate the bounding box to find exactly where the scarf is in 3D space
bbox = scarf_shape.bbox()
center = bbox.center()

print("--- SCARF DIAGNOSTICS ---")
print(f"Center Coordinates: {center}")
print(f"Size (Width, Height, Depth): {bbox.extents()}")
print("-------------------------\n")

print("Setting up dynamic camera and lighting...")

# --- 3. SCENE CONSTRUCTION ---
scene = mi.load_dict({
    'type': 'scene',
    'integrator': {
        'type': 'path',
    },
    'sensor': {
        'type': 'perspective',
        'fov': 40, # Narrow field of view for a flatter, more orthographic "sketch" look
        'to_world': mi.ScalarTransform4f.look_at(
            # Dynamically position the camera 25 units in front of the scarf's exact Z-center
            origin=[center[0], center[1], center[2] + 25.0],
            target=center, # Look directly at the calculated center
            up=[0, 1, 0]
        ),
        'film': {
            'type': 'hdrfilm',
            'width': 512, # 512x512 is standard for Stable Diffusion / ControlNet training
            'height': 512,
        },
    },
    'light': {
        'type': 'directional',
        # Angled slightly down and to the left to cast shadows across the fabric folds
        'direction': [-0.5, -0.5, -1.0], 
        'irradiance': {
            'type': 'rgb',
            'value': [3.0, 3.0, 3.0], 
        },
    },
    'scarf_object': {
        'type': 'obj',
        'filename': SCARF_PATH,
        # Applying a basic diffuse green material for the initial test
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.2, 0.7, 0.2], 
            }
        },
    },
})

print("Scene loaded! Starting render...")

# --- 4. RENDER EXECUTION ---
# Render the scene (256 samples per pixel for a cleaner image)
image = mi.render(scene, spp=256)

# Write the output image
mi.util.write_bitmap('output_scarf.png', image)
print("Success! Rendered image saved to output_scarf.png")