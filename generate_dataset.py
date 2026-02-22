import mitsuba as mi
import os
import random
import json

mi.set_variant('scalar_rgb')

# --- 1. PATH RESOLUTION & SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCARF_PATH = os.path.join(BASE_DIR, "WomensScarf", "10152_WomensScarf_v01_L3.obj")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset")

# Create a dataset folder if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Calculating mesh bounds...")
scarf_shape = mi.load_dict({'type': 'obj', 'filename': SCARF_PATH})
center = scarf_shape.bbox().center()

# We will store our text prompts here to save as a JSONL file later
metadata_records = []

# --- 2. THE RANDOMIZATION LOOP ---
NUM_SAMPLES = 10 # Change this to 1000+ when you are ready for full training

print(f"Starting generation of {NUM_SAMPLES} rendered pairs...")

for i in range(NUM_SAMPLES):
    # Randomize Light Direction (Spherical distribution)
    lx = random.uniform(-1.0, 1.0)
    ly = random.uniform(-1.0, 1.0)
    lz = random.uniform(-1.0, -0.1) # Keep light generally in front/above
    
    # Randomize Material (Color and Roughness)
    # Roughness: 0.1 (shiny silk) to 0.9 (matte wool)
    roughness = random.uniform(0.1, 0.9) 
    
    # Let's pick a random RGB color
    r, g, b = random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)
    
    # Generate the text prompt for ControlNet conditioning
    material_desc = "shiny silk" if roughness < 0.4 else "matte wool"
    prompt = f"a photorealistic 3D render of a {material_desc} scarf, physical rendering, detailed cloth folds"
    
    # --- 3. DYNAMIC SCENE CREATION ---
    scene_dict = {
        'type': 'scene',
        'integrator': {'type': 'path'},
        'sensor': {
            'type': 'perspective',
            'fov': 40,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[center[0], center[1], center[2] + 25.0],
                target=center,
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': 512,
                'height': 512,
            },
        },
        'light': {
            'type': 'directional',
            'direction': [lx, ly, lz],
            'irradiance': {'type': 'rgb', 'value': [3.0, 3.0, 3.0]},
        },
        'scarf_object': {
            'type': 'obj',
            'filename': SCARF_PATH,
            # Switched to the Principled BSDF for realistic cloth simulation
            'bsdf': {
                'type': 'principled',
                'base_color': {'type': 'rgb', 'value': [r, g, b]},
                'roughness': roughness,
            }
        },
    }
    
    # Load and render the specific variant
    scene = mi.load_dict(scene_dict)
    image = mi.render(scene, spp=128) # Kept at 128 for faster testing
    
    # Save the image
    image_filename = f"render_{i:04d}.png"
    image_path = os.path.join(OUTPUT_DIR, image_filename)
    mi.util.write_bitmap(image_path, image)
    
    # Record the metadata
    metadata_records.append({
        "file_name": image_filename,
        "text": prompt
    })
    
    print(f"[{i+1}/{NUM_SAMPLES}] Saved {image_filename} | Prompt: {prompt}")

# --- 4. SAVE METADATA ---
metadata_path = os.path.join(OUTPUT_DIR, "metadata.jsonl")
with open(metadata_path, 'w') as f:
    for record in metadata_records:
        f.write(json.dumps(record) + '\n')

print(f"\nSuccess! Dataset and metadata saved to the '{OUTPUT_DIR}' folder.")