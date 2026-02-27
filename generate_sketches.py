"""
generate_sketches.py
--------------------
Stage 2 of the ControlNet dataset pipeline.

Responsibility: For every frame produced by generate_dataset.py, load the
beauty render AND the AO pass, then produce a "shaded sketch" image that
serves as the ControlNet conditioning input:

  dataset/renders/render_XXXX.png   ← beauty (Canny source)
  dataset/ao/ao_XXXX.png            ← ambient occlusion pass
        ↓
  dataset/conditioning/conditioning_XXXX.png   ← shaded sketch output

The shaded sketch is formed by:
  1. Canny edge detection on the beauty render  → crisp structural lines
  2. Inverted AO map                            → soft cloth-fold shadows
  3. cv2.max blend of both layers              → final conditioning image

Run generate_dataset.py first to produce the renders and AO maps.
"""

import cv2
import numpy as np
import os
import glob

# ---------------------------------------------------------------------------
# 1. PATH SETUP
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR     = os.path.join(BASE_DIR, "dataset")
RENDERS_DIR     = os.path.join(DATASET_DIR, "renders")
AO_DIR          = os.path.join(DATASET_DIR, "ao")
CONDITION_DIR   = os.path.join(DATASET_DIR, "conditioning")

os.makedirs(CONDITION_DIR, exist_ok=True)

# Canny thresholds — lower values detect more interior cloth-fold lines
CANNY_LOW  = 50
CANNY_HIGH = 150

# How strongly the AO shading layer blends in (0.0 = off, 1.0 = full weight)
AO_WEIGHT  = 0.6

# ---------------------------------------------------------------------------
# 2. DISCOVER FRAMES
# ---------------------------------------------------------------------------
render_files = sorted(glob.glob(os.path.join(RENDERS_DIR, "render_*.png")))

if not render_files:
    print(f"No renders found in {RENDERS_DIR}.")
    print("Please run  python generate_dataset.py  first.")
    raise SystemExit(1)

print(f"Found {len(render_files)} render(s). Generating shaded sketches...\n")

processed = 0
skipped   = 0

# ---------------------------------------------------------------------------
# 3. PROCESSING LOOP
# ---------------------------------------------------------------------------
for render_path in render_files:
    basename  = os.path.basename(render_path)           # render_0000.png
    frame_str = basename.replace("render_", "").replace(".png", "")  # 0000
    ao_path   = os.path.join(AO_DIR,       f"ao_{frame_str}.png")
    out_path  = os.path.join(CONDITION_DIR, f"conditioning_{frame_str}.png")

    # ---- Load images with error handling ----
    beauty_bgr = cv2.imread(render_path)
    ao_gray    = cv2.imread(ao_path, cv2.IMREAD_GRAYSCALE)

    if beauty_bgr is None:
        print(f"  [ERROR] Could not load beauty render: {render_path}  — skipping.")
        skipped += 1
        continue
    if ao_gray is None:
        print(f"  [WARNING] Could not load AO map: {ao_path}  — falling back to Canny-only.")

    # ---- Step 1: Canny edge map (white lines on black background) ----
    beauty_gray    = cv2.cvtColor(beauty_bgr, cv2.COLOR_BGR2GRAY)
    beauty_blurred = cv2.GaussianBlur(beauty_gray, (5, 5), 0)
    canny_edges    = cv2.Canny(beauty_blurred, CANNY_LOW, CANNY_HIGH)
    # Result: uint8 [0 or 255], crisp structural lines ✓

    # ---- Step 2: Process AO pass → soft shadow shading layer ----
    if ao_gray is not None:
        # Raw AO:  bright = flat/unoccluded,  dark = deep fold
        # Invert so deep folds become bright (matches sketch convention)
        ao_inverted = cv2.bitwise_not(ao_gray)
        ao_shading  = (ao_inverted.astype(np.float32) * AO_WEIGHT).astype(np.uint8)
    else:
        ao_shading = np.zeros_like(canny_edges)   # no shading, lines only

    # ---- Step 3: Blend — take the brightest pixel from either layer ----
    # cv2.max ensures Canny lines are never dimmed by the shading layer
    shaded_sketch = cv2.max(canny_edges, ao_shading)

    # ---- Step 4: Save conditioning image ----
    cv2.imwrite(out_path, shaded_sketch)
    print(f"  [frame {frame_str}] ✓  {out_path}")
    processed += 1

# ---------------------------------------------------------------------------
# 4. SUMMARY
# ---------------------------------------------------------------------------
print(f"\n✓ Done!  {processed} conditioning image(s) saved to {CONDITION_DIR}")
if skipped:
    print(f"  ⚠  {skipped} frame(s) skipped due to missing render files.")