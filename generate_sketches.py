import cv2
import os
import glob
import numpy as np

# --- 1. PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CONDITIONING_DIR = os.path.join(DATASET_DIR, "conditioning")

# Create the output directory for the sketches
if not os.path.exists(CONDITIONING_DIR):
    os.makedirs(CONDITIONING_DIR)

# Find all the rendered PNGs from Stage 1
render_files = sorted(glob.glob(os.path.join(DATASET_DIR, "render_*.png")))

print(f"Found {len(render_files)} target renders. Starting OpenCV edge extraction...")

# --- 2. PROCESSING LOOP ---
for file_path in render_files:
    filename = os.path.basename(file_path)
    
    # Read the high-fidelity render
    img = cv2.imread(file_path)
    
    if img is None:
        print(f"Error loading {filename}. Skipping.")
        continue
        
    # Step A: Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step B: Gaussian Blur
    # Crucial for physically-based renders to smooth out path-tracing noise 
    # before edge detection, preventing a "dusty" sketch.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step C: Canny Edge Detection
    # The thresholds (50, 150) define how strict the edge finding is. 
    # Lower = more interior cloth fold lines.
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    
    # Note on ControlNet Architecture: 
    # The default ControlNet expects the conditioning image to have 
    # WHITE lines on a BLACK background. OpenCV's Canny does this automatically.
    
    # Step D: Save the conditioning image
    out_path = os.path.join(CONDITIONING_DIR, filename)
    cv2.imwrite(out_path, edges)
    
    print(f"Generated sketch for: {filename}")

print(f"\nSuccess! All conditioning sketches saved to the '{CONDITIONING_DIR}' folder.")