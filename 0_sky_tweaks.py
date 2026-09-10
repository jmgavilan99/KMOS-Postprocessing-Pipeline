#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:57:14 2026
Processes kmos_sky_tweak for all OB folders found inside each concatenation.

@author: jmgavilan
"""

import os
import glob
import subprocess

# ============================================================
# 1. EDIT THESE VARIABLES (only these!)
# ============================================================
pointing = "P0"                              # pointing name (e.g. P1, P2...)
cons = ["con1"]              # concatenation folders to process
SKY_FOLDER = "Sky"                           # name of the sky subdirectory inside each con

base_path = f"/home/data/KMOS/PILOT/reduced/P113/{pointing}"

# Full path to the esorex executable (needed if it's not in your PATH)
ESOREX = "/home/linuxbrew/.linuxbrew/bin/esorex"

# ============================================================
# 2. Function to find SINGLE_CUBES_KMOS path from a base directory
# ============================================================
def find_single_cubes(base_dir):
    """Recursively looks for a file matching *SINGLE_CUBES_KMOS*.fits,
       ignoring directories like 'tmp' or 'sky_tweak'."""
    ignore_dirs = {'sky_tweak', 'tmp'}
    
    os.chdir(base_dir)
    contents = os.listdir('.')
    folders = [c for c in contents 
               if os.path.isdir(c) and c not in ignore_dirs]
    
    # Look for the first folder that contains a subdirectory KMOS.*_tpl
    for fecha_dir in folders:
        path_fecha = os.path.join(base_dir, fecha_dir)
        kmos_dirs = glob.glob(os.path.join(path_fecha, "KMOS.*_tpl"))
        if kmos_dirs:
            # Found the correct folder
            kmos_dir = kmos_dirs[0]
            files = glob.glob(os.path.join(kmos_dir, "*SINGLE_CUBES_KMOS*.fits"))
            if files:
                return os.path.abspath(files[0])
    
    # Nothing found
    raise Exception(f"No SINGLE_CUBES_KMOS file found in {base_dir} "
                    f"(checked subdirectories: {folders})")

# ============================================================
# 3. Main loop over concatenations
# ============================================================
for con in cons:
    print(f"\n{'#'*60}")
    print(f"Processing concatenation: {con}")
    print(f"{'#'*60}")

    con_dir = os.path.join(base_path, con)

    # ----- 3a. Find the sky cube for this concatenation -----
    sky_dir = os.path.join(con_dir, SKY_FOLDER)
    print(f"Looking for sky cube in {sky_dir} ...")
    try:
        sky_path = find_single_cubes(sky_dir)
    except Exception as e:
        print(f"  ERROR finding sky cube for {con}: {e}")
        continue   # skip this whole concatenation
    print(f"Sky cube found: {sky_path}")

    # ----- 3b. Discover all OB folders inside this concatenation -----
    # We take every subdirectory whose name starts with "OB"
    # and is not the sky folder (or other special names).
    all_items = os.listdir(con_dir)
    ob_names = sorted([
        d for d in all_items
        if os.path.isdir(os.path.join(con_dir, d))
        and d.startswith("OB")
        and d != SKY_FOLDER
    ])

    if not ob_names:
        print(f"  No OB folders found in {con_dir} (checked for directories starting with 'OB')")
        continue

    print(f"Found {len(ob_names)} OB folder(s): {', '.join(ob_names)}")

    # ----- 3c. Process each OB -----
    for ob in ob_names:
        print(f"\n{'='*40}")
        print(f"Processing OB: {ob}")
        print(f"{'='*40}")

        cube_dir = os.path.join(con_dir, ob)
        print(f"Looking for science cube in {cube_dir} ...")
        try:
            cube_path = find_single_cubes(cube_dir)
        except Exception as e:
            print(f"  ERROR finding cube for {ob}: {e}")
            continue
        print(f"Science cube: {cube_path}")

        # Create clean output directory (inside the OB folder)
        clean_dir = os.path.join(cube_dir, "sky_tweak")
        os.makedirs(clean_dir, exist_ok=True)

        # Write single_cubes.sof
        sof_file = os.path.join(clean_dir, "single_cubes.sof")
        with open(sof_file, 'w') as f:
            f.write(f"{cube_path} CUBE_OBJECT\n")
            f.write(f"{sky_path} CUBE_SKY\n")

        # Run kmos_sky_tweak
        os.chdir(clean_dir)
        cmd_sky = [ESOREX, "kmos_sky_tweak", "single_cubes.sof"]
        result_sky = subprocess.run(cmd_sky, capture_output=True, text=True)
        if result_sky.returncode != 0:
            print(f"  ERROR in kmos_sky_tweak for {con}/{ob}:")
            print(result_sky.stderr)
            continue

        print("  kmos_sky_tweak completed successfully.")

        # Write decombine.sof
        sky_tweak_path = os.path.join(clean_dir, "SKY_TWEAK.fits")
        decombine_sof = os.path.join(clean_dir, "decombine.sof")
        with open(decombine_sof, 'w') as f:
            f.write(f"{sky_tweak_path} SINGLE_CUBES\n")

        # Run kmos_combine
        cmd_combine = [ESOREX, "kmos_combine", "--method=header", "decombine.sof"]
        result_combine = subprocess.run(cmd_combine, capture_output=True, text=True)
        if result_combine.returncode == 0:
            print("  kmos_combine completed successfully.")
        else:
            print(f"  ERROR in kmos_combine for {con}/{ob}:")
            print(result_combine.stderr)

print("\nAll requested concatenations processed.")