import os
import csv
import inspect
import importlib.util
import numpy as np

DATASET_KEY = "feynman/feynman_test_7"
ROOT_DIR = "."
YAML_PATH = os.path.join(ROOT_DIR, DATASET_KEY, "metadata_2.yaml")
DIST_CSV = os.path.join(ROOT_DIR, "feature_distributions.csv")
INFO_CSV = os.path.join(ROOT_DIR, "feynman", "srbench_feynman.csv")

def debug_run():
    print(f"Debugging {DATASET_KEY}...")
    
    # 1. Check Dist CSV
    ood_data = {}
    with open(DIST_CSV, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row[0] == DATASET_KEY:
                print("  [OK] Found in feature_distributions.csv")
                # Extract G
                try:
                    idx = row.index('G')
                    print(f"    - Found feature 'G' at index {idx}")
                    # Check range cols
                    print(f"    - Range vals: {row[idx+2]} to {row[idx+3]}")
                except ValueError:
                    print("    - Feature 'G' NOT found in row.")
                ood_data = row
                break
    if not ood_data:
        print("  [FAIL] NOT Found in feature_distributions.csv")

    # 2. Check Info CSV
    info_found = False
    with open(INFO_CSV, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row[0] == "test_7":
                print("  [OK] Found 'test_7' in srbench_feynman.csv")
                # Check vars
                print(f"    - Context: {row[3]}")
                # v1_name should be G?
                print(f"    - v1: {row[9]} ({row[19]})")
                info_found = True
                break
    if not info_found:
        print("  [FAIL] 'test_7' NOT Found in srbench_feynman.csv")

    # 3. Check Formula
    formula_path = os.path.join(ROOT_DIR, DATASET_KEY, "formula.py")
    if os.path.exists(formula_path):
        print("  [OK] formula.py exists")
    else:
        print("  [FAIL] formula.py missing")

    # 4. Dry Run YAML Update
    with open(YAML_PATH, 'r') as f:
        lines = f.readlines()
    
    print(f"  Read {len(lines)} lines from YAML.")
    
    # Simulate the loop simply
    new_lines = []
    in_features = False
    
    print("  --- Simulating YAML Processing ---")
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("features:"):
            in_features = True
            print("    Entered FEATURES block")
        
        if in_features and stripped.startswith("- name: G"):
            print("    Found feature 'G'")
        
        if in_features and stripped.startswith("ood_range:"):
            print(f"    Original OOD line: {stripped}")
            # Check if it looks like single interval
            if "[[" not in stripped:
                print("    -> Detected SINGLE interval (Old format)")
            else:
                print("    -> Detected DOUBLE interval (New format)")

    print("  --- End Simulation ---")

if __name__ == "__main__":
    debug_run()
