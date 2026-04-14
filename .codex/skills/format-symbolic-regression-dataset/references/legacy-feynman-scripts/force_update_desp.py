import csv
import os
import sys

# 仅处理 feynman 目录
ROOT_DIR = "."
BASE_DIR = os.path.join(ROOT_DIR, "feynman")
CSV_PATH = os.path.join(BASE_DIR, "srbench_feynman.csv")

def load_csv_info():
    """
    Load info from srbench_feynman.csv using indices.
    Returns a dict: key -> {
        'context': str,
        'target_symbol': str,
        'target_meaning': str,
        'vars': { var_name: description, ... }
    }
    Key matches the first column (Filename).
    """
    info_map = {}
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return {}

    with open(CSV_PATH, newline='') as f:
        reader = csv.reader(f)
        header = next(reader) # Skip header
        
        # Headers from previous inspection:
        # 0: Filename (Key)
        # 1: Output (Target Symbol)
        # 2: Meaning (Word) (Target Description)
        # 3: Context Description
        # 4: Filename (Duplicate)
        # ...
        # 9: v1_name
        # 19: v1_desp
        
        for row in reader:
            if not row: continue
            
            key = row[0].strip()
            target_symbol = row[1].strip()
            target_meaning = row[2].strip()
            context = row[3].strip()
            
            vars_map = {}
            
            # 10 variables max
            for i in range(10):
                name_idx = 9 + i
                desp_idx = 19 + i
                
                if name_idx >= len(row) or desp_idx >= len(row): break
                
                v_name = row[name_idx].strip()
                v_desp = row[desp_idx].strip()
                
                if v_name:
                    # Format: "Symbol, Description" or just "Description" if symbol redundant
                    # User asked for "v1_desp" to be put in description field.
                    # User example: "G, Gravitational constant"
                    if v_desp:
                        vars_map[v_name] = f"{v_name}, {v_desp}"
                    else:
                        vars_map[v_name] = v_name
            
            target_full = ""
            if target_symbol and target_meaning:
                target_full = f"{target_symbol}, {target_meaning}"
            elif target_meaning:
                target_full = target_meaning
            
            info_map[key] = {
                'context': context,
                'target_full': target_full,
                'vars': vars_map
            }
            
    return info_map

def derive_key_from_dirname(dirname):
    """
    Convert dirname to CSV key.
    feynman_test_1 -> test_1
    feynman_I_10_7 -> I.10.7
    """
    if not dirname.startswith('feynman_'):
        return None
    
    core = dirname[8:] # remove feynman_ 
    
    if core.startswith('test_'):
        return core
    
    # Standard Feynman: I_10_7 -> I.10.7
    # But watch out for suffixes like I_15_3t -> I.15.3t
    return core.replace('_', '.')

def update_metadata_yaml(file_path, info):
    if not os.path.exists(file_path):
        return

    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    
    in_dataset = False
    in_features = False
    in_target = False
    
    current_feature_name = None
    
    for line in lines:
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        
        # --- Block Detection ---
        if stripped.startswith("dataset:"):
            in_dataset = True; in_features = False; in_target = False
            new_lines.append(line); continue
        elif stripped.startswith("features:"):
            in_dataset = False; in_features = True; in_target = False
            new_lines.append(line); continue
        elif stripped.startswith("target:"):
            in_dataset = False; in_features = False; in_target = True
            new_lines.append(line); continue
        elif stripped.startswith("splits:") or stripped.startswith("ground_truth_formula:") or \
             stripped.startswith("license:") or stripped.startswith("citation:") or stripped.startswith("resources:"):
            in_dataset = False; in_features = False; in_target = False
            new_lines.append(line); continue
        # --- End Block Detection ---

        # --- Logic per Block ---
        
        # Dataset Description
        if in_dataset and stripped.startswith("description:"):
            if info['context']:
                new_lines.append(f"{indent}description: {info['context']}\n")
            else:
                new_lines.append(line)
            continue
            
        # Feature Description
        if in_features:
            if stripped.startswith("- name:"):
                parts = stripped.split(":", 1)
                if len(parts) == 2:
                    name_part = parts[1].strip()
                    if name_part.startswith(("'", '"')):
                        name_part = name_part[1:-1]
                    current_feature_name = name_part
                else:
                    current_feature_name = None
                new_lines.append(line)
                continue
            
            if stripped.startswith("description:"):
                # Found an existing description line, replace it
                if current_feature_name and current_feature_name in info['vars']:
                    new_lines.append(f"{indent}description: {info['vars'][current_feature_name]}\n")
                else:
                    new_lines.append(line)
                continue
            
            # If we reach the end of a feature block (e.g. next "- name" or end of features)
            # without seeing "description:", we might need to insert it.
            # But YAML parsing line-by-line makes this hard without lookahead.
            # However, generate_metadata_2.py ALREADY ensures a "description:" line exists 
            # (even if empty or wrong) IF it found info. 
            # If it didn't find info, it might not have added the line.
            # Let's handle the insertion case:
            # We insert description after 'ood_range' or 'type' or 'train_range' if it's missing?
            # Safer strategy: Look for the key attributes.
            pass

        # Target Description
        if in_target and stripped.startswith("description:"):
            if info['target_full']:
                new_lines.append(f"{indent}description: {info['target_full']}\n")
            else:
                new_lines.append(line)
            continue

        new_lines.append(line)
        
        # Insertion Logic:
        # If we are inside a feature block, and we just wrote 'ood_range', and we have a description, 
        # and the NEXT line is NOT 'description:', then we should insert it.
        # But we don't know the next line.
        # ALTERNATIVE: We blindly insert 'description' after 'ood_range', and if the next line IS description, 
        # we will skip/overwrite it in the next iteration? No, that duplicates.
        
        # Better approach:
        # Iterate lines. Store them.
        # If we are in a feature block, collect lines for ONE feature. 
        # Modify that block. Then write.
        # But that's a big refactor.
        
        # Pragmatic approach: 
        # 'generate_metadata_2.py' outputted 'ood_range' as the last item for a feature.
        # If we see 'ood_range:', we append 'description:' if we haven't seen it yet for this feature?
        # No, let's stick to: if 'description:' exists, replace it.
        # If it doesn't exist, we must append it.
        # Where to append? After `ood_range`.
        
        if in_features and stripped.startswith("ood_range:"):
            # Check if we have a description to add
            if current_feature_name and current_feature_name in info['vars']:
                # We want to add description here.
                # But wait, does the file already have a description line following this?
                # We can peek? No.
                # We can assume generate_metadata_2.py structure:
                #   - name: ...
                #     type: ...
                #     train_range: ...
                #     ood_range: ...
                #     description: ... (Optional)
                
                # If we append here, and the next line is 'description:', we will have two.
                # We can rely on the loop finding 'description:' next and replacing it?
                # No, because we just appended one.
                
                # Let's Read the whole file into a structure? No.
                
                # Let's use a flag `feature_desc_added`.
                pass

    # To handle insertion robustly:
    # We will do a second pass? Or just be smart.
    # Let's try to Detect if description is missing.
    # Actually, looking at feynman_test_1/metadata_2.yaml:
    # It DOES NOT have description lines for features.
    # So we MUST insert them.
    
    return new_lines

def update_yaml_robust(file_path, info):
    if not os.path.exists(file_path): return

    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    
    in_features = False
    in_target = False
    current_feature_name = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        
        # Block detection
        if stripped.startswith("dataset:"):
            # Handle Dataset Desc immediately
            new_lines.append(line)
            i += 1
            # Look ahead for description
            if i < len(lines) and lines[i].strip().startswith("description:"):
                if info['context']:
                    new_lines.append(f"{indent}  description: {info['context']}\n")
                else:
                    new_lines.append(lines[i]) # Keep existing
                i += 1
            else:
                # Insert if missing (and we have context)
                if info['context']:
                    new_lines.append(f"{indent}  description: {info['context']}\n")
            continue
            
        elif stripped.startswith("features:"):
            in_features = True; in_target = False
            new_lines.append(line)
            i += 1
            continue
        elif stripped.startswith("target:"):
            in_features = False; in_target = True
            new_lines.append(line)
            i += 1
            continue
        elif stripped.startswith("splits:") or stripped.startswith("ground_truth_formula:") or \
             stripped.startswith("license:") or stripped.startswith("citation:") or stripped.startswith("resources:"):
            in_features = False; in_target = False
            new_lines.append(line)
            i += 1
            continue
            
        # Feature Block Logic
        if in_features and stripped.startswith("- name:"):
            parts = stripped.split(":", 1)
            name_part = parts[1].strip() if len(parts) == 2 else None
            if name_part and name_part.startswith(("'","\"")):
                name_part = name_part[1:-1]
            current_feature_name = name_part
            new_lines.append(line)
            i += 1
            continue
            
        # We are inside a feature. Look for where to put description.
        # We usually put it at the end of the feature block (after ood_range).
        # Or if we see an existing description, replace it.
        if in_features and stripped.startswith("description:"):
            if current_feature_name and current_feature_name in info['vars']:
                new_lines.append(f"{indent}description: {info['vars'][current_feature_name]}\n")
            else:
                new_lines.append(line)
            i += 1
            continue
            
        if in_features and stripped.startswith("ood_range:"):
            new_lines.append(line)
            i += 1
            # Check if next line is description
            if i < len(lines) and lines[i].strip().startswith("description:"):
                # Next iteration will handle replacement
                continue
            else:
                # Next line is NOT description (e.g. next feature or end of block)
                # Insert description here!
                if current_feature_name and current_feature_name in info['vars']:
                    new_lines.append(f"{indent}description: {info['vars'][current_feature_name]}\n")
            continue

        # Target Block Logic
        if in_target and stripped.startswith("description:"):
            if info['target_full']:
                new_lines.append(f"{indent}description: {info['target_full']}\n")
            else:
                new_lines.append(line)
            i += 1
            continue
            
        if in_target and stripped.startswith("ood_range:"):
            new_lines.append(line)
            i += 1
            # Check if next line is description
            if i < len(lines) and lines[i].strip().startswith("description:"):
                continue
            else:
                if info['target_full']:
                    new_lines.append(f"{indent}description: {info['target_full']}\n")
            continue

        # Default
        new_lines.append(line)
        i += 1

    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    print(f"Updated {file_path}")

def main():
    csv_info = load_csv_info()
    print(f"Loaded info for {len(csv_info)} datasets from CSV.")
    
    if not os.path.exists(BASE_DIR):
        print("feynman directory not found.")
        return

    subdirs = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])
    
    for sub in subdirs:
        key = derive_key_from_dirname(sub)
        if not key: continue
        
        if key in csv_info:
            yaml_path = os.path.join(BASE_DIR, sub, "metadata_2.yaml")
            update_yaml_robust(yaml_path, csv_info[key])
        else:
            print(f"Skipping {sub} (Key: {key}) - Not found in CSV")

if __name__ == "__main__":
    main()
