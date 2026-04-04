import csv
import os
import random
from pathlib import Path

try:
    import pycuber as pc
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "pycuber is not installed in the active Python environment. "
        "Install it with: pip install pycuber"
    ) from exc

# Complete CFOP F2L Algorithms (Targeting the Front-Right Slot)
# The AI will be trained to recognize these states and output the corresponding label.
CASES = {
    "F2L_0_SOLVED": "",
    "F2L_1": "U R U' R'",
    "F2L_2": "U' F' U F",
    "F2L_3": "F' U' F",
    "F2L_4": "R U R'", # Also handles basic extraction natively
    "F2L_5_15": "R U R' U' R U R'", # Merged Case 5 and 15
    "F2L_6_16": "F' U' F U F' U' F", # Merged Case 6 and 16
    "F2L_7": "U R U' R' U' R U R' U' R U R'",
    "F2L_8_27": "R U2 R' U' R U R'", # Merged Case 8 and 27
    "F2L_9": "U' F' U2 F U F' U' F",
    "F2L_10": "U2 R U R' U R U' R'",
    "F2L_11": "U2 F' U' F U' F' U F",
    "F2L_12_29": "U R U' R' U R U' R' U R U' R'", # Merged Case 12 and 29
    "F2L_13": "U R U' R' U R U' R'",
    "F2L_14": "U' F' U F U' F' U F",
    "F2L_17": "R U' R' U R U' R'",
    "F2L_18": "F' U F U' F' U F",
    "F2L_19": "U R U' R' U' R U' R' U R U' R'",
    "F2L_20": "U' F' U F U F' U F U' F' U F",
    "F2L_21": "U R U R' U2 R U' R'",
    "F2L_22": "U' F' U' F U2 F' U F",
    "F2L_23": "U R U R' U' R U' R'",
    "F2L_24": "U' F' U' F U F' U F",
    "F2L_25": "U' R U' R' U R U R'",
    "F2L_26": "U F' U F U' F' U' F",
    "F2L_28": "F' U2 F U F' U' F",
    "F2L_30": "U' R U2 R' U R U' R'",
    "F2L_31": "U F' U2 F U' F' U F",
    "F2L_32": "R U R' U2 R U R' U' R U R'",
    "F2L_33": "F' U' F U2 F' U' F U F' U' F",
    "F2L_34": "U R U2 R' U R U' R'",
    "F2L_35": "U' F' U2 F U' F' U F",
    "F2L_36": "U2 R U2 R' U R U' R'",
    "F2L_37": "U2 F' U2 F U' F' U F",
    "F2L_38": "R U R' U' R U2 R' U' R U R'",
    "F2L_39": "R U' R' U F' U' F",
    "F2L_40": "R U R' U' R U' R' U d R' U' R",
    "F2L_41": "R U' R' d R' U2 R U R' U2 R",
    # EXTRACTION CASES: AI sees a trapped piece and outputs the command to free it.
    "ACTION_EXTRACT_RIGHT": "R U' R'",  # Setup puts piece in bottom-right slot
    "ACTION_EXTRACT_FRONT": "F' U F",    # Setup puts piece in bottom-front slot
    # NEW: The Flipped Edge extraction case
    "ACTION_EXTRACT_FLIPPED": "R U R' F R' F' R"
    
}

# Supports both color initials (w,y,r,o,g,b) and face initials (u,d,r,l,f,b)
# because some pycuber versions expose facelets by face letter.
COLOR_MAP = {
    "w": 0, "y": 1, "r": 2, "o": 3, "g": 4, "b": 5,
    "d": 0, "u": 1, "l": 3, "f": 4
}

def _sticker_to_color_key(sticker):
    """Normalizes PyCuber sticker objects and raw string facelets to a color key."""
    if hasattr(sticker, "colour"):
        value = sticker.colour
    else:
        value = sticker

    if isinstance(value, str):
        return value[0].lower()

    return str(value)[0].lower()

def flatten_cube(cube):
    """Converts cube stickers into a list of 54 integers for the Neural Network."""
    flattened = []
    for face in ["U", "D", "L", "R", "F", "B"]:
        for row in cube.get_face(face):
            for sticker in row:
                color_key = _sticker_to_color_key(sticker)
                flattened.append(COLOR_MAP[color_key])
    return flattened

def generate_dataset(num_samples_per_case=1500):
    dataset = []
    total_cases = len(CASES)
    
    for index, (case_id, alg) in enumerate(CASES.items(), start=1):
        if case_id == "F2L_0_SOLVED":
            setup_alg = pc.Formula("")
        else:
            setup_alg = pc.Formula(alg).reverse()
        
        # 1. OPTIMIZATION: Calculate the 4 perfect spatial states exactly ONCE.
        base_permutations = []
        for rotation in ["", "y", "y2", "y'"]:
            cube = pc.Cube() 
            
            # Apply the F2L problem
            cube(setup_alg)
            
            # Rotate the entire cube to shift the slot colors
            if rotation:
                cube(rotation)
            
            features = flatten_cube(cube)
            base_permutations.append(features + [case_id])

        # 2. DUPLICATION: Copy those perfect states in memory to fill the dataset
        for _ in range(num_samples_per_case):
            dataset.extend(base_permutations)

        # Because the math is done instantly, this print will now fire immediately!
        print(f"Processed {index}/{total_cases} cases ({case_id})", flush=True)
                
    return dataset

def save_dataset(num_samples_per_case=1000, output_filename="neuralcube_f2l_dataset.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_filename)
    estimated_rows = len(CASES) * num_samples_per_case * 4

    print(f"Generating F2L dataset with {num_samples_per_case} samples per case...")
    print(f"Estimated rows: {estimated_rows}")
    data = generate_dataset(num_samples_per_case)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"s{i}" for i in range(54)] + ["label"])
        writer.writerows(data)

    print(f"Dataset generated successfully. Generated {len(data)} training rows.")
    print(f"Saved CSV: {output_path}")


if __name__ == "__main__":
    # Increased to 1000 samples per case for better ML accuracy
    save_dataset(1000)