import csv
import os
import random

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
    "F2L_0_SOLVED": "", # State 42: The slot is already solved.

    # --- Basic Insertions (Pieces are paired in the U layer) ---
    "F2L_1": "U R U' R'",
    "F2L_2": "U' F' U F",
    "F2L_3": "F' U' F",
    "F2L_4": "R U R'",

    # --- White Sticker Facing Up (Edge in U layer) ---
    "F2L_5": "R U R' U' R U R'", # Edge is matched with center
    "F2L_6": "F' U' F U F' U' F",
    "F2L_7": "U R U' R' U' R U R' U' R U R'",
    "F2L_8": "R U2 R' U' R U R'",
    "F2L_9": "U' F' U2 F U F' U' F",
    "F2L_10": "U2 R U R' U R U' R'",
    "F2L_11": "U2 F' U' F U' F' U F",
    "F2L_12": "U R U' R' U R U' R' U R U' R'", # The triple sexy move

    # --- Corner in Bottom, Edge in Top ---
    "F2L_13": "U R U' R' U R U' R'",
    "F2L_14": "U' F' U F U' F' U F",
    "F2L_15": "R U R' U' R U R'",
    "F2L_16": "F' U' F U F' U' F",
    "F2L_17": "R U' R' U R U' R'",
    "F2L_18": "F' U F U' F' U F",

    # --- Edge in Bottom, Corner in Top ---
    "F2L_19": "U R U' R' U' R U' R' U R U' R'",
    "F2L_20": "U' F' U F U F' U F U' F' U F",
    "F2L_21": "U R U R' U2 R U' R'",
    "F2L_22": "U' F' U' F U2 F' U F",
    "F2L_23": "U R U R' U' R U' R'",
    "F2L_24": "U' F' U' F U F' U F",
    
    # --- Both Pieces in the U Layer (Separated) ---
    "F2L_25": "U' R U' R' U R U R'",
    "F2L_26": "U F' U F U' F' U' F",
    "F2L_27": "R U2 R' U' R U R'",
    "F2L_28": "F' U2 F U F' U' F",
    "F2L_29": "U R U' R' U R U' R' U R U' R'",
    "F2L_30": "U' R U2 R' U R U' R'",
    "F2L_31": "U F' U2 F U' F' U F",
    "F2L_32": "R U R' U2 R U R' U' R U R'",
    "F2L_33": "F' U' F U2 F' U' F U F' U' F",
    "F2L_34": "U R U2 R' U R U' R'",
    "F2L_35": "U' F' U2 F U' F' U F",
    "F2L_36": "U2 R U2 R' U R U' R'",
    "F2L_37": "U2 F' U2 F U' F' U F",
    
    # --- Both Pieces in the Correct Slot (But oriented incorrectly) ---
    "F2L_38": "R U R' U' R U2 R' U' R U R'",
    "F2L_39": "R U' R' U F' U' F",
    "F2L_40": "R U R' U' R U' R' U d R' U' R",
    "F2L_41": "R U' R' d R' U2 R U R' U2 R",

    # --- EXTRACTION CASES (For trapped pieces in wrong slots) ---
    # Sir, instead of adding 300+ cases, we train the ML to output an extraction move.
    # When pieces are trapped in the wrong slots (e.g. Back-Right or Front-Left), 
    # the AI simply predicts the move to pull them out into the U layer.
    "EXTRACT_FR": "R U R'",   # Pull piece out of Front-Right slot
    "EXTRACT_FL": "L' U' L",  # Pull piece out of Front-Left slot
    "EXTRACT_BR": "R' U' R",  # Pull piece out of Back-Right slot
    "EXTRACT_BL": "L U L'"    # Pull piece out of Back-Left slot
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

def generate_dataset(num_samples_per_case=500):
    dataset = []

    total_cases = len(CASES)
    for index, (case_id, solve_alg) in enumerate(CASES.items(), start=1):
        # Handle the solved state without applying an algorithm
        if case_id == "F2L_0_SOLVED":
            setup_alg = pc.Formula("")
        else:
            setup_alg = pc.Formula(solve_alg).reverse()
        
        for _ in range(num_samples_per_case):
            cube = pc.Cube() 
            
            # 1. Apply the reverse algorithm to create the F2L case
            cube(setup_alg)
            
            # 2. Apply random U face rotations (AUF) to prevent model overfitting
            auf = random.choice(["", "U", "U2", "U'"])
            cube(auf)
            
            # 3. Flatten for the ML Input Tensor
            features = flatten_cube(cube)
            dataset.append(features + [case_id])

        print(f"Processed {index}/{total_cases} cases ({case_id})")
            
    return dataset

def save_dataset(num_samples_per_case=1000, output_filename="neuralcube_f2l_dataset.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_filename)
    estimated_rows = len(CASES) * num_samples_per_case

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