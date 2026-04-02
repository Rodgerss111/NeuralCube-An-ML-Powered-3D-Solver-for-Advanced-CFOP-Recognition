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

# Complete CFOP PLL Algorithms (21 Cases + Solved State)
# Grouped by piece swapping behavior.
PLL_CASES = {
    "PLL_0_SOLVED": "",

    # --- Edges Only ---
    "PLL_H": "M2 U M2 U2 M2 U M2",
    "PLL_Ua": "R U' R U R U R U' R' U' R2",
    "PLL_Ub": "R2 U R U R' U' R' U' R' U R'",
    "PLL_Z": "M2 U M2 U M' U2 M2 U2 M'",

    # --- Corners Only ---
    "PLL_Aa": "l' U R' D2 R U' R' D2 R2",  # Using l instead of x for pycuber compatibility
    "PLL_Ab": "l' R' D2 R U R' D2 R U' R",
    "PLL_E": "x' R U' R' D R U R' D' R U R' D R U' R' D'",

    # --- Adjacent Corner Swap ---
    "PLL_T": "R U R' U' R' F R2 U' R' U' R U R' F'",
    "PLL_F": "R' U' F' R U R' U' R' F R2 U' R' U' R U R' U R",
    "PLL_Ja": "x R2 F R F' R U2 r' U r U2",
    "PLL_Jb": "R U R' F' R U R' U' R' F R2 U' R' U'",
    "PLL_Ra": "R U R' F' R U2 R' U2 R' F R U R U2 R'",
    "PLL_Rb": "R' U2 R U2 R' F R U R' U' R' F' R2",

    # --- Diagonal Corner Swap ---
    "PLL_V": "R' U R' U' y R' F' R2 U' R' U R' F R F",
    "PLL_Y": "F R U' R' U' R U R' F' R U R' U' R' F R F'",
    "PLL_Na": "R U R' U R U R' F' R U R' U' R' F R2 U' R' U2 R U' R'",
    "PLL_Nb": "R' U R U' R' F' U' F R U R' F R' F' R U' R",

    # --- G Permutations (The trickiest for human recognition!) ---
    "PLL_Ga": "R2 U R' U R' U' R U' R2 U' D R' U R D'",
    "PLL_Gb": "R' U' R U D' R2 U R' U R U' R U' R2 D",
    "PLL_Gc": "R2 U' R U' R U R' U R2 U D' R U' R' D",
    "PLL_Gd": "R U R' U' D R2 U' R U' R' U R' U R2 D'"
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

def flatten_last_layer(cube):
    """
    Extracts only the 21 stickers relevant to OLL and PLL.
    Reused from the OLL script for maximum efficiency.
    """
    flattened = []

    # 1. Get all 9 stickers on the Up (U) face
    up_face = cube.get_face("U")
    for i in range(3):
        for j in range(3):
            color_key = _sticker_to_color_key(up_face[i][j])
            flattened.append(COLOR_MAP[color_key])
            
    # 2. Get the top row (index 0) of the adjacent faces (F, R, B, L)
    for face in ["F", "R", "B", "L"]:
        face_grid = cube.get_face(face)
        for j in range(3):
            color_key = _sticker_to_color_key(face_grid[0][j])
            flattened.append(COLOR_MAP[color_key])
            
    return flattened

def generate_pll_dataset(num_samples_per_case=1000):
    dataset = []

    total_cases = len(PLL_CASES)
    for index, (case_id, solve_alg) in enumerate(PLL_CASES.items(), start=1):
        if case_id == "PLL_0_SOLVED":
            setup_alg = pc.Formula("")
        else:
            # Reverse the algorithm to generate the specific PLL case
            setup_alg = pc.Formula(solve_alg).reverse()
        
        for _ in range(num_samples_per_case):
            cube = pc.Cube() 
            cube(setup_alg)
            
            # Apply random U-face rotation (AUF)
            # Critical for PLL so the AI learns to recognize the pattern from any angle
            auf = random.choice(["", "U", "U2", "U'"])
            cube(auf)
            
            features = flatten_last_layer(cube)
            dataset.append(features + [case_id])

        print(f"Processed {index}/{total_cases} cases ({case_id})")
            
    return dataset

def save_dataset(num_samples_per_case=1000, output_filename="neuralcube_pll_dataset.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_filename)
    estimated_rows = len(PLL_CASES) * num_samples_per_case

    print(f"Generating PLL dataset with {num_samples_per_case} samples per case...")
    print(f"Estimated rows: {estimated_rows}")
    data = generate_pll_dataset(num_samples_per_case)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Headers: s0 to s20 for stickers, plus the label
        writer.writerow([f"s{i}" for i in range(21)] + ["label"])
        writer.writerows(data)

    print(f"PLL dataset generated successfully. Exported {len(data)} training rows.")
    print(f"Saved CSV: {output_path}")


if __name__ == "__main__":
    save_dataset(1000)