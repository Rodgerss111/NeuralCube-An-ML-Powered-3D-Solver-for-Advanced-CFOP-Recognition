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

# Complete CFOP OLL Algorithms (57 Cases + Solved State)
# Grouped by their visual "shape" on the U-face for easier reference.
OLL_CASES = {
    "OLL_0_SOLVED": "",

    # --- Dot Cases ---
    "OLL_1": "R U2 R2 F R F' U2 R' F R F'",
    "OLL_2": "F R U R' U' S R U R' U' f'",
    "OLL_3": "f R U R' U' f' U' F R U R' U' F'",
    "OLL_4": "f R U R' U' f' U F R U R' U' F'",
    "OLL_17": "R U R' U R' F R F' U2 R' F R F'",
    "OLL_18": "r U R' U R U2 r2 U' R U' R' U2 r",
    "OLL_19": "r' R U R U R' U' M' R' F R F'",
    "OLL_20": "r U R' U' M2 U R U' R' U' M'",

    # --- Square Cases ---
    "OLL_5": "r' U2 R U R' U r",
    "OLL_6": "r U2 R' U' R U' r'",

    # --- Small Lightning Bolts ---
    "OLL_7": "r U R' U R U2 r'",
    "OLL_8": "r' U' R U' R' U2 r",
    "OLL_11": "r' R2 U R' U R U2 R' U M'",
    "OLL_12": "M' R' U' R U' R' U2 R U' R r'",

    # --- Fish / Knight / Misc Dot-to-Cross ---
    "OLL_9": "R U R' U' R' F R2 U R' U' F'",
    "OLL_10": "R U R' U R' F R F' R U2 R'",
    "OLL_13": "F U R U2 R' U' R U R' F'",
    "OLL_14": "R' F R U R' F' R F U' F'",
    "OLL_15": "r' U' r R' U' R U r' U r",
    "OLL_16": "r U r' R U R' U' r U' r'",

    # --- Cross Cases (Edges already oriented) ---
    "OLL_21": "R U2 R' U' R U R' U' R U' R'",
    "OLL_22": "R U2 R2 U' R2 U' R2 U2 R",
    "OLL_23": "R2 D' R U2 R' D R U2 R",
    "OLL_24": "r U R' U' r' F R F'",
    "OLL_25": "F' r U R' U' r' F R",
    "OLL_26": "R U2 R' U' R U' R'",
    "OLL_27": "R U R' U R U2 R'",

    # --- All Corners Oriented ---
    "OLL_28": "r U R' U' r' R U R U' R'",
    "OLL_57": "R U R' U' M' U R U' r'",

    # --- Awkward Shapes ---
    "OLL_29": "R U R' U' R U' R' F' U' F R U R'",
    "OLL_30": "F R' F R2 U' R' U' R U R' F2'",
    "OLL_41": "R U R' U R U2 R' F R U R' U' F'",
    "OLL_42": "R' U' R U' R' U2 R F R U R' U' F'",

    # --- P Shapes ---
    "OLL_31": "R' U' F U R U' R' F' R",
    "OLL_32": "S R U R' U' R' F R f'",
    "OLL_43": "R' U' F' U F R",
    "OLL_44": "F U R U' R' F'",

    # --- W Shapes ---
    "OLL_36": "L' U' L U' L' U L U L F' L' F",
    "OLL_38": "R U R' U R U' R' U' R' F R F'",

    # --- C and T Shapes ---
    "OLL_33": "R U R' U' R' F R F'",
    "OLL_34": "R U R2 U' R' F R U R U' F'",
    "OLL_45": "F R U R' U' F'",
    "OLL_46": "R' U' R' F R F' U R",

    # --- L Shapes ---
    "OLL_47": "F' L' U' L U L' U' L U F",
    "OLL_48": "F R U R' U' R U R' U' F'",
    "OLL_49": "r U' r2 U r2 U r2 U' r",
    "OLL_50": "r' U r2 U' r2' U' r2 U r'",
    "OLL_53": "r' U' R U' R' U R U' R' U2 r",
    "OLL_54": "r U R' U R U' R' U R U2 r'",

    # --- I Shapes / Line Cases ---
    "OLL_51": "f R U R' U' R U R' U' f'",
    "OLL_52": "R U R' U R U' B U' B' R'",
    "OLL_55": "R U2 R2 U' R U' R' U2 F R F'",
    "OLL_56": "r' U' r U' R' U R U' R' U R r' U r",

    # --- Big Lightning + Fish variants ---
    "OLL_35": "R U2 R2 F R F' R U2 R'",
    "OLL_37": "F R' F' R U R U' R'",
    "OLL_39": "L F' L' U' L U F U' L'",
    "OLL_40": "R' F R U R' U' F' U R"
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
    This feature reduction makes the Neural Network much faster and more accurate.
    """
    flattened = []
    
    # 1. Get all 9 stickers on the Up (U) face
    up_face = cube.get_face("U")
    for i in range(3):
        for j in range(3):
            color_key = _sticker_to_color_key(up_face[i][j])
            flattened.append(COLOR_MAP[color_key])
            
    # 2. Get the top row (index 0) of the adjacent faces
    # Standard traversal: Front, Right, Back, Left
    for face in ["F", "R", "B", "L"]:
        face_grid = cube.get_face(face)
        for j in range(3):
            color_key = _sticker_to_color_key(face_grid[0][j])
            flattened.append(COLOR_MAP[color_key])
            
    return flattened # Returns a highly optimized 1x21 tensor

def generate_oll_dataset(num_samples_per_case=1000):
    dataset = []

    total_cases = len(OLL_CASES)
    for index, (case_id, solve_alg) in enumerate(OLL_CASES.items(), start=1):
        if case_id == "OLL_0_SOLVED":
            setup_alg = pc.Formula("")
        else:
            # We reverse the algorithm to generate the specific OLL case on a solved cube
            setup_alg = pc.Formula(solve_alg).reverse()
        
        for _ in range(num_samples_per_case):
            cube = pc.Cube() 
            cube(setup_alg)
            
            # Apply random U-face rotation (AUF)
            # This teaches the AI that orientation doesn't matter for recognizing the shape
            auf = random.choice(["", "U", "U2", "U'"])
            cube(auf)
            
            # Flatten the top layer into our 21-integer array
            features = flatten_last_layer(cube)
            dataset.append(features + [case_id])

        print(f"Processed {index}/{total_cases} cases ({case_id})")
            
    return dataset

def save_dataset(num_samples_per_case=1000, output_filename="neuralcube_oll_dataset.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_filename)
    estimated_rows = len(OLL_CASES) * num_samples_per_case

    print(f"Generating OLL dataset with {num_samples_per_case} samples per case...")
    print(f"Estimated rows: {estimated_rows}")
    data = generate_oll_dataset(num_samples_per_case)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Create column headers: s0 to s20 for stickers, plus the label
        writer.writerow([f"s{i}" for i in range(21)] + ["label"])
        writer.writerows(data)

    print(f"OLL dataset generated successfully. Exported {len(data)} training rows.")
    print(f"Saved CSV: {output_path}")


if __name__ == "__main__":
    save_dataset(1000)