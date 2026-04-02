import pycuber as pc
import csv
import random

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

COLOR_MAP = {"w": 0, "y": 1, "r": 2, "o": 3, "g": 4, "b": 5}

def flatten_cube(cube):
    """Converts cube stickers into a list of 54 integers for the Neural Network."""
    flattened = []
    for face in ["U", "D", "L", "R", "F", "B"]:
        for row in cube[face]:
            for sticker in row:
                flattened.append(COLOR_MAP[sticker.colour[0]])
    return flattened

def generate_dataset(num_samples_per_case=500):
    dataset = []
    
    for case_id, solve_alg in CASES.items():
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
            
    return dataset

# Execute and save
data = generate_dataset(1000) # Increased to 1000 samples per case for better ML accuracy
with open('neuralcube_f2l_dataset.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([f"s{i}" for i in range(54)] + ["label"])
    writer.writerows(data)

print(f"Dataset generated successfully, Sir! Generated {len(data)} training rows.")