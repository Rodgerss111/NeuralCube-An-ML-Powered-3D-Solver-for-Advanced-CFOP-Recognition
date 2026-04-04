import json
import sys
from pathlib import Path

SUPPORTED_PYTHON_MIN = (3, 9)
SUPPORTED_PYTHON_MAX_EXCLUSIVE = (3, 11)

if not (SUPPORTED_PYTHON_MIN <= sys.version_info[:2] < SUPPORTED_PYTHON_MAX_EXCLUSIVE):
    raise RuntimeError(
        "Unsupported Python version for this training script. "
        "Use Python 3.9 or 3.10 (TensorFlow-compatible), then reinstall dependencies."
    )

try:
    import pandas as pd
    import tensorflow as tf
    import tensorflowjs as tfjs
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Missing dependency: "
        f"{exc.name}. Install project dependencies from requirements.txt first."
    ) from exc

print("Initializing F2L Neural Network Training, Sir...")

# Resolve paths from this file so execution works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_CANDIDATES = [
    PROJECT_ROOT / 'data' / 'neuralcube_f2l_dataset.csv',
    PROJECT_ROOT / 'Utilities' / 'data' / 'neuralcube_f2l_dataset.csv',
    PROJECT_ROOT / 'Utilities' / 'data_generators' / 'neuralcube_f2l_dataset.csv',
]

DATASET_PATH = next((path for path in DATASET_CANDIDATES if path.exists()), None)
if DATASET_PATH is None:
    searched_paths = "\n".join(str(path) for path in DATASET_CANDIDATES)
    raise FileNotFoundError(
        "Could not find neuralcube_f2l_dataset.csv. Checked:\n"
        f"{searched_paths}"
    )

LABEL_MAP_PATH = Path(__file__).resolve().parent / 'f2l_label_map.json'
WEB_MODEL_PATH = Path(__file__).resolve().parent / 'web_model_f2l'

# 1. Ingest the CSV Data
# Assuming the CSV has 54 columns (s0-s53) and 1 label column ('label')
data = pd.read_csv(DATASET_PATH)

# Separate features (X) and labels (y)
X_raw = data.drop('label', axis=1).values
y_raw = data['label'].values

# --- NEW FIX: ONE-HOT ENCODE THE INPUTS ---
# Convert the integers (0-5) into binary categorical matrices
X_one_hot = tf.keras.utils.to_categorical(X_raw, num_classes=6) 
# Flatten the 3D matrix (samples, 54, 6) into a 2D matrix (samples, 324)
X_processed = X_one_hot.reshape(-1, 54 * 6)
# ------------------------------------------

# 2. Encode Labels to Integers (Keep your existing label code here...)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)

# Save the label mapping to a JSON file so your React app knows which number means what!
label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
with open(LABEL_MAP_PATH, 'w', encoding='utf-8') as f:
    json.dump(label_mapping, f)
print(f"Label mapping saved to {LABEL_MAP_PATH}")

# Convert labels to categorical format (One-Hot Encoding for Softmax)
num_classes = len(label_encoder.classes_)
y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)

# Split data using the NEW processed features
X_train, X_val, y_train, y_val = train_test_split(X_processed, y_categorical, test_size=0.2, random_state=42)

# 3. Build an OPTIMIZED Neural Network Architecture
model = tf.keras.Sequential([
    # Input layer
    Dense(512, input_shape=(324,)),
    BatchNormalization(),           # NEW: Stabilizes the math
    tf.keras.layers.Activation('relu'),
    Dropout(0.3),                   # Increased dropout slightly for the larger dataset
    
    # Hidden Layer 1
    Dense(256),
    BatchNormalization(),           # NEW
    tf.keras.layers.Activation('relu'),
    Dropout(0.2),
    
    # Hidden Layer 2
    Dense(128),
    BatchNormalization(),           # NEW
    tf.keras.layers.Activation('relu'),
    
    # Output layer
    Dense(num_classes, activation='softmax')
])

# 4. Compile with a Fine-Tuned Learning Rate
# Default is 0.001. We slow it down to 0.0003 for precision fine-tuning.
custom_optimizer = Adam(learning_rate=0.0003)

model.compile(optimizer=custom_optimizer, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# NEW: Early Stopping Callback
# This stops the training automatically if val_accuracy doesn't improve for 10 epochs
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# 5. Train the Model (Extended Epochs)
print("Beginning fine-tuned training process...")
history = model.fit(X_train, y_train, 
                    epochs=150, 
                    batch_size=128,  # NEW: Increased batch size for the larger dataset
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop]) # Plugs in the smart stopping logic

# 6. Export for the Web App (React)
# This saves the model into a folder that TensorFlow.js can read directly!
WEB_MODEL_PATH.mkdir(parents=True, exist_ok=True)
tfjs.converters.save_keras_model(model, str(WEB_MODEL_PATH))

print(f"\nTraining Complete! Model saved to '{WEB_MODEL_PATH}' folder.")