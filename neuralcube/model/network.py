"""
model/network.py
----------------
Keras model for F2L move prediction.

Input:  (324,) float32 — one-hot encoded cube state
Output: (18,)  float32 — softmax probability over 18 HTM moves

Architecture: Dense feedforward with BatchNorm + Dropout.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List

NUM_MOVES = 18       # 6 faces × 3 (CW, CCW, 180°)
INPUT_DIM = 324      # 54 facelets × 6 colors one-hot


def build_model(
    input_dim: int = INPUT_DIM,
    num_moves: int = NUM_MOVES,
    hidden_units: List[int] = None,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Build and compile the F2L move prediction model.

    Parameters
    ----------
    input_dim    : dimensionality of input vector (default 324)
    num_moves    : number of output classes (default 18)
    hidden_units : list of hidden layer sizes (default [512, 256, 128, 64])
    dropout_rate : dropout probability
    learning_rate: Adam learning rate

    Returns
    -------
    Compiled Keras model
    """
    if hidden_units is None:
        hidden_units = [512, 256, 128, 64]

    inputs = keras.Input(shape=(input_dim,), name="cube_state")
    x = inputs

    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, name=f"dense_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Activation("relu", name=f"relu_{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)

    outputs = layers.Dense(num_moves, activation="softmax", name="move_probs")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="F2L_Solver")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def load_model(path: str) -> keras.Model:
    """Load a saved Keras model from path (.h5 or SavedModel dir)."""
    return keras.models.load_model(path)


def model_summary(model: keras.Model):
    """Print a clean model summary."""
    model.summary(line_length=80)
    total = model.count_params()
    print(f"\nTotal parameters: {total:,}")
