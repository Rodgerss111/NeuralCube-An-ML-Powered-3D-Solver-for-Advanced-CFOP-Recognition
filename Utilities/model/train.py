"""
model/train.py
--------------
Training script for the NeuralCube F2L solver.

Usage:
  python -m model.train --data data/dataset --epochs 50 --out model/saved/f2l_model.h5

What it does:
  1. Loads dataset from .npy files
  2. Splits into train/val sets
  3. Builds the Keras model
  4. Trains with EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
  5. Saves the best model as .h5
  6. Prints final evaluation metrics
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List

from data.generator import load_dataset, build_dataset, save_dataset
from model.network import build_model, model_summary


def train(
    data_prefix: str = None,
    out_path: str = "model/saved/f2l_model.h5",
    epochs: int = 50,
    batch_size: int = 512,
    val_split: float = 0.1,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.3,
    hidden_units: List[int] = None,
    # Quick-generate data if no data_prefix given
    quick_samples: int = 50_000,
    seed: int = 42,
):
    """
    Full training pipeline.

    Parameters
    ----------
    data_prefix   : prefix of _X.npy / _y.npy files. If None, generates data.
    out_path      : where to save the best .h5 model
    epochs        : max training epochs
    batch_size    : mini-batch size
    val_split     : fraction of data for validation
    learning_rate : initial Adam lr
    dropout_rate  : dropout probability
    hidden_units  : list of hidden layer sizes
    quick_samples : samples to generate if data_prefix is None
    seed          : random seed
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # ── 1. Load or generate data ──────────────────────────────────────────
    if data_prefix and os.path.exists(f"{data_prefix}_X.npy"):
        X, y = load_dataset(data_prefix)
    else:
        print(f"No dataset found at '{data_prefix}'. Generating {quick_samples:,} samples...")
        X, y = build_dataset(samples=quick_samples, seed=seed, verbose=True)
        if data_prefix:
            save_dataset(X, y, data_prefix)

    print(f"Dataset: X={X.shape}, y={y.shape}")
    print(f"Move distribution: {np.bincount(y)}")

    # ── 2. Train/val split ────────────────────────────────────────────────
    n_val = int(len(X) * val_split)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    X_train, X_val = X[n_val:], X[:n_val]
    y_train, y_val = y[n_val:], y[:n_val]
    print(f"Train: {len(X_train):,}  |  Val: {len(X_val):,}")

    # ── 3. Build model ────────────────────────────────────────────────────
    model = build_model(
        hidden_units=hidden_units or [512, 256, 128, 64],
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )
    model_summary(model)

    # ── 4. Callbacks ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=out_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir="model/logs",
            histogram_freq=1,
        ),
    ]

    # ── 5. Train ──────────────────────────────────────────────────────────
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 6. Final evaluation ───────────────────────────────────────────────
    print("\nFinal evaluation on validation set:")
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"  Val Loss     : {loss:.4f}")
    print(f"  Val Accuracy : {acc:.4f} ({acc*100:.1f}%)")
    print(f"\nBest model saved to: {out_path}")

    return model, history


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuralCube F2L solver")
    parser.add_argument("--data",     type=str,   default=None,                  help="Dataset prefix (_X.npy / _y.npy)")
    parser.add_argument("--out",      type=str,   default="model/saved/f2l_model.h5", help="Output model path")
    parser.add_argument("--epochs",   type=int,   default=50,                    help="Max training epochs")
    parser.add_argument("--batch",    type=int,   default=512,                   help="Batch size")
    parser.add_argument("--lr",       type=float, default=1e-3,                  help="Learning rate")
    parser.add_argument("--dropout",  type=float, default=0.3,                   help="Dropout rate")
    parser.add_argument("--samples",  type=int,   default=50_000,                help="Samples to generate if no data")
    parser.add_argument("--seed",     type=int,   default=42,                    help="Random seed")
    args = parser.parse_args()

    train(
        data_prefix=args.data,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        dropout_rate=args.dropout,
        quick_samples=args.samples,
        seed=args.seed,
    )
