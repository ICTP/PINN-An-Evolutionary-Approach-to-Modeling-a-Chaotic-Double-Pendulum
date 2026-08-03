"""
train_mlp.py
============
Stage 2 — Data-Driven Learning

Trains an MLP  f(t) -> (theta1(t), theta2(t))  on RK4-generated data.

Architecture (shared across all NN stages):
  Input  : 1  (t)
  Hidden : 4 x 64, tanh
  Output : 2  (theta1, theta2), linear

Loss: MSE(theta_pred, theta_true)

Usage: import this module from the Jupyter notebook — see the
"Data-Driven Learning (MLP)" section of Double_Pendulum_Lab.ipynb.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

def build_mlp(n_hidden: int = 4, n_units: int = 64,
               activation: str = "tanh") -> tf.keras.Model:
    """Return the shared MLP architecture (Input=1, Output=2)."""
    inputs = tf.keras.Input(shape=(1,), name="t_input")
    x = inputs
    for i in range(n_hidden):
        x = tf.keras.layers.Dense(n_units, activation=activation,
                                   name=f"hidden_{i+1}")(x)
    outputs = tf.keras.layers.Dense(2, activation="linear", name="output")(x)
    return tf.keras.Model(inputs, outputs, name="MLP")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mlp(
    model: tf.keras.Model,
    t_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 5000,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 200,
    verbose: int = 1,
) -> tf.keras.callbacks.History:
    """Compile and fit the MLP with MSE loss.

    Parameters
    ----------
    t_train : (N,) or (N,1)  time values
    y_train : (N, 2)         [theta1, theta2] targets
    """
    t_train = np.asarray(t_train, dtype=np.float32).reshape(-1, 1)
    y_train = np.asarray(y_train, dtype=np.float32)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=patience // 2, min_lr=1e-6, verbose=1),
    ]

    return model.fit(
        t_train, y_train,
        epochs=epochs, batch_size=batch_size,
        validation_split=0.1, callbacks=callbacks, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_mlp(model: tf.keras.Model, path: str | Path) -> None:
    """Save model to disk (.keras format)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    print(f"MLP saved -> {path}")


def load_mlp(path: str | Path) -> tf.keras.Model:
    """Load a saved MLP from disk."""
    m = tf.keras.models.load_model(str(path))
    print(f"MLP loaded <- {path}")
    return m
