"""
train_cpinn.py
==============
Conservative PINN (cPINN) — Double Pendulum

Extends the standard PINN with a Hamiltonian energy-conservation term.
The Hamiltonian (total mechanical energy) of the frictionless double
pendulum:

  H = ½(m1+m2)L1²ω1² + ½m2L2²ω2² + m2·L1·L2·ω1·ω2·cos(θ1−θ2)
      + (m1+m2)·g·L1·(1−cosθ1) + m2·g·L2·(1−cosθ2)

must remain constant. The cPINN penalizes the variance of H:

  L_total = lambda_data   * L_data
           + lambda_phys  * L_physics
           + lambda_ic    * L_ic
           + lambda_energy* L_energy   (Var(H) ~= 0)   <- NEW

Applicability: most useful for long t_max, where the standard PINN
accumulates energy drift. Set --lambda-energy 0 if friction is present.

Usage: import this module from the Jupyter notebook — see the
"Conservative PINN (cPINN)" section of Double_Pendulum_Lab.ipynb.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import tensorflow as tf

from physics import DEFAULT_PHYS_PARAMS, angular_accelerations, hamiltonian
from train_mlp import build_mlp, save_mlp, load_mlp   # noqa: F401

# ---------------------------------------------------------------------------
# Physical constants (default)
# ---------------------------------------------------------------------------

PHYS = DEFAULT_PHYS_PARAMS

IC_DEFAULT = dict(
    theta1_0=np.pi / 3,
    omega1_0=0.0,
    theta2_0=np.pi / 4,
    omega2_0=0.0,
)

# ---------------------------------------------------------------------------
# ODE residual (identical to train_pinn.py), plus theta/omega for the
# Hamiltonian term
# ---------------------------------------------------------------------------

def physics_residual(
    model: tf.keras.Model,
    t_coll: tf.Tensor,
    m1: float = 1.0,
    m2: float = 1.0,
    L1: float = 1.0,
    L2: float = 1.0,
    g: float = 9.81,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """ODE residuals plus angular velocities/angles, for the Hamiltonian term.

    Returns
    -------
    res_omega1, res_omega2, omega1, omega2, theta1, theta2
    """
    t_coll = tf.cast(tf.reshape(t_coll, (-1, 1)), tf.float32)

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t_coll)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(t_coll)
            u      = model(t_coll, training=True)
            theta1 = u[:, 0:1]
            theta2 = u[:, 1:2]
        omega1 = tape1.gradient(theta1, t_coll)
        omega2 = tape1.gradient(theta2, t_coll)

    alpha1 = tape2.gradient(omega1, t_coll)
    alpha2 = tape2.gradient(omega2, t_coll)
    del tape1, tape2

    domega1_dt, domega2_dt = angular_accelerations(
        theta1, theta2, omega1, omega2, m1=m1, m2=m2, L1=L1, L2=L2, g=g
    )

    return (alpha1 - domega1_dt, alpha2 - domega2_dt,
            omega1, omega2, theta1, theta2)

# ---------------------------------------------------------------------------
# Composite loss (cPINN)
# ---------------------------------------------------------------------------

def cpinn_loss(
    model: tf.keras.Model,
    t_data: tf.Tensor,
    u_data: tf.Tensor,
    t_coll: tf.Tensor,
    t_ic: tf.Tensor,
    u_ic: tf.Tensor,
    lambda_data: float = 1.0,
    lambda_phys: float = 1.0,
    lambda_ic: float = 10.0,
    lambda_energy: float = 1.0,
    **phys_params: float,
) -> dict[str, tf.Tensor]:
    """cPINN loss: data + physics + IC + energy conservation."""
    pred_data = model(t_data, training=True)
    l_data    = tf.reduce_mean(tf.square(pred_data - u_data))

    pred_ic = model(t_ic, training=True)
    l_ic    = tf.reduce_mean(tf.square(pred_ic - u_ic))

    res1, res2, omega1, omega2, theta1, theta2 = physics_residual(
        model, t_coll, **phys_params
    )
    l_phys = tf.reduce_mean(tf.square(res1)) + tf.reduce_mean(tf.square(res2))

    # Variance of the Hamiltonian — should be 0 for a conservative system
    H        = hamiltonian(theta1, theta2, omega1, omega2, **phys_params)
    l_energy = tf.math.reduce_variance(H)

    total = (lambda_data     * l_data
             + lambda_phys   * l_phys
             + lambda_ic     * l_ic
             + lambda_energy * l_energy)

    return {
        "loss_total":  total,
        "loss_data":   l_data,
        "loss_phys":   l_phys,
        "loss_ic":     l_ic,
        "loss_energy": l_energy,
    }

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_cpinn(
    model: tf.keras.Model,
    t_train: np.ndarray,
    y_train: np.ndarray,
    ic: dict[str, float] | None = None,
    n_colloc: int = 1000,
    t_max: float = 10.0,
    epochs: int = 10_000,
    lr: float = 1e-3,
    lambda_data: float = 1.0,
    lambda_phys: float = 1.0,
    lambda_ic: float = 10.0,
    lambda_energy: float = 1.0,
    log_every: int = 500,
    phys_params: dict | None = None,
) -> list[dict[str, float]]:
    """Train the cPINN with physics + data + IC + energy conservation losses.

    Parameters
    ----------
    lambda_energy : weight of the energy term. Use 0 if friction is present.
    """
    if ic is None:
        ic = IC_DEFAULT
    if phys_params is None:
        phys_params = PHYS

    t_data = tf.constant(t_train.reshape(-1, 1), dtype=tf.float32)
    u_data = tf.constant(y_train[:, :2],         dtype=tf.float32)
    t_ic   = tf.constant([[0.0]],                dtype=tf.float32)
    u_ic   = tf.constant([[ic["theta1_0"], ic["theta2_0"]]], dtype=tf.float32)
    t_coll = tf.constant(
        np.linspace(0.0, t_max, n_colloc).reshape(-1, 1), dtype=tf.float32
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    history: list[dict[str, float]] = []

    @tf.function
    def train_step() -> dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            losses = cpinn_loss(
                model, t_data, u_data, t_coll, t_ic, u_ic,
                lambda_data=lambda_data,
                lambda_phys=lambda_phys,
                lambda_ic=lambda_ic,
                lambda_energy=lambda_energy,
                **phys_params,
            )
        grads = tape.gradient(losses["loss_total"], model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return losses

    for epoch in range(1, epochs + 1):
        losses = train_step()
        if epoch % log_every == 0 or epoch == 1:
            entry = {k: float(v) for k, v in losses.items()}
            entry["epoch"] = epoch
            history.append(entry)
            print(
                f"  Epoch {epoch:6d}  "
                f"total={entry['loss_total']:.4e}  "
                f"data={entry['loss_data']:.4e}  "
                f"phys={entry['loss_phys']:.4e}  "
                f"ic={entry['loss_ic']:.4e}  "
                f"energy={entry['loss_energy']:.4e}"
            )

    return history


def save_cpinn(model: tf.keras.Model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    print(f"cPINN saved -> {path}")


def load_cpinn(path: str | Path) -> tf.keras.Model:
    m = tf.keras.models.load_model(str(path))
    print(f"cPINN loaded <- {path}")
    return m
