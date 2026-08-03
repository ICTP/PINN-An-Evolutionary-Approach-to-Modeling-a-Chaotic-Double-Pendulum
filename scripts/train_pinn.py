"""
train_pinn.py
=============
Stage 3 — Physics-Informed Learning

Trains a Physics-Informed Neural Network (PINN) for a fixed initial
condition.  The same MLP architecture is used (f(t) -> (theta1, theta2)),
but the training objective now includes three terms:

  L_total = lambda_data  * L_data
           + lambda_phys * L_physics
           + lambda_ic   * L_ic

  L_data    = MSE(u_pred(t_data),  u_data)
  L_physics = residual of the double-pendulum ODEs, evaluated at
              collocation points via automatic differentiation
  L_ic      = MSE(u_pred(0), u_0)

The physics residual is computed with tf.GradientTape:
  d(theta1_pred)/dt  must equal  omega1_pred
  d(omega1_pred)/dt  must equal  N1/D   (from the ODE)
  (same for theta2, omega2)

Since the network only outputs (theta1, theta2), angular velocities
are obtained as d(theta_i)/dt via automatic differentiation.

Fixed training IC: theta1_0 = pi/3, theta2_0 = pi/4, omega1_0 = omega2_0 = 0

Usage: import this module from the Jupyter notebook — see the
"Physics-Informed Learning (PINN)" section of Double_Pendulum_Lab.ipynb.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import tensorflow as tf

# Reuse the shared architecture from train_mlp
from train_mlp import build_mlp, save_mlp, load_mlp   # noqa: F401
from physics import DEFAULT_PHYS_PARAMS, angular_accelerations


# ---------------------------------------------------------------------------
# Physical constants (default)
# ---------------------------------------------------------------------------

PHYS = DEFAULT_PHYS_PARAMS

# Fixed IC for Stage 3
IC_DEFAULT = dict(
    theta1_0=np.pi / 3,
    omega1_0=0.0,
    theta2_0=np.pi / 4,
    omega2_0=0.0,
)


# ---------------------------------------------------------------------------
# ODE residual via automatic differentiation
# ---------------------------------------------------------------------------

def physics_residual(
    model: tf.keras.Model,
    t_coll: tf.Tensor,
    m1: float = 1.0,
    m2: float = 1.0,
    L1: float = 1.0,
    L2: float = 1.0,
    g: float = 9.81,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Compute the ODE residuals at collocation points.

    Uses nested GradientTape to get first-order time derivatives of the
    network outputs (theta1, theta2), then computes second-order
    derivatives (angular accelerations) by differentiating those.

    Returns
    -------
    res_omega1 : residual of domega1/dt = N1/D
    res_omega2 : residual of domega2/dt = N2/D
    """
    t_coll = tf.cast(tf.reshape(t_coll, (-1, 1)), tf.float32)

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t_coll)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(t_coll)
            u = model(t_coll, training=True)      # (N, 2): [theta1, theta2]
            theta1 = u[:, 0:1]
            theta2 = u[:, 1:2]

        # First derivatives (angular velocities)
        omega1 = tape1.gradient(theta1, t_coll)   # dtheta1/dt
        omega2 = tape1.gradient(theta2, t_coll)   # dtheta2/dt

    # Second derivatives (angular accelerations)
    alpha1 = tape2.gradient(omega1, t_coll)        # d²theta1/dt²
    alpha2 = tape2.gradient(omega2, t_coll)        # d²theta2/dt²

    del tape1, tape2

    domega1_dt, domega2_dt = angular_accelerations(
        theta1, theta2, omega1, omega2, m1=m1, m2=m2, L1=L1, L2=L2, g=g
    )

    res_omega1 = alpha1 - domega1_dt
    res_omega2 = alpha2 - domega2_dt

    return res_omega1, res_omega2


# ---------------------------------------------------------------------------
# Composite PINN loss (one gradient-tape call)
# ---------------------------------------------------------------------------

def pinn_loss(
    model: tf.keras.Model,
    t_data: tf.Tensor,
    u_data: tf.Tensor,
    t_coll: tf.Tensor,
    t_ic: tf.Tensor,
    u_ic: tf.Tensor,
    lambda_data: float = 1.0,
    lambda_phys: float = 1.0,
    lambda_ic: float = 10.0,
    **phys_params: float,
) -> dict[str, tf.Tensor]:
    """Return a dictionary with individual and total loss terms.

    Parameters
    ----------
    t_data : (N_d, 1)  time points with labeled data
    u_data : (N_d, 2)  [theta1, theta2] labels
    t_coll : (N_c, 1)  collocation points for physics constraint
    t_ic   : (1, 1)    time = 0 for IC constraint
    u_ic   : (1, 2)    [theta1_0, theta2_0] initial values
    """
    # Data loss
    pred_data = model(t_data, training=True)
    l_data = tf.reduce_mean(tf.square(pred_data - u_data))

    # IC loss
    pred_ic = model(t_ic, training=True)
    l_ic = tf.reduce_mean(tf.square(pred_ic - u_ic))

    # Physics loss
    res1, res2 = physics_residual(model, t_coll, **phys_params)
    l_phys = tf.reduce_mean(tf.square(res1)) + tf.reduce_mean(tf.square(res2))

    total = lambda_data * l_data + lambda_phys * l_phys + lambda_ic * l_ic

    return {
        "loss_total": total,
        "loss_data":  l_data,
        "loss_phys":  l_phys,
        "loss_ic":    l_ic,
    }


# ---------------------------------------------------------------------------
# Training loop (custom — avoids recompilation on every step)
# ---------------------------------------------------------------------------

def train_pinn(
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
    log_every: int = 500,
    phys_params: dict | None = None,
) -> list[dict[str, float]]:
    """Train the PINN with physics + data + IC losses.

    Parameters
    ----------
    t_train : (N,)   time points from RK4
    y_train : (N, 2) [theta1, theta2] from RK4
    ic      : initial condition dict (theta1_0, omega1_0, theta2_0, omega2_0)

    Returns
    -------
    list of dict — loss history (one entry per log step)
    """
    if ic is None:
        ic = IC_DEFAULT
    if phys_params is None:
        phys_params = PHYS

    # ── Prepare tensors ───────────────────────────────────────────────────
    t_data = tf.constant(t_train.reshape(-1, 1), dtype=tf.float32)
    u_data = tf.constant(y_train[:, :2], dtype=tf.float32)

    t_ic = tf.constant([[0.0]], dtype=tf.float32)
    u_ic = tf.constant(
        [[ic["theta1_0"], ic["theta2_0"]]], dtype=tf.float32
    )

    # Collocation points: uniform in [0, t_max]
    t_coll = tf.constant(
        np.linspace(0.0, t_max, n_colloc).reshape(-1, 1), dtype=tf.float32
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    history: list[dict[str, float]] = []

    @tf.function
    def train_step() -> dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            losses = pinn_loss(
                model, t_data, u_data, t_coll, t_ic, u_ic,
                lambda_data=lambda_data,
                lambda_phys=lambda_phys,
                lambda_ic=lambda_ic,
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
                f"ic={entry['loss_ic']:.4e}"
            )

    return history


def save_pinn(model: tf.keras.Model, path: str | Path) -> None:
    """Save the trained PINN."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    print(f"PINN saved -> {path}")


def load_pinn(path: str | Path) -> tf.keras.Model:
    """Load a previously saved PINN."""
    m = tf.keras.models.load_model(str(path))
    print(f"PINN loaded <- {path}")
    return m
