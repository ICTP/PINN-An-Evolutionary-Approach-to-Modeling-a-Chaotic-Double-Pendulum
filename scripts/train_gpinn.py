"""
train_gpinn.py
==============
Gradient-enhanced PINN (gPINN) — Double Pendulum

Extends the standard PINN by also penalizing the time derivative of the
physics residuals. If R(t) = 0 is the standard constraint, gPINN adds
dR/dt = 0, which gives the optimizer extra gradient information and
speeds up convergence.

The double pendulum has two Euler-Lagrange residuals:

  R1(t) = alpha1 - f1(theta1, theta2, omega1, omega2)   (angular accel. 1)
  R2(t) = alpha2 - f2(theta1, theta2, omega1, omega2)   (angular accel. 2)

gPINN adds:

  L_grad = ||dR1/dt||^2 + ||dR2/dt||^2    <- NEW (3rd-order autograd)

Total loss:
  L_total = lambda_data * L_data
           + lambda_phys * L_physics
           + lambda_ic   * L_ic
           + lambda_grad * L_grad           <- NEW

Note: dR/dt requires 3rd-order differentiation in TensorFlow (three
nested GradientTapes), which increases the per-epoch compute cost —
offset by needing fewer epochs to converge.

Usage: import this module from the Jupyter notebook — see the
"Gradient-enhanced PINN (gPINN)" section of Double_Pendulum_Lab.ipynb.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import tensorflow as tf

from physics import DEFAULT_PHYS_PARAMS, angular_accelerations
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
# Physics residual and its time derivative (dR/dt)
# ---------------------------------------------------------------------------

def physics_and_gradient_residual(
    model: tf.keras.Model,
    t_coll: tf.Tensor,
    m1: float = 1.0,
    m2: float = 1.0,
    L1: float = 1.0,
    L2: float = 1.0,
    g: float = 9.81,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Euler-Lagrange residuals and their time derivatives.

    Uses three nested GradientTapes to reach the third derivative of
    theta with respect to t.

    Returns
    -------
    res1, res2         : standard residuals R1, R2   (same as PINN)
    dres1_dt, dres2_dt : time derivatives dR1/dt, dR2/dt   (new in gPINN)
    """
    t_coll = tf.cast(tf.reshape(t_coll, (-1, 1)), tf.float32)

    with tf.GradientTape(persistent=True) as tape3:
        tape3.watch(t_coll)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(t_coll)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(t_coll)
                u      = model(t_coll, training=True)
                theta1 = u[:, 0:1]
                theta2 = u[:, 1:2]
            omega1 = tape1.gradient(theta1, t_coll)   # dtheta1/dt
            omega2 = tape1.gradient(theta2, t_coll)   # dtheta2/dt
        alpha1 = tape2.gradient(omega1, t_coll)       # d2theta1/dt2
        alpha2 = tape2.gradient(omega2, t_coll)       # d2theta2/dt2

        domega1_dt, domega2_dt = angular_accelerations(
            theta1, theta2, omega1, omega2, m1=m1, m2=m2, L1=L1, L2=L2, g=g
        )

        # Standard residuals R1, R2
        res1 = alpha1 - domega1_dt
        res2 = alpha2 - domega2_dt

    # Time derivatives of the residuals: dR/dt
    dres1_dt = tape3.gradient(res1, t_coll)
    dres2_dt = tape3.gradient(res2, t_coll)

    del tape1, tape2, tape3

    return res1, res2, dres1_dt, dres2_dt

# ---------------------------------------------------------------------------
# Composite loss (gPINN)
# ---------------------------------------------------------------------------

def gpinn_loss(
    model: tf.keras.Model,
    t_data: tf.Tensor,
    u_data: tf.Tensor,
    t_coll: tf.Tensor,
    t_ic: tf.Tensor,
    u_ic: tf.Tensor,
    lambda_data: float = 1.0,
    lambda_phys: float = 1.0,
    lambda_ic: float = 10.0,
    lambda_grad: float = 0.1,
    **phys_params: float,
) -> dict[str, tf.Tensor]:
    """gPINN loss: data + physics + IC + residual gradient.

    Parameters
    ----------
    lambda_grad : weight of the dR/dt term. Small values (0.01-0.5) are
                  usually enough — the term already carries a lot of signal.
    """
    pred_data = model(t_data, training=True)
    l_data    = tf.reduce_mean(tf.square(pred_data - u_data))

    pred_ic = model(t_ic, training=True)
    l_ic    = tf.reduce_mean(tf.square(pred_ic - u_ic))

    res1, res2, dres1_dt, dres2_dt = physics_and_gradient_residual(
        model, t_coll, **phys_params
    )
    l_phys = tf.reduce_mean(tf.square(res1)) + tf.reduce_mean(tf.square(res2))

    # Penalize the residual gradient
    l_grad = (tf.reduce_mean(tf.square(dres1_dt))
              + tf.reduce_mean(tf.square(dres2_dt)))

    total = (lambda_data * l_data
             + lambda_phys * l_phys
             + lambda_ic   * l_ic
             + lambda_grad * l_grad)

    return {
        "loss_total": total,
        "loss_data":  l_data,
        "loss_phys":  l_phys,
        "loss_ic":    l_ic,
        "loss_grad":  l_grad,
    }

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_gpinn(
    model: tf.keras.Model,
    t_train: np.ndarray,
    y_train: np.ndarray,
    ic: dict[str, float] | None = None,
    n_colloc: int = 1000,
    t_max: float = 10.0,
    epochs: int = 8_000,
    lr: float = 1e-3,
    lambda_data: float = 1.0,
    lambda_phys: float = 1.0,
    lambda_ic: float = 10.0,
    lambda_grad: float = 0.1,
    log_every: int = 500,
    phys_params: dict | None = None,
) -> list[dict[str, float]]:
    """Train the gPINN with physics + residual gradient + data + IC losses.

    Parameters
    ----------
    lambda_grad : weight of dR/dt. Recommended: 0.01-0.5.
                  Large values can destabilize training.
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

    # NOTE: no @tf.function here — 3rd-order GradientTapes are incompatible
    # with static tracing on some TensorFlow versions.
    for epoch in range(1, epochs + 1):
        with tf.GradientTape() as outer_tape:
            losses = gpinn_loss(
                model, t_data, u_data, t_coll, t_ic, u_ic,
                lambda_data=lambda_data,
                lambda_phys=lambda_phys,
                lambda_ic=lambda_ic,
                lambda_grad=lambda_grad,
                **phys_params,
            )
        grads = outer_tape.gradient(losses["loss_total"], model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

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
                f"grad={entry['loss_grad']:.4e}"
            )

    return history


def save_gpinn(model: tf.keras.Model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    print(f"gPINN saved -> {path}")


def load_gpinn(path: str | Path) -> tf.keras.Model:
    m = tf.keras.models.load_model(str(path))
    print(f"gPINN loaded <- {path}")
    return m
