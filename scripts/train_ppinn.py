"""
train_ppinn.py
==============
Stage 4 — Parameterized Physics-Informed Learning

Extends the PINN to accept initial conditions as additional inputs:

    f(t, theta1_0, omega1_0, theta2_0, omega2_0) -> (theta1(t), theta2(t))

This allows one trained model to solve *any* initial value problem
within the training domain without retraining.

Architecture:
  Input  : 5  (t, theta1_0, omega1_0, theta2_0, omega2_0)
  Hidden : 4 x 64, tanh
  Output : 2  (theta1, theta2), linear

Loss:
  L_total = lambda_data  * L_data
           + lambda_phys * L_physics
           + lambda_ic   * L_ic

The IC loss now enforces:
    f(0, ic) == ic[:, [theta1_0, theta2_0]]

for all ICs in the mini-batch.

Usage: import this module from the Jupyter notebook — see the
"Parametric PINN (PPINN)" section of Double_Pendulum_Lab.ipynb.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import tensorflow as tf

from physics import DEFAULT_PHYS_PARAMS, angular_accelerations

PHYS = DEFAULT_PHYS_PARAMS


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

def build_ppinn(n_hidden: int = 4, n_units: int = 64,
                activation: str = "tanh") -> tf.keras.Model:
    """Return the Parametric PINN (Input=5, Output=2)."""
    inputs = tf.keras.Input(shape=(5,), name="t_ic_input")
    x = inputs
    for i in range(n_hidden):
        x = tf.keras.layers.Dense(n_units, activation=activation,
                                   name=f"hidden_{i+1}")(x)
    outputs = tf.keras.layers.Dense(2, activation="linear", name="output")(x)
    return tf.keras.Model(inputs, outputs, name="ParametricPINN")


# ---------------------------------------------------------------------------
# ODE residual for the PPINN
# ---------------------------------------------------------------------------

def ppinn_physics_residual(
    model: tf.keras.Model,
    t_coll: tf.Tensor,
    ics_coll: tf.Tensor,
    m1: float = 1.0, m2: float = 1.0,
    L1: float = 1.0, L2: float = 1.0,
    g:  float = 9.81,
) -> tuple[tf.Tensor, tf.Tensor]:
    """ODE residuals for the Parametric PINN at collocation points.

    Parameters
    ----------
    t_coll   : (N_c, 1)  collocation times — the only differentiation variable
    ics_coll : (N_c, 4)  [theta1_0, omega1_0, theta2_0, omega2_0] for each
                          collocation point (constant w.r.t. t)
    """
    t_coll   = tf.cast(tf.reshape(t_coll,   (-1, 1)), tf.float32)
    ics_coll = tf.cast(ics_coll, tf.float32)

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t_coll)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(t_coll)
            inp = tf.concat([t_coll, ics_coll], axis=1)  # (N_c, 5)
            u = model(inp, training=True)
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

    res_omega1 = alpha1 - domega1_dt
    res_omega2 = alpha2 - domega2_dt

    return res_omega1, res_omega2


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------

def ppinn_loss(
    model: tf.keras.Model,
    t_data: tf.Tensor,
    u_data: tf.Tensor,
    ics_data: tf.Tensor,
    t_coll: tf.Tensor,
    ics_coll: tf.Tensor,
    lambda_data: float = 1.0,
    lambda_phys: float = 1.0,
    lambda_ic:   float = 10.0,
    **phys_params: float,
) -> dict[str, tf.Tensor]:
    """Compute data, physics, and IC loss terms for the PPINN.

    Parameters
    ----------
    t_data    : (N_d, 1)  time points with data labels
    u_data    : (N_d, 2)  [theta1, theta2] labels
    ics_data  : (N_d, 4)  IC repeated for each data point
    t_coll    : (N_c, 1)  collocation times
    ics_coll  : (N_c, 4)  IC for each collocation point
    """
    # Data loss
    inp_data = tf.concat([t_data, ics_data], axis=1)
    pred_data = model(inp_data, training=True)
    l_data = tf.reduce_mean(tf.square(pred_data - u_data))

    # IC loss: f(0, ic) == [theta1_0, theta2_0]
    t_zero   = tf.zeros_like(t_data[:1])
    ic_batch = ics_data[:1]
    inp_ic   = tf.concat([t_zero, ic_batch], axis=1)
    pred_ic  = model(inp_ic, training=True)
    target_ic = tf.gather(ic_batch, [0, 2], axis=1)  # theta1_0, theta2_0
    l_ic = tf.reduce_mean(tf.square(pred_ic - target_ic))

    # Physics loss
    res1, res2 = ppinn_physics_residual(model, t_coll, ics_coll, **phys_params)
    l_phys = tf.reduce_mean(tf.square(res1)) + tf.reduce_mean(tf.square(res2))

    total = lambda_data * l_data + lambda_phys * l_phys + lambda_ic * l_ic

    return {
        "loss_total": total,
        "loss_data":  l_data,
        "loss_phys":  l_phys,
        "loss_ic":    l_ic,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_ppinn(
    model: tf.keras.Model,
    dataset: dict[str, np.ndarray],
    n_colloc_per_traj: int = 200,
    epochs: int = 20_000,
    lr: float = 1e-3,
    batch_trajs: int = 8,
    lambda_data: float = 1.0,
    lambda_phys: float = 1.0,
    lambda_ic:   float = 10.0,
    log_every: int = 1000,
    phys_params: dict | None = None,
) -> list[dict[str, float]]:
    """Train the Parametric PINN over multiple trajectories.

    Parameters
    ----------
    dataset : output of rk4_simulator.generate_dataset()
              keys: 't', 'trajectories', 'ics'
    """
    if phys_params is None:
        phys_params = PHYS

    t_arr   = dataset["t"].astype(np.float32)            # (N_steps,)
    trajs   = dataset["trajectories"].astype(np.float32) # (n_traj, N_steps, 4)
    ics_all = dataset["ics"].astype(np.float32)          # (n_traj, 4)
    n_traj, n_steps, _ = trajs.shape
    t_max = float(t_arr[-1])

    optimizer = tf.keras.optimizers.Adam(lr)
    history: list[dict[str, float]] = []
    rng = np.random.default_rng(0)

    @tf.function
    def train_step(t_d, u_d, ic_d, t_c, ic_c):
        with tf.GradientTape() as tape:
            losses = ppinn_loss(
                model, t_d, u_d, ic_d, t_c, ic_c,
                lambda_data=lambda_data,
                lambda_phys=lambda_phys,
                lambda_ic=lambda_ic,
                **phys_params,
            )
        grads = tape.gradient(losses["loss_total"], model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return losses

    for epoch in range(1, epochs + 1):
        # Sample a mini-batch of trajectories
        idx = rng.choice(n_traj, size=min(batch_trajs, n_traj), replace=False)

        # Build training tensors for this batch
        t_list, u_list, ic_list = [], [], []
        for i in idx:
            t_list.append(t_arr)
            u_list.append(trajs[i][:, [0, 2]])      # theta1, theta2  shape (N_steps, 2)
            ic_list.append(
                np.tile(ics_all[i], (n_steps, 1))  # repeat IC for all steps
            )

        t_data   = tf.constant(np.concatenate(t_list).reshape(-1, 1))
        u_data   = tf.constant(np.concatenate(u_list))
        ics_data = tf.constant(np.concatenate(ic_list))

        # Collocation: sample randomly across all ICs in batch
        t_coll_np  = rng.uniform(0, t_max, (n_colloc_per_traj * len(idx), 1)).astype(np.float32)
        ic_coll_np = ics_all[rng.choice(n_traj, size=len(t_coll_np), replace=True)]
        t_coll   = tf.constant(t_coll_np)
        ics_coll = tf.constant(ic_coll_np)

        losses = train_step(t_data, u_data, ics_data, t_coll, ics_coll)

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


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_ppinn(model: tf.keras.Model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    print(f"Parametric PINN saved -> {path}")


def load_ppinn(path: str | Path) -> tf.keras.Model:
    m = tf.keras.models.load_model(str(path))
    print(f"Parametric PINN loaded <- {path}")
    return m
