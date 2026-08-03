"""
train_xpinn.py
==============
Extended PINN (xPINN) — Double Pendulum

Applies temporal domain decomposition: the interval [0, t_max] is split
into N (non-overlapping) sub-domains, each with its own small MLP.
Interface losses enforce continuity of theta and omega at the junction
points:

  For each interface at t_i (i = 1..N-1):
    theta1_i(t_i) = theta1_{i+1}(t_i)     (angle continuity)
    omega1_i(t_i) = omega1_{i+1}(t_i)     (velocity continuity)
    (same for theta2, omega2)

Total loss:
  L_total = lambda_data * L_data           (MSE against RK4 data)
           + lambda_phys * L_physics         (EL residual per sub-domain)
           + lambda_ic   * L_ic             (initial condition, sub-domain 0)
           + lambda_intf * L_interface      (continuity at interfaces)

Advantage: each network only has to learn a short segment, which gives
better capture of the double pendulum's chaotic dynamics over large
t_max.

Usage: import this module from the Jupyter notebook — see the
"Extended PINN (xPINN)" section of Double_Pendulum_Lab.ipynb.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import tensorflow as tf

from physics import DEFAULT_PHYS_PARAMS, angular_accelerations
from train_mlp import build_mlp, save_mlp   # noqa: F401


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
# Sub-network construction
# ---------------------------------------------------------------------------

def build_subnetworks(n: int, n_hidden: int = 3, n_units: int = 32) -> list:
    """Create N small, identical MLPs — one per sub-domain.

    Each sub-network is a standard MLP f(t) -> (theta1, theta2). Smaller
    networks are used than the global PINN because each one only needs
    to learn a short segment of the time domain.

    Parameters
    ----------
    n        : number of sub-domains
    n_hidden : hidden layers per sub-network (default 3, vs 4 for the global PINN)
    n_units  : neurons per layer (default 32, vs 64 for the global PINN)
    """
    return [build_mlp(n_hidden=n_hidden, n_units=n_units) for _ in range(n)]

# ---------------------------------------------------------------------------
# Physics residual (per sub-network, over its own time segment)
# ---------------------------------------------------------------------------

def physics_residual_sub(
    model: tf.keras.Model,
    t_coll: tf.Tensor,
    m1: float = 1.0,
    m2: float = 1.0,
    L1: float = 1.0,
    L2: float = 1.0,
    g: float = 9.81,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Double-pendulum ODE residual within a single sub-domain.

    Identical to the one in train_pinn.py; returns (res_omega1, res_omega2).
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

    return alpha1 - domega1_dt, alpha2 - domega2_dt

# ---------------------------------------------------------------------------
# Angular velocities (needed for the interface conditions)
# ---------------------------------------------------------------------------

def _omega_at(model: tf.keras.Model, t_val: float) -> tuple[tf.Tensor, tf.Tensor]:
    """Evaluate omega1, omega2 = dtheta/dt at a given instant t_val."""
    t_pt = tf.constant([[t_val]], dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_pt)
        u      = model(t_pt, training=True)
        theta1 = u[:, 0:1]
        theta2 = u[:, 1:2]
    omega1 = tape.gradient(theta1, t_pt)
    omega2 = tape.gradient(theta2, t_pt)
    del tape
    return omega1, omega2

# ---------------------------------------------------------------------------
# Interface loss between sub-network i and sub-network i+1
# ---------------------------------------------------------------------------

def interface_loss(
    model_left: tf.keras.Model,
    model_right: tf.keras.Model,
    t_intf: float,
) -> tf.Tensor:
    """Enforce continuity of theta and omega at the interface point t_intf.

    L_intf = (theta1_L - theta1_R)^2 + (theta2_L - theta2_R)^2
           + (omega1_L - omega1_R)^2 + (omega2_L - omega2_R)^2
    """
    t_pt = tf.constant([[t_intf]], dtype=tf.float32)

    u_l = model_left(t_pt,  training=True)   # (1, 2)
    u_r = model_right(t_pt, training=True)

    # Angle continuity
    l_theta = tf.reduce_sum(tf.square(u_l - u_r))

    # Velocity continuity
    w1_l, w2_l = _omega_at(model_left,  t_intf)
    w1_r, w2_r = _omega_at(model_right, t_intf)
    l_omega = tf.square(w1_l - w1_r) + tf.square(w2_l - w2_r)
    l_omega = tf.reduce_sum(l_omega)

    return l_theta + l_omega

# ---------------------------------------------------------------------------
# Total xPINN loss
# ---------------------------------------------------------------------------

def xpinn_loss(
    subnetworks: list[tf.keras.Model],
    t_data_list: list[tf.Tensor],
    u_data_list: list[tf.Tensor],
    t_coll_list: list[tf.Tensor],
    t_interfaces: list[float],
    ic: dict[str, float],
    lambda_data: float = 1.0,
    lambda_phys: float = 1.0,
    lambda_ic: float = 10.0,
    lambda_intf: float = 10.0,
    **phys_params: float,
) -> dict[str, tf.Tensor]:
    """Composite xPINN loss across all sub-domains.

    Parameters
    ----------
    subnetworks  : list of N MLP models
    t_data_list  : data time points per sub-domain
    u_data_list  : [theta1, theta2] values per sub-domain
    t_coll_list  : collocation points per sub-domain
    t_interfaces : interface instants (len = N-1)
    """
    l_data = tf.constant(0.0)
    l_phys = tf.constant(0.0)
    l_ic   = tf.constant(0.0)
    l_intf = tf.constant(0.0)

    for i, model in enumerate(subnetworks):
        # Data
        if len(t_data_list[i]) > 0:
            pred = model(t_data_list[i], training=True)
            l_data += tf.reduce_mean(tf.square(pred - u_data_list[i]))

        # Physics
        r1, r2 = physics_residual_sub(model, t_coll_list[i], **phys_params)
        l_phys += tf.reduce_mean(tf.square(r1)) + tf.reduce_mean(tf.square(r2))

        # IC: only the first sub-network
        if i == 0:
            t_ic = tf.constant([[0.0]], dtype=tf.float32)
            u_ic = tf.constant(
                [[ic["theta1_0"], ic["theta2_0"]]], dtype=tf.float32
            )
            pred_ic = model(t_ic, training=True)
            l_ic = tf.reduce_mean(tf.square(pred_ic - u_ic))

    # Interfaces
    for i, t_intf in enumerate(t_interfaces):
        l_intf += interface_loss(subnetworks[i], subnetworks[i + 1], t_intf)

    n = float(len(subnetworks))
    total = (lambda_data  * l_data / n
             + lambda_phys * l_phys / n
             + lambda_ic   * l_ic
             + lambda_intf * l_intf)

    return {
        "loss_total":     total,
        "loss_data":      l_data / n,
        "loss_phys":      l_phys / n,
        "loss_ic":        l_ic,
        "loss_interface": l_intf,
    }

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_xpinn(
    subnetworks: list[tf.keras.Model],
    t_train: np.ndarray,
    y_train: np.ndarray,
    ic: dict[str, float] | None = None,
    n_colloc: int = 200,
    t_max: float = 10.0,
    epochs: int = 10_000,
    lr: float = 1e-3,
    lambda_data: float = 1.0,
    lambda_phys: float = 1.0,
    lambda_ic: float = 10.0,
    lambda_intf: float = 10.0,
    log_every: int = 500,
    phys_params: dict | None = None,
) -> list[dict[str, float]]:
    """Jointly train the N sub-networks of the xPINN.

    Parameters
    ----------
    subnetworks : list of N models (one per sub-domain)
    n_colloc    : collocation points *per sub-domain*
    """
    if ic is None:
        ic = IC_DEFAULT
    if phys_params is None:
        phys_params = PHYS

    n = len(subnetworks)
    boundaries = np.linspace(0.0, t_max, n + 1)   # [t0, t1, ..., tN]
    t_interfaces = list(boundaries[1:-1])            # interface instants

    # Split data and collocation points by sub-domain
    t_data_list: list[tf.Tensor] = []
    u_data_list: list[tf.Tensor] = []
    t_coll_list: list[tf.Tensor] = []

    for i in range(n):
        t0_i, t1_i = boundaries[i], boundaries[i + 1]

        mask = (t_train >= t0_i) & (t_train <= t1_i)
        t_seg = t_train[mask].reshape(-1, 1).astype(np.float32)
        y_seg = y_train[mask, :2].astype(np.float32)
        t_data_list.append(tf.constant(t_seg))
        u_data_list.append(tf.constant(y_seg))

        tc = np.linspace(t0_i, t1_i, n_colloc).reshape(-1, 1).astype(np.float32)
        t_coll_list.append(tf.constant(tc))

    # A single joint optimizer for all parameters
    all_vars = []
    for m in subnetworks:
        all_vars.extend(m.trainable_variables)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        with tf.GradientTape() as tape:
            losses = xpinn_loss(
                subnetworks,
                t_data_list, u_data_list, t_coll_list,
                t_interfaces, ic,
                lambda_data=lambda_data,
                lambda_phys=lambda_phys,
                lambda_ic=lambda_ic,
                lambda_intf=lambda_intf,
                **phys_params,
            )
        grads = tape.gradient(losses["loss_total"], all_vars)
        optimizer.apply_gradients(zip(grads, all_vars))

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
                f"interface={entry['loss_interface']:.4e}"
            )

    return history


def predict_xpinn(
    subnetworks: list[tf.keras.Model],
    t_query: np.ndarray,
    t_max: float,
) -> np.ndarray:
    """xPINN prediction over an array of query instants t_query.

    Each instant is routed to the sub-network that covers its segment.

    Returns
    -------
    pred : (N, 2)  [theta1, theta2] for each instant
    """
    n = len(subnetworks)
    boundaries = np.linspace(0.0, t_max, n + 1)
    pred = np.zeros((len(t_query), 2), dtype=np.float32)

    for i, model in enumerate(subnetworks):
        t0_i, t1_i = boundaries[i], boundaries[i + 1]
        if i < n - 1:
            mask = (t_query >= t0_i) & (t_query < t1_i)
        else:
            mask = (t_query >= t0_i) & (t_query <= t1_i)
        if mask.any():
            t_in = t_query[mask].reshape(-1, 1).astype(np.float32)
            pred[mask] = model(t_in, training=False).numpy()

    return pred


def save_xpinn(subnetworks: list[tf.keras.Model], base_path: str | Path) -> None:
    """Save each sub-network as base_path_0.keras, base_path_1.keras, ..."""
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(subnetworks):
        p = base.parent / f"{base.stem}_{i}.keras"
        m.save(str(p))
    print(f"xPINN ({len(subnetworks)} sub-networks) saved -> {base.parent}/")


def load_xpinn(base_path: str | Path, n: int) -> list[tf.keras.Model]:
    """Load N sub-networks from base_path_0.keras ... base_path_{N-1}.keras."""
    base = Path(base_path)
    subnetworks = []
    for i in range(n):
        p = base.parent / f"{base.stem}_{i}.keras"
        subnetworks.append(tf.keras.models.load_model(str(p)))
    print(f"xPINN loaded <- {base.parent}/  ({n} sub-networks)")
    return subnetworks
