"""
physics.py
==========
Shared Physical Model — Double Pendulum

Centralizes the equations-of-motion right-hand side and the Hamiltonian
(total mechanical energy) so the formulas are defined exactly once and
reused by every training script (train_pinn, train_cpinn, train_gpinn,
train_xpinn, train_ppinn) and by metrics.py's physics-residual check.

The numpy version used by the RK4 solver lives in rk4_simulator.py; this
module provides the TensorFlow equivalent so it can be embedded inside a
tf.GradientTape for physics-informed training.
"""

from __future__ import annotations

import tensorflow as tf

DEFAULT_PHYS_PARAMS: dict[str, float] = {
    "m1": 1.0,   # kg
    "m2": 1.0,   # kg
    "L1": 1.0,   # m
    "L2": 1.0,   # m
    "g": 9.81,   # m/s²
}


def angular_accelerations(
    theta1: tf.Tensor,
    theta2: tf.Tensor,
    omega1: tf.Tensor,
    omega2: tf.Tensor,
    m1: float = 1.0,
    m2: float = 1.0,
    L1: float = 1.0,
    L2: float = 1.0,
    g: float = 9.81,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Double-pendulum angular accelerations (domega1/dt, domega2/dt).

    Same equations as ``rk4_simulator.double_pendulum_ode``, expressed
    with TensorFlow ops so they can be evaluated inside a
    ``tf.GradientTape`` context for physics-informed training.
    """
    delta = theta1 - theta2
    sin_d = tf.sin(delta)
    cos_d = tf.cos(delta)
    denom = L1 * (2.0 * m1 + m2 - m2 * tf.cos(2.0 * delta))

    n1 = (
        -g * (2.0 * m1 + m2) * tf.sin(theta1)
        - m2 * g * tf.sin(theta1 - 2.0 * theta2)
        - 2.0 * sin_d * m2 * (omega2**2 * L2 + omega1**2 * L1 * cos_d)
    )
    n2 = 2.0 * sin_d * (
        omega1**2 * L1 * (m1 + m2)
        + g * (m1 + m2) * tf.cos(theta1)
        + omega2**2 * L2 * m2 * cos_d
    )

    domega1_dt = n1 / denom
    domega2_dt = n2 / (L2 * (2.0 * m1 + m2 - m2 * tf.cos(2.0 * delta)))
    return domega1_dt, domega2_dt


def hamiltonian(
    theta1: tf.Tensor,
    theta2: tf.Tensor,
    omega1: tf.Tensor,
    omega2: tf.Tensor,
    m1: float = 1.0,
    m2: float = 1.0,
    L1: float = 1.0,
    L2: float = 1.0,
    g: float = 9.81,
) -> tf.Tensor:
    """Total mechanical energy (Hamiltonian) of the frictionless double pendulum.

    Used by the Conservative PINN (cPINN) to penalize energy drift.
    """
    delta = theta1 - theta2
    kinetic_1 = 0.5 * (m1 + m2) * L1**2 * omega1**2
    kinetic_2 = 0.5 * m2 * L2**2 * omega2**2
    kinetic_coupling = m2 * L1 * L2 * omega1 * omega2 * tf.cos(delta)
    potential_1 = (m1 + m2) * g * L1 * (1.0 - tf.cos(theta1))
    potential_2 = m2 * g * L2 * (1.0 - tf.cos(theta2))
    return kinetic_1 + kinetic_2 + kinetic_coupling + potential_1 + potential_2
