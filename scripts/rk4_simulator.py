"""
rk4_simulator.py
================
Stage 1 — Classical Numerical Integration

Fourth-order Runge-Kutta (RK4) solver for the chaotic double pendulum.
Provides:
  - The exact ODE right-hand side (equations of motion).
  - A single RK4 integration step.
  - A full trajectory simulator.
  - A dataset generator that creates many trajectories from random
    initial conditions and saves them to disk for downstream use.

Physical model
--------------
Two rigid rods of lengths L1, L2 and point masses m1, m2 coupled
at the pivot.  The Lagrangian equations of motion are:

  dθ1/dt = ω1
  dω1/dt = N1 / D
  dθ2/dt = ω2
  dω2/dt = N2 / D

where

  Δ = θ1 − θ2
  D = L1 (2m1 + m2 − m2 cos(2Δ))

  N1 = −g(2m1 + m2) sin θ1
       − m2 g sin(θ1 − 2θ2)
       − 2 sin Δ m2 (ω2² L2 + ω1² L1 cos Δ)

  N2 = 2 sin Δ [ω1² L1 (m1 + m2)
                + g(m1 + m2) cos θ1
                + ω2² L2 m2 cos Δ]

All angles in radians.  Default parameters: m1=m2=1 kg, L1=L2=1 m,
g=9.81 m/s².

Author: generated for the MLAB Double Pendulum Lab
"""

from __future__ import annotations

import csv
import numpy as np
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Default physical parameters
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict[str, float] = {
    "m1": 1.0,   # kg
    "m2": 1.0,   # kg
    "L1": 1.0,   # m
    "L2": 1.0,   # m
    "g": 9.81,   # m/s²
}


# ---------------------------------------------------------------------------
# Equations of motion
# ---------------------------------------------------------------------------

def double_pendulum_ode(
    t: float,
    state: np.ndarray,
    m1: float = 1.0,
    m2: float = 1.0,
    L1: float = 1.0,
    L2: float = 1.0,
    g: float = 9.81,
) -> np.ndarray:
    """Return the time derivative of the double-pendulum state vector.

    Parameters
    ----------
    t:
        Current time (unused — autonomous system, kept for solver compatibility).
    state:
        Array ``[θ1, ω1, θ2, ω2]``.
    m1, m2:
        Masses of the two bobs (kg).
    L1, L2:
        Rod lengths (m).
    g:
        Gravitational acceleration (m/s²).

    Returns
    -------
    np.ndarray
        ``[dθ1/dt, dω1/dt, dθ2/dt, dω2/dt]``.
    """
    theta1, omega1, theta2, omega2 = state
    delta = theta1 - theta2

    sin_d = np.sin(delta)
    cos_d = np.cos(delta)

    # Shared denominator
    denom = L1 * (2.0 * m1 + m2 - m2 * np.cos(2.0 * delta))

    # Numerators
    n1 = (
        -g * (2.0 * m1 + m2) * np.sin(theta1)
        - m2 * g * np.sin(theta1 - 2.0 * theta2)
        - 2.0 * sin_d * m2 * (omega2**2 * L2 + omega1**2 * L1 * cos_d)
    )

    n2 = 2.0 * sin_d * (
        omega1**2 * L1 * (m1 + m2)
        + g * (m1 + m2) * np.cos(theta1)
        + omega2**2 * L2 * m2 * cos_d
    )

    dtheta1 = omega1
    domega1 = n1 / denom
    dtheta2 = omega2
    domega2 = n2 / (L2 * (2.0 * m1 + m2 - m2 * np.cos(2.0 * delta)))

    return np.array([dtheta1, domega1, dtheta2, domega2])


# ---------------------------------------------------------------------------
# RK4 integration step
# ---------------------------------------------------------------------------

def rk4_step(
    f: Any,
    t: float,
    y: np.ndarray,
    dt: float,
    **params: float,
) -> np.ndarray:
    """Advance the state ``y`` by one RK4 step of size ``dt``.

    Parameters
    ----------
    f:
        ODE right-hand side callable ``f(t, y, **params)``.
    t:
        Current time.
    y:
        Current state vector.
    dt:
        Integration step size (s).
    **params:
        Physical parameters forwarded to ``f``.

    Returns
    -------
    np.ndarray
        State at time ``t + dt``.
    """
    k1 = f(t,           y,              **params)
    k2 = f(t + dt / 2,  y + dt / 2 * k1, **params)
    k3 = f(t + dt / 2,  y + dt / 2 * k2, **params)
    k4 = f(t + dt,       y + dt * k3,     **params)

    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# Full trajectory simulator
# ---------------------------------------------------------------------------

def simulate(
    ic: np.ndarray,
    t_span: tuple[float, float] = (0.0, 10.0),
    dt: float = 0.01,
    **params: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the double pendulum from initial conditions.

    Parameters
    ----------
    ic:
        Initial state ``[θ1⁰, ω1⁰, θ2⁰, ω2⁰]``.
    t_span:
        ``(t_start, t_end)`` in seconds.
    dt:
        Integration time step (s).
    **params:
        Physical parameters (m1, m2, L1, L2, g).

    Returns
    -------
    t : np.ndarray, shape (N,)
        Time array.
    traj : np.ndarray, shape (N, 4)
        State trajectory ``[θ1, ω1, θ2, ω2]`` at each time step.
    """
    p = {**DEFAULT_PARAMS, **params}
    t_start, t_end = t_span
    n_steps = int(round((t_end - t_start) / dt))

    t_arr = np.linspace(t_start, t_end, n_steps + 1)
    traj = np.empty((n_steps + 1, 4))
    traj[0] = ic

    state = ic.copy().astype(float)
    for i in range(1, n_steps + 1):
        state = rk4_step(double_pendulum_ode, t_arr[i - 1], state, dt, **p)
        traj[i] = state

    return t_arr, traj


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

def generate_dataset(
    n_trajectories: int = 50,
    ic_range: dict[str, tuple[float, float]] | None = None,
    t_span: tuple[float, float] = (0.0, 10.0),
    dt: float = 0.01,
    seed: int = 42,
    save_path: str | Path | None = None,
    **params: float,
) -> dict[str, np.ndarray]:
    """Generate a dataset of double-pendulum trajectories.

    Each trajectory starts from a randomly sampled initial condition
    drawn uniformly from ``ic_range``.

    Parameters
    ----------
    n_trajectories:
        Number of trajectories to simulate.
    ic_range:
        Dictionary with keys ``'theta1'``, ``'theta2'``, ``'omega1'``,
        ``'omega2'``, each mapping to a ``(min, max)`` tuple.
        Defaults to ``±π/2`` for angles and ``0`` for angular velocities.
    t_span:
        Time interval ``(t_start, t_end)`` in seconds.
    dt:
        Integration step size (s).
    seed:
        Random seed for reproducibility.
    save_path:
        If provided, saves the dataset as a ``.npz`` file at this path.
    **params:
        Physical parameters forwarded to the ODE.

    Returns
    -------
    dict with keys:
        ``'t'``           — time array, shape ``(N_steps,)``
        ``'trajectories'``— shape ``(n_trajectories, N_steps, 4)``
        ``'ics'``         — initial conditions, shape ``(n_trajectories, 4)``
    """
    if ic_range is None:
        ic_range = {
            "theta1": (-np.pi / 2, np.pi / 2),
            "omega1": (0.0, 0.0),
            "theta2": (-np.pi / 2, np.pi / 2),
            "omega2": (0.0, 0.0),
        }

    rng = np.random.default_rng(seed)

    def _sample_ic() -> np.ndarray:
        th1 = rng.uniform(*ic_range["theta1"])
        om1 = rng.uniform(*ic_range["omega1"])
        th2 = rng.uniform(*ic_range["theta2"])
        om2 = rng.uniform(*ic_range["omega2"])
        return np.array([th1, om1, th2, om2])

    t_ref, _ = simulate(_sample_ic(), t_span=t_span, dt=dt, **params)
    n_steps = len(t_ref)

    trajectories = np.empty((n_trajectories, n_steps, 4))
    ics = np.empty((n_trajectories, 4))

    for i in range(n_trajectories):
        ic = _sample_ic()
        _, traj = simulate(ic, t_span=t_span, dt=dt, **params)
        trajectories[i] = traj
        ics[i] = ic
        print(f"  Simulated trajectory {i + 1:3d}/{n_trajectories}", end="\r")

    print()  # newline after progress

    dataset = {
        "t": t_ref,
        "trajectories": trajectories,
        "ics": ics,
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, **dataset)
        print(f"Dataset saved to {save_path}")
        _save_dataset_csv(dataset, save_path)

    return dataset


def _save_dataset_csv(dataset: dict[str, np.ndarray], npz_path: Path) -> None:
    """Write the dataset as two CSV files alongside the ``.npz``.

    Files produced
    --------------
    ``<stem>_trajectories.csv``
        Long-format table with one row per (trajectory, time-step).
        Columns: ``traj_idx, t, theta1, omega1, theta2, omega2``.

    ``<stem>_ics.csv``
        One row per trajectory.
        Columns: ``traj_idx, theta1_0, omega1_0, theta2_0, omega2_0``.

    Parameters
    ----------
    dataset  : dict returned by :func:`generate_dataset`.
    npz_path : Path of the ``.npz`` file — CSVs are placed in the same
               directory with the same stem.
    """
    base = npz_path.parent / npz_path.stem

    t            = dataset["t"]            # (N_steps,)
    trajectories = dataset["trajectories"] # (n_traj, N_steps, 4)
    ics          = dataset["ics"]          # (n_traj, 4)
    n_traj, n_steps, _ = trajectories.shape

    # ── trajectories CSV (long format) ───────────────────────────────────────
    traj_csv = Path(f"{base}_trajectories.csv")
    with traj_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["traj_idx", "t", "theta1", "omega1", "theta2", "omega2"])
        for i in range(n_traj):
            for k in range(n_steps):
                writer.writerow([
                    i,
                    f"{t[k]:.6f}",
                    f"{trajectories[i, k, 0]:.8f}",
                    f"{trajectories[i, k, 1]:.8f}",
                    f"{trajectories[i, k, 2]:.8f}",
                    f"{trajectories[i, k, 3]:.8f}",
                ])
    print(f"Trajectories CSV saved to {traj_csv}")

    # ── initial conditions CSV ────────────────────────────────────────────────
    ics_csv = Path(f"{base}_ics.csv")
    with ics_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["traj_idx", "theta1_0", "omega1_0", "theta2_0", "omega2_0"])
        for i in range(n_traj):
            writer.writerow([
                i,
                f"{ics[i, 0]:.8f}",
                f"{ics[i, 1]:.8f}",
                f"{ics[i, 2]:.8f}",
                f"{ics[i, 3]:.8f}",
            ])
    print(f"ICs CSV saved to {ics_csv}")
