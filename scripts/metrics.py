"""
metrics.py
==========
Performance Evaluation & Visualization

Computes and plots all metrics required by the lab:
  - RMSE and MAE (per angle, per method)
  - Training time (logged externally, displayed here)
  - Inference time (single-sample and batch)
  - Physical consistency: ODE residual on model predictions
  - Generalization: error on trajectories NOT used in training

Visualization:
  - Bar charts: RMSE, MAE, inference time
  - Error-over-time plots per method
  - Phase-space portrait comparison
  - Divergence-from-RK4 over time horizon
  - Summary metrics table (pandas DataFrame)

All plots are returned AND optionally saved to disk.

Usage: import this module from the Jupyter notebook.
"""

from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from physics import angular_accelerations


# ---------------------------------------------------------------------------
# Basic error metrics
# ---------------------------------------------------------------------------

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Root Mean Square Error per output column.

    Parameters
    ----------
    y_true, y_pred : (N, 2)  [theta1, theta2]

    Returns
    -------
    np.ndarray, shape (2,) — RMSE for theta1 and theta2
    """
    return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mean Absolute Error per output column."""
    return np.mean(np.abs(y_pred - y_true), axis=0)


def compute_error_over_time(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """Pointwise absolute error over time, averaged over both angles.

    Returns
    -------
    np.ndarray, shape (N,)
    """
    return np.mean(np.abs(y_pred - y_true), axis=1)


# ---------------------------------------------------------------------------
# Inference time
# ---------------------------------------------------------------------------

def measure_inference_time(
    model: tf.keras.Model,
    inputs: np.ndarray,
    n_runs: int = 200,
    warmup: int = 10,
) -> dict[str, float]:
    """Measure single-sample and batch inference time.

    Parameters
    ----------
    model  : trained Keras model
    inputs : sample input array (shape compatible with model input)
    n_runs : number of timed forward passes
    warmup : number of warmup passes (not timed)

    Returns
    -------
    dict with keys: 'single_ms', 'batch_ms'  (milliseconds)
    """
    single = inputs[:1]
    batch  = inputs

    # Warmup
    for _ in range(warmup):
        _ = model(single, training=False)
        _ = model(batch,  training=False)

    # Time single inference
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = model(single, training=False)
    single_ms = (time.perf_counter() - t0) / n_runs * 1000.0

    # Time batch inference
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = model(batch, training=False)
    batch_ms = (time.perf_counter() - t0) / n_runs * 1000.0

    return {"single_ms": single_ms, "batch_ms": batch_ms}


# ---------------------------------------------------------------------------
# Physical consistency
# ---------------------------------------------------------------------------

def compute_physics_residual(
    model: tf.keras.Model,
    t: np.ndarray,
    is_ppinn: bool = False,
    ic: np.ndarray | None = None,
    m1: float = 1.0, m2: float = 1.0,
    L1: float = 1.0, L2: float = 1.0,
    g:  float = 9.81,
) -> np.ndarray:
    """Evaluate the ODE residual on predictions made by a trained model.

    Differentiates the model output w.r.t. time using tf.GradientTape,
    then compares the derivatives to the ODE right-hand side.

    Parameters
    ----------
    model    : MLP, PINN, or Parametric PINN
    t        : (N,) time array
    is_ppinn : if True, concatenate ic to time inputs
    ic       : (4,) initial condition [theta1_0, omega1_0, theta2_0, omega2_0]
               required when is_ppinn=True

    Returns
    -------
    residuals : (N, 2) ODE residuals for [omega1_eq, omega2_eq]
    """
    t_tf = tf.constant(t.reshape(-1, 1), dtype=tf.float32)

    if is_ppinn:
        assert ic is not None, "ic must be provided for PPINN residual"
        ic_rep = tf.constant(
            np.tile(ic, (len(t), 1)), dtype=tf.float32
        )

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t_tf)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(t_tf)
            if is_ppinn:
                inp = tf.concat([t_tf, ic_rep], axis=1)
            else:
                inp = t_tf
            u = model(inp, training=False)
            theta1 = u[:, 0:1]
            theta2 = u[:, 1:2]

        omega1 = tape1.gradient(theta1, t_tf)
        omega2 = tape1.gradient(theta2, t_tf)

    alpha1 = tape2.gradient(omega1, t_tf)
    alpha2 = tape2.gradient(omega2, t_tf)
    del tape1, tape2

    domega1_dt, domega2_dt = angular_accelerations(
        theta1, theta2, omega1, omega2, m1=m1, m2=m2, L1=L1, L2=L2, g=g
    )

    res1 = alpha1 - domega1_dt
    res2 = alpha2 - domega2_dt

    residuals = tf.concat([res1, res2], axis=1).numpy()
    return residuals


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

_COLORS = {
    "RK4":   "#2196F3",
    "MLP":   "#F44336",
    "PINN":  "#4CAF50",
    "PPINN": "#FF9800",
}

_STYLE = dict(linewidth=1.5)


def plot_trajectories(
    t: np.ndarray,
    trajectories: dict[str, np.ndarray],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Overlay theta1 and theta2 for multiple methods.

    Parameters
    ----------
    t : (N,) time array
    trajectories : dict mapping method name -> (N, 2) [theta1, theta2]
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle("Double Pendulum Trajectory Comparison", fontsize=14, fontweight="bold")

    for ax, angle_idx, label in zip(axes, [0, 1], [r"$\theta_1$ (rad)", r"$\theta_2$ (rad)"]):
        for method, traj in trajectories.items():
            ls = "-" if method == "RK4" else "--"
            ax.plot(t, traj[:, angle_idx], ls=ls, color=_COLORS.get(method, "k"),
                    label=method, **_STYLE)
        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")

    return fig


def plot_error_over_time(
    t: np.ndarray,
    trajectories: dict[str, np.ndarray],
    reference: str = "RK4",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot absolute error relative to reference method over time."""
    ref = trajectories[reference]
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(f"Mean Absolute Error Relative to {reference}", fontsize=13, fontweight="bold")

    for method, traj in trajectories.items():
        if method == reference:
            continue
        err = compute_error_over_time(ref, traj)
        ax.plot(t, err, color=_COLORS.get(method, "k"), label=method, **_STYLE)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MAE (rad)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_phase_portrait(
    trajectories: dict[str, np.ndarray],
    angle_idx: int = 0,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Phase portrait (theta vs d(theta)/dt approximated by diff)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    angle_label = r"$\theta_1$" if angle_idx == 0 else r"$\theta_2$"
    fig.suptitle(f"Phase Portrait — {angle_label}", fontsize=13, fontweight="bold")

    for method, traj in trajectories.items():
        theta = traj[:, angle_idx]
        omega = np.gradient(theta)          # crude finite-difference
        ls = "-" if method == "RK4" else "--"
        ax.plot(theta, omega, ls=ls, color=_COLORS.get(method, "k"),
                label=method, **_STYLE, alpha=0.8)

    ax.set_xlabel(f"{angle_label} (rad)")
    ax.set_ylabel(rf"$\dot{{{angle_label[1:-1]}}}$ (rad/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_validation(
    t: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    ic_label: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Four-panel validation figure for a single model against RK4.

    Reusable across MLP / PINN / PPINN / cPINN / gPINN / xPINN: theta1
    and theta2 overlays, absolute error over time, and a metrics table
    (RMSE/MAE in radians and degrees).

    Parameters
    ----------
    t          : (N,) time array
    y_true     : (N, 2) RK4 reference [theta1, theta2]
    y_pred     : (N, 2) model prediction [theta1, theta2]
    model_name : label used in legends and the figure title (e.g. "PINN")
    ic_label   : optional initial-condition description for the suptitle
    """
    rmse = compute_rmse(y_true, y_pred)
    mae  = compute_mae(y_true, y_pred)
    err  = compute_error_over_time(y_true, y_pred)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    title = f"{model_name} Validation"
    if ic_label:
        title += f"  {ic_label}"
    fig.suptitle(title, fontweight="bold")

    axes[0, 0].plot(t, np.degrees(y_true[:, 0]), "b-",  lw=2,   label="RK4")
    axes[0, 0].plot(t, np.degrees(y_pred[:, 0]), "b--", lw=1.8, label=model_name, alpha=0.85)
    axes[0, 0].set(ylabel=r"$\theta_1$ (°)", title=r"$\theta_1$")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(t, np.degrees(y_true[:, 1]), "r-",  lw=2,   label="RK4")
    axes[0, 1].plot(t, np.degrees(y_pred[:, 1]), "r--", lw=1.8, label=model_name, alpha=0.85)
    axes[0, 1].set(ylabel=r"$\theta_2$ (°)", title=r"$\theta_2$")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].semilogy(t, err, "k-", lw=2)
    axes[1, 0].set(xlabel="Time (s)", ylabel="MAE (rad)", title="Error vs Time")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis("off")
    table_rows = [
        ["Metric", r"$\theta_1$", r"$\theta_2$"],
        ["RMSE (rad)", f"{rmse[0]:.6f}", f"{rmse[1]:.6f}"],
        ["MAE  (rad)", f"{mae[0]:.6f}",  f"{mae[1]:.6f}"],
        ["RMSE (°)",   f"{np.degrees(rmse[0]):.4f}", f"{np.degrees(rmse[1]):.4f}"],
        ["MAE  (°)",   f"{np.degrees(mae[0]):.4f}",  f"{np.degrees(mae[1]):.4f}"],
    ]
    tbl = axes[1, 1].table(
        cellText=table_rows[1:], colLabels=table_rows[0],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.2, 1.9)
    axes[1, 1].set_title("Validation Metrics", fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metrics_bar(
    metrics_df: pd.DataFrame,
    metric: str = "RMSE_theta1",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart for a single scalar metric across methods."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [_COLORS.get(m, "#607D8B") for m in metrics_df["Method"]]
    ax.bar(metrics_df["Method"], metrics_df[metric], color=colors, edgecolor="white", width=0.5)
    ax.set_ylabel(metric)
    ax.set_title(f"Comparison — {metric}", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Full report generator
# ---------------------------------------------------------------------------

def generate_metrics_report(
    models: dict[str, tf.keras.Model | None],
    t: np.ndarray,
    trajectories: dict[str, np.ndarray],
    training_times: dict[str, float],
    reference: str = "RK4",
    is_ppinn: dict[str, bool] | None = None,
    ics: dict[str, np.ndarray] | None = None,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compute all metrics and return a summary DataFrame.

    Parameters
    ----------
    models         : dict method_name -> Keras model (or None for RK4)
    t              : (N,) time array
    trajectories   : dict method_name -> (N, 2) [theta1, theta2]
    training_times : dict method_name -> training seconds
    reference      : name of the ground-truth method
    is_ppinn       : dict method_name -> bool (True for PPINN)
    ics            : dict method_name -> (4,) IC array (for PPINN residual)
    """
    if is_ppinn is None:
        is_ppinn = {k: False for k in models}
    if ics is None:
        ics = {}

    ref_traj = trajectories[reference]
    t_single = t[:1].reshape(1, 1).astype(np.float32)

    rows = []
    for method, traj in trajectories.items():
        rmse = compute_rmse(ref_traj, traj)
        mae  = compute_mae(ref_traj,  traj)

        model = models.get(method)
        if model is not None:
            # Build appropriate input for timing
            if is_ppinn.get(method, False) and method in ics:
                ic_rep = ics[method].reshape(1, -1).astype(np.float32)
                single_inp = np.concatenate(
                    [t.reshape(-1, 1), np.tile(ics[method], (len(t), 1))], axis=1
                ).astype(np.float32)
            else:
                single_inp = t.reshape(-1, 1).astype(np.float32)

            timing = measure_inference_time(model, single_inp)
            single_ms = timing["single_ms"]
            batch_ms  = timing["batch_ms"]
        else:
            single_ms = np.nan
            batch_ms  = np.nan

        rows.append({
            "Method":       method,
            "RMSE_theta1":  round(rmse[0], 6),
            "RMSE_theta2":  round(rmse[1], 6),
            "MAE_theta1":   round(mae[0], 6),
            "MAE_theta2":   round(mae[1], 6),
            "Train_time_s": round(training_times.get(method, np.nan), 2),
            "Infer_single_ms": round(single_ms, 4) if not np.isnan(single_ms) else "—",
            "Infer_batch_ms":  round(batch_ms, 4)  if not np.isnan(batch_ms)  else "—",
        })

    df = pd.DataFrame(rows)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Metrics report saved -> {save_path}")

    return df
