"""
simulation_viz.py
=================
Animated Simulation Visualization

Renders a side-by-side animation of the double pendulum trajectory for
multiple methods (e.g. RK4 reference + one or more trained models), plus
a handful of exploratory plots meant to be called directly from the
notebook:

  - animate_comparison   : side-by-side multi-method animation
  - animate_pendulum      : single-pendulum real-time animation with
                             live theta1(t)/theta2(t) panels
  - plot_training_curve   : generic training-loss plot (auto-detects
                             whichever loss_* components are present in
                             the history, so the same function works for
                             PINN/PPINN/cPINN/gPINN/xPINN)
  - plot_parameters       : six-panel single-trajectory analysis
                             (angles, velocities, energy, phase portraits,
                             chaotic trajectory of mass 2)
  - plot_parameters_all   : one independent plot_parameters figure per
                             model in a predictions dict
  - plot_chaos_sensitivity: overlay of nearly-identical initial conditions
                             to visualize exponential divergence

Usage: import this module from the Jupyter notebook.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def pendulum_cartesian(
    theta1: np.ndarray,
    theta2: np.ndarray,
    L1: float = 1.0,
    L2: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert angles to Cartesian coordinates.

    Returns (x1, y1, x2, y2) arrays.
    """
    x1 =  L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Colors & style
# ---------------------------------------------------------------------------

_COLORS = {
    "RK4":   "#2196F3",
    "MLP":   "#F44336",
    "PINN":  "#4CAF50",
    "PPINN": "#FF9800",
}
_TAIL = 30   # frames of trajectory tail to show


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def animate_comparison(
    t: np.ndarray,
    trajectories: dict[str, np.ndarray],
    L1: float = 1.0,
    L2: float = 1.0,
    fps: int = 30,
    speed: float = 1.0,
    save_path: str | Path | None = None,
) -> FuncAnimation:
    """Create a side-by-side animation of all methods.

    Parameters
    ----------
    t            : (N,) time array
    trajectories : dict method_name -> (N, 2) [theta1, theta2]
    fps          : frames per second for the saved video
    speed        : playback speed multiplier (>1 = faster)
    save_path    : if provided, save animation as .mp4
    """
    methods = list(trajectories.keys())
    n_methods = len(methods)

    # Subplots: one per method
    fig, axes = plt.subplots(
        1, n_methods, figsize=(4 * n_methods, 4),
        facecolor="#0f0f1a",
    )
    if n_methods == 1:
        axes = [axes]

    fig.suptitle("Double Pendulum — Method Comparison",
                 color="white", fontsize=13, fontweight="bold")

    lim = (L1 + L2) * 1.1
    lines, tails = [], []

    for ax, method in zip(axes, methods):
        color = _COLORS.get(method, "white")
        ax.set_facecolor("#0f0f1a")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(method, color=color, fontsize=11, fontweight="bold")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

        # Pivot
        ax.plot(0, 0, "o", color="white", markersize=4, zorder=5)

        line, = ax.plot([], [], "-", color=color, linewidth=2, zorder=4)
        dot1, = ax.plot([], [], "o", color=color, markersize=8, zorder=5)
        dot2, = ax.plot([], [], "o", color=color, markersize=8, zorder=5)
        tail, = ax.plot([], [], "-", color=color, linewidth=0.6,
                        alpha=0.4, zorder=3)

        lines.append((line, dot1, dot2))
        tails.append(tail)

    # Step through: one frame per `step` time steps
    step = max(1, int(round(speed)))
    frame_indices = np.arange(0, len(t), step)

    def init():
        for (line, d1, d2), tail in zip(lines, tails):
            line.set_data([], [])
            d1.set_data([], [])
            d2.set_data([], [])
            tail.set_data([], [])
        return [obj for grp in lines for obj in grp] + tails

    def update(frame_idx):
        i = frame_indices[frame_idx]
        for (line, d1, d2), tail, method in zip(lines, tails, methods):
            traj = trajectories[method]
            th1 = traj[:, 0]
            th2 = traj[:, 1]
            x1, y1, x2, y2 = pendulum_cartesian(th1, th2, L1, L2)

            line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
            d1.set_data([x1[i]], [y1[i]])
            d2.set_data([x2[i]], [y2[i]])

            i_tail = max(0, i - _TAIL)
            tail.set_data(x2[i_tail:i + 1], y2[i_tail:i + 1])

        return [obj for grp in lines for obj in grp] + tails

    anim = FuncAnimation(
        fig, update,
        frames=len(frame_indices),
        init_func=init,
        interval=int(1000 / fps),
        blit=True,
    )

    if save_path:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps, metadata={"title": "Double Pendulum"})
        anim.save(str(save_path), writer=writer, dpi=120)
        print(f"Animation saved -> {save_path}")

    return anim


# ---------------------------------------------------------------------------
# Training curves — generic across PINN / PPINN / cPINN / gPINN / xPINN
# ---------------------------------------------------------------------------

_LOSS_LABELS = {
    "loss_data":      "Data",
    "loss_phys":      "Physics",
    "loss_ic":        "IC",
    "loss_energy":    "Energy",
    "loss_grad":      "dR/dt",
    "loss_interface": "Interface",
}

_LOSS_COLORS = {
    "loss_data":      "#2196F3",
    "loss_phys":      "#4CAF50",
    "loss_ic":        "#F44336",
    "loss_energy":    "#E91E63",
    "loss_grad":      "#FFC107",
    "loss_interface": "#00BCD4",
}


def plot_training_curve(
    history: list[dict],
    title: str = "Training Curve",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Two-panel training-loss plot: all components, and component breakdown.

    Works for any physics-informed training history — it auto-detects
    whichever ``loss_*`` keys are present (``loss_data``, ``loss_phys``,
    ``loss_ic``, and any method-specific extra term such as
    ``loss_energy``, ``loss_grad``, or ``loss_interface``), so the same
    function serves PINN, PPINN, cPINN, gPINN, and xPINN without
    duplicating plotting code per model.

    Parameters
    ----------
    history : list of dicts returned by any ``train_*`` function.
              Each dict must contain ``'epoch'``, ``'loss_total'``, and
              one or more ``'loss_*'`` component keys.
    title   : Figure super-title.
    save_path : If given, the figure is saved to this path.

    Returns
    -------
    matplotlib Figure
    """
    epochs     = [e["epoch"]      for e in history]
    loss_total = [e["loss_total"] for e in history]
    component_keys = [k for k in history[0] if k not in ("epoch", "loss_total")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].plot(epochs, loss_total, "k-", lw=2, label="Total")
    for key in component_keys:
        vals  = [e[key] for e in history]
        label = _LOSS_LABELS.get(key, key.replace("loss_", "").capitalize())
        color = _LOSS_COLORS.get(key, None)
        axes[0].plot(epochs, vals, "--", lw=2, color=color, label=label)
        axes[1].plot(epochs, vals, "-",  lw=2, color=color, label=label)

    axes[0].set(xlabel="Epoch", ylabel="Loss", title="All components (log)")
    axes[0].set_yscale("log"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].set(xlabel="Epoch", ylabel="Loss", title="Component breakdown (log)")
    axes[1].set_yscale("log"); axes[1].grid(alpha=0.3); axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Single-trajectory parameter analysis
# ---------------------------------------------------------------------------

def _double_pendulum_energy(th1, w1, th2, w2, m1=1., m2=1., L1=1., L2=1., g=9.81):
    """Compute kinetic, potential, and total energy arrays."""
    x1, y1, x2, y2 = pendulum_cartesian(th1, th2, L1, L2)
    vx1 =  L1 * w1 * np.cos(th1);  vy1 =  L1 * w1 * np.sin(th1)
    vx2 = vx1 + L2 * w2 * np.cos(th2)
    vy2 = vy1 + L2 * w2 * np.sin(th2)
    kinetic   = 0.5*m1*(vx1**2+vy1**2) + 0.5*m2*(vx2**2+vy2**2)
    potential = m1*g*y1 + m2*g*y2
    return kinetic, potential, kinetic + potential


def _colored_trace(ax, x, y, t, cmap='plasma', alpha=0.65, lw=1.3):
    """Draw a time-colored trajectory line on *ax*."""
    from matplotlib.collections import LineCollection
    pts  = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap=cmap, alpha=alpha, linewidth=lw)
    lc.set_array(t)
    ax.add_collection(lc)
    ax.autoscale_view()


def plot_parameters(
    t: np.ndarray,
    traj: np.ndarray,
    title: str = "Double Pendulum",
    m1: float = 1.0, m2: float = 1.0,
    L1: float = 1.0, L2: float = 1.0,
    g:  float = 9.81,
    nn_pred: np.ndarray | None = None,
    nn_label: str = "NN",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Six-panel parameter plot for a double-pendulum trajectory.

    Parameters
    ----------
    t      : (N,) time array.
    traj   : (N, 4) array ``[theta1, omega1, theta2, omega2]`` (RK4 reference).
    title  : Figure title.
    nn_pred: (N, 2) optional NN prediction ``[theta1_nn, theta2_nn]`` to overlay.
    nn_label: Legend label for the NN prediction.
    save_path: If given, save the figure to this path.

    Returns
    -------
    matplotlib Figure
    """
    th1, w1, th2, w2 = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]
    kinetic, potential, total_energy = _double_pendulum_energy(
        th1, w1, th2, w2, m1, m2, L1, L2, g
    )
    x1, y1, x2, y2 = pendulum_cartesian(th1, th2, L1, L2)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f'Analysis — {title}', fontsize=14, fontweight='bold')

    # theta(t)
    axes[0, 0].plot(t, np.degrees(th1), 'b-', lw=2, label='theta1 (RK4)')
    axes[0, 0].plot(t, np.degrees(th2), 'r-', lw=2, label='theta2 (RK4)')
    if nn_pred is not None:
        axes[0, 0].plot(t, np.degrees(nn_pred[:, 0]), 'b--', lw=1.5,
                        alpha=0.75, label=f'theta1 ({nn_label})')
        axes[0, 0].plot(t, np.degrees(nn_pred[:, 1]), 'r--', lw=1.5,
                        alpha=0.75, label=f'theta2 ({nn_label})')
    axes[0, 0].set(xlabel='Time (s)', ylabel='Angle (°)',
                   title='Angular Positions')
    axes[0, 0].grid(alpha=0.3); axes[0, 0].legend(fontsize=8)

    # omega(t)
    axes[0, 1].plot(t, w1, 'b-', lw=2, label='omega1')
    axes[0, 1].plot(t, w2, 'r-', lw=2, label='omega2')
    axes[0, 1].set(xlabel='Time (s)', ylabel='Angular velocity (rad/s)',
                   title='Angular Velocities')
    axes[0, 1].grid(alpha=0.3); axes[0, 1].legend()

    # Energy
    axes[0, 2].plot(t, kinetic,      'b-', lw=2, label='Kinetic')
    axes[0, 2].plot(t, potential,    'r-', lw=2, label='Potential')
    axes[0, 2].plot(t, total_energy, 'k--', lw=2, label='Total')
    axes[0, 2].set(xlabel='Time (s)', ylabel='Energy (J)',
                   title='Energy Conservation')
    axes[0, 2].grid(alpha=0.3); axes[0, 2].legend()

    # Phase space theta1
    axes[1, 0].plot(np.degrees(th1), w1, color='steelblue', lw=1, alpha=0.85)
    axes[1, 0].plot(np.degrees(th1[0]), w1[0], 'go', ms=8, label='Start')
    axes[1, 0].set(xlabel='theta1 (°)', ylabel='omega1 (rad/s)',
                   title='Phase Space — mass 1')
    axes[1, 0].grid(alpha=0.3); axes[1, 0].legend()

    # Phase space theta2
    axes[1, 1].plot(np.degrees(th2), w2, color='tomato', lw=1, alpha=0.85)
    axes[1, 1].plot(np.degrees(th2[0]), w2[0], 'go', ms=8, label='Start')
    axes[1, 1].set(xlabel='theta2 (°)', ylabel='omega2 (rad/s)',
                   title='Phase Space — mass 2')
    axes[1, 1].grid(alpha=0.3); axes[1, 1].legend()

    # Chaotic trajectory of mass 2
    _colored_trace(axes[1, 2], x2, y2, t)
    axes[1, 2].plot(x2[0], y2[0], 'go', ms=8, label='Start')
    axes[1, 2].set(xlabel='x (m)', ylabel='y (m)',
                   title='Mass 2 Trajectory (chaotic)')
    axes[1, 2].set_aspect('equal')
    axes[1, 2].grid(alpha=0.3); axes[1, 2].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def _pred_to_traj(t: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Build a pseudo ``(N, 4)`` state array from an ``(N, 2)`` NN prediction.

    Angular velocities omega1, omega2 are recovered by numerical
    differentiation of the predicted angles using ``np.gradient`` (central
    differences, O(h^2)). This lets every model produce all six analysis
    panels independently.
    """
    th1 = pred[:, 0]
    th2 = pred[:, 1]
    w1  = np.gradient(th1, t)
    w2  = np.gradient(th2, t)
    return np.stack([th1, w1, th2, w2], axis=1)


def plot_parameters_all(
    t: np.ndarray,
    traj_rk4: np.ndarray,
    predictions: dict[str, np.ndarray],
    m1: float = 1.0, m2: float = 1.0,
    L1: float = 1.0, L2: float = 1.0,
    g:  float = 9.81,
    save_dir: str | Path | None = None,
) -> list[plt.Figure]:
    """Generate one fully independent 6-panel figure per model.

    Each model gets its own figure showing **only** its own output, with
    no overlay from other models.

    For NN models (which predict only ``[theta1, theta2]``), angular
    velocities are recovered via numerical differentiation so that all
    six panels can be computed: angles, velocities, energy conservation,
    phase portraits, and the chaotic trajectory of mass 2.

    Parameters
    ----------
    t           : (N,) time array.
    traj_rk4    : (N, 4) RK4 reference ``[theta1, omega1, theta2, omega2]``.
    predictions : ``{model_name: (N, 2)}`` — ``[theta1_pred, theta2_pred]`` per model.
    save_dir    : If given, each figure is saved as
                  ``<save_dir>/<name>_parameters.png``.

    Returns
    -------
    list of matplotlib Figures — one per model, starting with RK4.
    """
    save_dir = Path(save_dir) if save_dir else None
    figs: list[plt.Figure] = []

    # RK4 — exact state, no differentiation needed
    sp = str(save_dir / "RK4_parameters.png") if save_dir else None
    fig = plot_parameters(
        t, traj_rk4,
        title="RK4 — Numerical Reference",
        m1=m1, m2=m2, L1=L1, L2=L2, g=g,
        save_path=sp,
    )
    figs.append(fig)

    # One fully independent figure per NN model
    for name, pred in predictions.items():
        traj_nn = _pred_to_traj(t, pred)          # (N, 4) via np.gradient
        sp = str(save_dir / f"{name}_parameters.png") if save_dir else None
        fig = plot_parameters(
            t, traj_nn,
            title=f"{name} — individual analysis",
            m1=m1, m2=m2, L1=L1, L2=L2, g=g,
            save_path=sp,
        )
        figs.append(fig)

    return figs


def animate_pendulum(
    t: np.ndarray,
    traj: np.ndarray,
    title: str = "Double Pendulum",
    L1: float = 1.0,
    L2: float = 1.0,
    nn_pred: np.ndarray | None = None,
    nn_label: str = "NN",
    tail_len: int = 180,
    speed: int = 1,
) -> FuncAnimation:
    """Real-time animation of the double pendulum with live angle plots.

    Parameters
    ----------
    t       : (N,) time array.
    traj    : (N, 4) state array ``[theta1, omega1, theta2, omega2]``.
    title   : Window title.
    nn_pred : (N, 2) optional NN prediction ``[theta1_nn, theta2_nn]`` to overlay.
    nn_label: Legend label for the NN overlay.
    tail_len: Number of frames to keep in the trajectory tail.
    speed   : Frame-skip factor (1 = real-time, 2 = 2x faster).

    Returns
    -------
    FuncAnimation  (call ``plt.show()`` after to display interactively)
    """
    th1, w1, th2, w2 = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]
    x1, y1, x2, y2 = pendulum_cartesian(th1, th2, L1, L2)
    m = (L1 + L2) * 1.25
    has_nn = nn_pred is not None

    fig = plt.figure(figsize=(15, 6))
    fig.suptitle(f'Simulation — {title}', fontsize=14, fontweight='bold')

    ax_p  = fig.add_subplot(1, 3, 1)
    ax_t1 = fig.add_subplot(1, 3, 2)
    ax_t2 = fig.add_subplot(1, 3, 3)

    # Physical system panel
    ax_p.set_xlim(-m, m); ax_p.set_ylim(-m, m * 0.2)
    ax_p.set_aspect('equal'); ax_p.grid(alpha=0.25)
    ax_p.set_title('Physical System', fontweight='bold')
    ax_p.plot(0, 0, 'ko', ms=9)

    rod1,  = ax_p.plot([], [], 'o-', color='#2E86AB', lw=3, ms=11, label='m1')
    rod2,  = ax_p.plot([], [], 'o-', color='#E63946', lw=3, ms=11, label='m2')
    trace, = ax_p.plot([], [], '-',  color='#9B59B6', lw=1, alpha=0.4)
    info   = ax_p.text(0.02, 0.98, '', transform=ax_p.transAxes, va='top',
                       fontsize=8.5,
                       bbox=dict(boxstyle='round', fc='wheat', alpha=0.85))
    ax_p.legend(loc='lower right', fontsize=8)
    tx, ty = [], []

    # theta1(t) panel
    yl1 = np.degrees(np.abs(th1)).max() * 1.15
    ax_t1.set_xlim(0, t[-1]); ax_t1.set_ylim(-yl1, yl1)
    ax_t1.set(xlabel='Time (s)', ylabel='theta1 (°)', title='theta1 — Mass 1')
    ax_t1.grid(alpha=0.3); ax_t1.axhline(0, color='k', ls='--', alpha=0.2)
    l1, = ax_t1.plot([], [], 'b-', lw=2, label='RK4')
    p1, = ax_t1.plot([], [], 'bo', ms=6)
    if has_nn:
        ln1, = ax_t1.plot([], [], 'b--', lw=1.5, alpha=0.75, label=nn_label)
    ax_t1.legend(loc='upper right', fontsize=8)

    # theta2(t) panel
    yl2 = np.degrees(np.abs(th2)).max() * 1.15
    ax_t2.set_xlim(0, t[-1]); ax_t2.set_ylim(-yl2, yl2)
    ax_t2.set(xlabel='Time (s)', ylabel='theta2 (°)', title='theta2 — Mass 2')
    ax_t2.grid(alpha=0.3); ax_t2.axhline(0, color='k', ls='--', alpha=0.2)
    l2, = ax_t2.plot([], [], 'r-', lw=2, label='RK4')
    p2, = ax_t2.plot([], [], 'ro', ms=6)
    if has_nn:
        ln2, = ax_t2.plot([], [], 'r--', lw=1.5, alpha=0.75, label=nn_label)
    ax_t2.legend(loc='upper right', fontsize=8)

    artists = [rod1, rod2, trace, info, l1, p1, l2, p2]
    if has_nn:
        artists += [ln1, ln2]

    def init():
        for o in [rod1, rod2, trace, l1, p1, l2, p2]:
            o.set_data([], [])
        if has_nn:
            ln1.set_data([], []); ln2.set_data([], [])
        info.set_text('')
        return artists

    def animate(fi):
        i = min(fi * speed, len(t) - 1)
        rod1.set_data([0, x1[i]], [0, y1[i]])
        rod2.set_data([x1[i], x2[i]], [y1[i], y2[i]])
        tx.append(x2[i]); ty.append(y2[i])
        if len(tx) > tail_len: tx.pop(0); ty.pop(0)
        trace.set_data(tx, ty)
        info.set_text(
            f't  = {t[i]:.2f} s\n'
            f'theta1 = {np.degrees(th1[i]):+.1f}°   omega1 = {w1[i]:+.2f}\n'
            f'theta2 = {np.degrees(th2[i]):+.1f}°   omega2 = {w2[i]:+.2f}'
        )
        l1.set_data(t[:i], np.degrees(th1[:i]))
        p1.set_data([t[i]], [np.degrees(th1[i])])
        l2.set_data(t[:i], np.degrees(th2[:i]))
        p2.set_data([t[i]], [np.degrees(th2[i])])
        if has_nn:
            ln1.set_data(t[:i], np.degrees(nn_pred[:i, 0]))
            ln2.set_data(t[:i], np.degrees(nn_pred[:i, 1]))
        return artists

    n_frames = len(t) // speed
    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=n_frames, interval=16, blit=True, repeat=True)
    plt.tight_layout()
    return anim


def plot_chaos_sensitivity(
    t: np.ndarray,
    trajs: list[np.ndarray],
    labels: list[str] | None = None,
    title: str = "Chaos Sensitivity",
    L1: float = 1.0,
    L2: float = 1.0,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Three-panel chaos sensitivity plot.

    Parameters
    ----------
    t      : (N,) shared time array.
    trajs  : list of (N, 4) arrays, one per initial condition variant.
    labels : legend labels (default: 'Traj 1', 'Traj 2', ...).
    save_path: If given, save the figure to this path.

    Returns
    -------
    matplotlib Figure
    """
    if labels is None:
        labels = [f'Traj {k+1}' for k in range(len(trajs))]

    cols = plt.cm.tab10(np.linspace(0, 0.7, len(trajs)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for traj, lbl, c in zip(trajs, labels, cols):
        th1 = traj[:, 0]; th2 = traj[:, 2]
        x1, y1, x2, y2 = pendulum_cartesian(th1, th2, L1, L2)
        n = len(t)
        axes[0].plot(t, np.degrees(th1[:n]), color=c, lw=1.5, label=lbl)
        axes[1].plot(t, np.degrees(th2[:n]), color=c, lw=1.5)
        axes[2].plot(x2[:n], y2[:n], color=c, lw=0.8, alpha=0.75)

    axes[0].set(title='theta1 — Divergence', xlabel='Time (s)', ylabel='theta1 (°)')
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].set(title='theta2', xlabel='Time (s)', ylabel='theta2 (°)')
    axes[1].grid(alpha=0.3)
    axes[2].set(title='Mass 2 Trajectory', xlabel='x (m)', ylabel='y (m)')
    axes[2].set_aspect('equal'); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
