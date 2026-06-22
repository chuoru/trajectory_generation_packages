#!/usr/bin/env python3
##
# @file purepursuit_fine_sweep.py
#
# @brief Pure Pursuit closed-loop tracking on the three optimal corners
#        identified by analyze_fine_sweep.py.
#
# Workflow
# --------
#   1. Run differential_drive_path_segment_fine_sweep.py to produce CSVs.
#   2. Optionally run analyze_fine_sweep.py to inspect the sweep.
#   3. Run this script to simulate closed-loop tracking and compare
#      reference vs. tracked trajectories.
#
# Figures
# -------
#   fig_xy_tracking.png      -- reference vs tracked XY path for 3 corners
#   fig_cte.png              -- cross-track error vs time
#   fig_velocity_tracking.png -- reference v(t) vs tracked v(t)
#   fig_heading_error.png    -- heading error theta_tracked - theta_ref vs time
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/06/06

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.differential_drive import DifferentialDrive
from simulators.time_stepping import TimeStepping
from controllers.purepursuit.purepursuit import PurePursuit
from controllers.trajectory import Trajectory
from trajectory_generators.euler_jlap_coverage import EulerJLAPCoverage

# =============================================================================
# CONFIGURATION  --  edit these before running
# =============================================================================
DATA_DIR  = Path(__file__).resolve().parent / 'csv_output_fine_sweep'
SAVE_FIGS = True
FIG_DIR   = DATA_DIR / 'purepursuit_figures'

# Controller
SIM_DT             = 0.05   # [s]
LOOKAHEAD_DISTANCE = 0.15   # [m] tighter base; scales with speed via LOOKAHEAD_GAIN
LOOKAHEAD_GAIN     = 0.4    # L_d = 0.15 + 0.4*v  →  at v=0.5 m/s: L_d=0.35 m
K_SPEED            = 3.0    # proportional speed gain (was 0.6; τ reduced from 1.67 s to 0.33 s)
K_I_SPEED          = 0.5    # integral speed gain (eliminates steady-state velocity offset)
WHEEL_BASE         = 0.53   # [m] — matches fine_sweep ROBOT_PARAMS (2 * l=0.265)

# Observable state noise (1-sigma Gaussian, set to 0 to disable)
NOISE_STD_X     = 0.0    # [m]   position x noise
NOISE_STD_Y     = 0.0    # [m]   position y noise
NOISE_STD_THETA = 0.0    # [rad] heading noise

# Outlier filters (must match analyze_fine_sweep.py)
PP_FILTER_MULT = 1.5
TE_FILTER_MULT = 1.5
MT_FILTER_MULT = 5.0

# Academic color/linestyle scheme matching analyze_fine_sweep.py
STYLE_TIME   = ('black', '-')
STYLE_KNEE   = ('red',   '--')
STYLE_ENERGY = ('blue',  '-.')
COL_REF      = '#888888'

# Robot kinematics — must match JLAP_ROBOT_PARAMS in differential_drive_path_segment_fine_sweep.py
JLAP_DT = 0.05
JLAP_ROBOT_PARAMS = {
    'robot_mass':         50.4,
    'robot_width':        0.510,
    'wheel_radius':       0.3,
    'gear_ratio':         40.0,
    'rated_motor_torque': 1.3,
    'rated_motor_speed':  3500.0,
    'motor_inertia':      0.66e-4,
    'path_vel_lim':       0.5,
}


# =============================================================================
# DATA LOADING  (duplicated from analyze_fine_sweep.py)
# =============================================================================
def load_sweep_stats():
    path = DATA_DIR / 'sweep_statistics.csv'
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    return {
        'we': data[:, 0],
        'pp': data[:, 1],
        'te': data[:, 2],
        'mt': data[:, 3],
    }


def load_corners():
    corner_dir = DATA_DIR / 'corners_by_we'
    corners = {}
    for p in sorted(corner_dir.glob('corner_we_*.csv')):
        w_e = float(p.stem.replace('corner_we_', ''))
        data = np.loadtxt(p, delimiter=',', skiprows=1)
        corner = {
            'time':   data[:, 0],
            'x':      data[:, 1],
            'y':      data[:, 2],
            'theta':  data[:, 3],
            'v':      data[:, 4],
            'omega':  data[:, 5],
            'power':  data[:, 6],
            'jerk_r': data[:, 7] if data.shape[1] > 7 else None,
            'jerk_l': data[:, 8] if data.shape[1] > 8 else None,
        }
        corners[w_e] = _stitch_ramps(corner)
    return corners


def _stitch_ramps(corner):
    """Prepend an acceleration ramp (0 → v_h) and append a deceleration ramp
    (v_h → 0) so the full trajectory starts and ends at rest."""
    v_h = float(corner['v'][0])
    if v_h < 1e-3:
        return corner

    rp = JLAP_ROBOT_PARAMS
    gr  = rp['gear_ratio']
    tau = rp['rated_motor_torque']
    m   = rp['robot_mass']
    r   = rp['wheel_radius']
    I   = rp['motor_inertia']
    rated_torque = gr * tau
    inertia      = gr**2 * I
    a_max = 0.5 * rated_torque * r / (0.25 * m * r**2 + inertia)
    D = max(v_h**2 / a_max * 2.0, 0.3)

    th0 = corner['theta'][0]
    x0, y0 = corner['x'][0], corner['y'][0]
    pre = EulerJLAPCoverage(
        waypoints=[[x0 - D * np.cos(th0), y0 - D * np.sin(th0)], [x0, y0]],
        sampling_time=JLAP_DT, robot_params=rp,
        initial_vel=0.0, final_vel=v_h,
    ).generate_trajectory()

    thf = corner['theta'][-1]
    xf, yf = corner['x'][-1], corner['y'][-1]
    post = EulerJLAPCoverage(
        waypoints=[[xf, yf], [xf + D * np.cos(thf), yf + D * np.sin(thf)]],
        sampling_time=JLAP_DT, robot_params=rp,
        initial_vel=v_h, final_vel=0.0,
    ).generate_trajectory()

    t_c_start    = pre['time'][-1] + JLAP_DT
    t_post_start = t_c_start + (corner['time'][-1] - corner['time'][0]) + JLAP_DT
    time_full = np.concatenate([
        pre['time'],
        corner['time'] - corner['time'][0] + t_c_start,
        post['time']   - post['time'][0]   + t_post_start,
    ])
    return {
        'time':   time_full,
        'x':      np.concatenate([pre['states'][:, 0],  corner['x'],     post['states'][:, 0]]),
        'y':      np.concatenate([pre['states'][:, 1],  corner['y'],     post['states'][:, 1]]),
        'theta':  np.concatenate([pre['states'][:, 2],  corner['theta'], post['states'][:, 2]]),
        'v':      np.concatenate([pre['v'],              corner['v'],     post['v']]),
        'omega':  np.concatenate([pre['omega'],          corner['omega'], post['omega']]),
        'power':  np.concatenate([
            np.zeros(len(pre['time'])),
            corner['power'],
            np.zeros(len(post['time'])),
        ]),
        'jerk_r': None,
        'jerk_l': None,
    }


# =============================================================================
# FILTERING AND OPTIMUM SELECTION  (duplicated from analyze_fine_sweep.py)
# =============================================================================
def _pareto_front_idx(te, pp):
    n = len(te)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and te[j] <= te[i] and pp[j] <= pp[i]:
                if te[j] < te[i] or pp[j] < pp[i]:
                    dominated[i] = True
                    break
    idx = np.where(~dominated)[0]
    return idx[np.argsort(te[idx])]


def apply_filters(sw):
    we, pp, te, mt = sw['we'], sw['pp'], sw['te'], sw['mt']
    keep = np.ones(len(we), dtype=bool)

    if MT_FILTER_MULT < np.inf and keep.sum() >= 3:
        keep &= mt <= MT_FILTER_MULT * np.median(mt[keep])
    if keep.sum() >= 3:
        if PP_FILTER_MULT < np.inf or TE_FILTER_MULT < np.inf:
            pp_med = np.median(pp[keep])
            te_med = np.median(te[keep])
            if PP_FILTER_MULT < np.inf and TE_FILTER_MULT < np.inf:
                keep &= ~((pp > PP_FILTER_MULT * pp_med) | (te > TE_FILTER_MULT * te_med))
            elif PP_FILTER_MULT < np.inf:
                keep &= pp <= PP_FILTER_MULT * pp_med
            else:
                keep &= te <= TE_FILTER_MULT * te_med

    n_drop = int((~keep).sum())
    if n_drop:
        print(f"  Filters removed {n_drop} of {len(we)} points.")

    we_f, pp_f, te_f, mt_f = we[keep], pp[keep], te[keep], mt[keep]
    d1 = np.gradient(pp_f, we_f)
    d2 = np.gradient(d1, we_f)

    return {'we': we_f, 'pp': pp_f, 'te': te_f, 'mt': mt_f, 'd2': d2}


def pick_optima(sw):
    we, pp, te, mt = sw['we'], sw['pp'], sw['te'], sw['mt']

    time_idx   = int(np.argmin(mt))
    energy_idx = int(np.argmin(te))

    pf = _pareto_front_idx(te, pp)
    if len(pf) >= 3:
        te_n = (te - te.min()) / max(float(te.max() - te.min()), 1e-12)
        pp_n = (pp - pp.min()) / max(float(pp.max() - pp.min()), 1e-12)
        ax, ay = te_n[pf[0]], pp_n[pf[0]]
        bx, by = te_n[pf[-1]], pp_n[pf[-1]]
        denom = max(float(np.hypot(bx - ax, by - ay)), 1e-12)
        pf_mid = pf[1:-1]
        dist = np.abs((by - ay) * (te_n[pf_mid] - ax) - (bx - ax) * (pp_n[pf_mid] - ay)) / denom
        knee_idx = int(pf_mid[np.argmax(dist)])
    elif len(pf) >= 1:
        knee_idx = int(pf[len(pf) // 2])
    else:
        knee_idx = 0

    return {
        'time_idx':   time_idx,   'time_we':   float(we[time_idx]),
        'knee_idx':   knee_idx,   'knee_we':   float(we[knee_idx]),
        'energy_idx': energy_idx, 'energy_we': float(we[energy_idx]),
    }


def _nearest_corner(corners, target_we):
    best = min(corners.keys(), key=lambda k: abs(k - target_we))
    return corners[best], best


# =============================================================================
# TRAJECTORY CONSTRUCTION
# =============================================================================
def corner_to_trajectory(corner_dict, dt=SIM_DT):
    """Resample a corner dict to a uniform-dt Trajectory for the controller."""
    t_raw = corner_dict['time']
    t_uni = np.arange(t_raw[0], t_raw[-1], dt)
    x_uni  = np.interp(t_uni, t_raw, corner_dict['x'])
    y_uni  = np.interp(t_uni, t_raw, corner_dict['y'])
    th_uni = np.interp(t_uni, t_raw, corner_dict['theta'])
    v_uni  = np.interp(t_uni, t_raw, corner_dict['v'])
    w_uni  = np.interp(t_uni, t_raw, corner_dict['omega'])
    states   = np.column_stack([x_uni, y_uni, th_uni])  # (N, 3)
    controls = np.vstack([v_uni, w_uni])                  # (2, N)
    return Trajectory(x=states, u=controls, t=t_uni, sampling_time=dt)


# =============================================================================
# NOISY OBSERVER WRAPPER
# =============================================================================
class NoisyObserver:
    """Wraps a controller and adds independent Gaussian noise to each observed
    state component before forwarding to the controller.  True state
    propagation inside TimeStepping is unaffected."""

    def __init__(self, controller, noise_std, rng=None):
        self._ctrl = controller
        self._std  = np.asarray(noise_std, dtype=float)
        self._rng  = rng if rng is not None else np.random.default_rng()

    def initialize(self):
        self._ctrl.initialize()

    def execute(self, state, input, index):
        if np.any(self._std > 0.0):
            noisy = state + self._rng.normal(0.0, self._std)
        else:
            noisy = state
        return self._ctrl.execute(noisy, input, index)


# =============================================================================
# SIMULATION
# =============================================================================
def run_purepursuit(traj, noise_std=None, seed=None):
    """Run closed-loop Pure Pursuit on *traj*.

    @param traj<Trajectory>: Reference trajectory.
    @param noise_std<tuple>: (sigma_x, sigma_y, sigma_theta) in m / rad.
                             Defaults to the module-level NOISE_STD_* values.
    @param seed<int|None>: RNG seed for reproducibility.
    @return TimeStepping instance with x_out / u_out / t_out populated.
    """
    if noise_std is None:
        noise_std = (NOISE_STD_X, NOISE_STD_Y, NOISE_STD_THETA)

    model = DifferentialDrive(wheel_base=WHEEL_BASE)

    PurePursuit.lookahead_distance = LOOKAHEAD_DISTANCE
    PurePursuit.lookahead_gain     = LOOKAHEAD_GAIN
    PurePursuit.k                  = K_SPEED
    PurePursuit.k_i                = K_I_SPEED
    PurePursuit.k_ff               = 1.0

    sim        = TimeStepping(model, float(traj.t[-1]), traj.sampling_time)
    controller = NoisyObserver(
        PurePursuit(model, traj),
        noise_std,
        rng=np.random.default_rng(seed),
    )
    sim.run_with_controller(list(traj.x[0]), traj, controller)
    return sim


# =============================================================================
# TRACKING METRICS
# =============================================================================
def cross_track_error(ref_xy, trk_xy):
    """Point-to-polyline cross-track error for each tracked point.

    @param ref_xy<ndarray>: (N, 2) reference path.
    @param trk_xy<ndarray>: (M, 2) tracked path (M may differ from N).
    @return cte<ndarray>: (M,) signed-magnitude CTE [m].
    """
    cte = np.empty(len(trk_xy))
    for i, pt in enumerate(trk_xy):
        diffs = ref_xy - pt
        cte[i] = np.min(np.hypot(diffs[:, 0], diffs[:, 1]))
    return cte


def heading_error(ref_th, trk_th):
    """Wrap-corrected heading error trk - ref [rad]."""
    err = trk_th - ref_th
    return np.arctan2(np.sin(err), np.cos(err))


# =============================================================================
# FIGURE HELPERS
# =============================================================================
def _savefig(fig, name):
    if SAVE_FIGS:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        out = FIG_DIR / name
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f"  [saved] {name}")


def fig_xy_tracking(results):
    """XY path: reference (dashed grey) vs tracked (solid colour) for 3 corners."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), num='XY Tracking')
    titles = ['Time-optimal', 'Pareto Knee', 'Energy-optimal']
    for ax, (lbl, style, traj, sim) in zip(axes, results):
        col, ls = style
        ref_xy = traj.x[:, :2]
        trk_xy = sim.x_out[:2, :].T

        ax.set_aspect('equal')
        ax.plot(ref_xy[:, 0], ref_xy[:, 1], color=COL_REF, ls='--', lw=1.5,
                label='Reference')
        ax.plot(trk_xy[:, 0], trk_xy[:, 1], color=col, ls=ls, lw=2.0,
                label=f'Tracked  w_e={lbl}')
        ax.plot(*ref_xy[0],  'o', color='green', ms=7, zorder=5, label='Start')
        ax.plot(*ref_xy[-1], 's', color='red',   ms=7, zorder=5, label='Goal')

        ax.set_xlabel('x [m]', fontsize=11)
        ax.set_ylabel('y [m]', fontsize=11)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=8)

    fig.tight_layout()
    _savefig(fig, 'fig_xy_tracking.png')


def fig_cte(results):
    """Cross-track error vs time for the 3 optimal corners."""
    fig, ax = plt.subplots(figsize=(9, 4), num='Cross-Track Error')
    for lbl, style, traj, sim in results:
        col, ls = style
        trk_xy = sim.x_out[:2, :].T
        ref_xy = traj.x[:, :2]
        cte = cross_track_error(ref_xy, trk_xy)
        t   = sim.t_out[:len(cte)]
        ax.plot(t, cte * 1e2, color=col, ls=ls, lw=1.8,
                label=f'w_e={lbl}')

    ax.set_xlabel('time [s]', fontsize=12)
    ax.set_ylabel('CTE [cm]', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _savefig(fig, 'fig_cte.png')


def fig_velocity_tracking(results):
    """Reference vs tracked linear velocity for the 3 optimal corners."""
    fig, ax = plt.subplots(figsize=(9, 4), num='Velocity Tracking')
    for lbl, style, traj, sim in results:
        col, ls = style
        nt = min(traj.x.shape[0], sim.u_out.shape[1])
        t  = traj.t[:nt]
        ax.plot(t, traj.u[0, :nt], color=COL_REF, ls='--', lw=1.2)
        ax.plot(t, sim.u_out[0, :nt], color=col, ls=ls, lw=1.8,
                label=f'Tracked  w_e={lbl}')

    ax.set_xlabel('time [s]', fontsize=12)
    ax.set_ylabel('v [m/s]', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10)
    ax.plot([], [], color=COL_REF, ls='--', lw=1.2, label='Reference')
    ax.legend(fontsize=10)
    fig.tight_layout()
    _savefig(fig, 'fig_velocity_tracking.png')


def fig_heading_error(results):
    """Heading error (tracked - reference) vs time for the 3 optimal corners."""
    fig, ax = plt.subplots(figsize=(9, 4), num='Heading Error')
    for lbl, style, traj, sim in results:
        col, ls = style
        nt = min(traj.x.shape[0], sim.x_out.shape[1])
        t  = traj.t[:nt]
        he = heading_error(traj.x[:nt, 2], sim.x_out[2, :nt])
        ax.plot(t, np.rad2deg(he), color=col, ls=ls, lw=1.8,
                label=f'w_e={lbl}')

    ax.axhline(0.0, color='black', lw=0.8, ls=':')
    ax.set_xlabel('time [s]', fontsize=12)
    ax.set_ylabel('heading error [deg]', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _savefig(fig, 'fig_heading_error.png')


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"Data directory : {DATA_DIR}")
    print(f"Noise std      : x={NOISE_STD_X} m  y={NOISE_STD_Y} m  "
          f"theta={NOISE_STD_THETA} rad")
    print()

    sw_raw  = load_sweep_stats()
    corners = load_corners()
    print(f"  Loaded {len(sw_raw['we'])} sweep points, "
          f"{len(corners)} corner trajectories.")

    sw  = apply_filters(sw_raw)
    opt = pick_optima(sw)

    specs = [
        ('time_we',   STYLE_TIME,   'Time-opt'),
        ('knee_we',   STYLE_KNEE,   'Knee'),
        ('energy_we', STYLE_ENERGY, 'Energy-opt'),
    ]

    results = []
    for key_we, style, lbl_prefix in specs:
        target = opt[key_we]
        c, actual_we = _nearest_corner(corners, target)
        we_str = f'{actual_we:.4f}'
        traj = corner_to_trajectory(c)
        sim  = run_purepursuit(traj, seed=0)

        trk_xy = sim.x_out[:2, :].T
        ref_xy = traj.x[:, :2]
        cte    = cross_track_error(ref_xy, trk_xy)
        final_err = np.hypot(
            sim.x_out[0, -1] - traj.x[-1, 0],
            sim.x_out[1, -1] - traj.x[-1, 1],
        )

        print(f"  {lbl_prefix:12s}  w_e={we_str}  "
              f"max_CTE={cte.max()*1e2:.2f} cm  "
              f"final_err={final_err*1e2:.2f} cm")

        results.append((we_str, style, traj, sim))

    print()
    fig_xy_tracking(results)
    fig_cte(results)
    fig_velocity_tracking(results)
    fig_heading_error(results)

    plt.show()
    plt.close('all')


# =============================================================================
if __name__ == '__main__':
    main()
