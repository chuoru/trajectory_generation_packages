#!/usr/bin/env python3
##
# @file purepursuit_fine_sweep.py
#
# @brief Pure Pursuit closed-loop tracking with simulated GPS-IMU pose
#        feedback on the three optimal corners identified by
#        analyze_fine_sweep.py.
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
#   fig_acceleration.png     -- reference vs tracked linear acceleration
#   fig_jerk.png             -- reference vs tracked linear jerk
#   fig_power.png            -- reference vs tracked total motor power
#   comparison_report.html   -- side-by-side metrics table (HTML)
#   method_slides.html       -- landscape slide deck: control-law derivation
#                                with prose explanations (HTML)
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/06/06

import sys
import os
import io
import base64
import itertools
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
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
# Retuned against GPS-IMU pose feedback (see GPS/IMU block below): the old
# (0.15, 0.4, 3.0, 0.5) gains let the tight time-optimal corner go unstable
# under sensor noise (>150 deg heading error / near-total loss of tracking).
# LOOKAHEAD_DISTANCE is the main stability lever — below ~0.20 m the
# time-optimal corner reliably diverges; K_SPEED beyond ~6 gives diminishing
# returns since commanded acceleration is already saturating at 1 m/s^2.
SIM_DT             = 0.05   # [s]
LOOKAHEAD_DISTANCE = 0.25   # [m] base lookahead; scales with speed via LOOKAHEAD_GAIN
LOOKAHEAD_GAIN     = 0.4    # L_d = 0.25 + 0.4*v  →  at v=0.5 m/s: L_d=0.45 m
                             # NOTE: currently a no-op — TimeStepping.run_with_controller
                             # always passes input=[0,0] to the controller (see
                             # simulators/time_stepping.py), so the speed term never
                             # contributes. Left in place / documented for when that's fixed.
K_SPEED            = 6.0    # proportional speed gain (τ ≈ 0.17 s)
K_I_SPEED          = 0.1    # integral speed gain (eliminates steady-state velocity offset)
WHEEL_BASE         = 0.53   # [m] — matches fine_sweep ROBOT_PARAMS (2 * l=0.265)

# GPS-IMU pose feedback (replaces raw full-state noise model)
# SIM_DT above doubles as the IMU / control update rate (1/SIM_DT = 20 Hz).
GPS_RATE_HZ    = 10.0   # [Hz]  GPS fix rate (sample-and-hold between fixes)
GPS_STD_XY     = 0.03   # [m]   GPS position 1-sigma noise (per axis), RTK/DGPS-like
IMU_STD_THETA  = 0.01   # [rad] IMU (AHRS-fused) heading 1-sigma noise (~0.6 deg)
IMU_STD_V      = 0.02   # [m/s] odometry-derived velocity 1-sigma noise, used for
                         #       IMU-rate dead-reckoning between GPS fixes

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
    'robot_width':        0.53,
    'wheel_radius':       0.15,
    'gear_ratio':         40.0,
    'rated_motor_torque': 1.3,
    'rated_motor_speed':  3500.0,
    'motor_inertia':      0.66e-4,
    'path_vel_lim':       0.5,
}

# Per-wheel motor power model — matches differential_drive_path_segment_fine_sweep.py
# (ENERGY_COEFFS_RIGHT/LEFT, P_ELECTRONICS) so reference and tracked power are
# evaluated with the same fitted model.
HALF_WHEEL_BASE = WHEEL_BASE / 2.0
ENERGY_COEFFS_RIGHT = [
    0.302433145557389,
    31.887262598534413,
    2.4140287888312457,
    0.9658866923308425,
    0.8260871406535432,
    2.37456174658809e-08,
]
ENERGY_COEFFS_LEFT = [
    0.33789198669595977,
    28.204019732889346,
    2.5903002025839688,
    0.00847962183165042,
    6.412423386896174e-09,
    0.3614761744737831,
]
P_ELECTRONICS = 2.0   # constant hotel load [W]


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
# GPS-IMU POSE OBSERVER
# =============================================================================
class GpsImuObserver:
    """Wraps a controller with a simple loosely-coupled GPS-IMU pose
    estimator: IMU-rate (== control rate) heading + dead-reckoned position
    propagation, corrected by a lower-rate GPS position fix.  True state
    propagation inside TimeStepping is unaffected."""

    def __init__(self, controller, sim_dt, gps_rate_hz,
                 gps_std_xy, imu_std_theta, imu_std_v, rng=None):
        self._ctrl = controller
        self._dt = sim_dt
        self._gps_period_steps = max(1, round(1.0 / gps_rate_hz / sim_dt))
        self._gps_std = gps_std_xy
        self._imu_std_theta = imu_std_theta
        self._imu_std_v = imu_std_v
        self._rng = rng if rng is not None else np.random.default_rng()
        self._x_hat = None
        self._y_hat = None

    def initialize(self):
        self._x_hat = None
        self._y_hat = None
        self._ctrl.initialize()

    def execute(self, state, input, index):
        theta_hat = state[2] + self._rng.normal(0.0, self._imu_std_theta)

        if self._x_hat is None:
            self._x_hat, self._y_hat = state[0], state[1]
        else:
            v_meas = input[0] + self._rng.normal(0.0, self._imu_std_v)
            self._x_hat += v_meas * np.cos(theta_hat) * self._dt
            self._y_hat += v_meas * np.sin(theta_hat) * self._dt

        if index % self._gps_period_steps == 0:
            self._x_hat = state[0] + self._rng.normal(0.0, self._gps_std)
            self._y_hat = state[1] + self._rng.normal(0.0, self._gps_std)

        estimated_state = np.array([self._x_hat, self._y_hat, theta_hat])
        return self._ctrl.execute(estimated_state, input, index)


# =============================================================================
# SIMULATION
# =============================================================================
def run_purepursuit(traj, gps_rate_hz=None, gps_std_xy=None,
                     imu_std_theta=None, imu_std_v=None, seed=None):
    """Run closed-loop Pure Pursuit on *traj* with simulated GPS-IMU pose
    feedback.

    @param traj<Trajectory>: Reference trajectory.
    @param gps_rate_hz<float>: GPS fix rate [Hz]. Defaults to GPS_RATE_HZ.
    @param gps_std_xy<float>: GPS position 1-sigma noise [m]. Defaults to GPS_STD_XY.
    @param imu_std_theta<float>: IMU heading 1-sigma noise [rad]. Defaults to IMU_STD_THETA.
    @param imu_std_v<float>: IMU/odometry velocity 1-sigma noise [m/s]. Defaults to IMU_STD_V.
    @param seed<int|None>: RNG seed for reproducibility.
    @return TimeStepping instance with x_out / u_out / t_out populated.
    """
    gps_rate_hz   = GPS_RATE_HZ   if gps_rate_hz   is None else gps_rate_hz
    gps_std_xy    = GPS_STD_XY    if gps_std_xy    is None else gps_std_xy
    imu_std_theta = IMU_STD_THETA if imu_std_theta is None else imu_std_theta
    imu_std_v     = IMU_STD_V     if imu_std_v     is None else imu_std_v

    model = DifferentialDrive(wheel_base=WHEEL_BASE)

    PurePursuit.lookahead_distance = LOOKAHEAD_DISTANCE
    PurePursuit.lookahead_gain     = LOOKAHEAD_GAIN
    PurePursuit.k                  = K_SPEED
    PurePursuit.k_i                = K_I_SPEED
    PurePursuit.k_ff               = 1.0

    sim        = TimeStepping(model, float(traj.t[-1]), traj.sampling_time)
    controller = GpsImuObserver(
        PurePursuit(model, traj),
        sim_dt=traj.sampling_time,
        gps_rate_hz=gps_rate_hz,
        gps_std_xy=gps_std_xy,
        imu_std_theta=imu_std_theta,
        imu_std_v=imu_std_v,
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


def compute_motor_power(v, omega, t):
    """Total motor power [W] from path velocity/yaw-rate via the per-wheel
    energy model (ENERGY_COEFFS_RIGHT/LEFT, P_ELECTRONICS).

    @param v<ndarray>: path linear velocity [m/s].
    @param omega<ndarray>: path yaw rate [rad/s].
    @param t<ndarray>: time samples [s], used to differentiate wheel velocity.
    @return power<ndarray>: total motor power [W].
    """
    v_r = v + HALF_WHEEL_BASE * omega
    v_l = v - HALF_WHEEL_BASE * omega
    a_r = np.gradient(v_r, t)
    a_l = np.gradient(v_l, t)

    def _p(vw, aw, c):
        return np.maximum(
            c[0] * aw**2 + c[1] * vw**2
            + np.abs(c[2] * aw) + np.abs(c[3] * vw)
            + np.abs(c[4] * vw * aw) + c[5], 0.0)

    return _p(v_r, a_r, ENERGY_COEFFS_RIGHT) + _p(v_l, a_l, ENERGY_COEFFS_LEFT) + P_ELECTRONICS


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


def fig_acceleration(results):
    """Reference vs tracked linear acceleration for the 3 optimal corners."""
    fig, ax = plt.subplots(figsize=(9, 4), num='Acceleration Tracking')
    for lbl, style, traj, sim in results:
        col, ls = style
        nt = min(traj.x.shape[0], sim.u_out.shape[1])
        t  = traj.t[:nt]
        a_ref = np.gradient(traj.u[0, :nt], t)
        a_trk = np.gradient(sim.u_out[0, :nt], t)
        ax.plot(t, a_ref, color=COL_REF, ls='--', lw=1.2)
        ax.plot(t, a_trk, color=col, ls=ls, lw=1.8, label=f'Tracked  w_e={lbl}')

    ax.set_xlabel('time [s]', fontsize=12)
    ax.set_ylabel('a [m/s²]', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.plot([], [], color=COL_REF, ls='--', lw=1.2, label='Reference')
    ax.legend(fontsize=10)
    fig.tight_layout()
    _savefig(fig, 'fig_acceleration.png')


def fig_jerk(results):
    """Reference vs tracked linear jerk for the 3 optimal corners."""
    fig, ax = plt.subplots(figsize=(9, 4), num='Jerk Tracking')
    for lbl, style, traj, sim in results:
        col, ls = style
        nt = min(traj.x.shape[0], sim.u_out.shape[1])
        t  = traj.t[:nt]
        a_ref = np.gradient(traj.u[0, :nt], t)
        a_trk = np.gradient(sim.u_out[0, :nt], t)
        j_ref = np.gradient(a_ref, t)
        j_trk = np.gradient(a_trk, t)
        ax.plot(t, j_ref, color=COL_REF, ls='--', lw=1.2)
        ax.plot(t, j_trk, color=col, ls=ls, lw=1.8, label=f'Tracked  w_e={lbl}')

    ax.set_xlabel('time [s]', fontsize=12)
    ax.set_ylabel('jerk [m/s³]', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.plot([], [], color=COL_REF, ls='--', lw=1.2, label='Reference')
    ax.legend(fontsize=10)
    fig.tight_layout()
    _savefig(fig, 'fig_jerk.png')


def fig_power(results):
    """Reference vs tracked total motor power for the 3 optimal corners."""
    fig, ax = plt.subplots(figsize=(9, 4), num='Power Tracking')
    for lbl, style, traj, sim in results:
        col, ls = style
        nt = min(traj.x.shape[0], sim.u_out.shape[1])
        t  = traj.t[:nt]
        p_ref = compute_motor_power(traj.u[0, :nt], traj.u[1, :nt], t)
        p_trk = compute_motor_power(sim.u_out[0, :nt], sim.u_out[1, :nt], t)
        ax.plot(t, p_ref, color=COL_REF, ls='--', lw=1.2)
        ax.plot(t, p_trk, color=col, ls=ls, lw=1.8, label=f'Tracked  w_e={lbl}')

    ax.set_xlabel('time [s]', fontsize=12)
    ax.set_ylabel('power [W]', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.plot([], [], color=COL_REF, ls='--', lw=1.2, label='Reference')
    ax.legend(fontsize=10)
    fig.tight_layout()
    _savefig(fig, 'fig_power.png')


# =============================================================================
# COMPARISON TABLE
# =============================================================================
# Row groups shared by the terminal table (print_comparison_table) and the
# HTML export (export_comparison_table_html): (label, metrics-dict key, format).
_COMPARISON_ROW_GROUPS = [
    [
        ('mission time [s]', 'mission_time', '{:.3f}'),
        ('total energy [J]', 'total_energy', '{:.2f}'),
        ('peak power [W]',   'peak_power',   '{:.2f}'),
    ],
    [
        ('max CTE [cm]',           'max_cte',   '{:.3f}'),
        ('mean CTE [cm]',          'mean_cte',  '{:.3f}'),
        ('RMS CTE [cm]',           'rms_cte',   '{:.3f}'),
        ('final pos error [cm]',   'final_err', '{:.3f}'),
        ('max heading err [deg]',  'max_he',    '{:.3f}'),
        ('mean heading err [deg]', 'mean_he',   '{:.3f}'),
        ('max vel error [m/s]',    'max_verr',  '{:.4f}'),
        ('tracking duration [s]',  'track_dur', '{:.3f}'),
    ],
]


def print_comparison_table(metrics):
    """Print a side-by-side comparison table of reference + tracking metrics."""
    lbl_w, col_w = 24, 18
    sep = '+' + '-' * lbl_w + '+' + ('-' * col_w + '+') * 3

    def _row(lbl, vals, fmt='{:.3f}'):
        cells = ''.join(f'| {fmt.format(v):>{col_w - 2}} ' for v in vals)
        return f'| {lbl:<{lbl_w - 2}} {cells}|'

    def _hrow(lbl, vals):
        cells = ''.join(f'| {str(v).center(col_w - 2)} ' for v in vals)
        return f'| {lbl:<{lbl_w - 2}} {cells}|'

    labels  = [m['label']  for m in metrics]
    we_strs = [m['we_str'] for m in metrics]

    print(sep)
    print(_hrow('', labels))
    print(_hrow('w_e', we_strs))
    print(sep)
    for group in _COMPARISON_ROW_GROUPS:
        for lbl, key, fmt in group:
            print(_row(lbl, [m[key] for m in metrics], fmt))
        print(sep)


# =============================================================================
# LATEX (MATHTEXT) EQUATION RENDERING
# =============================================================================
# Method equations for the adaptive Pure Pursuit law in
# controllers/purepursuit/purepursuit.py, grouped for the HTML report.
# Rendered locally via matplotlib's mathtext (no LaTeX install / CDN needed).
_METHOD_EQUATIONS = [
    ('Adaptive lookahead distance', [
        r'$L_d(v) = L_{d,0} + k_{ld}\, v$',
    ]),
    ('Lookahead heading error', [
        r'$\alpha = \mathrm{atan2}(y_t - y,\ x_t - x) - \theta$',
        r'$\alpha \leftarrow \mathrm{atan2}(\sin\alpha,\ \cos\alpha)$',
    ]),
    ('Curvature-based steering law', [
        r'$\kappa = \frac{2\sin\alpha}{L_d(v)}$',
        r'$\omega = v\,\kappa = \frac{2v\sin\alpha}{L_d(v)}$',
    ]),
    ('Longitudinal speed control (feedforward + PI)', [
        r'$a_{ff,k} = k_{ff}\,\frac{v_{ref,k+1} - v_{ref,k}}{\Delta t}$',
        r'$e_{v,k} = v_{ref,k} - v_k$',
        r'$I_k = \mathrm{clip}\left(I_{k-1} + e_{v,k}\Delta t,\ -I_{max},\ I_{max}\right)$',
        r'$a_k = \mathrm{clip}\left(a_{ff,k} + K_p e_{v,k} + K_i I_k,\ -a_{max},\ a_{max}\right)$',
        r'$v_{k+1} = v_k + a_k\,\Delta t$',
    ]),
    ('Angular-rate slew limiting', [
        r'$\omega_k^{\mathrm{des}} = \frac{2v_{k+1}\alpha}{L_d(v)}$',
        r'$\dot\omega_k = \mathrm{clip}\left(\frac{\omega_k^{\mathrm{des}}-\omega_{k-1}}{\Delta t},\ -\dot\omega_{max},\ \dot\omega_{max}\right)$',
        r'$\omega_k = \omega_{k-1} + \dot\omega_k\,\Delta t$',
    ]),
]

# Symbol -> (description, units). Subscripts k / k-1 / k+1 denote the
# current / previous / next discrete controller sample and are described
# once here rather than per-symbol.
_SYMBOL_GLOSSARY = [
    (r'$L_d$',        'adaptive lookahead distance', 'm'),
    (r'$L_{d,0}$',    'base lookahead distance (config LOOKAHEAD_DISTANCE)', 'm'),
    (r'$k_{ld}$',     'lookahead gain (config LOOKAHEAD_GAIN)', 's'),
    (r'$v$',          'robot path (linear) velocity', 'm/s'),
    (r'$\omega$',     'robot angular velocity (yaw rate)', 'rad/s'),
    (r'$\theta$',     'robot heading', 'rad'),
    (r'$(x,y)$',      'robot position', 'm'),
    (r'$(x_t,y_t)$',  'lookahead target point on the reference path', 'm'),
    (r'$\alpha$',     'lookahead heading error, wrapped to (−π, π]', 'rad'),
    (r'$\kappa$',     'commanded path curvature', '1/m'),
    (r'$\Delta t$',   'controller sample time (config SIM_DT)', 's'),
    (r'$v_{ref}$',    'reference (path) velocity from the trajectory', 'm/s'),
    (r'$e_v$',        'velocity tracking error (v_ref − v)', 'm/s'),
    (r'$a$',          'commanded linear acceleration', 'm/s²'),
    (r'$a_{ff}$',     'feedforward acceleration term', 'm/s²'),
    (r'$k_{ff}$',     'feedforward gain (config k_ff)', '–'),
    (r'$K_p$',        'proportional speed gain (config K_SPEED)', '1/s'),
    (r'$K_i$',        'integral speed gain (config K_I_SPEED)', '1/s²'),
    (r'$I$',          'clamped integral of velocity error', 'm'),
    (r'$I_{max}$',    'integral clamp limit', 'm'),
    (r'$a_{max}$',    'maximum linear acceleration', 'm/s²'),
    (r'$\dot\omega$', 'angular acceleration (slew rate applied to ω)', 'rad/s²'),
    (r'$\dot\omega_{max}$', 'maximum angular acceleration (slew limit)', 'rad/s²'),
]

# Computer Modern renders like real LaTeX output (serif, formal) instead of
# matplotlib's default sans-serif mathtext.
plt.rcParams['mathtext.fontset'] = 'cm'


def _render_math_png(tex, fontsize=15, dpi=220):
    """Render a mathtext expression (e.g. r'$a=b$') to a base64 PNG data URI.

    @param tex<str>: Mathtext/LaTeX-subset expression, wrapped in $ ... $.
    @return<str>: 'data:image/png;base64,...' string for an <img src>.
    """
    fig = plt.figure()
    fig.text(0.0, 0.0, tex, fontsize=fontsize, color='black')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, transparent=True,
                bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('ascii')


def _build_glossary_html():
    """Render _SYMBOL_GLOSSARY to a <symbol, description, units> table body."""
    rows = []
    for sym, desc, units in _SYMBOL_GLOSSARY:
        sym_img = f'<img class="eq-sym" alt="{sym}" src="{_render_math_png(sym, fontsize=13)}">'
        rows.append(
            f'<tr><td class="sym">{sym_img}</td><td class="desc">{desc}</td><td>{units}</td></tr>'
        )
    return ''.join(rows)


def _build_method_section_html():
    """Render _METHOD_EQUATIONS to inline <img> tags grouped under headings."""
    groups_html = []
    for title, eqs in _METHOD_EQUATIONS:
        imgs = ''.join(
            f'<img class="eq" alt="{tex}" src="{_render_math_png(tex)}">'
            for tex in eqs
        )
        groups_html.append(
            f'<div class="eq-group"><div class="eq-title">{title}</div>{imgs}</div>'
        )
    return ''.join(groups_html)


def _gain_rows():
    """Controller-gain (label, formatted value) pairs, shared by the report
    and the slide deck. The trailing superscript marker flags the
    lookahead-gain no-op, footnoted in both places."""
    return [
        ('lookahead distance (base) [m]', f'{LOOKAHEAD_DISTANCE:.2f}'),
        ('lookahead gain [-]',            f'{LOOKAHEAD_GAIN:.2f}¹'),
        ('speed gain K [-]',              f'{K_SPEED:.2f}'),
        ('speed integral gain K_i [-]',   f'{K_I_SPEED:.2f}'),
        ('wheel base [m]',                f'{WHEEL_BASE:.2f}'),
    ]


def export_comparison_table_html(metrics, path):
    """Export the comparison table to a self-contained HTML report.

    @param metrics<list[dict]>: Per-corner metrics, as built in main().
    @param path<Path>: Output .html file path.
    """
    labels  = [m['label']  for m in metrics]
    we_strs = [m['we_str'] for m in metrics]

    def _tr(cells, header=False, group_start=False, best_idx=None):
        tag = 'th' if header else 'td'
        row_cls = ' class="group-start"' if group_start else ''
        tds = []
        for i, c in enumerate(cells):
            cell_cls = ' class="best"' if i == best_idx else ''
            tds.append(f'<{tag}{cell_cls}>{c}</{tag}>')
        return f'<tr{row_cls}>' + ''.join(tds) + '</tr>'

    # Every metric here is "lower is better" (time, energy, power, error, effort),
    # so the best column per row is simply the argmin — highlighted bold red.
    body_rows = [_tr(['w_e'] + we_strs)]
    for group in _COMPARISON_ROW_GROUPS:
        for i, (lbl, key, fmt) in enumerate(group):
            raw_vals = [m[key] for m in metrics]
            vals = [fmt.format(v) for v in raw_vals]
            best_idx = int(np.argmin(raw_vals)) + 1   # +1: cells[0] is the row label
            body_rows.append(_tr([lbl] + vals, group_start=(i == 0), best_idx=best_idx))

    gains_html = ''.join(_tr([lbl, val]) for lbl, val in _gain_rows())
    method_html = _build_method_section_html()
    glossary_html = _build_glossary_html()

    generated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pure Pursuit GPS-IMU Tracking Report</title>
<style>
  :root {{
    --bg: #ffffff;
    --surface: #ffffff;
    --text: #000000;
    --muted: #444444;
    --border: #dddddd;
    --border-strong: #999999;
    --accent: #000000;
    --accent-soft: #f2f2f2;
    --best: #d32f2f;
    color-scheme: light;
  }}

  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 2.5rem 1.5rem;
  }}
  .report {{
    max-width: 820px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }}
  .eyebrow {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
  }}
  h1 {{
    font-size: 1.5rem;
    font-weight: 600;
    margin: 0.25rem 0 0;
    text-wrap: balance;
  }}
  .meta {{
    color: var(--muted);
    font-size: 0.85rem;
    font-variant-numeric: tabular-nums;
  }}
  .meta b {{ color: var(--text); font-weight: 600; }}
  .table-wrap {{
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--surface);
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    min-width: 560px;
  }}
  th, td {{
    padding: 0.55rem 0.9rem;
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-family: ui-monospace, "SFMono-Regular", "Roboto Mono", Menlo, Consolas, monospace;
    font-size: 0.86rem;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  th:first-child, td:first-child {{
    text-align: left;
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    color: var(--muted);
    font-weight: 500;
  }}
  thead th {{
    background: var(--accent-soft);
    color: var(--text);
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    font-weight: 600;
    border-bottom: 1px solid var(--border-strong);
  }}
  thead th:first-child {{ color: var(--muted); font-weight: 600; }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tr.group-start td {{ border-top: 2px solid var(--border-strong); }}
  td.best {{ color: var(--best); font-weight: 700; }}
  .section-title {{ font-size: 0.95rem; font-weight: 600; margin: 0 0 0.5rem; }}
  .table-wrap.compact table {{ min-width: 0; max-width: 380px; }}
  .eq-group {{ margin: 0 0 0.9rem; }}
  .eq-group:last-child {{ margin-bottom: 0; }}
  .eq-title {{ font-size: 0.82rem; color: var(--muted); margin-bottom: 0.35rem; }}
  img.eq {{ display: block; max-width: 100%; height: auto; margin: 0.15rem 0 0.5rem; }}
  .table-wrap.glossary table {{ min-width: 0; max-width: 640px; }}
  .table-wrap.glossary th.sym, .table-wrap.glossary td.sym {{ text-align: center; width: 4.5rem; }}
  .table-wrap.glossary th.desc, .table-wrap.glossary td.desc {{
    text-align: left;
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    color: var(--text);
    white-space: normal;
  }}
  img.eq-sym {{ height: 18px; vertical-align: middle; }}
  .foot {{ color: var(--muted); font-size: 0.75rem; }}
</style>
</head>
<body>
<div class="report">
  <div>
    <div class="eyebrow">Simulation Report</div>
    <h1>Pure Pursuit &middot; GPS-IMU Tracking</h1>
  </div>
  <div class="meta">
    Generated <b>{generated}</b> &nbsp;&middot;&nbsp;
    GPS <b>{GPS_RATE_HZ:.1f} Hz</b> (std {GPS_STD_XY * 1e2:.1f} cm) &nbsp;&middot;&nbsp;
    IMU <b>{1.0 / SIM_DT:.1f} Hz</b> (theta std {np.rad2deg(IMU_STD_THETA):.2f} deg, v std {IMU_STD_V} m/s)
  </div>
  <div>
    <div class="section-title">Pure Pursuit gains</div>
    <div class="table-wrap compact">
      <table>
        <thead><tr><th>parameter</th><th>value</th></tr></thead>
        <tbody>{gains_html}</tbody>
      </table>
    </div>
    <div class="foot">&sup1; currently a no-op &mdash; TimeStepping.run_with_controller always passes
      input=[0,0] to the controller, so the speed-adaptive lookahead term never contributes.</div>
  </div>
  <div>
    <div class="section-title">Method &mdash; adaptive Pure Pursuit control law</div>
    {method_html}
  </div>
  <div>
    <div class="section-title">Symbol reference</div>
    <div class="table-wrap glossary">
      <table>
        <thead><tr><th class="sym">symbol</th><th class="desc">meaning</th><th>units</th></tr></thead>
        <tbody>{glossary_html}</tbody>
      </table>
    </div>
  </div>
  <div>
    <div class="section-title">Tracking comparison</div>
    <div class="table-wrap">
      <table>
        <thead>{_tr(['metric'] + labels, header=True)}</thead>
        <tbody>
          {''.join(body_rows)}
        </tbody>
      </table>
    </div>
  </div>
  <div class="foot">Source: examples/purepursuit_fine_sweep.py</div>
</div>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_doc, encoding='utf-8')
    print(f"  [saved] {path.name}")


# =============================================================================
# METHOD SLIDE DECK
# =============================================================================
# One equation per slide: (group, mathtext expression, prose explanation).
# Same derivation as _METHOD_EQUATIONS, unpacked into a per-equation
# walkthrough for presentation use.
_METHOD_SLIDES = [
    ('Adaptive lookahead distance',
     r'$L_d(v) = L_{d,0} + k_{ld}\, v$',
     "The lookahead distance L_d is not fixed — it grows linearly with the robot's "
     "current path velocity v. L_{d,0} is the base distance used at a standstill, and "
     "k_{ld} is the lookahead gain that sets how quickly the lookahead point moves outward "
     "as speed increases. This keeps the geometric steering law well-conditioned across the "
     "robot's operating speed range: a short lookahead at low speed lets it track tight "
     "corners closely, while a longer lookahead at high speed avoids the aggressive, "
     "oscillatory steering that a fixed close lookahead would otherwise produce."),
    ('Lookahead heading error',
     r'$\alpha = \mathrm{atan2}(y_t - y,\ x_t - x) - \theta$',
     "Once the lookahead point (x_t, y_t) on the reference path has been located, the "
     "controller computes alpha, the angle between the robot's current heading theta and "
     "the line of sight to that point. atan2 gives the bearing from the robot's position "
     "(x, y) to the target in the world frame; subtracting theta expresses that bearing "
     "relative to the direction the robot already faces."),
    ('Lookahead heading error',
     r'$\alpha \leftarrow \mathrm{atan2}(\sin\alpha,\ \cos\alpha)$',
     "Because angles wrap around at ±π, the raw difference above can land outside "
     "the (−π, π] range even when the true heading error is small. Passing "
     "alpha back through atan2(sin alpha, cos alpha) re-wraps it into the correct signed "
     "range, so the steering law always turns the robot the short way around."),
    ('Curvature-based steering law',
     r'$\kappa = \frac{2\sin\alpha}{L_d(v)}$',
     "This is the core geometric result of Pure Pursuit: the curvature kappa of the unique "
     "circular arc that starts at the robot's pose, is tangent to its heading theta, and "
     "passes through the lookahead point equals 2 sin(alpha) divided by the lookahead "
     "distance L_d. A larger heading error or a shorter lookahead distance both produce a "
     "tighter arc."),
    ('Curvature-based steering law',
     r'$\omega = v\,\kappa = \frac{2v\sin\alpha}{L_d(v)}$',
     "Angular velocity is curvature times forward speed, so combining the curvature law "
     "with the current velocity v gives the commanded yaw rate omega directly — the "
     "steering command sent to the robot alongside the commanded linear velocity."),
    ('Longitudinal speed control',
     r'$a_{ff,k} = k_{ff}\,\frac{v_{ref,k+1} - v_{ref,k}}{\Delta t}$',
     "Rather than relying purely on feedback, the speed controller anticipates the "
     "reference trajectory's own acceleration. a_ff is the finite-difference slope between "
     "the current and next reference sample, scaled by k_ff. This feedforward term lets the "
     "robot start accelerating or decelerating in step with the reference profile instead "
     "of lagging behind and reacting only after a velocity error has already appeared."),
    ('Longitudinal speed control',
     r'$e_{v,k} = v_{ref,k} - v_k$',
     "e_v is the plain feedback signal: how far the robot's current commanded speed v_k is "
     "from the reference speed v_ref,k at this step. It drives both the proportional and "
     "integral terms of the speed controller."),
    ('Longitudinal speed control',
     r'$I_k = \mathrm{clip}\left(I_{k-1} + e_{v,k}\Delta t,\ -I_{max},\ I_{max}\right)$',
     "The integral term I accumulates velocity error over time, removing any steady-state "
     "speed offset that the proportional term alone would leave behind. Because an "
     "unbounded integrator can wind up during large or prolonged errors — for example "
     "while the robot is still ramping up to speed — the accumulated value is clipped "
     "to ±I_max each step, a standard anti-windup safeguard."),
    ('Longitudinal speed control',
     r'$a_k = \mathrm{clip}\left(a_{ff,k} + K_p e_{v,k} + K_i I_k,\ -a_{max},\ a_{max}\right)$',
     "The three contributions are summed into a single commanded acceleration: the "
     "feedforward term a_ff, the proportional correction K_p e_v, and the integral "
     "correction K_i I. The result is clipped to ±a_max so the command never exceeds "
     "what the robot's drivetrain can physically deliver."),
    ('Longitudinal speed control',
     r'$v_{k+1} = v_k + a_k\,\Delta t$',
     "The commanded velocity for the next step is obtained by forward-Euler integrating "
     "the clipped acceleration over the controller's sample time Δt. v_{k+1} is what "
     "actually gets sent to the vehicle as the linear velocity command, and it also feeds "
     "back into the next step's curvature-based steering law."),
    ('Angular-rate slew limiting',
     r'$\omega_k^{\mathrm{des}} = \frac{2v_{k+1}\alpha}{L_d(v)}$',
     "Before the final angular velocity is issued, the steering law is re-evaluated using "
     "the just-updated velocity v_{k+1} rather than the stale v_k, giving the desired yaw "
     "rate omega^des for this step. Re-deriving it here keeps the linear and angular "
     "commands consistent with each other within the same control cycle."),
    ('Angular-rate slew limiting',
     r'$\dot\omega_k = \mathrm{clip}\left(\frac{\omega_k^{\mathrm{des}}-\omega_{k-1}}{\Delta t},\ -\dot\omega_{max},\ \dot\omega_{max}\right)$',
     "Jumping straight to omega^des could demand an unrealistically abrupt change in yaw "
     "rate. Instead, the controller computes the angular acceleration needed to reach it "
     "from the previous command omega_{k-1}, and clips that rate to ±ω̇_max "
     "— the same slew-rate limiting idea used for linear acceleration, now applied to "
     "steering."),
    ('Angular-rate slew limiting',
     r'$\omega_k = \omega_{k-1} + \dot\omega_k\,\Delta t$',
     "Finally, the rate-limited angular acceleration is integrated forward by one sample "
     "time to produce the angular velocity command omega_k that is actually sent to the "
     "robot. Together with v_{k+1} from the speed loop, this pair is the full control "
     "output returned to the simulator each step."),
]


def export_method_slides_html(path):
    """Export the adaptive Pure Pursuit control law as a landscape slide deck:
    one slide per conceptual group (_METHOD_SLIDES entries sharing the same
    group label), each equation followed inline by its prose explanation, so
    related equations sit together instead of one-equation-per-slide. Framed
    by a title slide and a closing controller-gains slide.

    @param path<Path>: Output .html file path.
    """
    def _group_slide(group, items):
        rows = ''.join(
            f'<div class="eq-row">'
            f'<img class="eq" alt="{tex}" src="{_render_math_png(tex, fontsize=22)}">'
            f'<p>{para}</p>'
            f'</div>'
            for _, tex, para in items
        )
        return (f'<section class="slide eq-slide">'
                f'<div class="eyebrow">{group}</div>'
                f'<div class="eq-rows">{rows}</div>'
                f'</section>')

    slides = [
        '<section class="slide title-slide">'
        '<div class="eyebrow">Method</div>'
        '<h1>Adaptive Pure Pursuit</h1>'
        '<p class="subtitle">Control-law derivation &middot; '
        'controllers/purepursuit/purepursuit.py</p>'
        '</section>'
    ]
    for group, items in itertools.groupby(_METHOD_SLIDES, key=lambda t: t[0]):
        slides.append(_group_slide(group, list(items)))

    gains_rows_html = ''.join(
        f'<tr><td>{lbl}</td><td>{val}</td></tr>' for lbl, val in _gain_rows()
    )
    slides.append(
        '<section class="slide closing-slide">'
        '<div class="eyebrow">Configuration</div>'
        '<h1>Controller gains</h1>'
        '<table class="gains"><thead><tr><th>parameter</th><th>value</th></tr></thead>'
        f'<tbody>{gains_rows_html}</tbody></table>'
        '<p class="foot-note">&sup1; currently a no-op &mdash; TimeStepping.run_with_controller '
        'always passes input=[0,0] to the controller, so the speed-adaptive lookahead term '
        'never contributes.</p>'
        '</section>'
    )

    slides_html = ''.join(slides)
    n_slides = len(slides)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Adaptive Pure Pursuit &mdash; Slides</title>
<style>
  :root {{
    --page-bg: #e7e7e5;
    --bg: #ffffff;
    --text: #000000;
    --muted: #555555;
    --rule: #dddddd;
    --accent: #8a1f1f;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ height: 100%; margin: 0; }}
  body {{
    background: var(--page-bg);
    font-family: Georgia, "Times New Roman", serif;
    color: var(--text);
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .stage {{
    position: relative;
    width: min(94vw, 167vh);
    aspect-ratio: 16 / 9;
    background: var(--bg);
    border-radius: 8px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.18);
    overflow: hidden;
  }}
  .slide {{
    position: absolute;
    inset: 0;
    display: none;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 5% 9% 12%;
  }}
  .slide.active {{ display: flex; }}
  .eyebrow {{
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.82vw;
    min-font-size: 0.7rem;
    color: var(--accent);
    margin-bottom: 1.1rem;
  }}
  .slide h1 {{ font-size: 2.6rem; font-weight: 600; margin: 0 0 0.8rem; text-wrap: balance; }}
  .slide .subtitle {{
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    color: var(--muted);
    font-size: 1rem;
  }}
  .slide p {{
    font-size: 1.28rem;
    line-height: 1.6;
    max-width: 68ch;
    color: #1a1a1a;
    margin: 0;
  }}
  .eq-slide {{
    align-items: flex-start;
    justify-content: flex-start;
    text-align: left;
    padding-top: 6.5%;
  }}
  .eq-rows {{
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 1.3rem;
    width: 100%;
    height: 100%;
    overflow-y: auto;
    padding-right: 0.4rem;
  }}
  .eq-row {{
    display: flex;
    align-items: center;
    gap: 2.4rem;
  }}
  .eq-row img.eq {{
    flex: 0 0 32%;
    max-width: 32%;
    height: auto;
  }}
  .eq-row p {{
    flex: 1 1 auto;
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    text-align: left;
    font-size: 1.05rem;
    line-height: 1.55;
    color: #1a1a1a;
  }}
  .closing-slide table.gains {{
    border-collapse: collapse;
    margin: 0.6rem 0 1rem;
    font-family: ui-monospace, "SFMono-Regular", "Roboto Mono", Menlo, Consolas, monospace;
    font-size: 1rem;
  }}
  .closing-slide table.gains th, .closing-slide table.gains td {{
    border: 1px solid var(--rule);
    padding: 0.4rem 1.1rem;
  }}
  .closing-slide table.gains th {{
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    background: #f2f2f2;
  }}
  .closing-slide .foot-note {{
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    font-size: 0.8rem;
    color: var(--muted);
    max-width: 60ch;
  }}
  .nav {{
    position: absolute;
    left: 0; right: 0; bottom: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.9rem 1.6rem;
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    font-size: 0.85rem;
    color: var(--muted);
  }}
  .nav button {{
    font: inherit;
    background: none;
    border: 1px solid var(--rule);
    border-radius: 4px;
    padding: 0.35rem 0.9rem;
    color: var(--text);
    cursor: pointer;
  }}
  .nav button:hover:not(:disabled) {{ background: #f2f2f2; }}
  .nav button:disabled {{ opacity: 0.3; cursor: default; }}
  .nav button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
</style>
</head>
<body>
<div class="stage" id="stage">
  {slides_html}
  <div class="nav">
    <button id="prevBtn" type="button" aria-label="Previous slide">&larr; Prev</button>
    <span id="counter">1 / {n_slides}</span>
    <button id="nextBtn" type="button" aria-label="Next slide">Next &rarr;</button>
  </div>
</div>
<script>
  const slides = document.querySelectorAll('.slide');
  const counter = document.getElementById('counter');
  const prevBtn = document.getElementById('prevBtn');
  const nextBtn = document.getElementById('nextBtn');
  let i = 0;
  function show(n) {{
    slides.forEach((s, idx) => s.classList.toggle('active', idx === n));
    counter.textContent = (n + 1) + ' / ' + slides.length;
    prevBtn.disabled = n === 0;
    nextBtn.disabled = n === slides.length - 1;
  }}
  function next() {{ if (i < slides.length - 1) {{ i++; show(i); }} }}
  function prev() {{ if (i > 0) {{ i--; show(i); }} }}
  prevBtn.addEventListener('click', prev);
  nextBtn.addEventListener('click', next);
  window.addEventListener('keydown', (e) => {{
    if (e.key === 'ArrowRight' || e.key === ' ') {{ e.preventDefault(); next(); }}
    if (e.key === 'ArrowLeft') {{ e.preventDefault(); prev(); }}
  }});
  show(0);
</script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_doc, encoding='utf-8')
    print(f"  [saved] {path.name}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"Data directory : {DATA_DIR}")
    print(f"GPS            : {GPS_RATE_HZ:.1f} Hz, std={GPS_STD_XY * 1e2:.1f} cm")
    print(f"IMU            : {1.0 / SIM_DT:.1f} Hz, "
          f"theta std={np.rad2deg(IMU_STD_THETA):.2f} deg, "
          f"v std={IMU_STD_V} m/s")
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
    metrics = []
    for key_we, style, lbl_prefix in specs:
        target = opt[key_we]
        c, actual_we = _nearest_corner(corners, target)
        we_str = f'{actual_we:.4f}'
        traj = corner_to_trajectory(c)
        sim  = run_purepursuit(traj, seed=0)

        nt     = min(traj.x.shape[0], sim.x_out.shape[1])
        trk_xy = sim.x_out[:2, :].T
        ref_xy = traj.x[:, :2]
        cte    = cross_track_error(ref_xy, trk_xy)
        he     = heading_error(traj.x[:nt, 2], sim.x_out[2, :nt])
        v_err  = np.abs(traj.u[0, :nt] - sim.u_out[0, :nt])
        final_err = np.hypot(
            sim.x_out[0, -1] - traj.x[-1, 0],
            sim.x_out[1, -1] - traj.x[-1, 1],
        )

        print(f"  {lbl_prefix:12s}  w_e={we_str}  "
              f"max_CTE={cte.max()*1e2:.2f} cm  "
              f"final_err={final_err*1e2:.2f} cm")

        metrics.append({
            'label':        lbl_prefix,
            'we_str':       we_str,
            'mission_time': float(c['time'][-1] - c['time'][0]),
            'total_energy': float(np.trapz(c['power'], c['time'])),
            'peak_power':   float(c['power'].max()),
            'max_cte':      float(cte.max()  * 1e2),
            'mean_cte':     float(cte.mean() * 1e2),
            'rms_cte':      float(np.sqrt((cte**2).mean()) * 1e2),
            'final_err':    float(final_err  * 1e2),
            'max_he':       float(np.rad2deg(np.abs(he).max())),
            'mean_he':      float(np.rad2deg(np.abs(he).mean())),
            'max_verr':     float(v_err.max()),
            'track_dur':    float(sim.t_out[-1]),
        })
        results.append((we_str, style, traj, sim))

    print()
    print_comparison_table(metrics)
    if SAVE_FIGS:
        export_comparison_table_html(metrics, FIG_DIR / 'comparison_report.html')
        export_method_slides_html(FIG_DIR / 'method_slides.html')
    print()
    fig_xy_tracking(results)
    fig_cte(results)
    fig_velocity_tracking(results)
    fig_heading_error(results)
    fig_acceleration(results)
    fig_jerk(results)
    fig_power(results)

    plt.show()
    plt.close('all')


# =============================================================================
if __name__ == '__main__':
    main()
