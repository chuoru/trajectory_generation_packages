#!/usr/bin/env python3
##
# @file analyze_fine_sweep.py
#
# @brief Offline analysis of fine-sweep CSV output.
#
# Loads sweep_statistics.csv and corners_by_we/ from a previous run of
# differential_drive_path_segment_fine_sweep.py, applies configurable
# outlier filters, re-identifies the three optimal w_e values, and
# regenerates the key figures.
#
# Workflow
# --------
#   1. Run differential_drive_path_segment_fine_sweep.py once to produce CSVs.
#   2. Edit FILTER_* thresholds below.
#   3. Run this script to regenerate figures without re-solving the OCP.
#
# Figures
# -------
#   fig_sweep_stats.png    -- peak power / energy / time / d2 vs w_e
#   fig_pareto.png         -- energy vs peak-power Pareto front
#   fig_corner_velocity.png -- v(t) for the three selected corners
#   fig_corner_power.png   -- P(t) for the three selected corners
#   fig_corner_xy.png      -- XY path for the three selected corners
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/05/08

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trajectory_generators.euler_jlap_coverage import EulerJLAPCoverage

# =============================================================================
# CONFIGURATION  --  edit these before running
# =============================================================================
DATA_DIR  = Path(__file__).resolve().parent / 'csv_output_fine_sweep'
SAVE_FIGS = True
FIG_DIR   = DATA_DIR / 'analysis_figures'

# Outlier filters applied to the loaded sweep statistics.
# Set a multiplier to np.inf to disable that filter.
PP_FILTER_MULT = 1.5    # drop points where peak_power  > X * median(peak_power)
TE_FILTER_MULT = 1.5    # drop points where total_energy > X * median(total_energy)
MT_FILTER_MULT = 5.0    # drop points where mission_time > X * median(mission_time)

STYLE_TIME   = ('black', '-')    # solid
STYLE_KNEE   = ('red',   '--')   # dashed
STYLE_ENERGY = ('blue',  '-.')   # dash-dot
COL_REF      = '#888888'

# Wheel jerk limit [m/s³] — must match J_LIM in differential_drive_path_segment_fine_sweep.py
J_LIM = 3.547

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
# DATA LOADING
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

    t_c_start   = pre['time'][-1] + JLAP_DT
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
# FILTERING AND OPTIMUM SELECTION
# =============================================================================
def _pareto_front_idx(te, pp):
    """Indices of non-dominated points in (te, pp) space, sorted by te ascending."""
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
    we, pp, te, mt, d2 = sw['we'], sw['pp'], sw['te'], sw['mt'], sw['d2']

    time_idx   = int(np.argmin(mt))
    energy_idx = int(np.argmin(te))

    # Pareto knee: restrict to the non-dominated front in (te, pp) space,
    # then find the point with max perpendicular distance from the chord
    # connecting the two extreme Pareto-optimal endpoints.
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
    """Return the corner dict whose w_e key is closest to target_we."""
    best = min(corners.keys(), key=lambda k: abs(k - target_we))
    return corners[best], best


# =============================================================================
# FIGURE HELPERS
# =============================================================================
def _savefig(fig, name):
    if SAVE_FIGS:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        out = FIG_DIR / name
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f"  [saved] {name}")


def fig_sweep_stats(sw, opt):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             num='Sweep Statistics')
    we, pp, te, mt, d2 = sw['we'], sw['pp'], sw['te'], sw['mt'], sw['d2']

    panels = [
        (axes[0, 0], pp, 'o', STYLE_KNEE[0],   'Peak Power [W]',   'knee_idx'),
        (axes[0, 1], te, 's', STYLE_ENERGY[0], 'Total Energy [J]', 'energy_idx'),
        (axes[1, 0], mt, '^', STYLE_TIME[0],   'Mission Time [s]', 'time_idx'),
        (axes[1, 1], d2, 'D', COL_REF,         'd²(P)/d(w_e)²',    'knee_idx'),
    ]
    for ax, data, marker, color, ylabel, opt_key in panels:
        ax.scatter(we, data, marker=marker, color=color, s=20)
        ki = opt[opt_key]
        ax.scatter([we[ki]], [data[ki]], color='black', s=90, zorder=6)
        ax.axvline(we[ki], color='black', ls='--', lw=1.0)
        ax.set_xscale('symlog', linthresh=5e-4)
        ax.set_xlabel('w_energy (symlog)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(labelsize=11)

    fig.tight_layout()
    _savefig(fig, 'fig_sweep_stats.png')


def fig_pareto(sw, opt):
    fig, ax = plt.subplots(figsize=(7, 5), num='Pareto Front')
    we, pp, te = sw['we'], sw['pp'], sw['te']
    sc = ax.scatter(te, pp, c=we, cmap='viridis', s=25, zorder=3)
    plt.colorbar(sc, ax=ax, label='w_energy')

    for key_i, key_we, style, lbl_prefix in [
        ('time_idx',   'time_we',   STYLE_TIME,   'Time-opt'),
        ('knee_idx',   'knee_we',   STYLE_KNEE,   'Knee'),
        ('energy_idx', 'energy_we', STYLE_ENERGY, 'Energy-opt'),
    ]:
        col, _ = style
        i = opt[key_i]
        lbl = f'{lbl_prefix}  w_e={opt[key_we]:.4f}'
        ax.scatter([te[i]], [pp[i]], color=col, s=110, zorder=5,
                   label=lbl, edgecolors='black', linewidths=0.8)

    ax.set_xlabel('Total Energy [J]', fontsize=12)
    ax.set_ylabel('Peak Power [W]', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _savefig(fig, 'fig_pareto.png')


def fig_corner_velocity(corners, opt):
    fig, ax = plt.subplots(figsize=(9, 4), num='Corner Velocity Profiles')
    for key_we, style, lbl_prefix in [
        ('time_we',   STYLE_TIME,   'Time-opt'),
        ('knee_we',   STYLE_KNEE,   'Knee'),
        ('energy_we', STYLE_ENERGY, 'Energy-opt'),
    ]:
        col, ls = style
        c, actual_we = _nearest_corner(corners, opt[key_we])
        ax.plot(c['time'], c['v'], lw=1.8, color=col, ls=ls,
                label=f'{lbl_prefix}  w_e={actual_we:.4f}')

    ax.set_xlabel('time [s]', fontsize=12)
    ax.set_ylabel('v [m/s]', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _savefig(fig, 'fig_corner_velocity.png')


def fig_corner_power(corners, opt):
    fig, ax = plt.subplots(figsize=(9, 4), num='Corner Power Profiles')
    for key_we, style, lbl_prefix in [
        ('time_we',   STYLE_TIME,   'Time-opt'),
        ('knee_we',   STYLE_KNEE,   'Knee'),
        ('energy_we', STYLE_ENERGY, 'Energy-opt'),
    ]:
        col, ls = style
        c, actual_we = _nearest_corner(corners, opt[key_we])
        ax.plot(c['time'], c['power'], lw=1.8, color=col, ls=ls,
                label=f'{lbl_prefix}  w_e={actual_we:.4f}')

    ax.set_xlabel('time [s]', fontsize=12)
    ax.set_ylabel('Power [W]', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _savefig(fig, 'fig_corner_power.png')


def fig_corner_acceleration(corners, opt):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                   num='Corner Acceleration Profiles')
    for key_we, style, lbl_prefix in [
        ('time_we',   STYLE_TIME,   'Time-opt'),
        ('knee_we',   STYLE_KNEE,   'Knee'),
        ('energy_we', STYLE_ENERGY, 'Energy-opt'),
    ]:
        col, ls = style
        c, actual_we = _nearest_corner(corners, opt[key_we])
        a_lin = np.gradient(c['v'],     c['time'])
        a_ang = np.gradient(c['omega'], c['time'])
        lbl = f'{lbl_prefix}  w_e={actual_we:.4f}'
        ax1.plot(c['time'], a_lin, lw=1.8, color=col, ls=ls, label=lbl)
        ax2.plot(c['time'], a_ang, lw=1.8, color=col, ls=ls)

    ax1.set_ylabel('Linear accel. [m/s²]', fontsize=12)
    ax2.set_ylabel('Angular accel. [rad/s²]', fontsize=12)
    ax2.set_xlabel('time [s]', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.tick_params(labelsize=11)
    ax2.tick_params(labelsize=11)
    fig.tight_layout()
    _savefig(fig, 'fig_corner_acceleration.png')


def fig_corner_jerk(corners, opt):
    # Check if wheel jerk was saved in the CSV (new format); fall back to
    # numerical differentiation of acceleration only if unavailable.
    sample_c = next(iter(corners.values()))
    has_wheel_jerk = sample_c.get('jerk_r') is not None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                   num='Corner Jerk Profiles')
    for key_we, style, lbl_prefix in [
        ('time_we',   STYLE_TIME,   'Time-opt'),
        ('knee_we',   STYLE_KNEE,   'Knee'),
        ('energy_we', STYLE_ENERGY, 'Energy-opt'),
    ]:
        col, ls = style
        c, actual_we = _nearest_corner(corners, opt[key_we])
        lbl = f'{lbl_prefix}  w_e={actual_we:.4f}'
        if has_wheel_jerk:
            ax1.plot(c['time'], c['jerk_r'], lw=1.8, color=col, ls=ls, label=lbl)
            ax2.plot(c['time'], c['jerk_l'], lw=1.8, color=col, ls=ls)
        else:
            a_lin = np.gradient(c['v'],     c['time'])
            a_ang = np.gradient(c['omega'], c['time'])
            ax1.plot(c['time'], np.gradient(a_lin, c['time']), lw=1.8, color=col, ls=ls, label=lbl)
            ax2.plot(c['time'], np.gradient(a_ang, c['time']), lw=1.8, color=col, ls=ls)

    for ax in (ax1, ax2):
        ax.axhline( J_LIM, color='red', lw=1.0, ls='--', label=f'±J_LIM={J_LIM:.3f}')
        ax.axhline(-J_LIM, color='red', lw=1.0, ls='--')

    ylabel = 'Wheel jerk right [m/s³]' if has_wheel_jerk else 'Linear jerk [m/s³]'
    ax1.set_ylabel(ylabel, fontsize=12)
    ax2.set_ylabel('Wheel jerk left [m/s³]' if has_wheel_jerk else 'Angular jerk [rad/s³]', fontsize=12)
    ax2.set_xlabel('time [s]', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.tick_params(labelsize=11)
    ax2.tick_params(labelsize=11)
    fig.tight_layout()
    _savefig(fig, 'fig_corner_jerk.png')


def fig_corner_xy(corners, opt):
    fig, ax = plt.subplots(figsize=(6, 6), num='Corner XY Paths')
    ax.set_aspect('equal')
    for key_we, style, lbl_prefix in [
        ('time_we',   STYLE_TIME,   'Time-opt'),
        ('knee_we',   STYLE_KNEE,   'Knee'),
        ('energy_we', STYLE_ENERGY, 'Energy-opt'),
    ]:
        col, ls = style
        c, actual_we = _nearest_corner(corners, opt[key_we])
        ax.plot(c['x'], c['y'], lw=1.8, color=col, ls=ls,
                label=f'{lbl_prefix}  w_e={actual_we:.4f}')

    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _savefig(fig, 'fig_corner_xy.png')


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"Data directory : {DATA_DIR}")
    print(f"Filters        : PP <= {PP_FILTER_MULT}×median  "
          f"TE <= {TE_FILTER_MULT}×median  "
          f"MT <= {MT_FILTER_MULT}×median")
    print()

    sw_raw  = load_sweep_stats()
    corners = load_corners()
    print(f"  Loaded {len(sw_raw['we'])} sweep points, "
          f"{len(corners)} corner trajectories.")

    sw  = apply_filters(sw_raw)
    opt = pick_optima(sw)

    print(f"\n  Time-optimal    w_e = {opt['time_we']:.6f}  "
          f"T = {sw['mt'][opt['time_idx']]:.3f} s")
    print(f"  Pareto knee     w_e = {opt['knee_we']:.6f}  "
          f"Peak P = {sw['pp'][opt['knee_idx']]:.3f} W")
    print(f"  Energy-optimal  w_e = {opt['energy_we']:.6f}  "
          f"E = {sw['te'][opt['energy_idx']]:.3f} J")
    print()

    fig_sweep_stats(sw, opt)
    fig_pareto(sw, opt)
    fig_corner_velocity(corners, opt)
    fig_corner_power(corners, opt)
    fig_corner_acceleration(corners, opt)
    fig_corner_jerk(corners, opt)
    fig_corner_xy(corners, opt)

    plt.show()
    plt.close('all')


# =============================================================================
if __name__ == '__main__':
    main()
