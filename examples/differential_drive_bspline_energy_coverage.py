#!/usr/bin/env python3
##
# @file differential_drive_bspline_energy_coverage.py
#
# @brief Comparison of time-optimal vs energy-aware B-spline coverage for a
# differential drive robot navigating an L-shaped path.
#
# Runs two solvers on the same waypoints:
#   (A) BSplineEnergyCoverage (w_energy=0) - minimises total traversal time
#   (B) BSplineEnergyCoverage             - minimises w_time * T + w_energy * E_total
#
# Text output — full comparison table (12 metrics, time-opt vs energy-aware).
#
# Figures:
#   Figure 1 — Coverage Trajectory
#               XY path + polyhedra corridor constraints + control points
#   Figure 2 — Kinematic Profiles
#               v(t), ω(t), linear acceleration, angular acceleration
#   Figure 3 — TJ108 Energy Analysis
#               P(t) per motor, wheel angular velocities,
#               energy bar comparison, energy density map
#   Figure 8 — Per-wheel angular velocity, acceleration, and jerk
#               (time-optimal vs energy-aware, right/left wheel overlay)
#
# For the dense w_energy sweep analysis (Pareto front, peak-power suppression,
# trajectory overlay), see differential_drive_bspline_energy_sweep.py.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/04/16

# Standard library
import sys
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Internal library
from trajectory_generators.bspline_energy_coverage import BSplineEnergyCoverage


# =============================================================================
# PAPER FIGURE EXPORT
# Set SAVE_FIGS = True to write paper-quality PNGs into the Writting directory.
# =============================================================================
SAVE_FIGS   = True
FIG_OUT_DIR = (pathlib.Path(__file__).resolve().parent.parent.parent
               / 'Writting' / 'energy_aware')

# w_energy weight used for the energy-aware run (Run B).
# ~0.22 balances the energy/time gradient ratio for this problem
# (E_motor/nt ≈ 1.4, T_mission/nt ≈ 0.31 → ratio ≈ 4.5 → w_e* ≈ 1/4.5).
W_ENERGY = 0.22


def _savefig(fig, filename):
    """Save fig to FIG_OUT_DIR/<filename> at 300 dpi when SAVE_FIGS is True."""
    if SAVE_FIGS:
        FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = FIG_OUT_DIR / filename
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f"  [paper] Saved {filename} -> {out}")


# =============================================================================
# Shared configuration
# =============================================================================
WAYPOINTS = [
    [0.0, 0.0,  0.0],           # start: heading east
    [1.0, 0.0,  0.0],           # corner
    [1.0, 1.0,  np.pi / 2],     # end: heading north
]

COMMON_KWARGS = dict(
    bound=0.17,
    n_ctrl_pts=6,
    spline_order=3,
    n_sampling=15,
    vel_max=[0.75, 0.75, 0.196],
    vel_min_lin=0.001,
    eps_nonh=0.001,
)

# TJ108 robot-specific parameters
ROBOT_PARAMS = {'l': 0.53 / 2, 'r': 0.3}

# TJ108 energy model coefficients — per-motor bench-test identification.
# [c1, c2, c3, c4, c5, c6] for:
#   P = c1 + c2*omega + c3*omega^2 + c4*omega^3 + c5*omega_dot + c6*omega_dot^2
ENERGY_COEFFS_RIGHT = [
    0.302433145557389,       # c1 – static / base load
    31.887262598534413,      # c2 – viscous (linear)
    2.4140287888312457,      # c3 – viscous (quadratic)
    0.9658866923308425,      # c4 – viscous (cubic)
    0.8260871406535432,      # c5 – inertial (linear in alpha)
    2.37456174658809e-08,    # c6 – inertial (quadratic in alpha)
]

ENERGY_COEFFS_LEFT = [
    0.33789198669595977,     # c1
    28.204019732889346,      # c2
    2.5903002025839688,      # c3
    0.00847962183165042,     # c4
    6.412423386896174e-09,   # c5
    0.3614761744737831,      # c6
]

P_ELECTRONICS = 2.0   # constant hotel load [W]

# Colours used across all figures
COL_A = 'steelblue'   # time-optimal
COL_B = 'tomato'      # energy-aware


# =============================================================================
# PAPER STYLE
# =============================================================================
def _set_paper_style():
    plt.rcParams.update({
        'font.size':       12,
        'axes.labelsize':  12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'axes.titlesize':  12,
        'axes.grid':       False,
    })


# =============================================================================
# Corridor (polyhedra) visualization
# Ported from Fabian's getPolyhedronConstr_overlap_plot.m
# =============================================================================
def _draw_polyhedra_corridors(ax, waypoints, bound,
                               color='gold', alpha=0.18,
                               edgecolor='steelblue', lw=0.8):
    """Draw rectangular corridor constraints (polyhedra) for each waypoint segment.

    Each corridor extends `bound` beyond both segment endpoints in the travel
    direction and ±bound in the normal direction, so consecutive corridors
    overlap at the waypoint vertices — matching Fabian's overlap formulation.

    @param ax<Axes>:         Target matplotlib axes.
    @param waypoints<list>:  List of [x, y, ...] waypoints.
    @param bound<float>:     Corridor half-width [m].
    """
    patches = []
    for i in range(len(waypoints) - 1):
        A = np.array(waypoints[i][:2], dtype=float)
        B = np.array(waypoints[i + 1][:2], dtype=float)

        d = B - A
        d = d / np.linalg.norm(d)
        n = np.array([-d[1], d[0]])

        corners = np.array([
            B + d * bound + n * bound,   # corner_up_right
            B + d * bound - n * bound,   # corner_up_left
            A - d * bound - n * bound,   # corner_down_left
            A - d * bound + n * bound,   # corner_down_right
        ])
        patches.append(MplPolygon(corners, closed=True))

    col = PatchCollection(patches, facecolor=color, alpha=alpha,
                          edgecolor=edgecolor, linewidth=lw, zorder=1)
    ax.add_collection(col)

    # Dummy fill for the legend entry (draws nothing, registers the label).
    ax.fill([], [], color=color, alpha=alpha + 0.15, edgecolor=edgecolor,
            linewidth=lw, label=f'Corridor (bound={bound:.2f} m)')


# =============================================================================
# Helpers — wheel kinematics
# =============================================================================
def _compute_wheel_kinematics(res, l, r, dt):
    """Compute per-wheel angular velocity, acceleration, and jerk.

    Uses analytical OCP derivatives (jerk_r / jerk_l) when present in the
    result dict (BSplineEnergyCoverage output); falls back to np.gradient.

    @param res<dict>: Trajectory result dict.
    @param l<float>:  Half-wheelbase [m].
    @param r<float>:  Wheel radius [m].
    @param dt<float>: Sampling interval for numerical differentiation [s].
    @return dict with keys: time, omega_r, omega_l, alpha_r, alpha_l, jerk_r, jerk_l.
    """
    t     = res.get('time_ik', res['time'])
    v     = res['v']
    omega = res['omega']

    omega_r = (v + l * omega) / r
    omega_l = (v - l * omega) / r

    if 'acc_path' in res and 'alpha' in res:
        alpha_r = (res['acc_path'] + l * res['alpha']) / r
        alpha_l = (res['acc_path'] - l * res['alpha']) / r
    else:
        alpha_r = np.gradient(omega_r, dt)
        alpha_l = np.gradient(omega_l, dt)

    if 'jerk_r' in res and 'jerk_l' in res:
        jerk_r = res['jerk_r']
        jerk_l = res['jerk_l']
    else:
        jerk_r = np.gradient(alpha_r, dt) * r
        jerk_l = np.gradient(alpha_l, dt) * r

    return dict(time=t, omega_r=omega_r, omega_l=omega_l,
                alpha_r=alpha_r, alpha_l=alpha_l,
                jerk_r=jerk_r, jerk_l=jerk_l)


# =============================================================================
# Helpers — power computation and metrics
# =============================================================================
def _compute_wheel_power(v_arr, omega_arr, robot_params, cr, cl, p_elec):
    """Evaluate the polynomial power model on IK-output (v, omega) arrays.

    @return (P_total, P_right, P_left, energy) all (M-1,) arrays + scalar [J].
    """
    l = robot_params['l']
    dt = 0.01   # matches ts_des in generate_trajectory

    v_r = v_arr + l * omega_arr
    v_l = v_arr - l * omega_arr
    a_r = np.gradient(v_r, dt)
    a_l = np.gradient(v_l, dt)

    def _P(v, a, c):
        raw = (c[0] * a**2 + c[1] * v**2
               + np.abs(c[2] * a) + np.abs(c[3] * v)
               + np.abs(c[4] * v * a) + c[5])
        return np.maximum(raw, 0.0)

    P_r = _P(v_r, a_r, cr)
    P_l = _P(v_l, a_l, cl)
    P_total = P_r + P_l + p_elec
    energy = float(np.trapz(P_total, dx=dt))
    return P_total, P_r, P_l, energy


def _compute_metrics(res, robot_params, cr, cl, p_elec):
    """Compute the full set of comparison metrics for one trajectory result.

    @return dict with numeric metrics AND per-motor power arrays for plotting.
    """
    v       = res['v']
    omega   = res['omega']
    states  = res['states']
    t_ik    = res['time_ik']
    dt      = 0.01

    # Path length along the optimised spline
    dx = np.diff(states[:, 0])
    dy = np.diff(states[:, 1])
    path_length = float(np.sum(np.sqrt(dx**2 + dy**2)))

    # Per-motor power on the IK time grid (consistent for both solvers)
    P_total, P_r, P_l, energy_ik = _compute_wheel_power(
        v, omega, robot_params, cr, cl, p_elec)

    # Prefer the optimizer's own energy value when available (more accurate).
    energy = res.get('energy', energy_ik)

    avg_power        = float(np.mean(P_total))
    peak_power       = float(np.max(P_total))
    energy_per_meter = energy / path_length if path_length > 1e-9 else float('inf')

    v_max     = float(np.max(np.abs(v)))
    omega_max = float(np.max(np.abs(omega)))

    l = robot_params['l']
    r = robot_params['r']
    omega_r     = (v + l * omega) / r
    omega_l     = (v - l * omega) / r
    omega_r_max = float(np.max(np.abs(omega_r)))
    omega_l_max = float(np.max(np.abs(omega_l)))

    a_lin   = np.gradient(v,     dt)
    a_ang   = np.gradient(omega, dt)
    a_rms   = float(np.sqrt(np.mean(a_lin**2)))
    alp_rms = float(np.sqrt(np.mean(a_ang**2)))

    return {
        # --- scalar metrics ---
        'total_time':       res['time'][-1],
        'path_length':      path_length,
        'energy':           energy,
        'energy_per_meter': energy_per_meter,
        'avg_power':        avg_power,
        'peak_power':       peak_power,
        'v_max':            v_max,
        'omega_max':        omega_max,
        'omega_r_max':      omega_r_max,
        'omega_l_max':      omega_l_max,
        'a_rms':            a_rms,
        'alpha_rms':        alp_rms,
        # --- arrays for plotting ---
        'P_total':  P_total,
        'P_right':  P_r,
        'P_left':   P_l,
        'omega_r':  omega_r,
        'omega_l':  omega_l,
        'a_lin':    a_lin,
        'a_ang':    a_ang,
        'time_ik':  t_ik,
    }


def _print_comparison(mA, mB):
    """Print a two-column comparison table with percentage change."""
    rows = [
        ('Total time',          's',      'total_time'),
        ('Path length',         'm',      'path_length'),
        ('Total energy',        'J',      'energy'),
        ('Energy / meter',      'J/m',    'energy_per_meter'),
        ('Average power',       'W',      'avg_power'),
        ('Peak power',          'W',      'peak_power'),
        ('Max |v|',             'm/s',    'v_max'),
        ('Max |ω|',             'rad/s',  'omega_max'),
        ('Max ω_right',         'rad/s',  'omega_r_max'),
        ('Max ω_left',          'rad/s',  'omega_l_max'),
        ('RMS lin. accel',      'm/s²',   'a_rms'),
        ('RMS ang. accel',      'rad/s²', 'alpha_rms'),
    ]

    hdr = (f"  {'Metric':<26} {'Unit':<8} "
           f"{'Time-opt':>10}  {'Energy-aware':>12}  {'Δ%':>7}")
    print()
    print(hdr)
    print("  " + "─" * 68)
    for label, unit, key in rows:
        a = mA[key]
        b = mB[key]
        if abs(a) > 1e-12:
            delta = (b - a) / abs(a) * 100
            sign = '+' if delta >= 0 else ''
            ds = f'{sign}{delta:.1f}%'
        else:
            ds = '   n/a'
        print(f"  {label:<26} {unit:<8} {a:>10.4f}  {b:>12.4f}  {ds:>7}")
    print()


# =============================================================================
# Main
# =============================================================================
def main():
    _set_paper_style()

    # --- Run A: time-optimal --------------------------------------------------
    print("=" * 60)
    print("Run A: time-optimal  (w_time=1.0, w_energy=0.0)")
    print("=" * 60)
    gen_time = BSplineEnergyCoverage(
        waypoints=WAYPOINTS,
        robot_params=ROBOT_PARAMS,
        energy_coeffs_right=ENERGY_COEFFS_RIGHT,
        energy_coeffs_left=ENERGY_COEFFS_LEFT,
        w_time=1.0,
        w_energy=0.0,
        p_electronics=P_ELECTRONICS,
        **COMMON_KWARGS,
    )
    res_time = gen_time.generate_trajectory()

    # --- Run B: energy-aware (warm-started from Run A) ------------------------
    print()
    print("=" * 60)
    print(f"Run B: energy-aware  (w_time=1.0, w_energy={W_ENERGY})")
    print("=" * 60)
    gen_energy = BSplineEnergyCoverage(
        waypoints=WAYPOINTS,
        robot_params=ROBOT_PARAMS,
        energy_coeffs_right=ENERGY_COEFFS_RIGHT,
        energy_coeffs_left=ENERGY_COEFFS_LEFT,
        w_time=1.0,
        w_energy=W_ENERGY,
        p_electronics=P_ELECTRONICS,
        **COMMON_KWARGS,
    )
    res_energy = gen_energy.generate_trajectory(warm_start=res_time)

    # Compute full metrics for both runs
    mA = _compute_metrics(res_time,   ROBOT_PARAMS,
                          ENERGY_COEFFS_RIGHT, ENERGY_COEFFS_LEFT, P_ELECTRONICS)
    mB = _compute_metrics(res_energy, ROBOT_PARAMS,
                          ENERGY_COEFFS_RIGHT, ENERGY_COEFFS_LEFT, P_ELECTRONICS)

    _print_comparison(mA, mB)

    _fig1_trajectories(res_time, res_energy, mA, mB)
    _fig2_kinematic_profiles(mA, mB)
    _fig3_energy_analysis(res_time, res_energy, mA, mB)
    _fig8_wheel_kinematics(res_time, res_energy)

    plt.show()
    plt.close('all')


# =============================================================================
# Figure 1 — Coverage Trajectory (with polyhedra corridor constraints)
# =============================================================================
def _fig1_trajectories(res_time, res_energy, mA, mB):
    fig, ax = plt.subplots(figsize=(6, 6),
                           num='Figure 1 — Coverage Trajectory')
    ax.set_aspect('equal')
    wps = np.array(WAYPOINTS)

    # ---- Polyhedra corridor constraints (Fabian's overlap formulation) ----
    _draw_polyhedra_corridors(ax, WAYPOINTS, COMMON_KWARGS['bound'])

    ax.plot(wps[:, 0], wps[:, 1], '--', color='gray',
            linewidth=1.5, label='Reference path', zorder=2)

    ax.plot(res_time['states'][:, 0],   res_time['states'][:, 1],
            '-', color=COL_A, linewidth=2, label='Time-optimal', zorder=4)
    ax.plot(res_time['ctrl_pts'][:, 0], res_time['ctrl_pts'][:, 1],
            'x', color=COL_A, markersize=7, markeredgewidth=1.5, zorder=5)

    ax.plot(res_energy['states'][:, 0],   res_energy['states'][:, 1],
            '-', color=COL_B, linewidth=2, label='Energy-aware', zorder=4)
    ax.plot(res_energy['ctrl_pts'][:, 0], res_energy['ctrl_pts'][:, 1],
            'x', color=COL_B, markersize=7, markeredgewidth=1.5, zorder=5)

    ax.plot(wps[:, 0], wps[:, 1], 'o', color='green',
            markersize=8, label='Waypoints', zorder=6)

    # Metric annotation box
    txt = (
        f"Time-optimal :  T = {mA['total_time']:.2f} s"
        f"  |  s = {mA['path_length']:.3f} m"
        f"  |  E = {mA['energy']:.2f} J\n"
        f"Energy-aware :  T = {mB['total_time']:.2f} s"
        f"  |  s = {mB['path_length']:.3f} m"
        f"  |  E = {mB['energy']:.2f} J"
    )
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.9))

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend(loc='upper left')
    fig.tight_layout()
    _savefig(fig, 'fig_bspline_trajectory.png')


# =============================================================================
# Figure 2 — Kinematic Profiles
# =============================================================================
def _fig2_kinematic_profiles(mA, mB):
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=False,
                             num='Figure 2 — Kinematic Profiles')

    # Reconstruct v and omega from wheel velocities stored in metrics dicts:
    #   omega_r = (v + l*omega) / r  →  v = r/2 * (omega_r + omega_l)
    #   omega_l = (v - l*omega) / r  →  omega = r/(2*l) * (omega_r - omega_l)
    l = ROBOT_PARAMS['l']
    r = ROBOT_PARAMS['r']
    for mX in (mA, mB):
        mX['v']     = r / 2.0 * (mX['omega_r'] + mX['omega_l'])
        mX['omega'] = r / (2.0 * l) * (mX['omega_r'] - mX['omega_l'])

    ylabels  = ['v [m/s]', 'ω [rad/s]', 'a [m/s²]', 'α [rad/s²]']
    subtitles = ['Linear velocity',
                 'Angular velocity',
                 'Linear acceleration  dv/dt',
                 'Angular acceleration  dω/dt']
    arr_keys  = ['v', 'omega', 'a_lin', 'a_ang']

    for ax, ylabel, subtitle, key in zip(axes, ylabels, subtitles, arr_keys):
        ax.plot(mA['time_ik'], mA[key],
                color=COL_A, linewidth=2, label='Time-optimal')
        ax.plot(mB['time_ik'], mB[key],
                color=COL_B, linewidth=2, label='Energy-aware')
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=9)
        ax.legend(fontsize=8, loc='upper right')

    axes[-1].set_xlabel('time [s]')
    fig.suptitle('Figure 2 — Kinematic Profiles', fontsize=12)
    fig.tight_layout()


# =============================================================================
# Figure 3 — TJ108 Energy Analysis
# =============================================================================
def _fig3_energy_analysis(res_time, res_energy, mA, mB):
    fig = plt.figure(figsize=(12, 10),
                     num='Figure 3 — TJ108 Energy Analysis')
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           height_ratios=[2, 2, 1.8], hspace=0.45, wspace=0.35)

    ax_p   = fig.add_subplot(gs[0, :])   # total power — full width
    ax_wr  = fig.add_subplot(gs[1, 0])   # wheel angular vel — right
    ax_wl  = fig.add_subplot(gs[1, 1])   # wheel angular vel — left
    ax_bar = fig.add_subplot(gs[2, 0])   # energy metrics bar
    ax_map = fig.add_subplot(gs[2, 1])   # energy density map

    t_A = mA['time_ik']
    t_B = mB['time_ik']

    # --- [0,:] Power P(t): total, right motor, left motor ------------------
    ax_p.plot(t_A, mA['P_total'], '-',  color=COL_A, linewidth=2,
              label='Time-opt  P_total')
    ax_p.plot(t_A, mA['P_right'], '--', color=COL_A, linewidth=1.2, alpha=0.75,
              label='Time-opt  P_right')
    ax_p.plot(t_A, mA['P_left'],  ':',  color=COL_A, linewidth=1.2, alpha=0.75,
              label='Time-opt  P_left')

    ax_p.plot(t_B, mB['P_total'], '-',  color=COL_B, linewidth=2,
              label='Energy-aware  P_total')
    ax_p.plot(t_B, mB['P_right'], '--', color=COL_B, linewidth=1.2, alpha=0.75,
              label='Energy-aware  P_right')
    ax_p.plot(t_B, mB['P_left'],  ':',  color=COL_B, linewidth=1.2, alpha=0.75,
              label='Energy-aware  P_left')

    ax_p.axhline(P_ELECTRONICS, color='gray', linestyle='-.', linewidth=1,
                 label=f'P_elec = {P_ELECTRONICS} W')

    # Annotate peak values
    ax_p.annotate(f"peak {mA['peak_power']:.1f} W",
                  xy=(t_A[np.argmax(mA['P_total'])], mA['peak_power']),
                  xytext=(8, 4), textcoords='offset points',
                  fontsize=7, color=COL_A)
    ax_p.annotate(f"peak {mB['peak_power']:.1f} W",
                  xy=(t_B[np.argmax(mB['P_total'])], mB['peak_power']),
                  xytext=(8, -12), textcoords='offset points',
                  fontsize=7, color=COL_B)

    ax_p.set_ylabel('Power [W]')
    ax_p.set_xlabel('time [s]')
    ax_p.set_title('Motor power  P(t) = P_right + P_left + P_elec',
                   fontsize=9)
    ax_p.legend(fontsize=7, ncol=3, loc='upper right')

    # --- [1,0] Wheel angular velocity — right wheel -------------------------
    ax_wr.plot(t_A, mA['omega_r'], color=COL_A, linewidth=2,
               label='Time-optimal')
    ax_wr.plot(t_B, mB['omega_r'], color=COL_B, linewidth=2,
               label='Energy-aware')
    ax_wr.set_ylabel('ω_right [rad/s]')
    ax_wr.set_xlabel('time [s]')
    ax_wr.set_title('Right-wheel angular velocity', fontsize=9)
    ax_wr.legend(fontsize=8)

    # --- [1,1] Wheel angular velocity — left wheel --------------------------
    ax_wl.plot(t_A, mA['omega_l'], color=COL_A, linewidth=2,
               label='Time-optimal')
    ax_wl.plot(t_B, mB['omega_l'], color=COL_B, linewidth=2,
               label='Energy-aware')
    ax_wl.set_ylabel('ω_left [rad/s]')
    ax_wl.set_xlabel('time [s]')
    ax_wl.set_title('Left-wheel angular velocity', fontsize=9)
    ax_wl.legend(fontsize=8)

    # --- [2,0] Grouped energy metrics bar -----------------------------------
    metric_labels = ['E_total\n[J]', 'E/meter\n[J/m]',
                     'P_avg\n[W]', 'P_peak\n[W]']
    vals_A = [mA['energy'],           mA['energy_per_meter'],
              mA['avg_power'],         mA['peak_power']]
    vals_B = [mB['energy'],           mB['energy_per_meter'],
              mB['avg_power'],         mB['peak_power']]

    x = np.arange(len(metric_labels))
    w = 0.35
    bars_A = ax_bar.bar(x - w / 2, vals_A, w, color=COL_A,
                        label='Time-optimal')
    bars_B = ax_bar.bar(x + w / 2, vals_B, w, color=COL_B,
                        label='Energy-aware')

    for bar, val in zip(bars_A, vals_A):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals_A + vals_B) * 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    for bar, val in zip(bars_B, vals_B):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals_A + vals_B) * 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7,
                    color='darkred')

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_labels, fontsize=8)
    ax_bar.set_title('Energy metrics comparison', fontsize=9)
    ax_bar.legend(fontsize=8)

    # --- [2,1] Energy density map — time-optimal path -----------------------
    t_spline = res_time['time']
    t_ik     = mA['time_ik']
    P_at_spline = np.interp(t_spline, t_ik, mA['P_total'],
                             left=mA['P_total'][0], right=mA['P_total'][-1])
    states_A = res_time['states']
    sc = ax_map.scatter(states_A[:, 0], states_A[:, 1],
                        c=P_at_spline, cmap='hot_r', s=15, zorder=3)
    ax_map.set_aspect('equal')
    ax_map.plot(np.array(WAYPOINTS)[:, 0], np.array(WAYPOINTS)[:, 1],
                'go', markersize=6, label='Waypoints')
    cb = fig.colorbar(sc, ax=ax_map, shrink=0.9)
    cb.set_label('Power [W]', fontsize=8)
    ax_map.set_xlabel('x [m]')
    ax_map.set_ylabel('y [m]')
    ax_map.set_title('Energy density (time-optimal path)', fontsize=9)
    ax_map.legend(fontsize=8)

    fig.suptitle('Figure 3 — TJ108 Energy Analysis', fontsize=12)


# =============================================================================
# Figure 8 — Per-wheel angular velocity, acceleration, and jerk
# =============================================================================
def _fig8_wheel_kinematics(res_time, res_energy):
    """Per-wheel kinematic profiles for time-optimal vs energy-aware solutions.

    3 rows × 2 columns:
      rows: angular velocity ω [rad/s], angular acceleration α [rad/s²], jerk [m/s³]
      cols: time-optimal, energy-aware
    Right wheel solid, left wheel dashed.
    """
    l  = ROBOT_PARAMS['l']
    r  = ROBOT_PARAMS['r']
    dt = 0.01   # BSpline IK time step [s]

    wk_A = _compute_wheel_kinematics(res_time,   l, r, dt)
    wk_B = _compute_wheel_kinematics(res_energy, l, r, dt)

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex='col',
                             num='Figure 8 — Per-Wheel Kinematics')
    fig.suptitle('Figure 8 — Per-Wheel Angular Velocity / Acceleration / Jerk',
                 fontsize=12)

    row_keys    = [('omega_r', 'omega_l'), ('alpha_r', 'alpha_l'), ('jerk_r', 'jerk_l')]
    row_ylabels = ['ω_wheel [rad/s]', 'α_wheel [rad/s²]', 'jerk [m/s³]']
    col_data    = [(wk_A, COL_A, 'Time-optimal'), (wk_B, COL_B, 'Energy-aware')]

    for col, (wk, color, title) in enumerate(col_data):
        for row, ((kr, kl), ylabel) in enumerate(zip(row_keys, row_ylabels)):
            ax = axes[row, col]
            ax.plot(wk['time'], wk[kr], '-',  color=color, lw=1.8, label='Right')
            ax.plot(wk['time'], wk[kl], '--', color=color, lw=1.8,
                    label='Left', alpha=0.75)
            ax.axhline(0, color='lightgray', lw=0.8, zorder=0)
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(title)
            if row == 2:
                ax.set_xlabel('time [s]')
            ax.legend(fontsize=8)

    fig.tight_layout()
    _savefig(fig, 'fig_bspline_wheel_kinematics.png')


# =============================================================================
if __name__ == '__main__':
    main()
