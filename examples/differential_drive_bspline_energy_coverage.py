#!/usr/bin/env python3
##
# @file differential_drive_bspline_energy_coverage.py
#
# @brief Comparison of time-optimal vs energy-aware B-spline coverage for a
# differential drive robot navigating an L-shaped path.
#
# Runs two solvers on the same waypoints:
#   (A) BSplineEnergyCoverage (w_energy=0) - minimizes total traversal time
#   (B) BSplineEnergyCoverage             - minimizes w_time * T + w_energy * E_total
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
#   Figure 4 — Time–Energy Pareto Front
#               Sweep of w_time / w_energy with E/m as marker color
#   Figure 5–7 — Dense w_energy sweep analysis (peak power, energy, time,
#                trajectory overlay, kinematic & power profiles)
#   Figure 8 — Per-wheel angular velocity, acceleration, and jerk
#               (time-optimal vs energy-aware, right/left wheel overlay)
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
from trajectory_generators.bspline_coverage import BSplineCoverage
from trajectory_generators.bspline_energy_coverage import BSplineEnergyCoverage


# =============================================================================
# PAPER FIGURE EXPORT
# Set SAVE_FIGS = True to write paper-quality PNGs into the Writting directory.
# =============================================================================
SAVE_FIGS   = True
FIG_OUT_DIR = (pathlib.Path(__file__).resolve().parent.parent.parent
               / 'Writting' / 'energy_aware')


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
    n_sampling=30,
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

    # --- Step 1: sweep to find the optimal w_energy --------------------------
    print("=" * 60)
    print("Step 1: w_energy sweep to identify optimal trade-off point")
    print("=" * 60)
    opt_we = _fig5_we_sweep()   # produces Figures 5, 6, 7 and returns opt_we

    # --- Step 2: Run A — time-optimal ----------------------------------------
    # Use BSplineEnergyCoverage(w_energy=0) so res_time['energy'] is computed
    # by the same B-spline-derivative formula as Run B — consistent metric.
    print("=" * 60)
    print("Step 2 — Run A: time-optimal (w_time=1.0, w_energy=0.0)")
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

    # --- Step 3: Run B — energy-aware at the sweep-identified optimum ---------
    print()
    print("=" * 60)
    print(f"Step 3 — Run B: energy-aware  (w_time=1.0, w_energy={opt_we:.4f})")
    print("=" * 60)
    gen_energy = BSplineEnergyCoverage(
        waypoints=WAYPOINTS,
        robot_params=ROBOT_PARAMS,
        energy_coeffs_right=ENERGY_COEFFS_RIGHT,
        energy_coeffs_left=ENERGY_COEFFS_LEFT,
        w_time=1.0,
        w_energy=opt_we,
        p_electronics=P_ELECTRONICS,
        **COMMON_KWARGS,
    )
    res_energy = gen_energy.generate_trajectory()

    # Compute full metrics for both
    mA = _compute_metrics(res_time,   ROBOT_PARAMS,
                          ENERGY_COEFFS_RIGHT, ENERGY_COEFFS_LEFT, P_ELECTRONICS)
    mB = _compute_metrics(res_energy, ROBOT_PARAMS,
                          ENERGY_COEFFS_RIGHT, ENERGY_COEFFS_LEFT, P_ELECTRONICS)

    _print_comparison(mA, mB)

    # Figures 1–4: comparison at the sweep-identified optimal w_energy
    _fig1_trajectories(res_time, res_energy, mA, mB)
    _fig2_kinematic_profiles(mA, mB)
    _fig3_energy_analysis(res_time, res_energy, mA, mB)
    _fig4_pareto()
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
# Figure 4 — Time–Energy Pareto Front
# =============================================================================
def _fig4_pareto():
    """Sweep w_energy / w_time to trace the Pareto front.

    Each marker is coloured by energy-per-meter efficiency [J/m].
    This runs multiple IPOPT solves — comment out the call in main() if needed.
    """
    weights = [
        (1.0, 0.00),
        (1.0, 0.10),
        (1.0, 0.25),
        (1.0, 0.50),
        (1.0, 1.00),
        (0.5, 1.00),
        (0.1, 1.00),
        (0.0, 1.00),
    ]

    times, energies, eff = [], [], []

    print()
    print("=" * 60)
    print("Figure 4: Pareto sweep  (8 IPOPT solves) …")
    print("=" * 60)

    prev_res = None
    for w_t, w_e in weights:
        # Always use BSplineEnergyCoverage so that res['energy'] is computed
        # by the same B-spline-derivative formula for every point.
        # Continuation warm-start: seed each solve from the previous solution
        # so IPOPT traces the Pareto branch smoothly instead of jumping to a
        # different local minimum.
        gen = BSplineEnergyCoverage(
            waypoints=WAYPOINTS,
            robot_params=ROBOT_PARAMS,
            energy_coeffs_right=ENERGY_COEFFS_RIGHT,
            energy_coeffs_left=ENERGY_COEFFS_LEFT,
            w_time=w_t, w_energy=w_e,
            p_electronics=P_ELECTRONICS,
            **COMMON_KWARGS,
        )
        res      = gen.generate_trajectory(warm_start=prev_res)
        prev_res = res
        energy   = float(res['energy'])
        T_total = float(res['time'][-1])

        # Path length
        st = res['states']
        path_len = float(np.sum(np.sqrt(np.diff(st[:, 0])**2
                                        + np.diff(st[:, 1])**2)))
        e_per_m = energy / path_len if path_len > 0 else float('inf')

        times.append(T_total)
        energies.append(energy)
        eff.append(e_per_m)
        print(f"  solving ({w_t:.2f}, {w_e:.2f})  done  →  "
              f"T={T_total:.3f} s,  E={energy:.3f} J,  "
              f"E/m={e_per_m:.3f} J/m")

    # --- Summary table -------------------------------------------------------
    print()
    print("Pareto front summary")
    hdr = (f"  {'w_time':>6}  {'w_energy':>8}  "
           f"{'Time [s]':>10}  {'Energy [J]':>10}  "
           f"{'E/meter [J/m]':>14}  {'Avg P [W]':>10}")
    print(hdr)
    print("  " + "─" * 66)

    for i, (w_t, w_e) in enumerate(weights):
        dt_approx = times[i]
        avg_p = energies[i] / dt_approx if dt_approx > 0 else float('nan')
        print(f"  {w_t:>6.2f}  {w_e:>8.2f}  "
              f"{times[i]:>10.3f}  {energies[i]:>10.3f}  "
              f"{eff[i]:>14.4f}  {avg_p:>10.3f}")
    print()

    fig, ax = plt.subplots(figsize=(7, 5),
                           num='Figure 4 — Time–Energy Pareto Front')

    sc = ax.scatter(times, energies, c=eff, cmap='viridis_r',
                    s=90, zorder=5)
    ax.plot(times, energies, '-', color='gray', linewidth=1.2,
            zorder=4, alpha=0.6)

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('Energy / meter [J/m]')

    for i, (w_t, w_e) in enumerate(weights):
        ax.annotate(f'({w_t:.1f},{w_e:.1f})',
                    (times[i], energies[i]),
                    textcoords='offset points', xytext=(6, 4), fontsize=7)

    ax.set_xlabel('Total mission time [s]')
    ax.set_ylabel('Total energy [J]')
    ax.set_title('Figure 4 — Time–Energy Pareto Front\n'
                 'Marker colour = energy efficiency [J/m]')
    fig.tight_layout()
    _savefig(fig, 'fig_bspline_pareto.png')


# =============================================================================
# Figure 5 — Dense w_energy Sweep: Peak Power Suppression & Optimal Trade-off
# =============================================================================
def _fig5_we_sweep():
    """Sweep w_energy over [0, 0.001 … 1.0] log-spaced (w_time fixed at 1.0).

    Log spacing is chosen because the objective balance point is
        w_e* = T_mission / E_total ≈ 0.026
    Linear spacing wastes resolution far from this point.  25 log-spaced
    decades from 1e-3 to 1.0 plus the w_e=0 anchor give 26 solves that
    cover the full Pareto front in one pass.

    Records Peak Power, Total Energy, and Mission Time for each solve.
    Computes the second derivative of the Peak Power suppression curve and
    identifies the w_energy where it is minimised — the elbow of the curve
    where further energy weighting yields diminishing returns on peak power.
    """
    we_values = np.concatenate([[0.0], np.logspace(-3, 0, 25)])

    # Tighter corner tolerance: halve the corridor bound so the spline must
    # stay closer to the reference path around corners.
    sweep_kwargs = {**COMMON_KWARGS, 'bound': 0.17}

    peak_powers    = []
    total_energies = []
    mission_times  = []
    all_states     = []
    all_time_traj  = []
    all_time_ik    = []
    all_v          = []
    all_omega      = []
    all_P_tot      = []
    all_P_r        = []
    all_P_l        = []

    print()
    print("=" * 60)
    print(f"Figure 5: Log-spaced w_energy sweep  ({len(we_values)} IPOPT solves, "
          f"w_e in [0, 1e-3 … 1.0]) …")
    print("=" * 60)
    print(f"  {'w_e':>10}  {'Time [s]':>10}  {'Energy [J]':>10}  "
          f"{'Peak P [W]':>10}")
    print("  " + "─" * 49)

    prev_res = None
    for w_e in we_values:
        gen = BSplineEnergyCoverage(
            waypoints=WAYPOINTS,
            robot_params=ROBOT_PARAMS,
            energy_coeffs_right=ENERGY_COEFFS_RIGHT,
            energy_coeffs_left=ENERGY_COEFFS_LEFT,
            w_time=1.0, w_energy=float(w_e),
            p_electronics=P_ELECTRONICS,
            **sweep_kwargs,
        )
        res      = gen.generate_trajectory(warm_start=prev_res)
        prev_res = res
        P_tot, P_r, P_l, _ = _compute_wheel_power(
            res['v'], res['omega'],
            ROBOT_PARAMS, ENERGY_COEFFS_RIGHT, ENERGY_COEFFS_LEFT,
            P_ELECTRONICS)
        T_total = float(res['time'][-1])
        peak_p  = float(np.max(P_tot))
        energy  = float(res['energy'])   # consistent optimizer metric

        peak_powers.append(peak_p)
        total_energies.append(energy)
        mission_times.append(T_total)
        all_states.append(res['states'])
        all_time_traj.append(res['time'])
        all_time_ik.append(res['time_ik'])
        all_v.append(res['v'])
        all_omega.append(res['omega'])
        all_P_tot.append(P_tot)
        all_P_r.append(P_r)
        all_P_l.append(P_l)
        print(f"  {w_e:>10.6f}  {T_total:>10.3f}  {energy:>10.3f}  {peak_p:>10.3f}")

    peak_powers    = np.array(peak_powers)
    total_energies = np.array(total_energies)
    mission_times  = np.array(mission_times)

    # --- Second derivative of Peak Power curve w.r.t. w_energy ---------------
    # Pass the non-uniform we_values array so np.gradient uses variable spacing.
    d1_pp = np.gradient(peak_powers, we_values)
    d2_pp = np.gradient(d1_pp,       we_values)

    # Optimal trade-off: w_energy where d2_pp is most negative (maximum concavity)
    opt_idx = int(np.argmin(d2_pp))
    opt_we  = float(we_values[opt_idx])

    print()
    print(f"  Optimal trade-off point: w_energy = {opt_we:.2f}  "
          f"(d²P/dwe² = {d2_pp[opt_idx]:.4f})")
    print(f"    → Peak Power  = {peak_powers[opt_idx]:.3f} W")
    print(f"    → Total Energy = {total_energies[opt_idx]:.3f} J")
    print(f"    → Mission Time = {mission_times[opt_idx]:.3f} s")
    print()

    # --- Plot -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             num='Figure 5 — w_energy Sweep')
    fig.suptitle('Figure 5 — Dense w_energy Sweep  (w_time = 1.0)',
                 fontsize=13)

    # Panel (0,0): Peak Power vs w_energy
    ax = axes[0, 0]
    ax.plot(we_values, peak_powers, 'o-', color='tomato', linewidth=1.8,
            markersize=4, label='Peak Power')
    ax.axvline(opt_we, color='black', linestyle='--', linewidth=1.2,
               label=f'Optimal w_e = {opt_we:.4f}')
    ax.scatter([opt_we], [peak_powers[opt_idx]], color='black', s=80, zorder=6)
    ax.set_xlabel('w_energy (log scale)')
    ax.set_ylabel('Peak Power [W]')
    ax.set_title('Peak Power Suppression')
    ax.set_xscale('log')
    ax.legend(fontsize=8)

    # Panel (0,1): Total Energy vs w_energy
    ax = axes[0, 1]
    ax.plot(we_values, total_energies, 's-', color='steelblue', linewidth=1.8,
            markersize=4, label='Total Energy')
    ax.axvline(opt_we, color='black', linestyle='--', linewidth=1.2,
               label=f'Optimal w_e = {opt_we:.4f}')
    ax.scatter([opt_we], [total_energies[opt_idx]], color='black', s=80, zorder=6)
    ax.set_xlabel('w_energy (log scale)')
    ax.set_ylabel('Total Energy [J]')
    ax.set_title('Total Energy vs w_energy')
    ax.set_xscale('log')
    ax.legend(fontsize=8)

    # Panel (1,0): Mission Time vs w_energy
    ax = axes[1, 0]
    ax.plot(we_values, mission_times, '^-', color='seagreen', linewidth=1.8,
            markersize=4, label='Mission Time')
    ax.axvline(opt_we, color='black', linestyle='--', linewidth=1.2,
               label=f'Optimal w_e = {opt_we:.4f}')
    ax.scatter([opt_we], [mission_times[opt_idx]], color='black', s=80, zorder=6)
    ax.set_xlabel('w_energy (log scale)')
    ax.set_ylabel('Mission Time [s]')
    ax.set_title('Mission Time vs w_energy')
    ax.set_xscale('log')
    ax.legend(fontsize=8)

    # Panel (1,1): Second derivative of Peak Power curve
    ax = axes[1, 1]
    ax.plot(we_values, d2_pp, 'D-', color='darkorange', linewidth=1.8,
            markersize=4, label='d²(Peak P)/d(w_e)²')
    ax.axvline(opt_we, color='black', linestyle='--', linewidth=1.2,
               label=f'min d² at w_e = {opt_we:.4f}')
    ax.scatter([opt_we], [d2_pp[opt_idx]], color='black', s=80, zorder=6)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
    ax.set_xlabel('w_energy (log scale)')
    ax.set_ylabel('d²(Peak P) / d(w_e)²  [W]')
    ax.set_title('2nd Derivative — Optimal Trade-off Detection')
    ax.set_xscale('log')
    ax.legend(fontsize=8)

    fig.tight_layout()

    _fig6_sweep_trajectories(we_values, all_states, all_time_traj, opt_idx)
    _fig7_sweep_profiles(we_values, all_time_ik, all_v, all_omega,
                         all_P_tot, all_P_r, all_P_l, opt_idx)

    return opt_we


# =============================================================================
# Figure 6 — Sweep: XY Trajectory Overlay + Heading Profile
# =============================================================================
def _fig6_sweep_trajectories(we_values, all_states, all_time_traj, opt_idx):
    """XY trajectory overlay and heading-angle profiles across the sweep."""
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=float(we_values[0]), vmax=float(we_values[-1]))
    wps  = np.array(WAYPOINTS)
    opt_we = float(we_values[opt_idx])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             num='Figure 6 — Sweep Trajectories')
    fig.suptitle('Figure 6 — Trajectory Overlay Across w_energy Sweep  '
                 '(w_time = 1.0,  bound = 0.08 m)', fontsize=12)

    # --- Panel (0): XY overlay with polyhedra corridors ----------------------
    ax = axes[0]
    ax.set_aspect('equal')

    _draw_polyhedra_corridors(ax, WAYPOINTS, COMMON_KWARGS['bound'])

    ax.plot(wps[:, 0], wps[:, 1], '--', color='gray', linewidth=1.5,
            zorder=2, label='Reference path')

    for i, (w_e, states) in enumerate(zip(we_values, all_states)):
        is_opt = (i == opt_idx)
        ax.plot(states[:, 0], states[:, 1], '-',
                color=cmap(norm(w_e)),
                linewidth=2.5 if is_opt else 0.9,
                alpha=1.0 if is_opt else 0.5,
                zorder=5 if is_opt else 3)

    # Re-draw optimal on top with a named line for the legend
    ax.plot(all_states[opt_idx][:, 0], all_states[opt_idx][:, 1], '-',
            color='red', linewidth=2.5, zorder=6,
            label=f'Optimal  w_e = {opt_we:.3f}')
    ax.plot(wps[:, 0], wps[:, 1], 'o', color='limegreen',
            markersize=8, zorder=7, label='Waypoints')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='w_energy')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('XY Trajectory Overlay')
    ax.legend(fontsize=8)

    # --- Panel (1): Heading angle θ(t) ----------------------------------------
    ax = axes[1]
    for i, (w_e, states, t_traj) in enumerate(
            zip(we_values, all_states, all_time_traj)):
        is_opt = (i == opt_idx)
        ax.plot(t_traj, np.rad2deg(states[:, 2]), '-',
                color=cmap(norm(w_e)),
                linewidth=2.5 if is_opt else 0.9,
                alpha=1.0 if is_opt else 0.5,
                zorder=5 if is_opt else 3)

    ax.plot(all_time_traj[opt_idx],
            np.rad2deg(all_states[opt_idx][:, 2]),
            '-', color='red', linewidth=2.5, zorder=6,
            label=f'Optimal  w_e = {opt_we:.3f}')

    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2.set_array([])
    fig.colorbar(sm2, ax=ax, label='w_energy')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Heading θ [deg]')
    ax.set_title('Heading Angle vs Time')
    ax.legend(fontsize=8)

    fig.tight_layout()


# =============================================================================
# Figure 7 — Sweep: Kinematic & Power Profile Overlay
# =============================================================================
def _fig7_sweep_profiles(we_values, all_time_ik, all_v, all_omega,
                          all_P_tot, all_P_r, all_P_l, opt_idx):
    """Forward velocity, angular velocity, power, and cumulative energy
    profiles overlaid for every w_energy in the sweep."""
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=float(we_values[0]), vmax=float(we_values[-1]))
    opt_we = float(we_values[opt_idx])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                             num='Figure 7 — Sweep Kinematic & Power Profiles')
    fig.suptitle('Figure 7 — Kinematic & Power Profiles Across w_energy Sweep  '
                 '(w_time = 1.0)', fontsize=12)

    ax_v, ax_w, ax_p, ax_e = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for i, (w_e, t_ik, v, omega, P_tot, P_r, P_l) in enumerate(
            zip(we_values, all_time_ik, all_v, all_omega,
                all_P_tot, all_P_r, all_P_l)):
        color  = cmap(norm(w_e))
        is_opt = (i == opt_idx)
        lw     = 2.0 if is_opt else 0.8
        alpha  = 1.0 if is_opt else 0.45
        zo     = 5 if is_opt else 3

        E_cum = np.cumsum(P_tot) * 0.01   # dt = 0.01 s
        ax_v.plot(t_ik, v,     '-', color=color, lw=lw, alpha=alpha, zorder=zo)
        ax_w.plot(t_ik, omega, '-', color=color, lw=lw, alpha=alpha, zorder=zo)
        ax_p.plot(t_ik, P_tot, '-', color=color, lw=lw, alpha=alpha, zorder=zo)
        ax_e.plot(t_ik, E_cum, '-', color=color, lw=lw, alpha=alpha, zorder=zo)

    # Optimal overlay with label + per-motor breakdown on power panel
    t_opt   = all_time_ik[opt_idx]
    v_opt   = all_v[opt_idx]
    w_opt   = all_omega[opt_idx]
    P_opt   = all_P_tot[opt_idx]
    Pr_opt  = all_P_r[opt_idx]
    Pl_opt  = all_P_l[opt_idx]
    E_opt   = np.cumsum(P_opt) * 0.01

    lbl = f'Optimal  w_e = {opt_we:.3f}'
    ax_v.plot(t_opt, v_opt, '-', color='red', lw=2.5, zorder=6, label=lbl)
    ax_w.plot(t_opt, w_opt, '-', color='red', lw=2.5, zorder=6, label=lbl)
    ax_p.plot(t_opt, P_opt, '-', color='red', lw=2.5, zorder=6, label=lbl)
    ax_p.plot(t_opt, Pr_opt, '--', color='red', lw=1.4, zorder=6,
              label='P_right (optimal)')
    ax_p.plot(t_opt, Pl_opt, ':',  color='red', lw=1.4, zorder=6,
              label='P_left  (optimal)')
    ax_e.plot(t_opt, E_opt, '-', color='red', lw=2.5, zorder=6, label=lbl)

    # Axis labels, titles, grid
    ax_v.set(xlabel='Time [s]', ylabel='v [m/s]',
             title='Forward Velocity v(t)')
    ax_w.set(xlabel='Time [s]', ylabel='ω [rad/s]',
             title='Angular Velocity ω(t)')
    ax_p.set(xlabel='Time [s]', ylabel='P [W]',
             title='Total Electrical Power P(t)')
    ax_e.set(xlabel='Time [s]', ylabel='E [J]',
             title='Cumulative Energy E(t)')

    for ax in axes.flat:
        ax.legend(fontsize=7)

    # One shared colorbar per figure column pair
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), label='w_energy',
                 fraction=0.02, pad=0.04)

    fig.tight_layout()


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
