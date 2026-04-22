#!/usr/bin/env python3
##
# @file differential_drive_path_segment_combined.py
#
# @brief Combined trajectory generation for a 5 m x 5 m L-shaped corner path.
#
# Path:  (0,0) -> (5,0) -> (5,5)   -- a single 90-degree left turn.
#
# Pipeline
# --------
#   1. PathSegment          -- computes the feasible circular arc geometry at
#                              the corner (standoff L_seg, deviation b, radius R).
#   2. EulerJLAPCoverage    -- jerk-limited velocity profile for the two straight
#                              segments that flank the arc entry/exit points.
#   3. BSplineEnergyCoverage -- energy-aware B-spline OCP through the corner
#                               waypoints; w_energy is swept log-spaced to
#                               identify the optimal peak-power trade-off.
#
# Figures
# -------
#   Figure 1 -- Segmented path overview (arc geometry + all 3 generator outputs)
#   Figure 2 -- JLAP kinematic profiles for both straight segments
#   Figure 3 -- w_energy sweep statistics (peak power / energy / time / d2)
#   Figure 4 -- Corner detail + full stitched trajectory
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/04/22

# Standard library
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Internal library
from trajectory_generators.path_segment import PathSegment
from trajectory_generators.euler_jlap_coverage import EulerJLAPCoverage
from trajectory_generators.bspline_energy_coverage import BSplineEnergyCoverage


# =============================================================================
# CONFIGURATION
# =============================================================================

WP_START  = [0.0, 0.0]
WP_CORNER = [5.0, 0.0]   # sharp corner vertex
WP_END    = [5.0, 5.0]

# Headings at the corner: east -> north
HEADING_IN  = 0.0           # [rad]
HEADING_OUT = np.pi / 2     # [rad]
BETA        = np.pi / 2     # 90-degree left turn [rad]

# PathSegment feasibility inputs
L_INPUT     = 1.0           # desired standoff from corner vertex [m]
B_INPUT     = 0.08          # desired deviation tolerance [m]
V_MAX_SEG   = 0.5           # [m/s]
A_MAX_SEG   = 1.0           # [m/s^2]
L_WHEELBASE = 0.35          # [m]

# EulerJLAP (straight segments) -- NewMiniAGV defaults
JLAP_ROBOT_PARAMS = {
    'robot_mass':         120.4,
    'robot_width':        0.510,
    'wheel_radius':       0.075,
    'gear_ratio':         40.0,
    'rated_motor_torque': 1.3,
    'rated_motor_speed':  3500.0,
    'motor_inertia':      0.66e-4,
    'path_vel_lim':       0.5,
}
JLAP_DT = 0.05   # [s]

# Handoff velocity: S1 exits at this speed; corner enters at this speed.
# Taken from path_vel_lim so it is always within the JLAP kinematic limits.
V_HANDOFF = JLAP_ROBOT_PARAMS['path_vel_lim']   # 0.5 m/s

# BSplineEnergyCoverage (corner)
# Reduced n_ctrl_pts / n_sampling vs the full-path example so the NLP
# stays tractable for the short corner segments (each ~1 m).
# vel_max[0] = V_HANDOFF so the feedrate ceiling matches the JLAP cruise
# speed: the time-optimal objective pushes entry and exit to this ceiling
# automatically, achieving velocity continuity without any equality constraint.
# vel_min_lin is kept small (0.01) so the BSpline is free to slow through the
# mid-arc where omega_max=0.196 rad/s limits speed to ~0.196 m/s on a 1 m arc.
BSPLINE_COMMON = dict(
    bound=0.25,
    n_ctrl_pts=4,
    spline_order=3,
    n_sampling=15,
    vel_max=[V_HANDOFF, V_HANDOFF, 0.196],
    vel_min_lin=0.01,
    eps_nonh=0.005,
)
ROBOT_PARAMS_BSPLINE = {'l': 0.53 / 2, 'r': 0.3}

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

COL_S1  = 'steelblue'
COL_C   = 'tomato'
COL_S2  = 'seagreen'
COL_OPT = 'darkorange'
COL_REF = 'gray'


# =============================================================================
# MAIN
# =============================================================================
def main():
    # ------------------------------------------------------------------
    # Step 1: PathSegment -- corner arc geometry
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: PathSegment - corner arc geometry")
    print("=" * 60)
    seg_info = _segment_corner()
    arc_entry = seg_info['arc_entry_world']
    arc_exit  = seg_info['arc_exit_world']
    print(f"  feasible = {seg_info['feasible']}   "
          f"L_seg = {seg_info['L_seg']:.3f} m   "
          f"R = {seg_info['R']:.3f} m   "
          f"b = {seg_info['b']:.4f} m")
    print(f"  arc entry : ({arc_entry[0]:.3f}, {arc_entry[1]:.3f})")
    print(f"  arc exit  : ({arc_exit[0]:.3f},  {arc_exit[1]:.3f})")
    print(f"  handoff velocity : {V_HANDOFF:.3f} m/s")

    # ------------------------------------------------------------------
    # Step 2a: EulerJLAP -- segment 1 (accelerates, exits at V_HANDOFF)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 2a: EulerJLAPCoverage - segment 1 (0 -> V_HANDOFF)")
    print("=" * 60)
    res_s1 = _run_jlap_seg(WP_START, arc_entry.tolist(),
                            initial_vel=0.0, final_vel=V_HANDOFF)
    print(f"  Segment 1: T = {res_s1['time'][-1]:.3f} s   "
          f"v_peak = {np.max(res_s1['v']):.3f} m/s   "
          f"v_exit = {res_s1['v'][-1]:.3f} m/s")

    # ------------------------------------------------------------------
    # Step 3: Corner B-spline -- sweep w_energy -> optimal
    #         (entry speed pinned to V_HANDOFF)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3: BSplineEnergyCoverage - w_energy sweep for corner")
    print("=" * 60)
    corner_wps = _build_corner_waypoints(arc_entry, arc_exit)
    sweep = _sweep_we(corner_wps)
    opt_we = sweep['opt_we']
    res_by_we = sweep['res_by_we']

    # Reuse sweep results -- no re-solve needed (avoids cold-start local minima).
    # Fall back to a fresh solve only if the sweep result for that w_e is absent.
    print()
    print(f"  Using sweep result for time-optimal  (w_energy = 0.0)")
    if 0.0 in res_by_we:
        res_corner_time = res_by_we[0.0]
    else:
        print("  (sweep result missing -- re-solving cold)")
        res_corner_time = _solve_corner(corner_wps, w_energy=0.0)

    print(f"  Using sweep result for energy-optimal (w_energy = {opt_we:.4f})")
    if opt_we in res_by_we:
        res_corner_opt = res_by_we[opt_we]
    else:
        print("  (sweep result missing -- re-solving warm)")
        res_corner_opt = _solve_corner(corner_wps, w_energy=opt_we,
                                       warm_start=res_corner_time)

    mA = _corner_metrics(res_corner_time)
    mB = _corner_metrics(res_corner_opt)
    _print_corner_comparison(mA, mB, opt_we)

    # ------------------------------------------------------------------
    # Step 2b: EulerJLAP -- segment 3 (enters at corner exit speed)
    # ------------------------------------------------------------------
    # Read corner exit velocity from the last two OCP state samples:
    # these are from the actual optimization solution and avoid the
    # numerical interpolation artifacts that appear in the IK v[-1].
    st = res_corner_opt['states']
    tm = res_corner_opt['time']
    dt_exit = float(tm[-1] - tm[-2])
    if dt_exit > 1e-8:
        dx_exit = float(st[-1, 0] - st[-2, 0])
        dy_exit = float(st[-1, 1] - st[-2, 1])
        v_corner_exit = float(np.sqrt(dx_exit**2 + dy_exit**2) / dt_exit)
    else:
        v_corner_exit = 0.0
    v_corner_exit = float(np.clip(v_corner_exit, 0.0,
                                  JLAP_ROBOT_PARAMS['path_vel_lim']))
    print()
    print("=" * 60)
    print(f"Step 2b: EulerJLAPCoverage - segment 3 "
          f"(V_entry={v_corner_exit:.3f} -> 0)")
    print("=" * 60)
    res_s2 = _run_jlap_seg(arc_exit.tolist(), WP_END,
                            initial_vel=v_corner_exit, final_vel=0.0)
    print(f"  Segment 3: T = {res_s2['time'][-1]:.3f} s   "
          f"v_peak = {np.max(res_s2['v']):.3f} m/s   "
          f"v_entry = {v_corner_exit:.3f} m/s")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    _fig1_segmented_path(seg_info, res_s1, res_s2, res_corner_opt)
    _fig2_jlap_profiles(res_s1, res_s2)
    _fig3_we_sweep(sweep)
    _fig4_corner_and_full(seg_info, res_corner_time, res_corner_opt,
                          mA, mB, res_s1, res_s2, opt_we)

    plt.show()
    plt.close('all')


# =============================================================================
# STEP 1 - PathSegment
# =============================================================================
def _segment_corner():
    """! Compute feasible arc geometry for the 90-degree corner.

    Arc is generated in local frame (entry at origin, incoming along +x),
    then transformed to world frame.

    @return dict merging PathSegment output with world-frame arc arrays.
    """
    ps = PathSegment(
        beta=BETA,
        L_input=L_INPUT,
        b_input=B_INPUT,
        v_max=V_MAX_SEG,
        a_max=A_MAX_SEG,
        L_wheelbase=L_WHEELBASE,
        n_samples=100,
    )
    res = ps.generate_segment()

    L_seg        = res['L_seg']
    corner       = np.array(WP_CORNER, dtype=float)
    in_dir       = np.array([np.cos(HEADING_IN), np.sin(HEADING_IN)])
    arc_entry    = corner - L_seg * in_dir

    # Local -> world: rotate by HEADING_IN, translate to arc_entry
    c, s  = np.cos(HEADING_IN), np.sin(HEADING_IN)
    R_mat = np.array([[c, -s], [s, c]])
    arc_local    = res['path_segment']            # (n_samples, 3)
    xy_world     = (R_mat @ arc_local[:, :2].T).T + arc_entry
    hdg_world    = arc_local[:, 2] + HEADING_IN
    arc_world    = np.column_stack([xy_world, hdg_world])
    arc_exit     = arc_world[-1, :2].copy()

    res.update({
        'arc_world':       arc_world,
        'arc_entry_world': arc_entry,
        'arc_exit_world':  arc_exit,
        'corner_vertex':   corner,
    })
    return res


# =============================================================================
# STEP 2 - EulerJLAP straight segments
# =============================================================================
def _run_jlap_seg(wp_start, wp_end, initial_vel=0.0, final_vel=0.0):
    """! Run EulerJLAPCoverage on a single straight segment.

    @param wp_start<list>: Start waypoint [x, y] or [x, y, theta].
    @param wp_end<list>:   End   waypoint [x, y] or [x, y, theta].
    @param initial_vel<float>: Entry speed [m/s].
    @param final_vel<float>:   Exit speed  [m/s].
    @return result dict from generate_trajectory().
    """
    return EulerJLAPCoverage(
        waypoints=[wp_start, wp_end],
        sampling_time=JLAP_DT,
        robot_params=JLAP_ROBOT_PARAMS,
        path_vel_step=0.01,
        epsilon_offset=0.1,
        lc_scale=0.4,
        initial_vel=initial_vel,
        final_vel=final_vel,
    ).generate_trajectory()


# =============================================================================
# STEP 3 - Corner B-spline helpers
# =============================================================================
def _build_corner_waypoints(arc_entry, arc_exit):
    """! Build the 3-waypoint list for the corner B-spline OCP.

    @return list of [x, y, theta] triples.
    """
    return [
        [arc_entry[0], arc_entry[1], HEADING_IN],
        [WP_CORNER[0], WP_CORNER[1], HEADING_IN],   # vertex, incoming heading
        [arc_exit[0],  arc_exit[1],  HEADING_OUT],
    ]


def _solve_corner(corner_wps, w_energy, warm_start=None):
    """! Run BSplineEnergyCoverage for the corner at a given w_energy.

    @return result dict from generate_trajectory().
    """
    return BSplineEnergyCoverage(
        waypoints=corner_wps,
        robot_params=ROBOT_PARAMS_BSPLINE,
        energy_coeffs_right=ENERGY_COEFFS_RIGHT,
        energy_coeffs_left=ENERGY_COEFFS_LEFT,
        w_time=1.0,
        w_energy=float(w_energy),
        p_electronics=P_ELECTRONICS,
        **BSPLINE_COMMON,
    ).generate_trajectory(warm_start=warm_start)


def _compute_corner_power(res):
    """! Evaluate TJ108 motor power from IK outputs of a corner result.

    @return (n_ik,) ndarray [W] - total electrical power including hotel load.
    """
    l, r = ROBOT_PARAMS_BSPLINE['l'], ROBOT_PARAMS_BSPLINE['r']
    v, omega = res['v'], res['omega']
    dt = 0.01

    omega_r = (v + l * omega) / r
    omega_l = (v - l * omega) / r
    dor = np.gradient(omega_r, dt)
    dol = np.gradient(omega_l, dt)

    def _p(ow, dw, c):
        return np.maximum(
            c[0] + c[1]*ow + c[2]*ow**2 + c[3]*ow**3
            + c[4]*dw + c[5]*dw**2, 0.0)

    return (_p(omega_r, dor, ENERGY_COEFFS_RIGHT)
            + _p(omega_l, dol, ENERGY_COEFFS_LEFT)
            + P_ELECTRONICS)


def _corner_metrics(res):
    """! Compute corner performance metrics.

    @return dict with scalar and array metrics.
    """
    P_tot  = _compute_corner_power(res)
    energy = float(res.get('energy', np.trapezoid(P_tot, dx=0.01)))
    st     = res['states']
    plen   = float(np.sum(np.sqrt(np.diff(st[:, 0])**2
                                  + np.diff(st[:, 1])**2)))
    return {
        'total_time':       float(res['time'][-1]),
        'path_length':      plen,
        'energy':           energy,
        'peak_power':       float(np.max(P_tot)),
        'avg_power':        float(np.mean(P_tot)),
        'energy_per_meter': energy / plen if plen > 1e-9 else float('inf'),
        'P_total':          P_tot,
        'time_ik':          res['time_ik'],
    }


def _sweep_we(corner_wps):
    """! Sweep w_energy in [0, 1e-3 ... 1] and identify optimal trade-off.

    Uses the minimum of d2(peak_power)/d(w_e)^2 as the optimal point.

    @return dict with sweep arrays and opt_we.
    """
    # Fixed sweep values, run HIGH to LOW so that w_e=0 (time-optimal) is
    # warm-started from w_e=0.005 and converges quickly.
    # Higher w_e values converge easily on their own (energy term helps LBFGS).
    # w_e=1.0 and above are avoided -- they give degenerate (very slow) solutions.
    we_values = np.array([0.1, 0.05, 0.01, 0.005, 0.0])

    peak_powers    = []
    total_energies = []
    mission_times  = []
    we_valid       = []   # only keep feasible solves for gradient analysis

    print(f"  Sweeping {len(we_values)} w_e values (w_time = 1.0) ...")
    print(f"  {'w_e':>10}  {'Time [s]':>10}  {'Energy [J]':>10}  {'Peak P [W]':>10}")
    print("  " + "-" * 49)

    prev_res = None
    res_by_we = {}  # cache results for re-use in main()
    for w_e in we_values:
        try:
            res = _solve_corner(corner_wps, w_e, warm_start=prev_res)
            P_tot  = _compute_corner_power(res)
            T_tot  = float(res['time'][-1])
            peak_p = float(np.max(P_tot))
            energy = float(res.get('energy', np.trapezoid(P_tot, dx=0.01)))
            if not (np.isfinite(peak_p) and np.isfinite(energy)):
                raise ValueError("non-finite result")
            prev_res = res
            res_by_we[w_e] = res
            peak_powers.append(peak_p)
            total_energies.append(energy)
            mission_times.append(T_tot)
            we_valid.append(w_e)
            print(f"  {w_e:>10.6f}  {T_tot:>10.3f}  {energy:>10.3f}  {peak_p:>10.3f}")
        except Exception as exc:
            print(f"  {w_e:>10.6f}  skipped ({exc})")
            prev_res = None   # reset warm-start on failure

    # Sort by ascending w_e for gradient analysis and plotting.
    sort_idx       = np.argsort(we_valid)
    peak_powers    = np.array(peak_powers)[sort_idx]
    total_energies = np.array(total_energies)[sort_idx]
    mission_times  = np.array(mission_times)[sort_idx]
    we_values      = np.array(we_valid)[sort_idx]

    d1 = np.gradient(peak_powers, we_values)
    d2 = np.gradient(d1, we_values)

    # Restrict optimum search to interior points: the first-order finite
    # differences at the two boundary points are one-sided and less accurate.
    # Also require opt_we > 0 so the comparison is always time-opt vs energy-opt.
    interior = np.arange(1, len(we_values) - 1)
    if len(interior) > 0:
        opt_idx = int(interior[np.argmin(d2[interior])])
    else:
        opt_idx = int(np.argmin(d2))
    # Ensure the selected opt_we differs from 0 for a meaningful comparison.
    if float(we_values[opt_idx]) == 0.0 and len(we_values) > 1:
        opt_idx = 1
    opt_we  = float(we_values[opt_idx])

    print()
    print(f"  Optimal w_e = {opt_we:.6f}   "
          f"(d2(P_peak)/d(w_e)2 = {d2[opt_idx]:.4f})")
    print(f"    Peak Power   = {peak_powers[opt_idx]:.3f} W")
    print(f"    Total Energy = {total_energies[opt_idx]:.3f} J")
    print(f"    Mission Time = {mission_times[opt_idx]:.3f} s")

    return {
        'we_values':      we_values,      # only feasible w_e points
        'peak_powers':    peak_powers,
        'total_energies': total_energies,
        'mission_times':  mission_times,
        'd2_pp':          d2,
        'opt_idx':        opt_idx,
        'opt_we':         opt_we,
        'res_by_we':      res_by_we,      # cached trajectory results
    }


# =============================================================================
# TEXT OUTPUT
# =============================================================================
def _print_corner_comparison(mA, mB, opt_we):
    rows = [
        ('Total time',    's',   'total_time'),
        ('Path length',   'm',   'path_length'),
        ('Total energy',  'J',   'energy'),
        ('Energy/meter',  'J/m', 'energy_per_meter'),
        ('Peak power',    'W',   'peak_power'),
        ('Avg power',     'W',   'avg_power'),
    ]
    print()
    print("  Corner: time-optimal vs energy-optimal")
    w_col = f'Opt (w_e={opt_we:.4f})'
    hdr = (f"  {'Metric':<20} {'Unit':<6} "
           f"{'Time-opt':>10}  {w_col:>18}  {'Delta%':>7}")
    print(hdr)
    print("  " + "─" * 65)
    for label, unit, key in rows:
        a, b = mA[key], mB[key]
        if abs(a) > 1e-12:
            d = (b - a) / abs(a) * 100
            ds = f"{'+'if d>=0 else ''}{d:.1f}%"
        else:
            ds = 'n/a'
        print(f"  {label:<20} {unit:<6} {a:>10.4f}  {b:>18.4f}  {ds:>7}")
    print()


# =============================================================================
# FIGURE 1 - Segmented path overview
# =============================================================================
def _fig1_segmented_path(seg_info, res_s1, res_s2, res_corner_opt):
    """! XY overlay: reference L-path, PathSegment arc, all three generator outputs."""
    fig, ax = plt.subplots(figsize=(7, 7),
                           num='Figure 1 - Segmented Path')
    ax.set_aspect('equal')

    ref = np.array([WP_START, WP_CORNER, WP_END])
    ax.plot(ref[:, 0], ref[:, 1], '--', color=COL_REF, linewidth=1.5,
            label='Reference L-path', zorder=2)
    ax.plot(ref[:, 0], ref[:, 1], 'o', color='black', markersize=8, zorder=6)

    # PathSegment arc
    arc = seg_info['arc_world']
    ax.plot(arc[:, 0], arc[:, 1], '-.', color='purple', linewidth=2.0,
            label=f'PathSegment arc  R={seg_info["R"]:.3f} m', zorder=3)

    # Arc entry / exit
    ae, ax_e = seg_info['arc_entry_world'], seg_info['arc_exit_world']
    ax.plot(*ae, 's', color='purple', markersize=9, zorder=7,
            label='Arc entry / exit')
    ax.plot(*ax_e, 's', color='purple', markersize=9, zorder=7)

    # JLAP segment 1
    s1 = res_s1['states']
    ax.plot(s1[:, 0], s1[:, 1], '-', color=COL_S1, linewidth=2.2,
            label='Segment 1  JLAP', zorder=4)

    # B-spline corner (energy-optimal)
    sc = res_corner_opt['states']
    ax.plot(sc[:, 0], sc[:, 1], '-', color=COL_C, linewidth=2.2,
            label='Corner  B-spline (opt w_e)', zorder=4)
    cpts = res_corner_opt['ctrl_pts']
    ax.plot(cpts[:, 0], cpts[:, 1], 'x', color=COL_C,
            markersize=8, markeredgewidth=1.5, zorder=5)

    # JLAP segment 2
    s2 = res_s2['states']
    ax.plot(s2[:, 0], s2[:, 1], '-', color=COL_S2, linewidth=2.2,
            label='Segment 2  JLAP', zorder=4)

    # Heading arrows (one per ~10% of each segment)
    for states, col in [(s1, COL_S1), (sc, COL_C), (s2, COL_S2)]:
        step = max(1, len(states) // 10)
        for st in states[::step]:
            dx = 0.14 * np.cos(st[2])
            dy = 0.14 * np.sin(st[2])
            ax.annotate('', xy=(st[0]+dx, st[1]+dy), xytext=(st[0], st[1]),
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.2))

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Figure 1 - Segmented 5 m × 5 m L-Path\n'
                 'PathSegment arc  |  JLAP straights  |  B-spline corner')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True)
    fig.tight_layout()


# =============================================================================
# FIGURE 2 - JLAP kinematic profiles
# =============================================================================
def _fig2_jlap_profiles(res_s1, res_s2):
    """! v(t), a(t), omega(t) for both straight segments side by side."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex='col',
                             num='Figure 2 - JLAP Kinematic Profiles')
    fig.suptitle('Figure 2 - JLAP Kinematic Profiles  (straight segments)',
                 fontsize=12)

    for col, (res, color, title) in enumerate([
        (res_s1, COL_S1, 'Segment 1  (start -> arc entry)'),
        (res_s2, COL_S2, 'Segment 2  (arc exit -> end)'),
    ]):
        t   = res['time']
        v   = res['v']
        acc = res['acc_path']
        om  = res['omega']

        axes[0, col].plot(t, v,   color=color, linewidth=1.8)
        axes[0, col].set_ylabel('v [m/s]')
        axes[0, col].set_title(title, fontsize=9)

        axes[1, col].plot(t, acc, color=color, linewidth=1.8)
        axes[1, col].set_ylabel('a [m/s²]')

        axes[2, col].plot(t, om,  color=color, linewidth=1.8)
        axes[2, col].set_ylabel('ω [rad/s]')
        axes[2, col].set_xlabel('time [s]')

        for ax in axes[:, col]:
            ax.grid(True)

    fig.tight_layout()


# =============================================================================
# FIGURE 3 - w_energy sweep
# =============================================================================
def _fig3_we_sweep(sweep):
    """! Four-panel sweep analysis: peak power, energy, time, 2nd derivative."""
    we      = sweep['we_values']
    pp      = sweep['peak_powers']
    te      = sweep['total_energies']
    mt      = sweep['mission_times']
    d2      = sweep['d2_pp']
    oi      = sweep['opt_idx']
    opt_we  = sweep['opt_we']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             num='Figure 3 - w_energy Sweep (Corner OCP)')
    fig.suptitle('Figure 3 - w_energy Sweep for Corner B-spline OCP  '
                 '(w_time = 1.0)', fontsize=12)

    specs = [
        (axes[0, 0], pp, 'o-', COL_C,     'Peak Power [W]',    'Peak Power Suppression'),
        (axes[0, 1], te, 's-', COL_S1,    'Total Energy [J]',  'Total Energy vs w_energy'),
        (axes[1, 0], mt, '^-', COL_S2,    'Mission Time [s]',  'Mission Time vs w_energy'),
        (axes[1, 1], d2, 'D-', COL_OPT,   'd2(Peak P)/d(w_e)2', '2nd Derivative - Optimal Trade-off'),
    ]

    for ax, data, marker, color, ylabel, title in specs:
        ax.plot(we, data, marker, color=color, linewidth=1.8, markersize=4,
                label=ylabel)
        ax.axvline(opt_we, color='black', linestyle='--', linewidth=1.2,
                   label=f'opt w_e = {opt_we:.4f}')
        ax.scatter([opt_we], [data[oi]], color='black', s=80, zorder=6)
        if ax is axes[1, 1]:
            ax.axhline(0, color=COL_REF, linewidth=0.8, linestyle=':')
        ax.set_xscale('symlog', linthresh=5e-4)
        ax.set_xlabel('w_energy (symlog scale)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, which='both')

    fig.tight_layout()


# =============================================================================
# FIGURE 4 - Corner detail + full stitched trajectory
# =============================================================================
def _fig4_corner_and_full(seg_info, res_corner_time, res_corner_opt,
                           mA, mB, res_s1, res_s2, opt_we):
    """! Corner comparison (XY + power) and full stitched path + velocity."""

    fig, axes = plt.subplots(2, 2, figsize=(13, 10),
                             num='Figure 4 - Corner Detail + Full Trajectory')
    fig.suptitle('Figure 4 - Corner Detail (B-spline) & Full Stitched Trajectory',
                 fontsize=12)

    # ------------------------------------------------------------------ [0,0]
    # Corner XY: time-optimal vs energy-optimal
    ax = axes[0, 0]
    ax.set_aspect('equal')

    # PathSegment reference arc
    arc = seg_info['arc_world']
    ax.plot(arc[:, 0], arc[:, 1], '-.', color='purple', linewidth=1.5,
            label='PathSegment arc', zorder=2)

    # Corner waypoints
    cwps = np.array(_build_corner_waypoints(seg_info['arc_entry_world'],
                                            seg_info['arc_exit_world']))
    ax.plot(cwps[:, 0], cwps[:, 1], 'o--', color=COL_REF, markersize=7,
            linewidth=1.0, label='Corner waypoints', zorder=3)

    # Time-optimal
    st_A = res_corner_time['states']
    ax.plot(st_A[:, 0], st_A[:, 1], '-', color=COL_S1, linewidth=2.0,
            label='Time-optimal  (w_e = 0)', zorder=4)

    # Energy-optimal
    st_B = res_corner_opt['states']
    ax.plot(st_B[:, 0], st_B[:, 1], '-', color=COL_OPT, linewidth=2.0,
            label=f'Energy-optimal', zorder=4)
    cpts = res_corner_opt['ctrl_pts']
    ax.plot(cpts[:, 0], cpts[:, 1], 'x', color=COL_OPT,
            markersize=7, markeredgewidth=1.5, zorder=5)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Corner B-spline: time-opt vs energy-opt', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True)

    # ------------------------------------------------------------------ [0,1]
    # Corner power P(t): time-optimal vs energy-optimal
    ax = axes[0, 1]
    ax.plot(mA['time_ik'], mA['P_total'], '-', color=COL_S1, linewidth=1.8,
            label=f"Time-opt   peak={mA['peak_power']:.1f} W")
    ax.plot(mB['time_ik'], mB['P_total'], '-', color=COL_OPT, linewidth=1.8,
            label=f"Energy-opt  peak={mB['peak_power']:.1f} W")
    ax.axhline(P_ELECTRONICS, color=COL_REF, linestyle=':', linewidth=1.0,
               label=f'P_elec = {P_ELECTRONICS} W')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('Power [W]')
    ax.set_title('Corner motor power P(t)', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True)

    # ------------------------------------------------------------------ [1,0]
    # Full stitched XY trajectory
    ax = axes[1, 0]
    ax.set_aspect('equal')

    ref = np.array([WP_START, WP_CORNER, WP_END])
    ax.plot(ref[:, 0], ref[:, 1], '--', color=COL_REF, linewidth=1.2,
            label='Reference L-path', zorder=2)

    s1 = res_s1['states']
    sc = res_corner_opt['states']
    s2 = res_s2['states']

    ax.plot(s1[:, 0], s1[:, 1], '-', color=COL_S1, linewidth=2.0,
            label='Segment 1 (JLAP)', zorder=4)
    ax.plot(sc[:, 0], sc[:, 1], '-', color=COL_C, linewidth=2.0,
            label='Corner (B-spline)', zorder=4)
    ax.plot(s2[:, 0], s2[:, 1], '-', color=COL_S2, linewidth=2.0,
            label='Segment 2 (JLAP)', zorder=4)

    # Arc entry / exit junction markers
    ae, ax_e = seg_info['arc_entry_world'], seg_info['arc_exit_world']
    ax.plot(*ae, 's', color='black', markersize=8, zorder=7,
            label='Arc junctions')
    ax.plot(*ax_e, 's', color='black', markersize=8, zorder=7)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Full stitched trajectory (XY)', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True)

    # ------------------------------------------------------------------ [1,1]
    # Full stitched velocity profile
    ax = axes[1, 1]
    T1 = float(res_s1['time'][-1])
    Tc = float(res_corner_opt['time'][-1])

    ax.plot(res_s1['time'], res_s1['v'],
            '-', color=COL_S1, linewidth=1.8, label='Segment 1 (JLAP)')
    ax.plot(res_corner_opt['time_ik'] + T1, res_corner_opt['v'],
            '-', color=COL_C, linewidth=1.8, label='Corner (B-spline)')
    ax.plot(res_s2['time'] + T1 + Tc, res_s2['v'],
            '-', color=COL_S2, linewidth=1.8, label='Segment 2 (JLAP)')

    # Junction markers
    ax.axvline(T1,      color='black', linestyle=':', linewidth=1.0)
    ax.axvline(T1 + Tc, color='black', linestyle=':', linewidth=1.0)
    ax.set_xlabel('time [s]  (segments stitched)')
    ax.set_ylabel('v [m/s]')
    ax.set_title('Full stitched velocity profile', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True)

    fig.tight_layout()


# =============================================================================
if __name__ == '__main__':
    main()
