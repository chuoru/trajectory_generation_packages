#!/usr/bin/env python3
##
# @file differential_drive_path_segment_fine_sweep.py
#
# @brief Fine w_energy sweep + three-solution comparison on a 5 m x 5 m L-shaped path.
#
# Path:  (0,0) -> (5,0) -> (5,5)   -- a single 90-degree right turn.
#
# Pipeline
# --------
#   1. PathSegment          -- computes the feasible circular arc geometry at
#                              the corner (standoff L_seg, deviation b, radius R).
#   2. EulerJLAPCoverage    -- jerk-limited velocity profile for the two straight
#                              segments that flank the arc entry/exit points.
#   3. BSplineEnergyCoverage -- energy-aware B-spline OCP through the corner
#                               waypoints; w_energy is swept over ~21 log-spaced
#                               values to identify three characteristic solutions:
#                                 A) Time-optimal  -- minimum mission time
#                                 B) Best tradeoff -- Pareto knee (min d2P/dwe2)
#                                 C) Energy-optimal -- minimum total energy
#
# Figures
# -------
#   Figure 1 -- Segmented path overview (arc geometry + all 3 generator outputs)
#   Figure 2 -- JLAP kinematic profiles for both straight segments
#   Figure 3 -- w_energy sweep statistics (peak power / energy / time / d2)
#   Figure 4 -- Corner XY + power + stitched trajectory (three solutions)
#   Figure 5 -- Per-wheel kinematics for all three segments
#   Figure 6 -- Pareto front with colorbar and three annotated markers
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/05/06

# Standard library
import sys
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as _cm
import matplotlib.colors as _mcolors

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Internal library
from trajectory_generators.path_segment import PathSegment
from trajectory_generators.euler_jlap_coverage import EulerJLAPCoverage
from trajectory_generators.bspline_energy_coverage import BSplineEnergyCoverage


# =============================================================================
# PAPER FIGURE EXPORT
# =============================================================================
SAVE_FIGS   = True
FIG_OUT_DIR = (pathlib.Path(__file__).resolve().parent.parent.parent
               / 'Writting' / 'energy_aware_fine_sweep')


def _savefig(fig, filename):
    """Save fig to FIG_OUT_DIR/<filename> at 300 dpi when SAVE_FIGS is True."""
    if SAVE_FIGS:
        FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = FIG_OUT_DIR / filename
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f"  [paper] Saved {filename} -> {out}")


# =============================================================================
# CONFIGURATION
# =============================================================================

WP_START  = [0.0, 5.0]
WP_CORNER = [0.0, 10.0]   # sharp corner vertex
WP_END    = [5.0, 10.0]

# Headings at the corner: north -> east (right turn)
HEADING_IN  = np.pi / 2     # [rad]
HEADING_OUT = 0.0           # [rad]
BETA        = -np.pi / 2    # 90-degree right turn [rad]

# Straight lead-in / lead-out added to BSpline corner region so the OCP
# starts and ends on a straight section, giving smooth curvature ramp-up.
L_TRANSITION = 0.5   # m

# PathSegment feasibility inputs
L_INPUT     = 1.0           # desired standoff from corner vertex [m]
B_INPUT     = 0.08          # desired deviation tolerance [m]
V_MAX_SEG   = 0.5           # [m/s]
A_MAX_SEG   = 1.0           # [m/s^2]
L_WHEELBASE = 0.35          # [m]

# EulerJLAP (straight segments) -- NewMiniAGV defaults
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
JLAP_DT = 0.05   # [s]

# Jerk limit derived from motor parameters
_j_rated_torque = JLAP_ROBOT_PARAMS['gear_ratio'] * JLAP_ROBOT_PARAMS['rated_motor_torque']
_j_inertia      = JLAP_ROBOT_PARAMS['gear_ratio']**2 * JLAP_ROBOT_PARAMS['motor_inertia']
_j_r            = JLAP_ROBOT_PARAMS['wheel_radius']
_j_m            = JLAP_ROBOT_PARAMS['robot_mass']
_A_LIM          = (0.5 * _j_rated_torque * _j_r
                   / (0.25 * _j_m * _j_r**2 + _j_inertia))   # [m/s²]
J_LIM = _A_LIM / (40.0 * JLAP_DT)                            # [m/s³]

# Handoff velocity
V_HANDOFF     = JLAP_ROBOT_PARAMS['path_vel_lim']   # 0.5 m/s
V_HANDOFF_MIN = 0.10   # minimum fallback handoff velocity [m/s]

# Angular acc/jerk limits derived from JLAP linear limits and wheelbase.
_ANG_ACC_MAX  = _A_LIM / L_WHEELBASE   # [rad/s²]
_ANG_JERK_MAX = J_LIM  / L_WHEELBASE   # [rad/s³]


def _make_bspline_common(v_h):
    """Return BSplineEnergyCoverage kwargs parameterised by handoff speed v_h."""
    return dict(
        bound=0.25,
        n_ctrl_pts=6,
        spline_order=3,
        n_sampling=20,
        vel_max=[v_h, v_h, 0.196],
        vel_min_lin=0.01,
        eps_nonh=0.005,
        omega_entry=0.0,
        omega_exit=0.0,
        acc_max= [_A_LIM, _A_LIM, _ANG_ACC_MAX],
        jerk_max=[J_LIM,  J_LIM,  _ANG_JERK_MAX],
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
COL_ENE = 'mediumorchid'   # energy-optimal corner solution
COL_REF = 'gray'


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
# MAIN
# =============================================================================
def main():
    _set_paper_style()
    # ------------------------------------------------------------------
    # Step 1: PathSegment -- corner arc geometry
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: PathSegment - corner arc geometry")
    print("=" * 60)
    seg_info      = _segment_corner()
    arc_entry     = seg_info['arc_entry_world']
    arc_exit      = seg_info['arc_exit_world']
    arc_entry_ext = seg_info['arc_entry_ext_world']
    arc_exit_ext  = seg_info['arc_exit_ext_world']
    print(f"  feasible = {seg_info['feasible']}   "
          f"L_seg = {seg_info['L_seg']:.3f} m   "
          f"R = {seg_info['R']:.3f} m   "
          f"b = {seg_info['b']:.4f} m")
    print(f"  arc entry : ({arc_entry[0]:.3f}, {arc_entry[1]:.3f})  "
          f"handoff: ({arc_entry_ext[0]:.3f}, {arc_entry_ext[1]:.3f})")
    print(f"  arc exit  : ({arc_exit[0]:.3f},  {arc_exit[1]:.3f})   "
          f"handoff: ({arc_exit_ext[0]:.3f}, {arc_exit_ext[1]:.3f})")

    corner_wps = _build_corner_waypoints(arc_entry_ext, arc_exit_ext)

    # ------------------------------------------------------------------
    # Step 1b: Find the maximum handoff speed that keeps corner jerk <= J_LIM
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 1b: Adaptive V_HANDOFF search (corner jerk constraint)")
    print("=" * 60)
    v_handoff = _find_smooth_v_handoff(corner_wps)
    print(f"  Active V_HANDOFF = {v_handoff:.3f} m/s")

    # ------------------------------------------------------------------
    # Step 2a: EulerJLAP -- segment 1 (accelerates, exits at v_handoff)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Step 2a: EulerJLAPCoverage - segment 1 (0 -> {v_handoff:.3f})")
    print("=" * 60)
    res_s1 = _run_jlap_seg(WP_START, arc_entry_ext.tolist(),
                            initial_vel=0.0, final_vel=v_handoff)
    print(f"  Segment 1: T = {res_s1['time'][-1]:.3f} s   "
          f"v_peak = {np.max(res_s1['v']):.3f} m/s   "
          f"v_exit = {res_s1['v'][-1]:.3f} m/s")

    a_s1_exit     = float(res_s1['acc_path'][-1])
    alpha_s1_exit = float(res_s1['alpha'][-1])

    # ------------------------------------------------------------------
    # Step 3: Fine w_energy sweep (21 pts) -> three corner solutions
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3: BSplineEnergyCoverage - fine w_energy sweep (20 pts)")
    print("=" * 60)
    sweep = _sweep_we(corner_wps,
                      a_entry=a_s1_exit, alpha_entry=alpha_s1_exit,
                      v_handoff=v_handoff)
    opt_we        = sweep['opt_we']
    opt_we_energy = sweep['opt_we_energy']
    we_time_ref   = sweep['we_time_ref']
    res_by_we     = sweep['res_by_we']

    # Retrieve all three corner solutions from the sweep cache.
    res_corner_time   = res_by_we[we_time_ref]
    res_corner_opt    = res_by_we[opt_we]
    res_corner_energy = res_by_we[opt_we_energy]

    mA = _corner_metrics(res_corner_time)
    mB = _corner_metrics(res_corner_opt)
    mC = _corner_metrics(res_corner_energy)
    _print_corner_comparison(mA, mB, mC, opt_we, opt_we_energy, we_time_ref)

    # ------------------------------------------------------------------
    # Step 2b: EulerJLAP -- segment 2 (enters at corner exit speed)
    # All three corners are pinned to v_exit = v_handoff, so one Segment 2
    # suffices; use the knee (opt) exit velocity as the entry speed.
    # ------------------------------------------------------------------
    v_corner_exit = float(max(0.0, res_corner_opt['v'][-1]))
    print()
    print("=" * 60)
    print(f"Step 2b: EulerJLAPCoverage - segment 2 "
          f"(V_entry={v_corner_exit:.3f} -> 0)")
    print("=" * 60)
    res_s2 = _run_jlap_seg(arc_exit_ext.tolist(), WP_END,
                            initial_vel=v_corner_exit, final_vel=0.0)
    print(f"  Segment 2: T = {res_s2['time'][-1]:.3f} s   "
          f"v_peak = {np.max(res_s2['v']):.3f} m/s   "
          f"v_entry = {v_corner_exit:.3f} m/s")

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------
    _export_csv(res_s1, res_s2, res_corner_time, res_corner_opt, res_corner_energy,
                sweep=sweep)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    _fig1_segmented_path(seg_info, res_s1, res_s2, res_corner_opt)
    _fig2_jlap_profiles(res_s1, res_s2)
    _fig3_we_sweep(sweep)
    _fig4_corner_and_full(seg_info,
                          res_corner_time, res_corner_opt, res_corner_energy,
                          mA, mB, mC,
                          res_s1, res_s2,
                          opt_we, opt_we_energy, we_time_ref)
    _fig5_wheel_kinematics(res_s1, res_s2, res_corner_opt)
    _fig6_pareto_front(sweep)

    plt.show()
    plt.close('all')


# =============================================================================
# STEP 1 - PathSegment
# =============================================================================
def _segment_corner():
    """! Compute feasible arc geometry for the 90-degree corner."""
    ps = PathSegment(
        beta=abs(BETA),
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

    c, s  = np.cos(HEADING_IN), np.sin(HEADING_IN)
    R_mat = np.array([[c, -s], [s, c]])
    arc_local    = res['path_segment'].copy()
    if BETA < 0:
        arc_local[:, 1] = -arc_local[:, 1]
        arc_local[:, 2] = -arc_local[:, 2]
    xy_world     = (R_mat @ arc_local[:, :2].T).T + arc_entry
    hdg_world    = arc_local[:, 2] + HEADING_IN
    arc_world    = np.column_stack([xy_world, hdg_world])
    arc_exit     = arc_world[-1, :2].copy()

    out_dir_vec   = np.array([np.cos(HEADING_OUT), np.sin(HEADING_OUT)])
    arc_entry_ext = arc_entry - L_TRANSITION * in_dir
    arc_exit_ext  = arc_exit  + L_TRANSITION * out_dir_vec

    res.update({
        'arc_world':           arc_world,
        'arc_entry_world':     arc_entry,
        'arc_exit_world':      arc_exit,
        'arc_entry_ext_world': arc_entry_ext,
        'arc_exit_ext_world':  arc_exit_ext,
        'corner_vertex':       corner,
    })
    return res


# =============================================================================
# STEP 2 - EulerJLAP straight segments
# =============================================================================
def _run_jlap_seg(wp_start, wp_end, initial_vel=0.0, final_vel=0.0):
    """! Run EulerJLAPCoverage on a single straight segment."""
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
    """! Build the 3-waypoint list for the corner B-spline OCP."""
    return [
        [arc_entry[0], arc_entry[1], HEADING_IN],
        [WP_CORNER[0], WP_CORNER[1], HEADING_IN],
        [arc_exit[0],  arc_exit[1],  HEADING_OUT],
    ]


def _solve_corner(corner_wps, w_energy, warm_start=None,
                  v_entry=None, v_exit=None,
                  a_entry=0.0, a_exit=0.0,
                  alpha_entry=0.0, alpha_exit=0.0):
    """! Run BSplineEnergyCoverage for the corner at a given w_energy."""
    v_entry = float(v_entry) if v_entry is not None else V_HANDOFF
    v_exit  = float(v_exit)  if v_exit  is not None else V_HANDOFF
    return BSplineEnergyCoverage(
        waypoints=corner_wps,
        robot_params=ROBOT_PARAMS_BSPLINE,
        energy_coeffs_right=ENERGY_COEFFS_RIGHT,
        energy_coeffs_left=ENERGY_COEFFS_LEFT,
        w_time=1.0,
        w_energy=float(w_energy),
        p_electronics=P_ELECTRONICS,
        v_entry=v_entry,
        v_exit=v_exit,
        a_entry=a_entry,
        a_exit=a_exit,
        alpha_entry=alpha_entry,
        alpha_exit=alpha_exit,
        **_make_bspline_common(v_entry),
    ).generate_trajectory(warm_start=warm_start)


def _compute_corner_power(res):
    """! Evaluate TJ108 motor power from IK outputs of a corner result."""
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
    """! Compute corner performance metrics."""
    P_tot  = _compute_corner_power(res)
    energy = float(res.get('energy', np.trapz(P_tot, dx=0.01)))
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


def _compute_wheel_kinematics(res, l, r, dt):
    """Compute per-wheel angular velocity, acceleration, and jerk."""
    t = res.get('time_ik', res['time'])
    v, omega = res['v'], res['omega']

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


def _sweep_we(corner_wps, a_entry=0.0, alpha_entry=0.0, v_handoff=None):
    """! Sweep w_energy over 40 linearly-spaced values and identify three optimal points.

    Returns time-optimal, Pareto-knee (best tradeoff), and energy-optimal solutions.

    @param a_entry<float>:     Forward acceleration at corner entry [m/s²].
    @param alpha_entry<float>: Angular acceleration at corner entry [rad/s²].
    @param v_handoff<float|None>: Entry/exit speed for the corner [m/s].
    @return dict with sweep arrays and three optimal w_e values.
    """
    v_h = float(v_handoff) if v_handoff is not None else V_HANDOFF

    # 40 linearly-spaced points in [0.0, 1.0] swept high-to-low for warm-starting.
    # Any points that fail to converge are caught by the exception handler and skipped.
    we_values = np.linspace(0.0, 1.0, 40)[::-1]

    peak_powers    = []
    total_energies = []
    mission_times  = []
    we_valid       = []

    print(f"  Sweeping {len(we_values)} w_e values (w_time = 1.0, v_h = {v_h:.3f} m/s) ...")
    print(f"  {'w_e':>10}  {'Time [s]':>10}  {'Energy [J]':>10}  {'Peak P [W]':>10}")
    print("  " + "-" * 49)

    prev_res = None
    res_by_we = {}
    for w_e in we_values:
        try:
            res = _solve_corner(corner_wps, w_e, warm_start=prev_res,
                                v_entry=v_h, v_exit=v_h,
                                a_entry=a_entry, alpha_entry=alpha_entry)
            P_tot  = _compute_corner_power(res)
            T_tot  = float(res['time'][-1])
            peak_p = float(np.max(P_tot))
            energy = float(res.get('energy', np.trapz(P_tot, dx=0.01)))
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
            prev_res = None

    if len(peak_powers) < 2:
        raise RuntimeError(
            f"Sweep produced {len(peak_powers)} feasible result(s); need at least 2 "
            "to select an optimal w_e. Check IPOPT output above for convergence failures."
        )

    # Sort by ascending w_e for gradient analysis and plotting.
    sort_idx       = np.argsort(we_valid)
    peak_powers    = np.array(peak_powers)[sort_idx]
    total_energies = np.array(total_energies)[sort_idx]
    mission_times  = np.array(mission_times)[sort_idx]
    we_values      = np.array(we_valid)[sort_idx]

    # Drop non-converged outliers (mission time > 5× median).
    if len(mission_times) >= 3:
        t_med  = np.median(mission_times)
        valid  = mission_times <= 5.0 * t_med
        if valid.sum() < len(mission_times):
            n_drop = (~valid).sum()
            print(f"  NOTE: dropping {n_drop} non-converged sweep point(s) "
                  f"(mission time > 5× median={t_med:.2f} s).")
            peak_powers    = peak_powers[valid]
            total_energies = total_energies[valid]
            mission_times  = mission_times[valid]
            we_values      = we_values[valid]
            for w_e in list(res_by_we):
                if w_e not in we_values:
                    del res_by_we[w_e]

    # Drop points that are outliers in peak power OR total energy (> 1.5× median).
    if len(peak_powers) >= 3:
        pp_med = np.median(peak_powers)
        te_med = np.median(total_energies)
        valid_both = ~((peak_powers > 1.5 * pp_med) | (total_energies > 1.5 * te_med))
        if valid_both.sum() < len(peak_powers):
            n_drop = (~valid_both).sum()
            print(f"  NOTE: dropping {n_drop} outlier(s) "
                  f"(peak P > 1.5× median={pp_med:.2f} W OR energy > 1.5× median={te_med:.2f} J).")
            peak_powers    = peak_powers[valid_both]
            total_energies = total_energies[valid_both]
            mission_times  = mission_times[valid_both]
            we_values      = we_values[valid_both]
            for w_e in list(res_by_we):
                if w_e not in we_values:
                    del res_by_we[w_e]

    d1 = np.gradient(peak_powers, we_values)
    d2 = np.gradient(d1, we_values)

    # Pareto knee: minimum d2(P_peak)/d(w_e)^2 at interior points.
    interior = np.arange(1, len(we_values) - 1)
    if len(interior) > 0:
        opt_idx = int(interior[np.argmin(d2[interior])])
    else:
        opt_idx = int(np.argmin(d2))
    if float(we_values[opt_idx]) == 0.0 and len(we_values) > 1:
        opt_idx = 1
    opt_we = float(we_values[opt_idx])

    # Time-optimal: sweep result with shortest mission time.
    time_ref_idx = int(np.argmin(mission_times))
    we_time_ref  = float(we_values[time_ref_idx])
    if we_time_ref != 0.0:
        print(f"  NOTE: w_e=0.0 did not yield minimum time in sweep; "
              f"using w_e={we_time_ref:.4f} as time reference.")

    # Energy-optimal: sweep result with minimum total energy.
    energy_opt_idx = int(np.argmin(total_energies))
    opt_we_energy  = float(we_values[energy_opt_idx])

    print()
    print(f"  Time-optimal   w_e = {we_time_ref:.6f}  "
          f"T = {mission_times[time_ref_idx]:.3f} s")
    print(f"  Pareto knee    w_e = {opt_we:.6f}  "
          f"Peak P = {peak_powers[opt_idx]:.3f} W")
    print(f"  Energy-optimal w_e = {opt_we_energy:.6f}  "
          f"E = {total_energies[energy_opt_idx]:.3f} J")

    return {
        'we_values':      we_values,
        'peak_powers':    peak_powers,
        'total_energies': total_energies,
        'mission_times':  mission_times,
        'd2_pp':          d2,
        'opt_idx':        opt_idx,
        'opt_we':         opt_we,
        'we_time_ref':    we_time_ref,
        'time_ref_idx':   time_ref_idx,
        'energy_opt_idx': energy_opt_idx,
        'opt_we_energy':  opt_we_energy,
        'res_by_we':      res_by_we,
        'v_handoff':      v_h,
    }


def _find_smooth_v_handoff(corner_wps, a_entry=0.0, alpha_entry=0.0):
    """! Find the largest V_HANDOFF that keeps corner transition jerk within J_LIM."""
    l_ref = ROBOT_PARAMS_BSPLINE['l']
    r_ref = ROBOT_PARAMS_BSPLINE['r']
    dt_c  = 0.01
    jerk_wheel_lim = J_LIM

    candidates = [
        V_HANDOFF,
        V_HANDOFF * 0.8,
        V_HANDOFF * 0.6,
        V_HANDOFF * 0.4,
        V_HANDOFF_MIN,
    ]

    prev_res = None
    for v_h in candidates:
        v_h = max(float(v_h), V_HANDOFF_MIN)
        try:
            res = _solve_corner(corner_wps, w_energy=0.01,
                                warm_start=prev_res,
                                v_entry=v_h, v_exit=v_h,
                                a_entry=a_entry, alpha_entry=alpha_entry)
            wk = _compute_wheel_kinematics(res, l_ref, r_ref, dt_c)
            n_edge = min(3, len(wk['jerk_r']))
            j_entry = max(np.max(np.abs(wk['jerk_r'][:n_edge])),
                          np.max(np.abs(wk['jerk_l'][:n_edge])))
            j_exit  = max(np.max(np.abs(wk['jerk_r'][-n_edge:])),
                          np.max(np.abs(wk['jerk_l'][-n_edge:])))
            j_max = max(j_entry, j_exit)
            print(f"  V_HANDOFF probe {v_h:.3f} m/s -> "
                  f"edge jerk = {j_max:.2f} m/s^3  "
                  f"(limit {jerk_wheel_lim:.2f} m/s^3)")
            prev_res = res
            if j_max <= jerk_wheel_lim:
                if v_h < V_HANDOFF:
                    print(f"  NOTE: V_HANDOFF reduced to {v_h:.3f} m/s "
                          f"to satisfy jerk limit.")
                return v_h
        except Exception as exc:
            print(f"  V_HANDOFF probe {v_h:.3f} m/s failed: {exc}")
            prev_res = None

    print(f"  WARNING: all V_HANDOFF probes exceeded jerk limit; "
          f"using minimum {V_HANDOFF_MIN:.3f} m/s.")
    return V_HANDOFF_MIN


# =============================================================================
# TEXT OUTPUT
# =============================================================================
def _print_corner_comparison(mA, mB, mC, opt_we, opt_we_energy, we_time_ref=0.0):
    rows = [
        ('Total time',    's',   'total_time'),
        ('Path length',   'm',   'path_length'),
        ('Total energy',  'J',   'energy'),
        ('Energy/meter',  'J/m', 'energy_per_meter'),
        ('Peak power',    'W',   'peak_power'),
        ('Avg power',     'W',   'avg_power'),
    ]
    print()
    print("  Corner: time-optimal  |  best tradeoff  |  energy-optimal")
    t_col = f'Time-opt (w_e={we_time_ref:.4f})'
    k_col = f'Knee (w_e={opt_we:.4f})'
    e_col = f'Energy-opt (w_e={opt_we_energy:.4f})'
    hdr = (f"  {'Metric':<20} {'Unit':<6} "
           f"{t_col:>22}  {k_col:>18}  {e_col:>24}")
    print(hdr)
    print("  " + "-" * 95)
    for label, unit, key in rows:
        a, b, c = mA[key], mB[key], mC[key]
        def _d(ref, val):
            if abs(ref) > 1e-12:
                d = (val - ref) / abs(ref) * 100
                return f"{'+'if d>=0 else ''}{d:.1f}%"
            return 'n/a'
        print(f"  {label:<20} {unit:<6} {a:>22.4f}  {b:>18.4f}  {c:>24.4f}"
              f"   [{_d(a,b)} / {_d(a,c)}]")
    print()


# =============================================================================
# FIGURE 1 - Segmented path overview
# =============================================================================
def _draw_coverage_background(ax):
    """Draw rectangular inward-spiral reference on 10 m × 10 m field."""
    kw = dict(color='silver', lw=1.2, ls='--', zorder=1)
    field, strip = 10.0, 1.0

    pts = [(0.0, 0.0)]
    xlo, ylo, xhi, yhi = 0.0, 0.0, field, field
    while True:
        pts.append((xhi, ylo)); ylo += strip
        if ylo > yhi: break
        pts.append((xhi, yhi)); xhi -= strip
        if xlo > xhi: break
        pts.append((xlo, yhi)); yhi -= strip
        if ylo > yhi: break
        pts.append((xlo, ylo)); xlo += strip
        if xlo > xhi: break

    xs, ys = zip(*pts)
    ax.plot(xs, ys, **kw)
    ax.annotate('', xy=(5.3, 0.0), xytext=(4.7, 0.0),
                arrowprops=dict(arrowstyle='->', color='silver',
                                lw=1.0, mutation_scale=12))


def _fig1_segmented_path(seg_info, res_s1, res_s2, res_corner_opt):
    """! Zoomed-out boustrophedon overview with inset corner-detail cutout."""
    fig, ax = plt.subplots(figsize=(8, 8),
                           num='Figure 1 - Segmented Path (Fine Sweep)')
    ax.set_aspect('equal')
    _draw_coverage_background(ax)

    ref = np.array([WP_START, WP_CORNER, WP_END])
    ax.plot(ref[:, 0], ref[:, 1], 'o', color='black', markersize=7, zorder=6)

    s1  = res_s1['states']
    sc  = res_corner_opt['states']
    s2  = res_s2['states']
    arc = seg_info['arc_world']

    ax.plot(s1[:, 0], s1[:, 1], '-', color=COL_S1, linewidth=2.0,
            label='Segment 1  (JLAP)', zorder=4)
    ax.plot(sc[:, 0], sc[:, 1], '-', color=COL_C,  linewidth=2.0,
            label='Corner  (B-spline, knee $w_e$)', zorder=4)
    ax.plot(s2[:, 0], s2[:, 1], '-', color=COL_S2, linewidth=2.0,
            label='Segment 2  (JLAP)', zorder=4)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)

    xi1, xi2 = -0.5, 2.8
    yi1, yi2 = 7.5, 10.5
    axins = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
    axins.set_xlim(xi1, xi2)
    axins.set_ylim(yi1, yi2)
    axins.set_aspect('equal')

    ae, ax_e       = seg_info['arc_entry_world'],     seg_info['arc_exit_world']
    ae_ext, ax_ext = seg_info['arc_entry_ext_world'], seg_info['arc_exit_ext_world']
    cpts = res_corner_opt['ctrl_pts']

    axins.plot(arc[:, 0], arc[:, 1], '-.', color='purple', linewidth=1.8, zorder=3,
               label=f'PathSeg arc  R={seg_info["R"]:.2f} m')
    axins.plot(*ae,     's', color='purple', markersize=7, zorder=7,
               label='Arc tangent pts')
    axins.plot(*ax_e,   's', color='purple', markersize=7, zorder=7)
    axins.plot(*ae_ext, 'D', color='dimgray', markersize=6, zorder=7,
               label='Handoff pts')
    axins.plot(*ax_ext, 'D', color='dimgray', markersize=6, zorder=7)

    mask_s1 = s1[:, 1] >= yi1
    if mask_s1.any():
        axins.plot(s1[mask_s1, 0], s1[mask_s1, 1], '-',
                   color=COL_S1, linewidth=2.0, zorder=4)

    axins.plot(sc[:, 0], sc[:, 1], '-', color=COL_C, linewidth=2.0, zorder=4)
    axins.plot(cpts[:, 0], cpts[:, 1], 'x', color=COL_C,
               markersize=7, markeredgewidth=1.5, zorder=5)

    mask_s2 = s2[:, 0] <= xi2
    if mask_s2.any():
        axins.plot(s2[mask_s2, 0], s2[mask_s2, 1], '-',
                   color=COL_S2, linewidth=2.0, zorder=4)

    for states, col, mask in [
        (s1, COL_S1, mask_s1), (sc, COL_C, np.ones(len(sc), dtype=bool)),
        (s2, COL_S2, mask_s2),
    ]:
        sub = states[mask]
        if len(sub) == 0:
            continue
        step = max(1, len(sub) // 7)
        for st in sub[::step]:
            dx = 0.08 * np.cos(st[2])
            dy = 0.08 * np.sin(st[2])
            axins.annotate('', xy=(st[0]+dx, st[1]+dy), xytext=(st[0], st[1]),
                           arrowprops=dict(arrowstyle='->', color=col, lw=1.0))

    axins.set_xlabel('x [m]', fontsize=9)
    axins.set_ylabel('y [m]', fontsize=9)
    axins.tick_params(labelsize=8)
    axins.legend(fontsize=7, loc='lower right')
    ax.indicate_inset_zoom(axins, edgecolor='black', linewidth=1.2)

    fig.tight_layout()
    _savefig(fig, 'fig_overview.png')


# =============================================================================
# FIGURE 2 - JLAP kinematic profiles
# =============================================================================
def _fig2_jlap_profiles(res_s1, res_s2):
    """! Left-wheel v / a / jerk for Segment 1 (left panel) and Segment 2 (right panel)."""
    r = JLAP_ROBOT_PARAMS['wheel_radius']
    l = 0.5 * JLAP_ROBOT_PARAMS['robot_width']

    wk1 = _compute_wheel_kinematics(res_s1, l, r, JLAP_DT)
    wk2 = _compute_wheel_kinematics(res_s2, l, r, JLAP_DT)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4),
                             num='Figure 2 - JLAP Kinematic Profiles (Fine Sweep)')

    for ax, wk in zip(axes, [wk1, wk2]):
        t = wk['time']
        ax.plot(t, wk['omega_l'] * r, '-',  color='steelblue',    lw=1.8, label='v [m/s]')
        ax.plot(t, wk['alpha_l'] * r, '--', color='darkorange',   lw=1.8, label='a [m/s²]')
        ax.plot(t, wk['jerk_l'],      ':',  color='mediumorchid', lw=1.8, label='j [m/s³]')
        ax.axhline(0, color='lightgray', lw=0.8, zorder=0)
        ax.set_xlabel('time [s]', fontsize=13)
        ax.set_ylabel('m/s  /  m/s²  /  m/s³', fontsize=13)
        ax.tick_params(labelsize=12)

    axes[0].legend(fontsize=12)
    fig.tight_layout()
    _savefig(fig, 'fig_jlap.png')


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
                             num='Figure 3 - w_energy Sweep (Corner OCP) (Fine Sweep)')

    specs = [
        (axes[0, 0], pp, 'o', COL_C,     'Peak Power [W]',    'Peak Power Suppression'),
        (axes[0, 1], te, 's', COL_S1,    'Total Energy [J]',  'Total Energy vs w_energy'),
        (axes[1, 0], mt, '^', COL_S2,    'Mission Time [s]',  'Mission Time vs w_energy'),
        (axes[1, 1], d2, 'D', COL_OPT,   'd2(Peak P)/d(w_e)2', '2nd Derivative - Optimal Trade-off'),
    ]

    for ax, data, marker, color, ylabel, title in specs:
        ax.scatter(we, data, marker=marker, color=color, s=16, label=ylabel)
        ax.axvline(opt_we, color='black', linestyle='--', linewidth=1.2,
                   label=f'opt w_e = {opt_we:.4f}')
        ax.scatter([opt_we], [data[oi]], color='black', s=80, zorder=6)
        if ax is axes[1, 1]:
            ax.axhline(0, color=COL_REF, linewidth=0.8, linestyle=':')
        ax.set_xscale('symlog', linthresh=5e-4)
        ax.set_xlabel('w_energy (symlog scale)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10)

    fig.tight_layout()


# =============================================================================
# FIGURE 4 - Corner detail + full stitched trajectory (three solutions)
# =============================================================================
def _fig4_corner_and_full(seg_info,
                           res_corner_time, res_corner_opt, res_corner_energy,
                           mA, mB, mC,
                           res_s1, res_s2,
                           opt_we, opt_we_energy, we_time_ref=0.0):
    """! Corner comparison (XY + power) and full stitched path + velocity.

    Shows all three corner solutions: time-optimal, Pareto knee, energy-optimal.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10),
                             num='Figure 4 - Corner Detail + Full Trajectory (Fine Sweep)')

    # ------------------------------------------------------------------ [0,0]
    # Corner XY: all three solutions
    ax = axes[0, 0]
    ax.set_aspect('equal')

    arc = seg_info['arc_world']
    ax.plot(arc[:, 0], arc[:, 1], '-.', color='purple', linewidth=1.5,
            label='PathSegment arc', zorder=2)

    cwps = np.array(_build_corner_waypoints(seg_info['arc_entry_ext_world'],
                                            seg_info['arc_exit_ext_world']))
    ax.plot(cwps[:, 0], cwps[:, 1], 'o--', color=COL_REF, markersize=7,
            linewidth=1.0, label='Corner waypoints', zorder=3)

    st_A = res_corner_time['states']
    ax.plot(st_A[:, 0], st_A[:, 1], '-', color=COL_S1, linewidth=2.0,
            label=f'Time-opt  (w_e={we_time_ref:.4f})', zorder=4)

    st_B = res_corner_opt['states']
    ax.plot(st_B[:, 0], st_B[:, 1], '-', color=COL_OPT, linewidth=2.0,
            label=f'Knee  (w_e={opt_we:.4f})', zorder=4)
    cpts = res_corner_opt['ctrl_pts']
    ax.plot(cpts[:, 0], cpts[:, 1], 'x', color=COL_OPT,
            markersize=7, markeredgewidth=1.5, zorder=5)

    st_C = res_corner_energy['states']
    ax.plot(st_C[:, 0], st_C[:, 1], '-', color=COL_ENE, linewidth=2.0,
            label=f'Energy-opt  (w_e={opt_we_energy:.4f})', zorder=4)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend(fontsize=9)

    # ------------------------------------------------------------------ [0,1]
    # Power P(t): all three corner solutions
    ax = axes[0, 1]
    ax.plot(mA['time_ik'], mA['P_total'], '-', color=COL_S1, linewidth=1.8,
            label=f"Time-opt (w_e={we_time_ref:.4f})  peak={mA['peak_power']:.1f} W")
    ax.plot(mB['time_ik'], mB['P_total'], '-', color=COL_OPT, linewidth=1.8,
            label=f"Knee  peak={mB['peak_power']:.1f} W")
    ax.plot(mC['time_ik'], mC['P_total'], '-', color=COL_ENE, linewidth=1.8,
            label=f"Energy-opt  peak={mC['peak_power']:.1f} W")
    ax.axhline(P_ELECTRONICS, color=COL_REF, linestyle=':', linewidth=1.0,
               label=f'P_elec = {P_ELECTRONICS} W')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('Power [W]')
    ax.legend(fontsize=9)

    # ------------------------------------------------------------------ [1,0]
    # Full stitched XY trajectory (knee solution)
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
            label='Corner (B-spline, knee)', zorder=4)
    ax.plot(s2[:, 0], s2[:, 1], '-', color=COL_S2, linewidth=2.0,
            label='Segment 2 (JLAP)', zorder=4)

    ae, ax_e       = seg_info['arc_entry_world'],     seg_info['arc_exit_world']
    ae_ext, ax_ext = seg_info['arc_entry_ext_world'], seg_info['arc_exit_ext_world']
    ax.plot(*ae,     's', color='black',   markersize=8, zorder=7,
            label='Arc tangent points')
    ax.plot(*ax_e,   's', color='black',   markersize=8, zorder=7)
    ax.plot(*ae_ext, 'D', color='dimgray', markersize=7, zorder=7,
            label='JLAP handoff points')
    ax.plot(*ax_ext, 'D', color='dimgray', markersize=7, zorder=7)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend(fontsize=9)

    # ------------------------------------------------------------------ [1,1]
    # Stitched velocity profile: all three corner solutions
    ax = axes[1, 1]
    T1 = float(res_s1['time'][-1])
    Tc = float(res_corner_opt['time'][-1])

    for res_c, col, lbl in [
        (res_corner_time,   COL_S1,  f'Time-opt (w_e={we_time_ref:.4f})'),
        (res_corner_opt,    COL_OPT, f'Knee (w_e={opt_we:.4f})'),
        (res_corner_energy, COL_ENE, f'Energy-opt (w_e={opt_we_energy:.4f})'),
    ]:
        Tc_i = float(res_c['time'][-1])
        ax.plot(res_c['time_ik'] + T1, res_c['v'],
                '-', color=col, linewidth=1.5, label=lbl)

    ax.plot(res_s1['time'], res_s1['v'],
            '-', color=COL_S1, linewidth=2.2, label='Segment 1 (JLAP)',
            zorder=5)
    ax.plot(res_s2['time'] + T1 + Tc, res_s2['v'],
            '-', color=COL_S2, linewidth=2.2, label='Segment 2 (JLAP)',
            zorder=5)

    ax.axvline(T1,      color='black', linestyle=':', linewidth=1.0)
    ax.axvline(T1 + Tc, color='black', linestyle=':', linewidth=1.0)
    ax.set_xlabel('time [s]  (segments stitched, knee offset)')
    ax.set_ylabel('v [m/s]')
    ax.legend(fontsize=9)

    fig.tight_layout()
    _savefig(fig, 'fig_corner_detail.png')


# =============================================================================
# FIGURE 5 - Per-wheel kinematics
# =============================================================================
def _fig5_wheel_kinematics(res_s1, res_s2, res_corner_opt):
    """! Per-wheel angular velocity, acceleration, and jerk for all three segments."""
    l_ref = ROBOT_PARAMS_BSPLINE['l']
    r_ref = ROBOT_PARAMS_BSPLINE['r']

    wk_s1 = _compute_wheel_kinematics(res_s1,         l_ref, r_ref, JLAP_DT)
    wk_c  = _compute_wheel_kinematics(res_corner_opt, l_ref, r_ref, 0.01)
    wk_s2 = _compute_wheel_kinematics(res_s2,         l_ref, r_ref, JLAP_DT)

    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex='col',
                             num='Figure 5 - Per-Wheel Kinematics (Fine Sweep)')

    segments = [
        (wk_s1, COL_S1, 'Segment 1 (JLAP)'),
        (wk_c,  COL_C,  'Corner (B-spline knee)'),
        (wk_s2, COL_S2, 'Segment 2 (JLAP)'),
    ]
    row_keys    = [('omega_r', 'omega_l'), ('alpha_r', 'alpha_l'), ('jerk_r', 'jerk_l')]
    row_ylabels = ['ω_wheel [rad/s]', 'α_wheel [rad/s²]', 'jerk [m/s³]']

    for col, (wk, color, title) in enumerate(segments):
        for row, ((kr, kl), ylabel) in enumerate(zip(row_keys, row_ylabels)):
            ax = axes[row, col]
            ax.plot(wk['time'], wk[kr], '-',  color=color, linewidth=1.8, label='Right')
            ax.plot(wk['time'], wk[kl], '--', color=color, linewidth=1.8,
                    label='Left', alpha=0.75)
            if row == 2:
                ax.axhline( J_LIM, color='red', linewidth=1.0, linestyle=':', label=f'+J_LIM={J_LIM:.2f}')
                ax.axhline(-J_LIM, color='red', linewidth=1.0, linestyle=':')
            ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel('time [s]')
            ax.legend(fontsize=10)

    fig.tight_layout()
    _savefig(fig, 'fig_wheel_kinematics.png')


# =============================================================================
# FIGURE 6 - Pareto front (paper figure)
# =============================================================================
def _fig6_pareto_front(sweep):
    """! Pareto front with colorbar and three annotated special markers.

    Uses a viridis colorbar keyed on log10(w_e) instead of per-point labels
    to keep the figure readable with many sweep points.
    """
    we     = sweep['we_values']
    pp     = sweep['peak_powers']
    te     = sweep['total_energies']
    oi     = sweep['opt_idx']
    tri    = sweep['time_ref_idx']
    eoi    = sweep['energy_opt_idx']
    opt_we = sweep['opt_we']
    we_ref = sweep['we_time_ref']
    we_ene = sweep['opt_we_energy']

    fig, ax = plt.subplots(figsize=(7, 5),
                           num='Figure 6 - Pareto Front: Peak Power vs. Total Energy (Fine Sweep)')

    # Scatter all points, coloured by log10(w_e).
    log_we = np.where(we > 0, np.log10(we), np.log10(1e-5))
    sc = ax.scatter(te, pp, c=log_we, cmap='viridis', s=55, zorder=3,
                    vmin=np.min(log_we), vmax=np.max(log_we))
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('log₁₀(w_energy)')

    # Time-optimal marker
    ax.scatter([te[tri]], [pp[tri]], marker='^', s=150, color=COL_S1,
               zorder=6, label=f'Time-opt  ($w_e$={we_ref:.4f})')
    ax.annotate(f'time-opt\n$w_e$={we_ref:.4f}',
                xy=(te[tri], pp[tri]),
                xytext=(6, 6), textcoords='offset points',
                fontsize=8, color=COL_S1)

    # Pareto knee marker
    ax.axvline(te[oi], color=COL_OPT, ls='--', lw=1.0, alpha=0.7, zorder=2)
    ax.axhline(pp[oi], color=COL_OPT, ls='--', lw=1.0, alpha=0.7, zorder=2)
    ax.scatter([te[oi]], [pp[oi]], marker='o', s=180, color=COL_OPT,
               zorder=6, label=f'Knee  ($w_e$={opt_we:.4f})')
    ax.annotate(f'knee\n$w_e$={opt_we:.4f}',
                xy=(te[oi], pp[oi]),
                xytext=(6, -22), textcoords='offset points',
                fontsize=8, color=COL_OPT)

    # Energy-optimal marker
    ax.scatter([te[eoi]], [pp[eoi]], marker='s', s=150, color=COL_ENE,
               zorder=6, label=f'Energy-opt  ($w_e$={we_ene:.4f})')
    ax.annotate(f'energy-opt\n$w_e$={we_ene:.4f}',
                xy=(te[eoi], pp[eoi]),
                xytext=(6, 6), textcoords='offset points',
                fontsize=8, color=COL_ENE)

    ax.set_xlabel('Total Energy [J]')
    ax.set_ylabel('Peak Motor Power [W]')
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()
    _savefig(fig, 'fig_pareto.png')


# =============================================================================
# CSV EXPORT
# =============================================================================
def _export_csv(res_s1, res_s2,
                res_corner_time, res_corner_opt, res_corner_energy,
                sweep=None,
                out_dir=None):
    """! Write per-segment, stitched trajectory, and sweep CSV files.

    Files written:
      segment1_jlap.csv        -- straight segment 1 (JLAP)
      corner_time_opt.csv      -- corner B-spline (time-optimal)
      corner_knee.csv          -- corner B-spline (Pareto knee)
      corner_energy_opt.csv    -- corner B-spline (energy-optimal)
      segment2_jlap.csv        -- straight segment 2 (JLAP)
      trajectory_stitched.csv  -- all three on a single time axis (knee solution)
      sweep_statistics.csv     -- per-w_e metrics for offline filtering/analysis
      corners_by_we/           -- per-w_e corner trajectory (one file per w_e)
    """
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), 'csv_output_fine_sweep')
    os.makedirs(out_dir, exist_ok=True)

    def _save(fname, header, arrays):
        path = os.path.join(out_dir, fname)
        data = np.column_stack(arrays)
        np.savetxt(path, data, delimiter=',',
                   header=','.join(header), comments='', fmt='%.8f')
        print(f"  Saved {fname}  ({data.shape[0]} rows x {data.shape[1]} cols)")

    print()
    print("=" * 60)
    print("CSV export")
    print("=" * 60)

    l_ref = ROBOT_PARAMS_BSPLINE['l']
    r_ref = ROBOT_PARAMS_BSPLINE['r']

    # ------------------------------------------------------------------
    # Segment 1 - JLAP
    # ------------------------------------------------------------------
    s1 = res_s1['states']
    omr_s1   = (res_s1['v'] + l_ref * res_s1['omega']) / r_ref
    oml_s1   = (res_s1['v'] - l_ref * res_s1['omega']) / r_ref
    alr_s1   = (res_s1['acc_path'] + l_ref * res_s1['alpha']) / r_ref
    all_s1   = (res_s1['acc_path'] - l_ref * res_s1['alpha']) / r_ref
    jrkr_s1  = np.gradient(alr_s1, JLAP_DT) * r_ref
    jrkl_s1  = np.gradient(all_s1, JLAP_DT) * r_ref
    _save('segment1_jlap.csv',
          ['time', 'x', 'y', 'theta', 'v', 'acc_path', 'omega', 'alpha',
           'omega_r', 'omega_l', 'alpha_r', 'alpha_l', 'jerk_r', 'jerk_l'],
          [res_s1['time'], s1[:, 0], s1[:, 1], s1[:, 2],
           res_s1['v'], res_s1['acc_path'], res_s1['omega'], res_s1['alpha'],
           omr_s1, oml_s1, alr_s1, all_s1, jrkr_s1, jrkl_s1])

    # ------------------------------------------------------------------
    # Corner solutions (time-opt, knee, energy-opt)
    # ------------------------------------------------------------------
    corner_specs = [
        ('corner_time_opt.csv',   res_corner_time),
        ('corner_knee.csv',       res_corner_opt),
        ('corner_energy_opt.csv', res_corner_energy),
    ]
    for fname, res_c in corner_specs:
        t_ik  = res_c['time_ik']
        t_ocp = res_c['time']
        sc    = res_c['states']
        x_c   = np.interp(t_ik, t_ocp, sc[:, 0])
        y_c   = np.interp(t_ik, t_ocp, sc[:, 1])
        th_c  = np.interp(t_ik, t_ocp, sc[:, 2])
        pwr_c = np.interp(t_ik, t_ocp, res_c['power'])
        alr_c  = (res_c['acc_path'] + l_ref * res_c['alpha']) / r_ref
        all_c  = (res_c['acc_path'] - l_ref * res_c['alpha']) / r_ref
        jrkr_c = res_c['jerk_r']
        jrkl_c = res_c['jerk_l']
        _save(fname,
              ['time', 'x', 'y', 'theta', 'v', 'acc_path', 'omega', 'alpha',
               'omega_r', 'omega_l', 'alpha_r', 'alpha_l', 'jerk_r', 'jerk_l', 'power'],
              [t_ik, x_c, y_c, th_c,
               res_c['v'], res_c['acc_path'], res_c['omega'], res_c['alpha'],
               res_c['omega_r'], res_c['omega_l'],
               alr_c, all_c, jrkr_c, jrkl_c, pwr_c])

    # ------------------------------------------------------------------
    # Segment 2 - JLAP
    # ------------------------------------------------------------------
    s2 = res_s2['states']
    omr_s2   = (res_s2['v'] + l_ref * res_s2['omega']) / r_ref
    oml_s2   = (res_s2['v'] - l_ref * res_s2['omega']) / r_ref
    alr_s2   = (res_s2['acc_path'] + l_ref * res_s2['alpha']) / r_ref
    all_s2   = (res_s2['acc_path'] - l_ref * res_s2['alpha']) / r_ref
    jrkr_s2  = np.gradient(alr_s2, JLAP_DT) * r_ref
    jrkl_s2  = np.gradient(all_s2, JLAP_DT) * r_ref
    _save('segment2_jlap.csv',
          ['time', 'x', 'y', 'theta', 'v', 'acc_path', 'omega', 'alpha',
           'omega_r', 'omega_l', 'alpha_r', 'alpha_l', 'jerk_r', 'jerk_l'],
          [res_s2['time'], s2[:, 0], s2[:, 1], s2[:, 2],
           res_s2['v'], res_s2['acc_path'], res_s2['omega'], res_s2['alpha'],
           omr_s2, oml_s2, alr_s2, all_s2, jrkr_s2, jrkl_s2])

    # ------------------------------------------------------------------
    # Stitched trajectory (knee solution)
    # ------------------------------------------------------------------
    t_ik_knee  = res_corner_opt['time_ik']
    t_ocp_knee = res_corner_opt['time']
    sc_knee    = res_corner_opt['states']
    x_ck   = np.interp(t_ik_knee, t_ocp_knee, sc_knee[:, 0])
    y_ck   = np.interp(t_ik_knee, t_ocp_knee, sc_knee[:, 1])
    th_ck  = np.interp(t_ik_knee, t_ocp_knee, sc_knee[:, 2])
    T1 = float(res_s1['time'][-1])
    Tc = float(t_ocp_knee[-1])

    seg_id = np.concatenate([
        np.ones(len(res_s1['time'])),
        np.full(len(t_ik_knee), 2),
        np.full(len(res_s2['time']) - 1, 3),
    ])
    _save('trajectory_stitched.csv',
          ['time', 'x', 'y', 'theta', 'v', 'omega', 'omega_r', 'omega_l', 'segment'],
          [np.concatenate([res_s1['time'], t_ik_knee + T1, res_s2['time'][1:] + T1 + Tc]),
           np.concatenate([s1[:, 0], x_ck, s2[1:, 0]]),
           np.concatenate([s1[:, 1], y_ck, s2[1:, 1]]),
           np.concatenate([s1[:, 2], th_ck, s2[1:, 2]]),
           np.concatenate([res_s1['v'],              res_corner_opt['v'],       res_s2['v'][1:]]),
           np.concatenate([res_s1['omega'],           res_corner_opt['omega'],   res_s2['omega'][1:]]),
           np.concatenate([omr_s1,                   res_corner_opt['omega_r'], omr_s2[1:]]),
           np.concatenate([oml_s1,                   res_corner_opt['omega_l'], oml_s2[1:]]),
           seg_id])

    # ------------------------------------------------------------------
    # Sweep statistics + per-w_e corner trajectories
    # ------------------------------------------------------------------
    if sweep is not None:
        _save('sweep_statistics.csv',
              ['w_e', 'peak_power_W', 'total_energy_J', 'mission_time_s', 'd2_peak_power'],
              [sweep['we_values'], sweep['peak_powers'],
               sweep['total_energies'], sweep['mission_times'], sweep['d2_pp']])

        corners_dir = os.path.join(out_dir, 'corners_by_we')
        os.makedirs(corners_dir, exist_ok=True)
        for w_e, res_c in sweep['res_by_we'].items():
            t_ik  = res_c['time_ik']
            t_ocp = res_c['time']
            sc    = res_c['states']
            x_c   = np.interp(t_ik, t_ocp, sc[:, 0])
            y_c   = np.interp(t_ik, t_ocp, sc[:, 1])
            th_c  = np.interp(t_ik, t_ocp, sc[:, 2])
            pwr_c = np.interp(t_ik, t_ocp, res_c['power'])
            fname = f'corner_we_{w_e:.6f}.csv'
            path  = os.path.join(corners_dir, fname)
            data  = np.column_stack([t_ik, x_c, y_c, th_c,
                                     res_c['v'], res_c['omega'], pwr_c])
            np.savetxt(path, data, delimiter=',',
                       header='time,x,y,theta,v,omega,power',
                       comments='', fmt='%.8f')
        print(f"  Saved {len(sweep['res_by_we'])} corner files -> {corners_dir}/")

    print(f"  Output directory: {out_dir}")


# =============================================================================
if __name__ == '__main__':
    main()
