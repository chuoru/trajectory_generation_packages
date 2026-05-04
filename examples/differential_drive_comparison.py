#!/usr/bin/env python3
##
# @file differential_drive_comparison.py
#
# @brief Three-method trajectory comparison on a 5 m × 5 m L-shaped path.
#
# Path:  (0,0) → (5,0) → (5,5)   -- 90-degree left turn.
#
# Methods
# -------
#   A. EulerJLAPCoverage  -- full 3-waypoint path; Euler spiral corner
#                            smoothing + JLAP jerk-limited velocity profiling.
#   B. Segmented pipeline -- PathSegment arc geometry; EulerJLAP for both
#                            straight segments; BSpline corner OCP at
#                            we_time_ref (time-optimal corner).
#   C. Segmented pipeline -- same architecture as B, but corner BSpline OCP
#                            at sweep-optimal w_energy (energy-aware corner).
#
# Methods B and C share the same PathSegment geometry, JLAP straight segments,
# and v_handoff.  The w_e sweep is run once; B uses the corner result at
# we_time_ref and C uses the corner result at opt_we.  This makes B vs C a
# direct measurement of energy-awareness at the corner only.
#
# Power metrics are evaluated with the same TJ108 wheel-level model for all
# three methods so that energy and peak-power numbers are directly comparable.
#
# Figures
# -------
#   Figure 1 -- XY trajectory overlay
#   Figure 2 -- Velocity profiles v(t) on a common absolute time axis
#   Figure 3 -- Power profiles P(t)
#   Figure 4 -- Summary metrics bar chart (4 KPIs × 3 methods)
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/05/03

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
WP_CORNER = [5.0, 0.0]
WP_END    = [5.0, 5.0]

HEADING_IN  = 0.0
HEADING_OUT = np.pi / 2
BETA        = np.pi / 2

# Straight lead-in / lead-out added to BSpline corner region so the OCP
# starts and ends on a straight section, giving smooth curvature ramp-up.
L_TRANSITION = 0.5   # m

# PathSegment feasibility inputs (Methods B & C corner geometry)
L_INPUT     = 1.0
B_INPUT     = 0.08
V_MAX_SEG   = 0.5
A_MAX_SEG   = 1.0
L_WHEELBASE = 0.35

# EulerJLAP robot params (all three methods use these for straight segments)
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
JLAP_DT = 0.05

# Derived motion limits (mirrors differential_drive_path_segment_combined.py)
_j_rated_torque = JLAP_ROBOT_PARAMS['gear_ratio'] * JLAP_ROBOT_PARAMS['rated_motor_torque']
_j_inertia      = JLAP_ROBOT_PARAMS['gear_ratio']**2 * JLAP_ROBOT_PARAMS['motor_inertia']
_j_r            = JLAP_ROBOT_PARAMS['wheel_radius']
_j_m            = JLAP_ROBOT_PARAMS['robot_mass']
_A_LIM          = (0.5 * _j_rated_torque * _j_r
                   / (0.25 * _j_m * _j_r**2 + _j_inertia))
J_LIM           = _A_LIM / (40.0 * JLAP_DT)

_ANG_ACC_MAX  = _A_LIM / L_WHEELBASE
_ANG_JERK_MAX = J_LIM  / L_WHEELBASE

V_HANDOFF     = JLAP_ROBOT_PARAMS['path_vel_lim']
V_HANDOFF_MIN = 0.10

# BSpline OCP robot geometry (Methods B and C corner)
ROBOT_PARAMS_BSPLINE = {'l': 0.53 / 2, 'r': 0.3}

# TJ108 energy model — same for all three methods
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
P_ELECTRONICS = 2.0

COL_A   = 'steelblue'
COL_B   = 'tomato'
COL_C   = 'seagreen'
COL_REF = 'gray'


def _make_bspline_common(v_h):
    """Corner OCP kwargs factory — consistent with combined pipeline."""
    return dict(
        bound=0.25,
        n_ctrl_pts=6,
        spline_order=3,
        n_sampling=40,
        vel_max=[v_h, v_h, 0.196],
        vel_min_lin=0.01,
        eps_nonh=0.005,
        omega_entry=0.0,
        omega_exit=0.0,
        acc_max=[_A_LIM, _A_LIM, _ANG_ACC_MAX],
        jerk_max=[J_LIM, J_LIM,  _ANG_JERK_MAX],
    )


# =============================================================================
# COVERAGE TOLERANCE SUMMARY
# =============================================================================
def _print_coverage_tolerances():
    print("\n" + "=" * 60)
    print("Coverage tolerance summary")
    print("=" * 60)
    print("  Method A  (EulerJLAP, full path)")
    print(f"    epsilon_offset = {0.25:.3f} m  (Euler spiral corner lateral tolerance)")
    print(f"    lc_scale       = {0.4:.3f}    (max corner reach fraction)")
    print("  Methods B & C  (BSpline corner OCP)")
    print(f"    bound          = {0.25:.3f} m  (BSpline corridor half-width)")
    print(f"    eps_nonh       = {0.005:.4f}  (nonholonomic relaxation)")
    print("  Methods B & C  (EulerJLAP straight segments)")
    print(f"    epsilon_offset = {0.1:.3f} m  (same as Method A) ✓")
    print()


# =============================================================================
# MAIN
# =============================================================================
def main():
    # ------------------------------------------------------------------
    # Method A: EulerJLAP full path
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Method A: EulerJLAPCoverage — full 3-waypoint path")
    print("=" * 60)
    res_a = _run_method_a()
    print(f"  T = {res_a['time'][-1]:.3f} s   "
          f"v_peak = {np.max(res_a['v']):.3f} m/s")

    # ------------------------------------------------------------------
    # Methods B & C: shared segmented pipeline
    # The sweep is run once; B takes the time-optimal corner (we_time_ref)
    # and C takes the energy-optimal corner (opt_we).
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Methods B & C: Segmented pipeline  (shared PathSegment + JLAP + sweep)")
    print("  B → corner at we_time_ref  (time-optimal)")
    print("  C → corner at opt_we       (energy-optimal)")
    print("=" * 60)
    seg  = _run_segmented_pipeline()
    we_b = seg['sweep']['we_time_ref']
    we_c = seg['sweep']['opt_we']
    res_b = _build_segmented_result(seg, we_b)
    res_c = _build_segmented_result(seg, we_c)
    T_b   = res_b['T_s1'] + res_b['T_corner'] + res_b['T_s2']
    T_c   = res_c['T_s1'] + res_c['T_corner'] + res_c['T_s2']
    print(f"\n  Method B  w_e = {we_b:.4f}   T = {T_b:.3f} s")
    print(f"  Method C  w_e = {we_c:.4f}   T = {T_c:.3f} s")

    # ------------------------------------------------------------------
    # Power (uniform TJ108 model)
    # ------------------------------------------------------------------
    pm_a = _power_for_jlap(res_a, JLAP_DT)
    pm_b = _power_for_segmented(res_b)
    pm_c = _power_for_segmented(res_c)

    # ------------------------------------------------------------------
    # Metrics and comparison table
    # ------------------------------------------------------------------
    m_a = _compute_metrics(res_a['states'],
                           float(res_a['time'][-1]), pm_a['time'], pm_a['P'])
    m_b = _compute_metrics(_stitch_states(res_b), T_b, pm_b['time'], pm_b['P'])
    m_c = _compute_metrics(_stitch_states(res_c), T_c, pm_c['time'], pm_c['P'])
    _print_comparison(m_a, m_b, m_c, we_b, we_c)
    _print_coverage_tolerances()

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    _fig1_xy_overlay(res_a, res_b, res_c, we_b, we_c)
    _fig2_velocity(res_a, res_b, res_c, we_b, we_c)
    _fig3_power(pm_a, pm_b, pm_c, m_a, m_b, m_c, res_b, res_c, we_b, we_c)
    _fig4_bars(m_a, m_b, m_c, we_b, we_c)
    _fig5_acc_jerk(res_a, res_b, res_c, we_b, we_c)

    plt.show()
    plt.close('all')


def _stitch_states(res_seg):
    """Concatenate state arrays from the three sub-segments into one array."""
    return np.concatenate([
        res_seg['res_s1']['states'],
        res_seg['res_corner']['states'][1:],
        res_seg['res_s2']['states'][1:],
    ], axis=0)


def _stitch_acc_jerk(res_seg):
    """Return (t_abs, a_abs, j_abs) stitched across all three sub-segments."""
    T_s1     = res_seg['T_s1']
    T_corner = res_seg['T_corner']
    a_s1 = res_seg['res_s1']['acc_path']
    a_c  = res_seg['res_corner']['acc_path']
    a_s2 = res_seg['res_s2']['acc_path']
    j_s1 = np.gradient(a_s1, JLAP_DT)
    j_c  = np.gradient(a_c,  0.01)
    j_s2 = np.gradient(a_s2, JLAP_DT)
    t_abs = np.concatenate([
        res_seg['res_s1']['time'],
        res_seg['res_corner']['time_ik'] + T_s1,
        res_seg['res_s2']['time'][1:]    + T_s1 + T_corner,
    ])
    a_abs = np.concatenate([a_s1, a_c,    a_s2[1:]])
    j_abs = np.concatenate([j_s1, j_c,    j_s2[1:]])
    return t_abs, a_abs, j_abs


# =============================================================================
# METHOD A — EulerJLAPCoverage, full path
# =============================================================================
def _run_method_a():
    return EulerJLAPCoverage(
        waypoints=[WP_START, WP_CORNER, WP_END],
        sampling_time=JLAP_DT,
        robot_params=JLAP_ROBOT_PARAMS,
        path_vel_step=0.01,
        epsilon_offset=0.25,
        lc_scale=0.4,
        initial_vel=0.0,
        final_vel=0.0,
    ).generate_trajectory()


# =============================================================================
# METHODS B & C — shared segmented pipeline
# =============================================================================
def _run_segmented_pipeline():
    """Run PathSegment + JLAP straights + w_e sweep.

    All corners in the sweep pin v_exit = v_handoff, so the same segment-2
    JLAP result is valid for both B and C.

    @return dict consumed by _build_segmented_result().
    """
    seg_info      = _segment_corner()
    arc_entry_ext = seg_info['arc_entry_ext_world']
    arc_exit_ext  = seg_info['arc_exit_ext_world']
    corner_wps    = _build_corner_waypoints(arc_entry_ext, arc_exit_ext)

    v_handoff = _find_smooth_v_handoff(corner_wps)
    print(f"  Active V_HANDOFF = {v_handoff:.3f} m/s")

    # JLAP segments end/start at the buffered handoff points (L_TRANSITION m
    # before/after the arc tangent points) so the handoff falls inside the
    # cruise phase — acceleration and jerk are ~0 at both junctions.
    res_s1        = _run_jlap_seg(WP_START, arc_entry_ext.tolist(),
                                   initial_vel=0.0, final_vel=v_handoff)
    a_s1_exit     = float(res_s1['acc_path'][-1])
    alpha_s1_exit = float(res_s1['alpha'][-1])
    print(f"  Segment 1: T = {res_s1['time'][-1]:.3f} s")

    print()
    sweep = _sweep_we(corner_wps,
                      a_entry=a_s1_exit, alpha_entry=alpha_s1_exit,
                      v_handoff=v_handoff)

    # v_exit is pinned to v_handoff for every corner in the sweep,
    # so segment 2 entry speed is shared between B and C.
    res_s2 = _run_jlap_seg(arc_exit_ext.tolist(), WP_END,
                            initial_vel=v_handoff, final_vel=0.0)
    print(f"  Segment 2: T = {res_s2['time'][-1]:.3f} s")

    return dict(
        seg_info=seg_info,
        res_s1=res_s1,
        res_s2=res_s2,
        sweep=sweep,
        v_handoff=v_handoff,
        T_s1=float(res_s1['time'][-1]),
        T_s2=float(res_s2['time'][-1]),
    )


def _build_segmented_result(seg, w_e):
    """Assemble a complete segmented trajectory dict using the corner at w_e.

    @param seg<dict>: Output of _run_segmented_pipeline().
    @param w_e<float>: Key into seg['sweep']['res_by_we'].
    @return dict with keys: seg_info, res_s1, res_s2, res_corner,
                             v_handoff, T_s1, T_corner, T_s2.
    """
    res_corner = seg['sweep']['res_by_we'][w_e]
    return dict(
        seg_info=seg['seg_info'],
        res_s1=seg['res_s1'],
        res_s2=seg['res_s2'],
        res_corner=res_corner,
        v_handoff=seg['v_handoff'],
        T_s1=seg['T_s1'],
        T_corner=float(res_corner['time'][-1]),
        T_s2=seg['T_s2'],
    )


# =============================================================================
# PATH SEGMENT HELPERS (verbatim from differential_drive_path_segment_combined)
# =============================================================================
def _segment_corner():
    ps = PathSegment(
        beta=BETA,
        L_input=L_INPUT,
        b_input=B_INPUT,
        v_max=V_MAX_SEG,
        a_max=A_MAX_SEG,
        L_wheelbase=L_WHEELBASE,
        n_samples=100,
    )
    res       = ps.generate_segment()
    L_seg     = res['L_seg']
    corner    = np.array(WP_CORNER, dtype=float)
    in_dir    = np.array([np.cos(HEADING_IN), np.sin(HEADING_IN)])
    arc_entry = corner - L_seg * in_dir

    c, s    = np.cos(HEADING_IN), np.sin(HEADING_IN)
    R_mat   = np.array([[c, -s], [s, c]])
    arc_local = res['path_segment']
    xy_world  = (R_mat @ arc_local[:, :2].T).T + arc_entry
    hdg_world = arc_local[:, 2] + HEADING_IN
    arc_world = np.column_stack([xy_world, hdg_world])
    arc_exit  = arc_world[-1, :2].copy()

    out_dir       = np.array([np.cos(HEADING_OUT), np.sin(HEADING_OUT)])
    arc_entry_ext = arc_entry - L_TRANSITION * in_dir
    arc_exit_ext  = arc_exit  + L_TRANSITION * out_dir

    res.update({
        'arc_world':           arc_world,
        'arc_entry_world':     arc_entry,
        'arc_exit_world':      arc_exit,
        'arc_entry_ext_world': arc_entry_ext,
        'arc_exit_ext_world':  arc_exit_ext,
        'corner_vertex':       corner,
    })
    return res


def _build_corner_waypoints(arc_entry, arc_exit):
    return [
        [arc_entry[0], arc_entry[1], HEADING_IN],
        [WP_CORNER[0], WP_CORNER[1], HEADING_IN],
        [arc_exit[0],  arc_exit[1],  HEADING_OUT],
    ]


def _run_jlap_seg(wp_start, wp_end, initial_vel=0.0, final_vel=0.0):
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


def _solve_corner(corner_wps, w_energy, warm_start=None,
                  v_entry=None, v_exit=None,
                  a_entry=0.0, a_exit=0.0,
                  alpha_entry=0.0, alpha_exit=0.0):
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


def _compute_wheel_kinematics(res, l, r, dt):
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
        jerk_r, jerk_l = res['jerk_r'], res['jerk_l']
    else:
        jerk_r = np.gradient(alpha_r, dt) * r
        jerk_l = np.gradient(alpha_l, dt) * r
    return dict(time=t, omega_r=omega_r, omega_l=omega_l,
                alpha_r=alpha_r, alpha_l=alpha_l,
                jerk_r=jerk_r, jerk_l=jerk_l)


def _find_smooth_v_handoff(corner_wps, a_entry=0.0, alpha_entry=0.0):
    l_ref = ROBOT_PARAMS_BSPLINE['l']
    r_ref = ROBOT_PARAMS_BSPLINE['r']
    dt_c  = 0.01
    jerk_lim = J_LIM

    candidates = [V_HANDOFF, V_HANDOFF * 0.8, V_HANDOFF * 0.6,
                  V_HANDOFF * 0.4, V_HANDOFF_MIN]
    prev_res = None
    for v_h in candidates:
        v_h = max(float(v_h), V_HANDOFF_MIN)
        try:
            res = _solve_corner(corner_wps, w_energy=0.01,
                                warm_start=prev_res,
                                v_entry=v_h, v_exit=v_h,
                                a_entry=a_entry, alpha_entry=alpha_entry)
            wk     = _compute_wheel_kinematics(res, l_ref, r_ref, dt_c)
            n_edge = min(3, len(wk['jerk_r']))
            j_max  = max(
                np.max(np.abs(wk['jerk_r'][:n_edge])),
                np.max(np.abs(wk['jerk_l'][:n_edge])),
                np.max(np.abs(wk['jerk_r'][-n_edge:])),
                np.max(np.abs(wk['jerk_l'][-n_edge:])),
            )
            print(f"  V_HANDOFF probe {v_h:.3f} m/s -> "
                  f"edge jerk = {j_max:.2f}  (limit {jerk_lim:.2f})")
            prev_res = res
            if j_max <= jerk_lim:
                if v_h < V_HANDOFF:
                    print(f"  NOTE: V_HANDOFF reduced to {v_h:.3f} m/s")
                return v_h
        except Exception as exc:
            print(f"  V_HANDOFF probe {v_h:.3f} m/s failed: {exc}")
            prev_res = None
    print(f"  WARNING: using minimum {V_HANDOFF_MIN:.3f} m/s")
    return V_HANDOFF_MIN


def _sweep_we(corner_wps, a_entry=0.0, alpha_entry=0.0, v_handoff=None):
    v_h = float(v_handoff) if v_handoff is not None else V_HANDOFF
    we_values = np.array([0.1, 0.05, 0.01, 0.005, 0.0])

    peak_powers, total_energies, mission_times, we_valid = [], [], [], []
    prev_res  = None
    res_by_we = {}

    print(f"  Sweeping {len(we_values)} w_e values (v_h = {v_h:.3f} m/s) ...")
    print(f"  {'w_e':>10}  {'Time [s]':>10}  {'Energy [J]':>10}  {'Peak P [W]':>10}")
    print("  " + "-" * 49)

    for w_e in we_values:
        try:
            res    = _solve_corner(corner_wps, w_e, warm_start=prev_res,
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
            f"Sweep produced {len(peak_powers)} feasible result(s); need at least 2.")

    sort_idx       = np.argsort(we_valid)
    peak_powers    = np.array(peak_powers)[sort_idx]
    total_energies = np.array(total_energies)[sort_idx]
    mission_times  = np.array(mission_times)[sort_idx]
    we_values      = np.array(we_valid)[sort_idx]

    if len(mission_times) >= 3:
        t_med = np.median(mission_times)
        valid = mission_times <= 5.0 * t_med
        if valid.sum() < len(mission_times):
            n_drop = (~valid).sum()
            print(f"  NOTE: dropping {n_drop} non-converged sweep point(s).")
            peak_powers    = peak_powers[valid]
            total_energies = total_energies[valid]
            mission_times  = mission_times[valid]
            we_values      = we_values[valid]
            for w_e in list(res_by_we):
                if w_e not in we_values:
                    del res_by_we[w_e]

    opt_idx = int(np.argmin(peak_powers))
    opt_we  = float(we_values[opt_idx])

    if 0.0 in res_by_we:
        time_ref_idx = int(np.where(we_values == 0.0)[0][0])
        we_time_ref  = 0.0
    else:
        time_ref_idx = int(np.argmin(mission_times))
        we_time_ref  = float(we_values[time_ref_idx])
        print(f"  NOTE: w_e=0.0 did not converge; "
              f"using w_e={we_time_ref:.4f} as time reference.")

    print(f"\n  opt w_e = {opt_we:.4f}   peak P = {peak_powers[opt_idx]:.2f} W"
          f"   E = {total_energies[opt_idx]:.2f} J   T = {mission_times[opt_idx]:.2f} s")

    return {
        'we_values':      we_values,
        'peak_powers':    peak_powers,
        'total_energies': total_energies,
        'mission_times':  mission_times,
        'opt_idx':        opt_idx,
        'opt_we':         opt_we,
        'we_time_ref':    we_time_ref,
        'time_ref_idx':   time_ref_idx,
        'res_by_we':      res_by_we,
        'v_handoff':      v_h,
    }


# =============================================================================
# POWER COMPUTATION (uniform TJ108 model for all three methods)
# =============================================================================
def _compute_power_uniform(v, omega, dt):
    """Evaluate TJ108 total electrical power from (v, omega) at sample interval dt."""
    l, r = ROBOT_PARAMS_BSPLINE['l'], ROBOT_PARAMS_BSPLINE['r']
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


def _power_for_jlap(res, dt):
    """Power dict for a full-path JLAP result (Method A)."""
    P = _compute_power_uniform(res['v'], res['omega'], dt)
    return {'time': res['time'], 'P': P,
            'energy': float(np.trapz(P, dx=dt))}


def _power_for_segmented(res_seg):
    """Power dict for a segmented result (Methods B or C)."""
    res_s1     = res_seg['res_s1']
    res_s2     = res_seg['res_s2']
    res_corner = res_seg['res_corner']
    T_s1       = res_seg['T_s1']
    T_corner   = res_seg['T_corner']

    P_s1 = _compute_power_uniform(res_s1['v'],    res_s1['omega'],    JLAP_DT)
    P_c  = _compute_power_uniform(res_corner['v'], res_corner['omega'], 0.01)
    P_s2 = _compute_power_uniform(res_s2['v'],    res_s2['omega'],    JLAP_DT)

    t_abs = np.concatenate([
        res_s1['time'],
        res_corner['time_ik'] + T_s1,
        res_s2['time'][1:]    + T_s1 + T_corner,
    ])
    P_abs = np.concatenate([P_s1, P_c, P_s2[1:]])

    return {'time': t_abs, 'P': P_abs,
            'energy': float(np.trapz(P_abs, t_abs))}


# =============================================================================
# METRICS
# =============================================================================
def _compute_metrics(states, total_time, t_P, P):
    """Compute scalar performance metrics from states and power arrays."""
    dx    = np.diff(states[:, 0])
    dy    = np.diff(states[:, 1])
    plen  = float(np.sum(np.sqrt(dx**2 + dy**2)))
    energy = float(np.trapz(P, t_P))
    return {
        'total_time':       total_time,
        'path_length':      plen,
        'energy':           energy,
        'peak_power':       float(np.max(P)),
        'avg_power':        float(np.mean(P)),
        'energy_per_meter': energy / plen if plen > 1e-9 else float('inf'),
    }


# =============================================================================
# COMPARISON TABLE
# =============================================================================
def _print_comparison(m_a, m_b, m_c, we_b, we_c):
    rows = [
        ('Total time',   's',   'total_time'),
        ('Path length',  'm',   'path_length'),
        ('Total energy', 'J',   'energy'),
        ('Peak power',   'W',   'peak_power'),
        ('Avg power',    'W',   'avg_power'),
        ('Energy/meter', 'J/m', 'energy_per_meter'),
    ]

    def _delta(val, base):
        if abs(base) > 1e-12:
            d = (val - base) / abs(base) * 100
            return f"{'+'if d>=0 else ''}{d:.1f}%"
        return 'n/a'

    col_b = f'B (w_e={we_b:.3f})'
    col_c = f'C (w_e={we_c:.3f})'
    hdr = (f"\n  {'Metric':<20} {'Unit':<5} "
           f"{'Method A':>12}  {col_b:>15}  {'Δ(B-A)':>8}  "
           f"{col_c:>15}  {'Δ(C-A)':>8}")
    print(hdr)
    print("  " + "─" * 98)
    for label, unit, key in rows:
        a, b, c = m_a[key], m_b[key], m_c[key]
        print(f"  {label:<20} {unit:<5} {a:>12.4f}  {b:>15.4f}  "
              f"{_delta(b, a):>8}  {c:>15.4f}  {_delta(c, a):>8}")
    print()
    print(f"  Method A: EulerJLAPCoverage (full path, Euler spiral corner)")
    print(f"  Method B: Segmented pipeline, corner BSpline w_e={we_b:.4f} (time-optimal)")
    print(f"  Method C: Segmented pipeline, corner BSpline w_e={we_c:.4f} (energy-optimal)")
    print()


# =============================================================================
# FIGURE 1 — XY trajectory overlay
# =============================================================================
def _fig1_xy_overlay(res_a, res_b, res_c, we_b, we_c):
    fig, ax = plt.subplots(figsize=(7, 7),
                           num='Figure 1 - XY Trajectory Overlay')
    ax.set_aspect('equal')

    ref = np.array([WP_START, WP_CORNER, WP_END])
    ax.plot(ref[:, 0], ref[:, 1], '--', color=COL_REF, lw=1.5,
            label='Reference L-path', zorder=2)
    ax.plot(ref[:, 0], ref[:, 1], 'o', color='black', ms=8, zorder=6)

    # Method A
    s_a = res_a['states']
    ax.plot(s_a[:, 0], s_a[:, 1], '-', color=COL_A, lw=2.2,
            label='A: EulerJLAP', zorder=4)
    step = max(1, len(s_a) // 12)
    for st in s_a[::step]:
        ax.annotate('', xy=(st[0] + 0.14*np.cos(st[2]),
                             st[1] + 0.14*np.sin(st[2])),
                    xytext=(st[0], st[1]),
                    arrowprops=dict(arrowstyle='->', color=COL_A, lw=1.1))

    # Methods B and C share seg1 and seg2 — draw them once in a neutral colour
    s_s1 = res_b['res_s1']['states']
    s_s2 = res_b['res_s2']['states']
    ax.plot(s_s1[:, 0], s_s1[:, 1], '-', color='dimgray', lw=1.8,
            label='Shared seg1 & seg2  (JLAP)', zorder=3)
    ax.plot(s_s2[:, 0], s_s2[:, 1], '-', color='dimgray', lw=1.8, zorder=3)

    # Corner B
    s_cb   = res_b['res_corner']['states']
    cpts_b = res_b['res_corner']['ctrl_pts']
    ax.plot(s_cb[:, 0], s_cb[:, 1], '-', color=COL_B, lw=2.2,
            label=f'B: corner w_e={we_b:.3f} (time-opt)', zorder=5)
    ax.plot(cpts_b[:, 0], cpts_b[:, 1], 'x', color=COL_B,
            ms=7, markeredgewidth=1.5, zorder=6)

    # Corner C
    s_cc   = res_c['res_corner']['states']
    cpts_c = res_c['res_corner']['ctrl_pts']
    ax.plot(s_cc[:, 0], s_cc[:, 1], '-', color=COL_C, lw=2.2,
            label=f'C: corner w_e={we_c:.3f} (energy-opt)', zorder=5)
    ax.plot(cpts_c[:, 0], cpts_c[:, 1], '+', color=COL_C,
            ms=9, markeredgewidth=1.5, zorder=6)

    # Arc entry/exit markers
    ae     = res_b['seg_info']['arc_entry_world']
    ax_e   = res_b['seg_info']['arc_exit_world']
    ae_ext = res_b['seg_info']['arc_entry_ext_world']
    ax_ext = res_b['seg_info']['arc_exit_ext_world']
    ax.plot(*ae,     's', color='black',  ms=8, zorder=7, label='Arc tangent points')
    ax.plot(*ax_e,   's', color='black',  ms=8, zorder=7)
    ax.plot(*ae_ext, 'D', color='dimgray', ms=7, zorder=7, label='JLAP handoff points')
    ax.plot(*ax_ext, 'D', color='dimgray', ms=7, zorder=7)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Figure 1 — XY Trajectory Overlay\n'
                 'A: EulerJLAP  |  B: corner time-opt  |  C: corner energy-opt')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True)
    fig.tight_layout()


# =============================================================================
# FIGURE 2 — Velocity profiles
# =============================================================================
def _fig2_velocity(res_a, res_b, res_c, we_b, we_c):
    fig, ax = plt.subplots(figsize=(10, 4),
                           num='Figure 2 - Velocity Profiles')

    ax.plot(res_a['time'], res_a['v'],
            '-', color=COL_A, lw=1.8, label='A: EulerJLAP')

    xform = ax.get_xaxis_transform()

    for res_seg, col, lbl in [
        (res_b, COL_B, f'B: Segmented corner w_e={we_b:.3f} (time-opt)'),
        (res_c, COL_C, f'C: Segmented corner w_e={we_c:.3f} (energy-opt)'),
    ]:
        T_s1     = res_seg['T_s1']
        T_corner = res_seg['T_corner']
        t_abs = np.concatenate([
            res_seg['res_s1']['time'],
            res_seg['res_corner']['time_ik'] + T_s1,
            res_seg['res_s2']['time'][1:]    + T_s1 + T_corner,
        ])
        v_abs = np.concatenate([
            res_seg['res_s1']['v'],
            res_seg['res_corner']['v'],
            res_seg['res_s2']['v'][1:],
        ])
        ax.plot(t_abs, v_abs, '-', color=col, lw=1.8, label=lbl)

    # Junction markers (S1|C is the same for B and C since they share seg1)
    T_s1 = res_b['T_s1']
    ax.axvline(T_s1, color=COL_REF, ls=':', lw=1.0)
    ax.text(T_s1, 0.94, 'S1|C', fontsize=7,
            color=COL_REF, ha='center', transform=xform)
    # C|S2 may differ slightly between B and C if corner times differ
    for res_seg, col, tag in [(res_b, COL_B, 'C|S2 B'),
                               (res_c, COL_C, 'C|S2 C')]:
        t_j = res_seg['T_s1'] + res_seg['T_corner']
        ax.axvline(t_j, color=col, ls=':', lw=0.9)
    ax.text(res_b['T_s1'] + res_b['T_corner'], 0.94, 'C|S2 B', fontsize=7,
            color=COL_B, ha='center', transform=xform)
    ax.text(res_c['T_s1'] + res_c['T_corner'], 0.86, 'C|S2 C', fontsize=7,
            color=COL_C, ha='center', transform=xform)

    ax.set_xlabel('time [s]')
    ax.set_ylabel('v [m/s]')
    ax.set_title('Figure 2 — Velocity Profiles v(t)  (all three methods)')
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()


# =============================================================================
# FIGURE 3 — Power profiles
# =============================================================================
def _fig3_power(pm_a, pm_b, pm_c, m_a, m_b, m_c, res_b, res_c, we_b, we_c):
    fig, ax = plt.subplots(figsize=(10, 4),
                           num='Figure 3 - Power Profiles')

    ax.plot(pm_a['time'], pm_a['P'], '-', color=COL_A, lw=1.8,
            label=f"A: EulerJLAP  peak={m_a['peak_power']:.1f} W")
    ax.plot(pm_b['time'], pm_b['P'], '-', color=COL_B, lw=1.8,
            label=f"B: corner w_e={we_b:.3f}  peak={m_b['peak_power']:.1f} W")
    ax.plot(pm_c['time'], pm_c['P'], '-', color=COL_C, lw=1.8,
            label=f"C: corner w_e={we_c:.3f}  peak={m_c['peak_power']:.1f} W")

    ax.axhline(P_ELECTRONICS, color=COL_REF, ls=':', lw=1.0,
               label=f'P_elec = {P_ELECTRONICS} W')

    for pm, m, col in [(pm_a, m_a, COL_A), (pm_b, m_b, COL_B),
                       (pm_c, m_c, COL_C)]:
        idx = int(np.argmax(pm['P']))
        ax.annotate(f"{m['peak_power']:.1f} W",
                    xy=(pm['time'][idx], pm['P'][idx]),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=7, color=col)

    # Junction markers for B and C
    for res_seg, col in [(res_b, COL_B), (res_c, COL_C)]:
        T_s1     = res_seg['T_s1']
        T_corner = res_seg['T_corner']
        ax.axvline(T_s1,          color=col, ls=':', lw=0.8)
        ax.axvline(T_s1+T_corner, color=col, ls=':', lw=0.8)

    ax.set_xlabel('time [s]')
    ax.set_ylabel('Power [W]')
    ax.set_title('Figure 3 — Motor Power P(t)  (TJ108 model, uniform across methods)')
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()


# =============================================================================
# FIGURE 4 — Summary metrics bar chart
# =============================================================================
def _fig4_bars(m_a, m_b, m_c, we_b, we_c):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7),
                             num='Figure 4 - Summary Metrics')
    fig.suptitle('Figure 4 — Performance Metrics Comparison\n'
                 f'A: EulerJLAP  |  B: corner w_e={we_b:.3f}  '
                 f'|  C: corner w_e={we_c:.3f}',
                 fontsize=10)

    specs = [
        (axes[0, 0], 'total_time',       'Total Time [s]'),
        (axes[0, 1], 'energy',           'Total Energy [J]'),
        (axes[1, 0], 'peak_power',       'Peak Power [W]'),
        (axes[1, 1], 'energy_per_meter', 'Energy / Meter [J/m]'),
    ]

    x_lbl  = ['A', f'B\nw_e={we_b:.3f}', f'C\nw_e={we_c:.3f}']
    x      = np.arange(3)
    width  = 0.55
    colors = [COL_A, COL_B, COL_C]

    for ax, key, title in specs:
        vals = [m_a[key], m_b[key], m_c[key]]
        bars = ax.bar(x, vals, width, color=colors)
        top  = max(vals)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + top * 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(x_lbl)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, top * 1.15)
        ax.grid(axis='y')

    fig.tight_layout()


# =============================================================================
# FIGURE 5 — Acceleration and Jerk profiles
# =============================================================================
def _fig5_acc_jerk(res_a, res_b, res_c, we_b, we_c):
    fig, (ax_a, ax_j) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                      num='Figure 5 - Acceleration and Jerk Profiles')
    fig.suptitle('Figure 5 — Linear Acceleration and Jerk Profiles  (all three methods)',
                 fontsize=10)

    xform_a = ax_a.get_xaxis_transform()
    xform_j = ax_j.get_xaxis_transform()

    # Method A
    j_a = np.gradient(res_a['acc_path'], JLAP_DT)
    ax_a.plot(res_a['time'], res_a['acc_path'], '-', color=COL_A, lw=1.8,
              label='A: EulerJLAP')
    ax_j.plot(res_a['time'], j_a,               '-', color=COL_A, lw=1.8,
              label='A: EulerJLAP')

    # Methods B and C
    for res_seg, col, lbl in [
        (res_b, COL_B, f'B: Segmented corner w_e={we_b:.3f} (time-opt)'),
        (res_c, COL_C, f'C: Segmented corner w_e={we_c:.3f} (energy-opt)'),
    ]:
        t_abs, a_abs, j_abs = _stitch_acc_jerk(res_seg)
        ax_a.plot(t_abs, a_abs, '-', color=col, lw=1.8, label=lbl)
        ax_j.plot(t_abs, j_abs, '-', color=col, lw=1.8, label=lbl)

    # Zero reference lines
    ax_a.axhline(0, color='k', lw=0.5, ls=':')
    ax_j.axhline(0, color='k', lw=0.5, ls=':')

    # Segment junction markers (shared S1 entry, separate S2 entries for B and C)
    T_s1 = res_b['T_s1']
    for ax, xform in [(ax_a, xform_a), (ax_j, xform_j)]:
        ax.axvline(T_s1, color=COL_REF, ls=':', lw=1.0)
        ax.text(T_s1, 0.97, 'S1|C', fontsize=7, color=COL_REF,
                ha='center', transform=xform)
        for res_seg, col, tag, y in [(res_b, COL_B, 'C|S2 B', 0.97),
                                     (res_c, COL_C, 'C|S2 C', 0.89)]:
            t_j = res_seg['T_s1'] + res_seg['T_corner']
            ax.axvline(t_j, color=col, ls=':', lw=0.9)
            ax.text(t_j, y, tag, fontsize=7, color=col, ha='center', transform=xform)

    ax_a.set_ylabel('Acceleration [m/s²]')
    ax_a.legend(fontsize=8)
    ax_a.grid(True)

    ax_j.set_xlabel('time [s]')
    ax_j.set_ylabel('Jerk [m/s³]')
    ax_j.legend(fontsize=8)
    ax_j.grid(True)

    fig.tight_layout()


# =============================================================================
if __name__ == '__main__':
    main()
