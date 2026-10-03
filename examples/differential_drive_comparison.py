#!/usr/bin/env python3
##
# @file differential_drive_comparison.py
#
# @brief Three-method trajectory comparison on a 5 m × 5 m L-shaped path.
#
# Path:  (0,0) -> (5,0) -> (5,5)   -- 90-degree left turn.
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
import pathlib
import pickle
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
L_WHEELBASE = 0.53

# EulerJLAP robot params (all three methods use these for straight segments)
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
ROBOT_PARAMS_BSPLINE = {'l': 0.53 / 2, 'r': 0.15}

# TJ108 energy model -- same for all three methods
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

# Corner w_e for Methods B (time-optimal), D (saturation-selected, the
# fine sweep's w_e* = 2/39 grid point) and C (energy-optimal endpoint).
WE_B, WE_D, WE_C = 0.0, 0.05128205, 0.4872

COL_A   = 'steelblue'
COL_B   = 'tomato'
COL_C   = 'seagreen'
COL_D   = 'darkorange'
COL_REF = 'gray'

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


def _make_bspline_common(v_h):
    """Corner OCP kwargs factory -- consistent with combined pipeline."""
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
    print(f"    epsilon_offset = {0.1:.3f} m  (same as Method A) OK")
    print()


# =============================================================================
# PAPER STYLE
# =============================================================================
def _set_paper_style():
    # Font sizes are tuned for figures generated at figsize~(10,4) or
    # ~(8,8) inches but embedded at width=9cm in the two-column paper
    # (scale factor ~0.35-0.44x), so source sizes must be ~2.5-3x the
    # target printed size to stay legible after shrinking.
    plt.rcParams.update({
        'font.size':       24,
        'axes.labelsize':  24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'axes.titlesize':  24,
        'axes.grid':       False,
    })


# Printed widths of the paper's figure slots (\includegraphics width=).
FIG_W_9CM = 9.0 / 2.54
FIG_W_7CM = 7.0 / 2.54


def _set_print_style():
    """Style for figures drawn at their printed width (FIG_W_9CM /
    FIG_W_7CM), so font sizes here are the final printed sizes."""
    plt.rcParams.update({
        'font.size':         8,
        'axes.labelsize':    8,
        'xtick.labelsize':   7,
        'ytick.labelsize':   7,
        'legend.fontsize':   7,
        'axes.titlesize':    8,
        'axes.linewidth':    0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'lines.linewidth':   1.0,
        'lines.markersize':  4,
        'legend.frameon':    False,
        'axes.grid':         False,
        # purepursuit_fine_sweep sets 'cm' at import; keep math in the
        # same sans font as the text across all paper figures.
        'mathtext.fontset':  'dejavusans',
    })


# =============================================================================
# MAIN
# =============================================================================
def main():
    _set_print_style()
    # ------------------------------------------------------------------
    # Method A: EulerJLAP full path
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Method A: EulerJLAPCoverage -- full 3-waypoint path")
    print("=" * 60)
    res_a = _run_method_a()
    print(f"  T = {res_a['time'][-1]:.3f} s   "
          f"v_peak = {np.max(res_a['v']):.3f} m/s")

    # ------------------------------------------------------------------
    # Methods B, D & C: shared segmented pipeline
    # The sweep is run once at three fixed target weights matching the
    # fine-sweep's characteristic solutions (differential_drive_path_
    # segment_fine_sweep.py + analyze_fine_sweep.py):
    #   B -> w_e=0       (time-optimal)
    #   D -> w_e=0.0513  (peak-power saturation -- the automated-selection criterion)
    #   C -> w_e=0.4872  (energy-optimal sweep endpoint)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Methods B, D & C: Segmented pipeline  (shared PathSegment + JLAP + sweep)")
    print(f"  B -> corner at w_e={WE_B:.4f}   (time-optimal)")
    print(f"  D -> corner at w_e={WE_D:.4f}   (saturation-selected)")
    print(f"  C -> corner at w_e={WE_C:.4f}   (energy-optimal)")
    print("=" * 60)
    seg  = _run_segmented_pipeline()
    we_b, we_d, we_c = WE_B, WE_D, WE_C
    res_b = _build_segmented_result(seg, we_b)
    res_d = _build_segmented_result(seg, we_d)
    res_c = _build_segmented_result(seg, we_c)
    T_b   = res_b['T_s1'] + res_b['T_corner'] + res_b['T_s2']
    T_d   = res_d['T_s1'] + res_d['T_corner'] + res_d['T_s2']
    T_c   = res_c['T_s1'] + res_c['T_corner'] + res_c['T_s2']
    print(f"\n  Method B  w_e = {we_b:.4f}   T = {T_b:.3f} s")
    print(f"  Method D  w_e = {we_d:.4f}   T = {T_d:.3f} s")
    print(f"  Method C  w_e = {we_c:.4f}   T = {T_c:.3f} s")

    # ------------------------------------------------------------------
    # Power (uniform TJ108 model)
    # ------------------------------------------------------------------
    pm_a = _power_for_jlap(res_a, JLAP_DT)
    pm_b = _power_for_segmented(res_b)
    pm_d = _power_for_segmented(res_d)
    pm_c = _power_for_segmented(res_c)

    # ------------------------------------------------------------------
    # Metrics and comparison table
    # ------------------------------------------------------------------
    m_a = _compute_metrics(res_a['states'],
                           float(res_a['time'][-1]), pm_a['time'], pm_a['P'])
    m_b = _compute_metrics(_stitch_states(res_b), T_b, pm_b['time'], pm_b['P'])
    m_d = _compute_metrics(_stitch_states(res_d), T_d, pm_d['time'], pm_d['P'])
    m_c = _compute_metrics(_stitch_states(res_c), T_c, pm_c['time'], pm_c['P'])
    _print_comparison(m_a, m_b, m_d, m_c, we_b, we_d, we_c)

    # ------------------------------------------------------------------
    # Figures (generated before any further console output so PNGs are
    # written even if a subsequent print raises a codec error)
    # ------------------------------------------------------------------
    _fig1_xy_overlay(res_a, res_b, res_d, res_c, we_b, we_d, we_c)
    _fig2_velocity(res_a, res_b, res_d, res_c, we_b, we_d, we_c)
    _fig3_power(pm_a, pm_b, pm_d, pm_c, m_a, m_b, m_d, m_c,
                res_b, res_d, res_c, we_b, we_d, we_c)
    _fig4_bars(m_a, m_b, m_c, we_b, we_c)
    _fig5_acc_jerk(res_a, res_b, res_c, we_b, we_c)
    _fig6_junction_zoom(res_b, res_c, we_b, we_c)

    _print_coverage_tolerances()

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
# METHOD A -- EulerJLAPCoverage, full path
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
# METHODS B, D & C -- shared segmented pipeline
# =============================================================================
_CACHE_PATH = (pathlib.Path(__file__).resolve().parent
               / 'csv_output' / 'segmented_pipeline_cache.pkl')


def _run_segmented_pipeline(use_cache=True):
    """Run PathSegment + JLAP straights + w_e sweep.

    All corners in the sweep pin v_exit = v_handoff, so the same segment-2
    JLAP result is valid for B, D and C.

    w_e=0.4872 (Method C) sits in a numerically difficult region of the OCP
    and can take ~1-2 hours to converge even with careful warm-starting
    (see _sweep_we). To avoid paying this cost in every script that needs
    the same B/D/C corner solutions (section5_purepursuit.py,
    fig_corridor_regen.py), the result is cached to disk; pass
    use_cache=False to force a fresh solve.

    @return dict consumed by _build_segmented_result().
    """
    if use_cache and _CACHE_PATH.exists():
        print(f"  Loading cached segmented pipeline from {_CACHE_PATH}")
        with open(_CACHE_PATH, 'rb') as f:
            seg = pickle.load(f)
        if _ensure_we_in_cache(seg, WE_D):
            with open(_CACHE_PATH, 'wb') as f:
                pickle.dump(seg, f)
            print(f"  Updated cached segmented pipeline -> {_CACHE_PATH}")
        return seg

    seg_info      = _segment_corner()
    arc_entry_ext = seg_info['arc_entry_ext_world']
    arc_exit_ext  = seg_info['arc_exit_ext_world']
    corner_wps    = _build_corner_waypoints(arc_entry_ext, arc_exit_ext)

    v_handoff = _find_smooth_v_handoff(corner_wps)
    print(f"  Active V_HANDOFF = {v_handoff:.3f} m/s")

    # JLAP segments end/start at the buffered handoff points (L_TRANSITION m
    # before/after the arc tangent points) so the handoff falls inside the
    # cruise phase -- acceleration and jerk are ~0 at both junctions.
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

    result = dict(
        seg_info=seg_info,
        res_s1=res_s1,
        res_s2=res_s2,
        sweep=sweep,
        v_handoff=v_handoff,
        T_s1=float(res_s1['time'][-1]),
        T_s2=float(res_s2['time'][-1]),
    )

    if use_cache:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_PATH, 'wb') as f:
            pickle.dump(result, f)
        print(f"  Cached segmented pipeline -> {_CACHE_PATH}")

    return result


def _ensure_we_in_cache(seg, w_e):
    """Solve the corner at w_e and add it to seg['sweep']['res_by_we'] if a
    cache written before w_e was part of the sweep lacks it.

    Warm-starts from the nearest cached weight; B and C stay untouched.

    @return True if seg was modified (caller should re-pickle it).
    """
    res_by_we = seg['sweep']['res_by_we']
    if w_e in res_by_we:
        return False

    seg_info   = seg['seg_info']
    corner_wps = _build_corner_waypoints(seg_info['arc_entry_ext_world'],
                                         seg_info['arc_exit_ext_world'])
    we_warm = min(res_by_we, key=lambda k: abs(k - w_e))
    print(f"  Solving corner at w_e={w_e:.4f} (warm start from w_e={we_warm:.4f}) ...")
    res = _solve_corner(corner_wps, w_e, warm_start=res_by_we[we_warm],
                        v_entry=seg['v_handoff'], v_exit=seg['v_handoff'],
                        a_entry=float(seg['res_s1']['acc_path'][-1]),
                        alpha_entry=float(seg['res_s1']['alpha'][-1]),
                        max_iter=15000)
    status = res.get('return_status', 'unknown')
    if status != 'Solve_Succeeded':
        raise RuntimeError(f"w_e={w_e:.4f}: IPOPT did not converge (status={status})")

    print(f"  w_e={w_e:.4f}: T_corner = {res['time'][-1]:.3f} s   "
          f"corner E (OCP) = {float(res['energy']):.3f} J")
    res_by_we[w_e] = res
    return True


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
                  alpha_entry=0.0, alpha_exit=0.0, max_iter=10000):
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
        max_iter=max_iter,
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
    # Target points matching the fine-sweep's characteristic solutions
    # (differential_drive_path_segment_fine_sweep.py + analyze_fine_sweep.py):
    # time-optimal, knee (Method D), and the energy-optimal sweep endpoint
    # (Method C). Intermediate stepping stones between the knee and the
    # energy-optimal endpoint are included purely to keep each warm-start
    # jump small -- solving w_e=0.4872 directly from a warm start at 0.0769
    # (or cold) hits IPOPT's iteration cap without converging ("Maximum
    # Number of Iterations Exceeded"); the original 40-point fine sweep
    # reached this same weight fine via small sequential steps, which this
    # mirrors. Intermediate points are not kept as Method results.
    we_values  = np.array([WE_B, WE_D, 0.0769, 0.15, 0.25, 0.35, WE_C])
    we_targets = {WE_B, WE_D, 0.0769, WE_C}

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
                                   a_entry=a_entry, alpha_entry=alpha_entry,
                                   max_iter=15000)
            status = res.get('return_status', 'unknown')
            if status != 'Solve_Succeeded':
                raise RuntimeError(f"IPOPT did not converge (status={status})")
            P_tot  = _compute_corner_power(res)
            T_tot  = float(res['time'][-1])
            peak_p = float(np.max(P_tot))
            energy = float(res.get('energy', np.trapezoid(P_tot, dx=0.01)))
            if not (np.isfinite(peak_p) and np.isfinite(energy)):
                raise ValueError("non-finite result")
            prev_res = res
            tag = '' if w_e in we_targets else '  (stepping stone, not kept)'
            if w_e in we_targets:
                res_by_we[w_e] = res
                peak_powers.append(peak_p)
                total_energies.append(energy)
                mission_times.append(T_tot)
                we_valid.append(w_e)
            print(f"  {w_e:>10.6f}  {T_tot:>10.3f}  {energy:>10.3f}  {peak_p:>10.3f}{tag}")
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
# POWER COMPUTATION (uniform polynomial model for all three methods)
# =============================================================================
def _compute_power_uniform(v, omega, dt):
    """Evaluate total electrical power from (v, omega) at sample interval dt,
    re-deriving acceleration via finite differences.

    CAUTION: only accurate when (v, omega) are natively sampled on a truly
    uniform dt grid (e.g. EulerJLAPCoverage's own output). Do NOT use this on
    B-spline OCP corner results resampled via Pchip onto `time_ik` -- that
    grid's last interval is generally shorter than `dt` (the interpolation
    domain is forced to end exactly at t_real[-1]), and re-differentiating an
    already-interpolated signal with np.gradient assuming uniform spacing
    produces large spurious values at the array boundary (verified: a
    141x-higher apparent peak than the true analytic acceleration for one
    sweep point). Use _compute_power_from_accel with the OCP's own
    Pchip-interpolated acc_path/alpha fields for corner segments instead.
    """
    l = ROBOT_PARAMS_BSPLINE['l']
    v_r = v + l * omega
    v_l = v - l * omega
    a_r = np.gradient(v_r, dt)
    a_l = np.gradient(v_l, dt)

    def _p(vw, aw, c):
        return np.maximum(
            c[0]*aw**2 + c[1]*vw**2
            + np.abs(c[2]*aw) + np.abs(c[3]*vw)
            + np.abs(c[4]*vw*aw) + c[5], 0.0)

    return (_p(v_r, a_r, ENERGY_COEFFS_RIGHT)
            + _p(v_l, a_l, ENERGY_COEFFS_LEFT)
            + P_ELECTRONICS)


def _compute_power_from_accel(v, omega, acc_path, alpha):
    """Evaluate total electrical power from (v, omega, acc_path, alpha),
    using an already-computed acceleration/angular-acceleration field
    instead of re-differentiating velocity. Correct for both JLAP results
    (acc_path/alpha are the JLAP profile's own analytic values) and B-spline
    OCP corner results (acc_path/alpha are Pchip-interpolated directly from
    the OCP's analytically-exact per-node acceleration, not re-derived from
    velocity -- see _compute_power_uniform's docstring for why that
    distinction matters)."""
    l = ROBOT_PARAMS_BSPLINE['l']
    v_r = v + l * omega
    v_l = v - l * omega
    a_r = acc_path + l * alpha
    a_l = acc_path - l * alpha

    def _p(vw, aw, c):
        return np.maximum(
            c[0]*aw**2 + c[1]*vw**2
            + np.abs(c[2]*aw) + np.abs(c[3]*vw)
            + np.abs(c[4]*vw*aw) + c[5], 0.0)

    return (_p(v_r, a_r, ENERGY_COEFFS_RIGHT)
            + _p(v_l, a_l, ENERGY_COEFFS_LEFT)
            + P_ELECTRONICS)


def _power_for_jlap(res, dt):
    """Power dict for a full-path JLAP result (Method A)."""
    P = _compute_power_from_accel(res['v'], res['omega'],
                                  res['acc_path'], res['alpha'])
    return {'time': res['time'], 'P': P,
            'energy': float(np.trapezoid(P, dx=dt))}


def _power_for_segmented(res_seg):
    """Power dict for a segmented result (Methods B, D or C)."""
    res_s1     = res_seg['res_s1']
    res_s2     = res_seg['res_s2']
    res_corner = res_seg['res_corner']
    T_s1       = res_seg['T_s1']
    T_corner   = res_seg['T_corner']

    P_s1 = _compute_power_from_accel(res_s1['v'], res_s1['omega'],
                                     res_s1['acc_path'], res_s1['alpha'])
    P_c  = _compute_power_from_accel(res_corner['v'], res_corner['omega'],
                                     res_corner['acc_path'], res_corner['alpha'])
    P_s2 = _compute_power_from_accel(res_s2['v'], res_s2['omega'],
                                     res_s2['acc_path'], res_s2['alpha'])

    t_abs = np.concatenate([
        res_s1['time'],
        res_corner['time_ik'] + T_s1,
        res_s2['time'][1:]    + T_s1 + T_corner,
    ])
    P_abs = np.concatenate([P_s1, P_c, P_s2[1:]])

    return {'time': t_abs, 'P': P_abs,
            'energy': float(np.trapezoid(P_abs, t_abs))}


# =============================================================================
# METRICS
# =============================================================================
def _compute_metrics(states, total_time, t_P, P):
    """Compute scalar performance metrics from states and power arrays."""
    dx    = np.diff(states[:, 0])
    dy    = np.diff(states[:, 1])
    plen  = float(np.sum(np.sqrt(dx**2 + dy**2)))
    energy = float(np.trapezoid(P, t_P))
    # Time-weighted average power (= energy / duration). NOT np.mean(P): the
    # segmented pipeline's P array is sampled non-uniformly in time (corner
    # at dt=0.01, JLAP straights at dt=0.05), so a plain element-wise mean
    # over-weights the densely-sampled corner region relative to its actual
    # share of the mission duration.
    return {
        'total_time':       total_time,
        'path_length':      plen,
        'energy':           energy,
        'peak_power':       float(np.max(P)),
        'avg_power':        energy / total_time if total_time > 1e-9 else float('nan'),
        'energy_per_meter': energy / plen if plen > 1e-9 else float('inf'),
    }


# =============================================================================
# COMPARISON TABLE
# =============================================================================
def _print_comparison(m_a, m_b, m_d, m_c, we_b, we_d, we_c):
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
    col_d = f'D (w_e={we_d:.3f})'
    col_c = f'C (w_e={we_c:.3f})'
    hdr = (f"\n  {'Metric':<20} {'Unit':<5} "
           f"{'Method A':>12}  {col_b:>15}  {col_d:>15}  {col_c:>15}")
    print(hdr)
    print("  " + "-" * 98)
    for label, unit, key in rows:
        a, b, d, c = m_a[key], m_b[key], m_d[key], m_c[key]
        print(f"  {label:<20} {unit:<5} {a:>12.4f}  {b:>15.4f}  "
              f"{d:>15.4f}  {c:>15.4f}")
    print()
    print(f"  Method A: EulerJLAPCoverage (full path, Euler spiral corner)")
    print(f"  Method B: Segmented pipeline, corner BSpline w_e={we_b:.4f} (time-optimal)")
    print(f"  Method D: Segmented pipeline, corner BSpline w_e={we_d:.4f} (saturation-selected)")
    print(f"  Method C: Segmented pipeline, corner BSpline w_e={we_c:.4f} (energy-optimal)")
    print()


# =============================================================================
# FIGURE 1 -- XY trajectory overlay
# =============================================================================
def _method_labels(we_b, we_d, we_c):
    """Legend labels shared by the comparison figures (paper naming)."""
    return {
        'A': 'A: Euler spiral (prior)',
        'B': f'B: $w_e$={we_b:g} (time-opt)',
        'D': f'D: $w_e$={we_d:.4f}',
        'C': f'C: $w_e$={we_c:.4f} (energy-opt)',
    }


def _fig1_xy_overlay(res_a, res_b, res_d, res_c, we_b, we_d, we_c):
    lbl = _method_labels(we_b, we_d, we_c)
    fig, ax = plt.subplots(figsize=(FIG_W_9CM, 4.3), constrained_layout=True,
                           num='Figure 1 - XY Trajectory Overlay')
    ax.set_aspect('equal')

    ref = np.array([WP_START, WP_CORNER, WP_END])
    ax.plot(ref[:, 0], ref[:, 1], '--', color=COL_REF, lw=0.8,
            label='Reference L-path', zorder=2)
    ax.plot(ref[:, 0], ref[:, 1], 'o', color='black', ms=3.5, zorder=6)

    # Method A
    s_a = res_a['states']
    ax.plot(s_a[:, 0], s_a[:, 1], '-', color=COL_A, lw=1.2,
            label=lbl['A'], zorder=4)
    step = max(1, len(s_a) // 12)
    for st in s_a[::step]:
        ax.annotate('', xy=(st[0] + 0.14*np.cos(st[2]),
                             st[1] + 0.14*np.sin(st[2])),
                    xytext=(st[0], st[1]),
                    arrowprops=dict(arrowstyle='->', color=COL_A, lw=0.7))

    # Methods B, D and C share seg1 and seg2 -- draw them once in a neutral colour
    s_s1 = res_b['res_s1']['states']
    s_s2 = res_b['res_s2']['states']
    ax.plot(s_s1[:, 0], s_s1[:, 1], '-', color='dimgray', lw=1.0,
            label='Shared seg. 1/2 (JLAP)', zorder=3)
    ax.plot(s_s2[:, 0], s_s2[:, 1], '-', color='dimgray', lw=1.0, zorder=3)

    # Corners B, D, C
    for res_seg, col, key, marker in [
        (res_b, COL_B, 'B', 'x'),
        (res_d, COL_D, 'D', '^'),
        (res_c, COL_C, 'C', '+'),
    ]:
        s_c   = res_seg['res_corner']['states']
        cpts  = res_seg['res_corner']['ctrl_pts']
        ax.plot(s_c[:, 0], s_c[:, 1], '-', color=col, lw=1.2, label=lbl[key], zorder=5)
        ax.plot(cpts[:, 0], cpts[:, 1], marker, color=col,
                ms=4, markeredgewidth=0.9, zorder=6)

    # Arc entry/exit markers
    ae     = res_b['seg_info']['arc_entry_world']
    ax_e   = res_b['seg_info']['arc_exit_world']
    ae_ext = res_b['seg_info']['arc_entry_ext_world']
    ax_ext = res_b['seg_info']['arc_exit_ext_world']
    ax.plot(*ae,     's', color='black',  ms=3.5, zorder=7, label='Arc tangent points')
    ax.plot(*ax_e,   's', color='black',  ms=3.5, zorder=7)
    ax.plot(*ae_ext, 'D', color='dimgray', ms=3.2, zorder=7, label='JLAP handoff points')
    ax.plot(*ax_ext, 'D', color='dimgray', ms=3.2, zorder=7)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center',
               ncol=2, fontsize=7, handlelength=1.8, columnspacing=1.0)
    _savefig(fig, 'fig_comparison_xy.png')


# =============================================================================
# FIGURE 2 -- Velocity profiles
# =============================================================================
def _fig2_velocity(res_a, res_b, res_d, res_c, we_b, we_d, we_c):
    lbl = _method_labels(we_b, we_d, we_c)
    fig, ax = plt.subplots(figsize=(FIG_W_9CM, 2.6), constrained_layout=True,
                           num='Figure 2 - Velocity Profiles')

    ax.plot(res_a['time'], res_a['v'], '-', color=COL_A, label=lbl['A'])

    specs = [(res_b, COL_B, 'B'), (res_d, COL_D, 'D'), (res_c, COL_C, 'C')]

    for res_seg, col, key in specs:
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
        ax.plot(t_abs, v_abs, '-', color=col, label=lbl[key])

    # Junction markers (S1|C is the same for B, D and C since they share
    # seg1; C|S2 differs with each method's corner time). The caption
    # explains them, so they carry no text labels.
    ax.axvline(res_b['T_s1'], color=COL_REF, ls=':', lw=0.8)
    for res_seg, col, _ in specs:
        ax.axvline(res_seg['T_s1'] + res_seg['T_corner'], color=col, ls=':', lw=0.8)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('v [m/s]')
    fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center',
               ncol=2, fontsize=7, handlelength=1.8, columnspacing=1.0)
    _savefig(fig, 'fig_comparison_v.png')


# =============================================================================
# FIGURE 3 -- Power profiles
# =============================================================================
def _fig3_power(pm_a, pm_b, pm_d, pm_c, m_a, m_b, m_d, m_c,
                res_b, res_d, res_c, we_b, we_d, we_c):
    lbl = _method_labels(we_b, we_d, we_c)
    fig, ax = plt.subplots(figsize=(FIG_W_9CM, 2.8), constrained_layout=True,
                           num='Figure 3 - Power Profiles')

    for pm, m, col, key in [(pm_a, m_a, COL_A, 'A'), (pm_b, m_b, COL_B, 'B'),
                            (pm_d, m_d, COL_D, 'D'), (pm_c, m_c, COL_C, 'C')]:
        ax.plot(pm['time'], pm['P'], '-', color=col,
                label=f"{lbl[key]}: {m['peak_power']:.1f} W peak")

    ax.axhline(P_ELECTRONICS, color=COL_REF, ls=':', lw=0.8,
               label=rf'$P_{{\mathrm{{elec}}}}$ = {P_ELECTRONICS:g} W')

    # Junction markers for B, D and C
    for res_seg, col in [(res_b, COL_B), (res_d, COL_D), (res_c, COL_C)]:
        T_s1     = res_seg['T_s1']
        T_corner = res_seg['T_corner']
        ax.axvline(T_s1,          color=col, ls=':', lw=0.6)
        ax.axvline(T_s1+T_corner, color=col, ls=':', lw=0.6)

    ax.set_ylim(0, 24)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Power [W]')
    fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center',
               ncol=1, fontsize=7, handlelength=1.8)
    _savefig(fig, 'fig_comparison_power.png')


# =============================================================================
# FIGURE 4 -- Summary metrics bar chart
# =============================================================================
def _fig4_bars(m_a, m_b, m_c, we_b, we_c):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7),
                             num='Figure 4 - Summary Metrics')

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
        ax.set_ylim(0, top * 1.15)

    fig.tight_layout()


# =============================================================================
# FIGURE 5 -- Acceleration and Jerk profiles
# =============================================================================
def _fig5_acc_jerk(res_a, res_b, res_c, we_b, we_c):
    fig, (ax_a, ax_j) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                      num='Figure 5 - Acceleration and Jerk Profiles')

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
        ax.text(T_s1, 0.97, 'S1|C', fontsize=8, color=COL_REF,
                ha='center', transform=xform)
        for res_seg, col, tag, y in [(res_b, COL_B, 'C|S2 B', 0.97),
                                     (res_c, COL_C, 'C|S2 C', 0.89)]:
            t_j = res_seg['T_s1'] + res_seg['T_corner']
            ax.axvline(t_j, color=col, ls=':', lw=0.9)
            ax.text(t_j, y, tag, fontsize=8, color=col, ha='center', transform=xform)

    ax_a.set_ylabel('Acceleration [m/s²]')
    ax_a.legend(fontsize=9)


    ax_j.set_xlabel('time [s]')
    ax_j.set_ylabel('Jerk [m/s³]')
    ax_j.legend(fontsize=9)


    fig.tight_layout()


# =============================================================================
# FIGURE 6 - Velocity continuity at junctions (paper figure)
# =============================================================================
def _fig6_junction_zoom(res_b, res_c, we_b, we_c):
    """! Zoomed velocity profiles around both segment-junction points.

    Left panel: S1 -> Corner junction (±ZOOM_DUR s relative to junction time).
    Right panel: Corner -> S2 junction.

    Overlays Methods B and C so the reader can verify C1 velocity continuity
    at both handoff points.  Saved as fig_junction_zoom.png when SAVE_FIGS=True.

    @param res_b<dict>: Segmented result dict for Method B.
    @param res_c<dict>: Segmented result dict for Method C.
    @param we_b<float>: w_e value for Method B.
    @param we_c<float>: w_e value for Method C.
    """
    ZOOM_DUR = 0.5   # seconds shown on each side of the junction

    fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                             num='Figure 6 - Velocity Continuity at Junctions')

    for res_seg, col, lbl in [
        (res_b, COL_B, f'B: $w_e$={we_b:.3f}  (time-opt)'),
        (res_c, COL_C, f'C: $w_e$={we_c:.3f}  (energy-opt)'),
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

        # Left panel: S1 -> Corner junction at t = T_s1
        mask_l = (t_abs >= T_s1 - ZOOM_DUR) & (t_abs <= T_s1 + ZOOM_DUR)
        axes[0].plot(t_abs[mask_l] - T_s1, v_abs[mask_l],
                     '-', color=col, linewidth=1.8, label=lbl)

        # Right panel: Corner -> S2 junction at t = T_s1 + T_corner
        t_j2   = T_s1 + T_corner
        mask_r = (t_abs >= t_j2 - ZOOM_DUR) & (t_abs <= t_j2 + ZOOM_DUR)
        axes[1].plot(t_abs[mask_r] - t_j2, v_abs[mask_r],
                     '-', color=col, linewidth=1.8, label=lbl)

    for ax, title in [
        (axes[0], 'Seg 1 -> Corner  (junction at t = 0)'),
        (axes[1], 'Corner -> Seg 2  (junction at t = 0)'),
    ]:
        ax.axvline(0, color='black', linewidth=0.9, linestyle='--',
                   label='junction')
        ax.set_xlabel('Time relative to junction [s]')
        ax.set_ylabel('v [m/s]')
        ax.legend(fontsize=10)
    
    fig.tight_layout()
    _savefig(fig, 'fig_junction_zoom.png')


# =============================================================================
if __name__ == '__main__':
    main()
