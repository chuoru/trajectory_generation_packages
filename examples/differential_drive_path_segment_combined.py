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
from models.differential_drive import DifferentialDrive
from simulators.time_stepping import TimeStepping
from controllers.purepursuit import PurePursuit
from controllers.trajectory import Trajectory


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

# Jerk limit derived from motor parameters -- mirrors EulerJLAPCoverage.__init__:102-104
_j_rated_torque = JLAP_ROBOT_PARAMS['gear_ratio'] * JLAP_ROBOT_PARAMS['rated_motor_torque']
_j_inertia      = JLAP_ROBOT_PARAMS['gear_ratio']**2 * JLAP_ROBOT_PARAMS['motor_inertia']
_j_r            = JLAP_ROBOT_PARAMS['wheel_radius']
_j_m            = JLAP_ROBOT_PARAMS['robot_mass']
_A_LIM          = (0.5 * _j_rated_torque * _j_r
                   / (0.25 * _j_m * _j_r**2 + _j_inertia))   # [m/s²]
J_LIM = _A_LIM / (40.0 * JLAP_DT)                            # [m/s³]  ~= 3.55

# Handoff velocity: S1 exits at this speed; corner enters at this speed.
# Taken from path_vel_lim so it is always within the JLAP kinematic limits.
V_HANDOFF     = JLAP_ROBOT_PARAMS['path_vel_lim']   # 0.5 m/s
V_HANDOFF_MIN = 0.10   # minimum fallback handoff velocity [m/s]

# Angular acc/jerk limits derived from JLAP linear limits and wheelbase.
_ANG_ACC_MAX  = _A_LIM / L_WHEELBASE   # [rad/s²]
_ANG_JERK_MAX = J_LIM  / L_WHEELBASE   # [rad/s³]


def _make_bspline_common(v_h):
    """Return BSplineEnergyCoverage kwargs parameterised by handoff speed v_h.

    acc_max / jerk_max are set to the same physical limits the JLAP segments
    use so that the corner OCP cannot produce profiles with unrealistically
    large acceleration or jerk.  vel_max[0] tracks v_h so the feedrate limit
    stays consistent with the chosen handoff velocity.
    """
    return dict(
        bound=0.25,
        n_ctrl_pts=6,
        spline_order=3,
        n_sampling=15,
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

    # BSpline corner covers arc_entry_ext -> arc_exit_ext (includes L_TRANSITION
    # straight lead-in/out) so the OCP starts/ends on a straight section.
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

    # Extract actual exit acceleration/alpha for tight boundary matching.
    a_s1_exit     = float(res_s1['acc_path'][-1])
    alpha_s1_exit = float(res_s1['alpha'][-1])

    # ------------------------------------------------------------------
    # Step 3: Corner B-spline -- sweep w_energy -> optimal
    #         Entry acceleration matched to actual S1 exit for C1 continuity.
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3: BSplineEnergyCoverage - w_energy sweep for corner")
    print("=" * 60)
    sweep = _sweep_we(corner_wps,
                      a_entry=a_s1_exit, alpha_entry=alpha_s1_exit,
                      v_handoff=v_handoff)
    opt_we      = sweep['opt_we']
    we_time_ref = sweep['we_time_ref']
    res_by_we   = sweep['res_by_we']

    # Reuse sweep results -- no re-solve needed (avoids cold-start local minima).
    print()
    print(f"  Using sweep result for time reference (w_energy = {we_time_ref:.4f})")
    res_corner_time = res_by_we[we_time_ref]

    print(f"  Using sweep result for energy-optimal (w_energy = {opt_we:.4f})")
    if opt_we in res_by_we:
        res_corner_opt = res_by_we[opt_we]
    else:
        print("  (sweep result missing -- re-solving warm)")
        res_corner_opt = _solve_corner(corner_wps, w_energy=opt_we,
                                       warm_start=res_corner_time,
                                       v_entry=v_handoff, v_exit=v_handoff,
                                       a_entry=a_s1_exit,
                                       alpha_entry=alpha_s1_exit)

    mA = _corner_metrics(res_corner_time)
    mB = _corner_metrics(res_corner_opt)
    _print_corner_comparison(mA, mB, opt_we, we_time_ref)

    # ------------------------------------------------------------------
    # Step 2b: EulerJLAP -- segment 2 (enters at corner exit speed)
    # ------------------------------------------------------------------
    v_corner_exit = float(max(0.0, res_corner_opt['v'][-1]))
    print()
    print("=" * 60)
    print(f"Step 2b: EulerJLAPCoverage - segment 2 "
          f"(V_entry={v_handoff:.3f} -> 0)  "
          f"(corner actual exit = {v_corner_exit:.3f} m/s)")
    print("=" * 60)
    res_s2 = _run_jlap_seg(arc_exit_ext.tolist(), WP_END,
                            initial_vel=v_corner_exit, final_vel=0.0)
    print(f"  Segment 2: T = {res_s2['time'][-1]:.3f} s   "
          f"v_peak = {np.max(res_s2['v']):.3f} m/s   "
          f"v_entry = {v_corner_exit:.3f} m/s")

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------
    _export_csv(res_s1, res_s2, res_corner_opt)

    # ------------------------------------------------------------------
    # Pure Pursuit closed-loop tracking
    # To load from the exported CSV instead, use:
    #   csv_path = os.path.join(os.path.dirname(__file__), 'csv_output',
    #                           'trajectory_stitched.csv')
    #   ref_traj = Trajectory.from_stitched_csv(csv_path, dt=0.05)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 4: PurePursuit closed-loop tracking")
    print("=" * 60)
    ref_traj  = _build_reference_trajectory(res_s1, res_s2, res_corner_opt, dt=0.05)
    sim_pp    = _run_purepursuit(ref_traj)
    print(f"  Reference waypoints : {len(ref_traj.x)}")
    print(f"  Simulation steps    : {sim_pp.x_out.shape[1]}")
    trk_end   = sim_pp.x_out[:2, -1]
    ref_end   = ref_traj.x[-1, :2]
    print(f"  Final position error: {np.hypot(*(trk_end - ref_end)) * 1e3:.1f} mm")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    _fig1_segmented_path(seg_info, res_s1, res_s2, res_corner_opt)
    _fig2_jlap_profiles(res_s1, res_s2)
    _fig3_we_sweep(sweep)
    _fig4_corner_and_full(seg_info, res_corner_time, res_corner_opt,
                          mA, mB, res_s1, res_s2, opt_we, we_time_ref)
    _fig5_wheel_kinematics(res_s1, res_s2, res_corner_opt)
    _fig6_pareto_front(sweep)
    _fig7_purepursuit(ref_traj, sim_pp)

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
        beta=abs(BETA),          # PathSegment requires beta in (0, pi)
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

    # Local -> world: rotate by HEADING_IN, translate to arc_entry.
    # For a right turn (BETA < 0) mirror the local arc about the x-axis
    # (negate y and heading) before rotating, which flips the curve direction.
    c, s  = np.cos(HEADING_IN), np.sin(HEADING_IN)
    R_mat = np.array([[c, -s], [s, c]])
    arc_local    = res['path_segment'].copy()     # (n_samples, 3)
    if BETA < 0:
        arc_local[:, 1] = -arc_local[:, 1]       # mirror y
        arc_local[:, 2] = -arc_local[:, 2]       # mirror heading
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


def _solve_corner(corner_wps, w_energy, warm_start=None,
                  v_entry=None, v_exit=None,
                  a_entry=0.0, a_exit=0.0,
                  alpha_entry=0.0, alpha_exit=0.0):
    """! Run BSplineEnergyCoverage for the corner at a given w_energy.

    @param v_entry<float|None>: Pinned entry speed [m/s]. None -> V_HANDOFF.
    @param v_exit<float|None>:  Pinned exit speed  [m/s]. None -> V_HANDOFF.
    @param a_entry<float>: Pinned entry forward acceleration [m/s²].
    @param a_exit<float>:  Pinned exit  forward acceleration [m/s²].
    @param alpha_entry<float>: Pinned entry angular acceleration [rad/s²].
    @param alpha_exit<float>:  Pinned exit  angular acceleration [rad/s²].
    @return result dict from generate_trajectory().
    """
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
    """! Evaluate motor power from IK outputs of a corner result.

    @return (n_ik,) ndarray [W] - total electrical power including hotel load.
    """
    l = ROBOT_PARAMS_BSPLINE['l']
    v, omega = res['v'], res['omega']
    dt = 0.01

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


def _corner_metrics(res):
    """! Compute corner performance metrics.

    @return dict with scalar and array metrics.
    """
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
    """Compute per-wheel angular velocity, acceleration, and jerk.

    @param res<dict>: Trajectory result dict (JLAP or B-spline).
    @param l<float>: Half-wheelbase [m].
    @param r<float>: Wheel radius [m].
    @param dt<float>: Sampling interval for numerical differentiation [s].
    @return dict with keys: time, omega_r, omega_l, alpha_r, alpha_l, jerk_r, jerk_l.
    """
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

    # jerk in m/s³: use OCP-level values if available (bspline), else np.gradient * r
    if 'jerk_r' in res and 'jerk_l' in res:
        jerk_r = res['jerk_r']   # already m/s³ from OCP ddds_val
        jerk_l = res['jerk_l']
    else:
        jerk_r = np.gradient(alpha_r, dt) * r
        jerk_l = np.gradient(alpha_l, dt) * r

    return dict(time=t, omega_r=omega_r, omega_l=omega_l,
                alpha_r=alpha_r, alpha_l=alpha_l,
                jerk_r=jerk_r, jerk_l=jerk_l)


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


def _sweep_we(corner_wps, a_entry=0.0, alpha_entry=0.0, v_handoff=None):
    """! Sweep w_energy in [0, 1e-3 ... 1] and identify optimal trade-off.

    Uses the minimum of d2(peak_power)/d(w_e)^2 as the optimal point.

    @param a_entry<float>:     Forward acceleration at corner entry [m/s²].
                               Pass the actual JLAP exit value for continuity.
    @param alpha_entry<float>: Angular acceleration at corner entry [rad/s²].
    @param v_handoff<float|None>: Entry/exit speed for the corner [m/s].
                               None defaults to V_HANDOFF.
    @return dict with sweep arrays and opt_we.
    """
    v_h = float(v_handoff) if v_handoff is not None else V_HANDOFF

    # Sweep values run HIGH to LOW (warm-start chain).
    # Hand-picked to skip the 0.04-0.09 range which causes IPOPT to exceed
    # 5000 iterations on this problem geometry.
    we_values = np.array([0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0])

    peak_powers    = []
    total_energies = []
    mission_times  = []
    we_valid       = []   # only keep feasible solves for gradient analysis

    print(f"  Sweeping {len(we_values)} w_e values (w_time = 1.0, v_h = {v_h:.3f} m/s) ...")
    print(f"  {'w_e':>10}  {'Time [s]':>10}  {'Energy [J]':>10}  {'Peak P [W]':>10}")
    print("  " + "-" * 49)

    prev_res = None
    res_by_we = {}  # cache results for re-use in main()
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
            prev_res = None   # reset warm-start on failure

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

    # Drop non-converged outliers: IPOPT can return finite but garbage values
    # when hitting max_iter (mission time inflates by 10×+). Filter by median.
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
            # Purge from cache too so main() doesn't try to use them.
            for w_e in list(res_by_we):
                if w_e not in we_values:
                    del res_by_we[w_e]

    d1 = np.gradient(peak_powers, we_values)
    d2 = np.gradient(d1, we_values)

    # Time reference and energy-optimal anchor points (needed for Pareto knee chord).
    # w_e=0.0 is the pure time-optimal formulation but its flat Hessian causes
    # LBFGS to sometimes converge to a suboptimal local minimum.  Using the
    # sweep minimum is always a valid (and often tighter) time reference.
    time_ref_idx   = int(np.argmin(mission_times))
    we_time_ref    = float(we_values[time_ref_idx])
    _energy_anchor = int(np.argmin(total_energies))

    # Pareto knee: restrict to the non-dominated front in (te, pp) space,
    # then find the point with max perpendicular distance from the chord
    # connecting the two extreme Pareto-optimal endpoints.
    _pf = _pareto_front_idx(total_energies, peak_powers)
    if len(_pf) >= 3:
        _te_n = (total_energies - total_energies.min()) / max(float(total_energies.max() - total_energies.min()), 1e-12)
        _pp_n = (peak_powers    - peak_powers.min())    / max(float(peak_powers.max()    - peak_powers.min()),    1e-12)
        _ax, _ay = _te_n[_pf[0]], _pp_n[_pf[0]]
        _bx, _by = _te_n[_pf[-1]], _pp_n[_pf[-1]]
        _denom = max(float(np.hypot(_bx - _ax, _by - _ay)), 1e-12)
        _pf_mid = _pf[1:-1]
        _dist  = np.abs((_by - _ay) * (_te_n[_pf_mid] - _ax) - (_bx - _ax) * (_pp_n[_pf_mid] - _ay)) / _denom
        opt_idx = int(_pf_mid[np.argmax(_dist)])
    elif len(_pf) >= 1:
        opt_idx = int(_pf[len(_pf) // 2])
    else:
        opt_idx = _energy_anchor
    # Ensure the selected opt_we differs from 0 for a meaningful comparison.
    if float(we_values[opt_idx]) == 0.0 and len(we_values) > 1:
        opt_idx = 1
    opt_we  = float(we_values[opt_idx])
    if we_time_ref != 0.0:
        print(f"  NOTE: w_e=0.0 did not yield minimum time in sweep; "
              f"using w_e={we_time_ref:.4f} as time reference.")

    print()
    print(f"  Optimal w_e = {opt_we:.6f}   (Pareto knee)")
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
        'we_time_ref':    we_time_ref,    # w_e giving minimum mission time
        'time_ref_idx':   time_ref_idx,
        'res_by_we':      res_by_we,      # cached trajectory results
        'v_handoff':      v_h,            # handoff speed used for this sweep
    }


def _find_smooth_v_handoff(corner_wps, a_entry=0.0, alpha_entry=0.0):
    """! Find the largest V_HANDOFF that keeps corner transition jerk within J_LIM.

    Tries V_HANDOFF, 80%, 60%, 40%, V_HANDOFF_MIN in sequence, solving the
    corner OCP at w_energy=0.01 (a light energy-regularised formulation that
    converges robustly without a warm start).  Returns the first speed that
    satisfies the per-wheel jerk limit, or V_HANDOFF_MIN as a safe fallback.

    @param a_entry<float>:     Forward acceleration at corner entry [m/s²].
    @param alpha_entry<float>: Angular acceleration at corner entry [rad/s²].
    @return float: chosen handoff velocity [m/s].
    """
    l_ref = ROBOT_PARAMS_BSPLINE['l']
    r_ref = ROBOT_PARAMS_BSPLINE['r']
    dt_c  = 0.01   # BSpline IK time step [s]
    jerk_wheel_lim = J_LIM   # m/s³ -- _compute_wheel_kinematics now returns m/s³

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
            # Max per-wheel angular jerk at entry (first 3 samples) and
            # exit (last 3 samples) -- the region adjacent to the JLAP segments.
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
def _print_corner_comparison(mA, mB, opt_we, we_time_ref=0.0):
    rows = [
        ('Total time',    's',   'total_time'),
        ('Path length',   'm',   'path_length'),
        ('Total energy',  'J',   'energy'),
        ('Energy/meter',  'J/m', 'energy_per_meter'),
        ('Peak power',    'W',   'peak_power'),
        ('Avg power',     'W',   'avg_power'),
    ]
    print()
    print("  Corner: time reference vs energy-optimal")
    t_col = f'Time-ref (w_e={we_time_ref:.4f})'
    w_col = f'Opt (w_e={opt_we:.4f})'
    hdr = (f"  {'Metric':<20} {'Unit':<6} "
           f"{t_col:>22}  {w_col:>18}  {'Delta%':>7}")
    print(hdr)
    print("  " + "-" * 78)
    for label, unit, key in rows:
        a, b = mA[key], mB[key]
        if abs(a) > 1e-12:
            d = (b - a) / abs(a) * 100
            ds = f"{'+'if d>=0 else ''}{d:.1f}%"
        else:
            ds = 'n/a'
        print(f"  {label:<20} {unit:<6} {a:>22.4f}  {b:>18.4f}  {ds:>7}")
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

    # Travel-direction arrow on the first eastward leg
    ax.annotate('', xy=(5.3, 0.0), xytext=(4.7, 0.0),
                arrowprops=dict(arrowstyle='->', color='silver',
                                lw=1.0, mutation_scale=12))


def _fig1_segmented_path(seg_info, res_s1, res_s2, res_corner_opt):
    """! Zoomed-out boustrophedon overview with inset corner-detail cutout."""
    fig, ax = plt.subplots(figsize=(8, 8),
                           num='Figure 1 - Segmented Path')
    ax.set_aspect('equal')

    # ---- background boustrophedon strips ----
    _draw_coverage_background(ax)

    # ---- reference L-path waypoints ----
    ref = np.array([WP_START, WP_CORNER, WP_END])
    ax.plot(ref[:, 0], ref[:, 1], 'o', color='black', markersize=7, zorder=6)

    # ---- computed trajectory on main axes (overview, no arrows for clarity) ----
    s1  = res_s1['states']
    sc  = res_corner_opt['states']
    s2  = res_s2['states']
    arc = seg_info['arc_world']

    ax.plot(s1[:, 0], s1[:, 1], '-', color=COL_S1, linewidth=2.0,
            label='Segment 1  (JLAP)', zorder=4)
    ax.plot(sc[:, 0], sc[:, 1], '-', color=COL_C,  linewidth=2.0,
            label='Corner  (B-spline, opt $w_e$)', zorder=4)
    ax.plot(s2[:, 0], s2[:, 1], '-', color=COL_S2, linewidth=2.0,
            label='Segment 2  (JLAP)', zorder=4)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)

    # ================================================================
    # INSET: zoom into corner region where PathSegment planning occurs
    # ================================================================
    # Data extent of the zoom window (corner now at (0, 10))
    xi1, xi2 = -0.5, 2.8
    yi1, yi2 = 7.5, 10.5

    # Place inset in the upper-right area (clear of the computed path)
    axins = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
    axins.set_xlim(xi1, xi2)
    axins.set_ylim(yi1, yi2)
    axins.set_aspect('equal')

    # PathSegment arc
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

    # Tail of S1 approaching corner (travelling north — filter by y)
    mask_s1 = s1[:, 1] >= yi1
    if mask_s1.any():
        axins.plot(s1[mask_s1, 0], s1[mask_s1, 1], '-',
                   color=COL_S1, linewidth=2.0, zorder=4)

    # Full corner B-spline + control polygon
    axins.plot(sc[:, 0], sc[:, 1], '-', color=COL_C, linewidth=2.0, zorder=4)
    axins.plot(cpts[:, 0], cpts[:, 1], 'x', color=COL_C,
               markersize=7, markeredgewidth=1.5, zorder=5)

    # Head of S2 leaving corner (travelling east — filter by x)
    mask_s2 = s2[:, 0] <= xi2
    if mask_s2.any():
        axins.plot(s2[mask_s2, 0], s2[mask_s2, 1], '-',
                   color=COL_S2, linewidth=2.0, zorder=4)

    # Heading arrows on inset (every ~15% of each segment within window)
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

    # Connect inset to zoom region on main axes
    ax.indicate_inset_zoom(axins, edgecolor='black', linewidth=1.2)

    fig.tight_layout()
    _savefig(fig, 'fig_overview.png')


# =============================================================================
# FIGURE 2 - JLAP kinematic profiles
# =============================================================================
def _fig2_jlap_profiles(res_s1, res_s2):
    """! Per-wheel v / a / jerk for Segment 1 on two side-by-side axes."""
    r = JLAP_ROBOT_PARAMS['wheel_radius']
    l = 0.5 * JLAP_ROBOT_PARAMS['robot_width']

    wk = _compute_wheel_kinematics(res_s1, l, r, JLAP_DT)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4),
                             num='Figure 2 - JLAP Kinematic Profiles')

    wheel_keys = [
        ('omega_l', 'alpha_l', 'jerk_l', 'Left Wheel'),
        ('omega_r', 'alpha_r', 'jerk_r', 'Right Wheel'),
    ]
    for ax, (k_v, k_a, k_j, side) in zip(axes, wheel_keys):
        t = wk['time']
        ax.plot(t, wk[k_v] * r, '-',  color=COL_S1, lw=1.8, label='v [m/s]')
        ax.plot(t, wk[k_a] * r, '--', color=COL_S1, lw=1.8, label='a [m/s²]')
        ax.plot(t, wk[k_j],     ':',  color=COL_S1, lw=1.8, label='j [m/s³]')
        ax.axhline(0, color='lightgray', lw=0.8, zorder=0)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('m/s  /  m/s²  /  m/s³')
        ax.set_title(side)

    axes[0].legend(fontsize=8)
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
                             num='Figure 3 - w_energy Sweep (Corner OCP)')

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
        ax.legend(fontsize=10)

    fig.tight_layout()


# =============================================================================
# FIGURE 4 - Corner detail + full stitched trajectory
# =============================================================================
def _fig4_corner_and_full(seg_info, res_corner_time, res_corner_opt,
                           mA, mB, res_s1, res_s2, opt_we, we_time_ref=0.0):
    """! Corner comparison (XY + power) and full stitched path + velocity."""

    fig, axes = plt.subplots(2, 2, figsize=(13, 10),
                             num='Figure 4 - Corner Detail + Full Trajectory')

    # ------------------------------------------------------------------ [0,0]
    # Corner XY: time-optimal vs energy-optimal
    ax = axes[0, 0]
    ax.set_aspect('equal')

    # PathSegment reference arc
    arc = seg_info['arc_world']
    ax.plot(arc[:, 0], arc[:, 1], '-.', color='purple', linewidth=1.5,
            label='PathSegment arc', zorder=2)

    # Corner waypoints
    cwps = np.array(_build_corner_waypoints(seg_info['arc_entry_ext_world'],
                                            seg_info['arc_exit_ext_world']))
    ax.plot(cwps[:, 0], cwps[:, 1], 'o--', color=COL_REF, markersize=7,
            linewidth=1.0, label='Corner waypoints', zorder=3)

    # Time reference
    st_A = res_corner_time['states']
    ax.plot(st_A[:, 0], st_A[:, 1], '-', color=COL_S1, linewidth=2.0,
            label=f'Time-ref  (w_e={we_time_ref:.4f})', zorder=4)

    # Energy-optimal
    st_B = res_corner_opt['states']
    ax.plot(st_B[:, 0], st_B[:, 1], '-', color=COL_OPT, linewidth=2.0,
            label=f'Energy-optimal', zorder=4)
    cpts = res_corner_opt['ctrl_pts']
    ax.plot(cpts[:, 0], cpts[:, 1], 'x', color=COL_OPT,
            markersize=7, markeredgewidth=1.5, zorder=5)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend(fontsize=10)

    # ------------------------------------------------------------------ [0,1]
    # Corner power P(t): time reference vs energy-optimal
    ax = axes[0, 1]
    ax.plot(mA['time_ik'], mA['P_total'], '-', color=COL_S1, linewidth=1.8,
            label=f"Time-ref (w_e={we_time_ref:.4f})  peak={mA['peak_power']:.1f} W")
    ax.plot(mB['time_ik'], mB['P_total'], '-', color=COL_OPT, linewidth=1.8,
            label=f"Energy-opt  peak={mB['peak_power']:.1f} W")
    ax.axhline(P_ELECTRONICS, color=COL_REF, linestyle=':', linewidth=1.0,
               label=f'P_elec = {P_ELECTRONICS} W')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('Power [W]')
    ax.legend(fontsize=10)

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

    # Arc tangent points and JLAP handoff points
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
    ax.legend(fontsize=10)

    # ------------------------------------------------------------------ [1,1]
    # Full stitched velocity profile
    ax = axes[1, 1]
    T1 = float(res_s1['time'][-1])
    Tc = float(res_corner_opt['time'][-1])

    # time_ik starts at 0 (linspace); no extra prepend needed.
    corner_t = res_corner_opt['time_ik']
    corner_v = res_corner_opt['v']

    ax.plot(res_s1['time'], res_s1['v'],
            '-', color=COL_S1, linewidth=1.8, label='Segment 1 (JLAP)')
    ax.plot(corner_t + T1, corner_v,
            '-', color=COL_C, linewidth=1.8, label='Corner (B-spline)')
    ax.plot(res_s2['time'] + T1 + Tc, res_s2['v'],
            '-', color=COL_S2, linewidth=1.8, label='Segment 2 (JLAP)')

    # Junction markers
    ax.axvline(T1,      color='black', linestyle=':', linewidth=1.0)
    ax.axvline(T1 + Tc, color='black', linestyle=':', linewidth=1.0)
    ax.set_xlabel('time [s]  (segments stitched)')
    ax.set_ylabel('v [m/s]')
    ax.legend(fontsize=10)

    fig.tight_layout()


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
                             num='Figure 5 - Per-Wheel Kinematics')

    segments = [
        (wk_s1, COL_S1, 'Segment 1 (JLAP)'),
        (wk_c,  COL_C,  'Corner (B-spline opt)'),
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


# =============================================================================
# FIGURE 6 - Pareto front (paper figure)
# =============================================================================
def _fig6_pareto_front(sweep):
    """! Pareto front: peak power vs. total energy, parametric on w_energy.

    Each sweep point is plotted as a scatter marker coloured by log10(w_e).
    The Pareto knee (opt_we) and the time-reference point (we_time_ref) are
    annotated.  This figure is saved as fig_pareto.png when SAVE_FIGS is True.
    """
    we     = sweep['we_values']
    pp     = sweep['peak_powers']
    te     = sweep['total_energies']
    oi     = sweep['opt_idx']
    tri    = sweep['time_ref_idx']
    opt_we = sweep['opt_we']
    we_ref = sweep['we_time_ref']

    fig, ax = plt.subplots(figsize=(6, 4.5),
                           num='Figure 6 - Pareto Front: Peak Power vs. Total Energy')

    # Connect points by a thin grey line in ascending energy order.
    sort_e = np.argsort(te)
    ax.plot(te[sort_e], pp[sort_e], '-', color='lightgray',
            linewidth=1.0, zorder=1)

    # Scatter all points with uniform colour.
    ax.scatter(te, pp, c='dimgray', s=55, zorder=3)

    # Time-reference marker.
    ax.scatter([te[tri]], [pp[tri]], marker='^', s=120, color='steelblue',
               zorder=5, label=f'Time-ref  ($w_e$={we_ref:.4f})')
    ax.annotate(f'time-ref\n$w_e$={we_ref:.4f}',
                xy=(te[tri], pp[tri]),
                xytext=(6, 6), textcoords='offset points',
                fontsize=8, color='steelblue')

    # Dashed crosshair at knee point.
    ax.axvline(te[oi], color='darkorange', ls='--', lw=1.0, alpha=0.7, zorder=2)
    ax.axhline(pp[oi], color='darkorange', ls='--', lw=1.0, alpha=0.7, zorder=2)

    # Knee marker.
    ax.scatter([te[oi]], [pp[oi]], marker='o', s=150, color='darkorange',
               zorder=5, label=f'Knee  ($w_e$={opt_we:.4f})')
    ax.annotate(f'knee\n$w_e$={opt_we:.4f}',
                xy=(te[oi], pp[oi]),
                xytext=(6, -22), textcoords='offset points',
                fontsize=8, color='darkorange')

    # Label remaining points — alternate above/below to avoid crowding.
    others = [(i, t_e, p_p, w_e)
              for i, (t_e, p_p, w_e) in enumerate(zip(te, pp, we))
              if i not in (oi, tri)]
    for k, (i, t_e, p_p, w_e) in enumerate(others):
        dy = 8 if k % 2 == 0 else -14
        ax.annotate(f'$w_e$={w_e:.4f}', xy=(t_e, p_p),
                    xytext=(4, dy), textcoords='offset points',
                    fontsize=8, color='dimgray')

    ax.set_xlabel('Total Energy [J]')
    ax.set_ylabel('Peak Motor Power [W]')
    ax.legend(fontsize=10)
    fig.tight_layout()
    _savefig(fig, 'fig_pareto.png')


# =============================================================================
# CSV EXPORT
# =============================================================================
def _export_csv(res_s1, res_s2, res_corner_opt,
                out_dir=None):
    """! Write per-segment and stitched trajectory CSV files.

    Files written:
      segment1_jlap.csv    -- straight segment 1 (JLAP)
      corner_bspline.csv   -- corner B-spline (energy-optimal)
      segment2_jlap.csv    -- straight segment 2 (JLAP)
      trajectory_stitched.csv -- all three on a single time axis

    Columns for JLAP segments:
        time, x, y, theta, v, acc_path, omega, alpha, omega_r, omega_l

    Columns for corner:
        time, x, y, theta, v, omega, omega_r, omega_l, power

    Columns for stitched:
        time, x, y, theta, v, omega, omega_r, omega_l, segment
        (segment: 1=seg1, 2=corner, 3=seg2)
    """
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), 'csv_output')
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

    # Unified wheel-kinematic parameters for output (r=0.3 m, l=0.265 m).
    # JLAP internally uses its own wheel_radius for dynamics; we re-express the
    # output omega_r/omega_l using the BSpline robot geometry so all three
    # segments share the same scale in the stitched CSV.
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
    jrkr_s1  = np.gradient(alr_s1, JLAP_DT) * r_ref   # m/s³: d(alpha_r)/dt * r
    jrkl_s1  = np.gradient(all_s1, JLAP_DT) * r_ref
    _save('segment1_jlap.csv',
          ['time', 'x', 'y', 'theta', 'v', 'acc_path', 'omega', 'alpha',
           'omega_r', 'omega_l', 'alpha_r', 'alpha_l', 'jerk_r', 'jerk_l'],
          [res_s1['time'], s1[:, 0], s1[:, 1], s1[:, 2],
           res_s1['v'], res_s1['acc_path'], res_s1['omega'], res_s1['alpha'],
           omr_s1, oml_s1, alr_s1, all_s1, jrkr_s1, jrkl_s1])

    # ------------------------------------------------------------------
    # Corner - B-spline (kinematic outputs are on time_ik, denser than OCP grid)
    # ------------------------------------------------------------------
    t_ik  = res_corner_opt['time_ik']
    t_ocp = res_corner_opt['time']
    sc    = res_corner_opt['states']
    x_c   = np.interp(t_ik, t_ocp, sc[:, 0])
    y_c   = np.interp(t_ik, t_ocp, sc[:, 1])
    th_c  = np.interp(t_ik, t_ocp, sc[:, 2])
    pwr_c = np.interp(t_ik, t_ocp, res_corner_opt['power'])
    # Wheel accelerations from analytical B-spline IK derivatives.
    # Wheel jerks come directly from the OCP 3rd derivative (stored in the
    # result dict by BSplineEnergyCoverage) -- no numerical differentiation,
    # so they are guaranteed to respect the OCP wheel-jerk constraint.
    alr_c  = (res_corner_opt['acc_path'] + l_ref * res_corner_opt['alpha']) / r_ref
    all_c  = (res_corner_opt['acc_path'] - l_ref * res_corner_opt['alpha']) / r_ref
    jrkr_c = res_corner_opt['jerk_r']
    jrkl_c = res_corner_opt['jerk_l']

    _save('corner_bspline.csv',
          ['time', 'x', 'y', 'theta', 'v', 'acc_path', 'omega', 'alpha',
           'omega_r', 'omega_l', 'alpha_r', 'alpha_l', 'jerk_r', 'jerk_l', 'power'],
          [t_ik, x_c, y_c, th_c,
           res_corner_opt['v'], res_corner_opt['acc_path'],
           res_corner_opt['omega'], res_corner_opt['alpha'],
           res_corner_opt['omega_r'], res_corner_opt['omega_l'],
           alr_c, all_c, jrkr_c, jrkl_c, pwr_c])

    # ------------------------------------------------------------------
    # Segment 2 - JLAP
    # ------------------------------------------------------------------
    s2 = res_s2['states']
    omr_s2   = (res_s2['v'] + l_ref * res_s2['omega']) / r_ref
    oml_s2   = (res_s2['v'] - l_ref * res_s2['omega']) / r_ref
    alr_s2   = (res_s2['acc_path'] + l_ref * res_s2['alpha']) / r_ref
    all_s2   = (res_s2['acc_path'] - l_ref * res_s2['alpha']) / r_ref
    jrkr_s2  = np.gradient(alr_s2, JLAP_DT) * r_ref   # m/s³: d(alpha_r)/dt * r
    jrkl_s2  = np.gradient(all_s2, JLAP_DT) * r_ref
    _save('segment2_jlap.csv',
          ['time', 'x', 'y', 'theta', 'v', 'acc_path', 'omega', 'alpha',
           'omega_r', 'omega_l', 'alpha_r', 'alpha_l', 'jerk_r', 'jerk_l'],
          [res_s2['time'], s2[:, 0], s2[:, 1], s2[:, 2],
           res_s2['v'], res_s2['acc_path'], res_s2['omega'], res_s2['alpha'],
           omr_s2, oml_s2, alr_s2, all_s2, jrkr_s2, jrkl_s2])

    # ------------------------------------------------------------------
    # Stitched - continuous time axis across all three segments
    # ------------------------------------------------------------------
    T1 = float(res_s1['time'][-1])
    Tc = float(t_ocp[-1])

    # Corner endpoint (time_ik[-1] == Tc) has omega=0 by constraint; include it.
    # S2 sample[0] is at t=0 -> offset to T1+Tc, which duplicates the corner
    # endpoint timestamp.  Drop S2's first sample to avoid the duplicate.
    seg_id = np.concatenate([
        np.ones(len(res_s1['time'])),
        np.full(len(t_ik), 2),
        np.full(len(res_s2['time']) - 1, 3),
    ])
    _save('trajectory_stitched.csv',
          ['time', 'x', 'y', 'theta', 'v', 'omega', 'omega_r', 'omega_l', 'segment'],
          [np.concatenate([res_s1['time'], t_ik + T1, res_s2['time'][1:] + T1 + Tc]),
           np.concatenate([s1[:, 0], x_c, s2[1:, 0]]),
           np.concatenate([s1[:, 1], y_c, s2[1:, 1]]),
           np.concatenate([s1[:, 2], th_c, s2[1:, 2]]),
           np.concatenate([res_s1['v'],       res_corner_opt['v'],       res_s2['v'][1:]]),
           np.concatenate([res_s1['omega'],    res_corner_opt['omega'],   res_s2['omega'][1:]]),
           np.concatenate([omr_s1,             res_corner_opt['omega_r'], omr_s2[1:]]),
           np.concatenate([oml_s1,             res_corner_opt['omega_l'], oml_s2[1:]]),
           seg_id])

    print(f"  Output directory: {out_dir}")


# =============================================================================
# PURE PURSUIT TRACKING
# =============================================================================
def _build_reference_trajectory(res_s1, res_s2, res_corner_opt, dt=0.05):
    """! Stitch the three segments into a uniform-dt Trajectory for controllers.

    The raw segments have non-uniform time grids (JLAP: 0.05 s, BSpline IK: 0.01 s).
    This function concatenates them on a raw time axis, then resamples to dt.

    @param res_s1<dict>:          EulerJLAP result for segment 1.
    @param res_s2<dict>:          EulerJLAP result for segment 2.
    @param res_corner_opt<dict>:  BSplineEnergy result for the corner.
    @param dt<float>:             Target sampling interval [s].
    @return Trajectory instance ready for PurePursuit / FeedForward.
    """
    T1   = float(res_s1['time'][-1])
    t_ik = res_corner_opt['time_ik']          # starts at 0
    t_ocp = res_corner_opt['time']
    Tc   = float(t_ocp[-1])
    sc   = res_corner_opt['states']

    x_c  = np.interp(t_ik, t_ocp, sc[:, 0])
    y_c  = np.interp(t_ik, t_ocp, sc[:, 1])
    th_c = np.interp(t_ik, t_ocp, sc[:, 2])

    t_raw  = np.concatenate([res_s1['time'],
                              t_ik + T1,
                              res_s2['time'][1:] + T1 + Tc])
    x_raw  = np.concatenate([res_s1['states'][:, 0],  x_c,  res_s2['states'][1:, 0]])
    y_raw  = np.concatenate([res_s1['states'][:, 1],  y_c,  res_s2['states'][1:, 1]])
    th_raw = np.concatenate([res_s1['states'][:, 2],  th_c, res_s2['states'][1:, 2]])
    v_raw  = np.concatenate([res_s1['v'],  res_corner_opt['v'],  res_s2['v'][1:]])
    w_raw  = np.concatenate([res_s1['omega'], res_corner_opt['omega'], res_s2['omega'][1:]])

    t_uni  = np.arange(t_raw[0], t_raw[-1], dt)
    x_uni  = np.interp(t_uni, t_raw, x_raw)
    y_uni  = np.interp(t_uni, t_raw, y_raw)
    th_uni = np.interp(t_uni, t_raw, th_raw)
    v_uni  = np.interp(t_uni, t_raw, v_raw)
    w_uni  = np.interp(t_uni, t_raw, w_raw)

    states   = np.column_stack([x_uni, y_uni, th_uni])  # (N, 3)
    controls = np.vstack([v_uni, w_uni])                  # (2, N)

    return Trajectory(x=states, u=controls, t=t_uni, sampling_time=dt)


def _run_purepursuit(ref_traj):
    """! Run closed-loop Pure Pursuit tracking on the reference trajectory.

    @param ref_traj<Trajectory>: Uniform-dt reference from _build_reference_trajectory.
    @return TimeStepping instance with x_out / u_out / t_out populated.
    """
    wheel_base = 2.0 * ROBOT_PARAMS_BSPLINE['l']
    model      = DifferentialDrive(wheel_base=wheel_base)
    sim        = TimeStepping(model, float(ref_traj.t[-1]), ref_traj.sampling_time)
    controller = PurePursuit(model, ref_traj)

    initial_position = list(ref_traj.x[0])
    sim.run_with_controller(initial_position, ref_traj, controller)
    return sim


def _fig7_purepursuit(ref_traj, sim):
    """! Four-panel tracking analysis: XY path, speed, angular velocity, cross-track error."""
    t = sim.t_out
    ref_xy  = ref_traj.x[:, :2]                    # (N, 2) reference positions
    trk_xy  = sim.x_out[:2, :].T                   # (N, 2) tracked positions

    # Cross-track error: distance from each tracked point to nearest reference point.
    cte = np.array([
        np.min(np.hypot(ref_xy[:, 0] - pt[0], ref_xy[:, 1] - pt[1]))
        for pt in trk_xy
    ])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                             num='Figure 7 - PurePursuit Tracking')

    # ---- [0,0] XY path ----
    ax = axes[0, 0]
    ax.set_aspect('equal')
    ax.plot(ref_traj.x[:, 0], ref_traj.x[:, 1], '--',
            color=COL_REF, linewidth=1.5, label='Reference', zorder=2)
    ax.plot(trk_xy[:, 0], trk_xy[:, 1], '-',
            color=COL_OPT, linewidth=2.0, label='PurePursuit tracked', zorder=3)
    ax.plot(*ref_traj.x[0, :2],  'o', color='green',  markersize=8, zorder=4, label='Start')
    ax.plot(*ref_traj.x[-1, :2], 's', color='red',    markersize=8, zorder=4, label='Goal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend(fontsize=10)

    # ---- [0,1] Linear velocity ----
    ax = axes[0, 1]
    ax.plot(ref_traj.t, ref_traj.u[0, :], '--',
            color=COL_REF, linewidth=1.5, label='Reference v')
    ax.plot(t, sim.u_out[0, :], '-',
            color=COL_OPT, linewidth=1.8, label='Tracked v')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('v [m/s]')
    ax.legend(fontsize=10)

    # ---- [1,0] Angular velocity ----
    ax = axes[1, 0]
    ax.plot(ref_traj.t, ref_traj.u[1, :], '--',
            color=COL_REF, linewidth=1.5, label='Reference ω')
    ax.plot(t, sim.u_out[1, :], '-',
            color=COL_OPT, linewidth=1.8, label='Tracked ω')
    ax.axhline(0, color='lightgray', linewidth=0.8, zorder=0)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('ω [rad/s]')
    ax.legend(fontsize=10)

    # ---- [1,1] Cross-track error ----
    ax = axes[1, 1]
    ax.plot(t, cte * 1e3, '-', color='tomato', linewidth=1.8)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('Cross-track error [mm]')
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    _savefig(fig, 'fig_purepursuit.png')


# =============================================================================
if __name__ == '__main__':
    main()
