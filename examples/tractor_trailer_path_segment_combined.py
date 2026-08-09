#!/usr/bin/env python3
##
# @file tractor_trailer_path_segment_combined.py
#
# @brief Combined trajectory for a tractor-trailer on a 5 m x 5 m L-shaped corner.
#
# Path:  (0,5) -> (0,10) -> (5,10)   -- 90-degree right turn.
#
# Pipeline
# --------
#   1. PathSegment                         -- circular arc geometry at the corner.
#   2. EulerJLAPCoverage                   -- jerk-limited profiles for the two
#                                             straight segments flanking the arc.
#   3. BSplineEnergyTractorTrailerCoverage -- corner OCP run twice:
#        Run A: time-optimal  (w_energy = 0)
#        Run B: energy-aware  (w_energy = W_ENERGY, warm-started from A)
#
# Figures
# -------
#   Figure 1 -- Segmented path overview (arc + JLAP + energy-opt corner)
#   Figure 2 -- JLAP kinematic profiles for both straight segments
#   Figure 3 -- Corner comparison: time-opt vs energy-opt (XY + power + hitch)
#   Figure 4 -- Full stitched trajectory (XY + velocity + hitch angle)
#   Figure 5 -- PurePursuit closed-loop tracking of tractor reference path
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/06/23

import sys
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from trajectory_generators.path_segment import PathSegment
from trajectory_generators.euler_jlap_coverage import EulerJLAPCoverage
from trajectory_generators.bspline_energy_tractor_trailer_coverage import (
    BSplineEnergyTractorTrailerCoverage,
)
from models.differential_drive import DifferentialDrive
from simulators.time_stepping import TimeStepping
from controllers.purepursuit import PurePursuit
from controllers.trajectory import Trajectory


# =============================================================================
# PAPER FIGURE EXPORT
# =============================================================================
SAVE_FIGS   = True
FIG_OUT_DIR = (pathlib.Path(__file__).resolve().parent.parent.parent
               / 'Writting' / 'energy_aware')


def _savefig(fig, filename):
    if SAVE_FIGS:
        FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = FIG_OUT_DIR / filename
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f"  [paper] Saved {filename} -> {out}")


# =============================================================================
# CONFIGURATION
# =============================================================================

WP_START  = [0.0, 5.0]
WP_CORNER = [0.0, 10.0]
WP_END    = [5.0, 10.0]

HEADING_IN  = np.pi / 2    # north  [rad]
HEADING_OUT = 0.0          # east   [rad]
BETA        = -np.pi / 2   # 90-degree right turn [rad]

L_TRANSITION = 0.5         # straight lead-in/out for B-spline OCP [m]

# PathSegment feasibility inputs
L_INPUT   = 1.0
B_INPUT   = 0.08
V_MAX_SEG = 0.5
A_MAX_SEG = 1.0
L_WHEELBASE = 0.35         # [m] -- used only for PathSegment radius estimate

# Tractor-trailer geometry
LB        = 0.2            # tractor rear axle -> hitch [m]
LF        = 0.8            # hitch -> trailer rear axle [m]
GAMMA_MAX = 0.785          # hitch angle limit [rad]
TRACTOR_WHEEL_BASE = 0.35  # 2 * robot_params['l'] = 2 * 0.175 [m]

# EulerJLAP robot params -- NewMiniAGV motor model, path_vel_lim capped at TT vel_max.
JLAP_ROBOT_PARAMS = {
    'robot_mass':         50.4,
    'robot_width':        0.510,
    'wheel_radius':       0.3,
    'gear_ratio':         40.0,
    'rated_motor_torque': 1.3,
    'rated_motor_speed':  3500.0,
    'motor_inertia':      0.66e-4,
    'path_vel_lim':       0.2,   # match TT vel_max [m/s]
}
JLAP_DT = 0.05   # [s]

# Derived acc/jerk limits from JLAP motor model
_j_rated_torque = JLAP_ROBOT_PARAMS['gear_ratio'] * JLAP_ROBOT_PARAMS['rated_motor_torque']
_j_inertia      = JLAP_ROBOT_PARAMS['gear_ratio']**2 * JLAP_ROBOT_PARAMS['motor_inertia']
_j_r            = JLAP_ROBOT_PARAMS['wheel_radius']
_j_m            = JLAP_ROBOT_PARAMS['robot_mass']
_A_LIM          = (0.5 * _j_rated_torque * _j_r
                   / (0.25 * _j_m * _j_r**2 + _j_inertia))
J_LIM = _A_LIM / (40.0 * JLAP_DT)

# Corner handoff velocity: TT vel_max = 0.2 m/s
V_HANDOFF = JLAP_ROBOT_PARAMS['path_vel_lim']

# Angular limits derived from linear limits and wheelbase
_ANG_ACC_MAX  = _A_LIM / L_WHEELBASE
_ANG_JERK_MAX = J_LIM  / L_WHEELBASE

# Energy-aware weight for Run B
W_ENERGY = 1.0

COL_S1  = 'steelblue'
COL_C_A = 'tomato'
COL_C_B = 'darkorange'
COL_S2  = 'seagreen'
COL_REF = 'gray'


def _make_bspline_tt(v_h):
    return dict(
        bound=0.25,
        n_ctrl_pts=6,
        spline_order=3,
        n_sampling=15,
        vel_max=[v_h, v_h, 0.196],
        vel_min_lin=0.01,
        eps_nonh=0.001,
        acc_max=[_A_LIM, _A_LIM, _ANG_ACC_MAX],
        jerk_max=[J_LIM, J_LIM, _ANG_JERK_MAX],
        eps_hitch=0.05,
        length_back=LB,
        length_front=LF,
        gamma_max=GAMMA_MAX,
        gamma_entry=0.0,
        gamma_exit=0.0,
        p_electronics=2.0,
    )


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
# HELPERS
# =============================================================================
def tractor_xy(states):
    """Derive tractor rear-axle position from trailer states [x, y, theta, gamma]."""
    γ = states[:, 3]
    θ = states[:, 2]
    xt = states[:, 0] + LF * np.cos(θ) + LB * np.cos(θ - γ)
    yt = states[:, 1] + LF * np.sin(θ) + LB * np.sin(θ - γ)
    return xt, yt


def _corner_metrics(res):
    P_tot  = res['power']   # TT returns total power on OCP time grid
    energy = float(res.get('energy', np.trapz(P_tot, res['time'])))
    st     = res['states']
    plen   = float(np.sum(np.sqrt(np.diff(st[:, 0])**2 + np.diff(st[:, 1])**2)))
    return {
        'total_time':       float(res['time'][-1]),
        'path_length':      plen,
        'energy':           energy,
        'peak_power':       float(np.max(P_tot)),
        'avg_power':        float(np.mean(P_tot)),
        'energy_per_meter': energy / plen if plen > 1e-9 else float('inf'),
        'P_total':          P_tot,
        'time_ocp':         res['time'],
    }


def _print_comparison(mA, mB):
    rows = [
        ('Total time',    's',   'total_time'),
        ('Path length',   'm',   'path_length'),
        ('Total energy',  'J',   'energy'),
        ('Energy/meter',  'J/m', 'energy_per_meter'),
        ('Peak power',    'W',   'peak_power'),
        ('Avg power',     'W',   'avg_power'),
    ]
    print()
    print("  Corner: time-optimal vs energy-aware")
    hdr = (f"  {'Metric':<20} {'Unit':<6} "
           f"{'Time-optimal':>22}  {'Energy-aware':>18}  {'Delta%':>7}")
    print(hdr)
    print("  " + "-" * 78)
    for label, unit, key in rows:
        a, b = mA[key], mB[key]
        d = (b - a) / abs(a) * 100 if abs(a) > 1e-12 else float('nan')
        ds = f"{'+'if d>=0 else ''}{d:.1f}%" if np.isfinite(d) else 'n/a'
        print(f"  {label:<20} {unit:<6} {a:>22.4f}  {b:>18.4f}  {ds:>7}")
    print()


# =============================================================================
# STEP 1 - PathSegment
# =============================================================================
def _segment_corner():
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

    L_seg     = res['L_seg']
    corner    = np.array(WP_CORNER, dtype=float)
    in_dir    = np.array([np.cos(HEADING_IN), np.sin(HEADING_IN)])
    arc_entry = corner - L_seg * in_dir

    c, s  = np.cos(HEADING_IN), np.sin(HEADING_IN)
    R_mat = np.array([[c, -s], [s, c]])
    arc_local = res['path_segment'].copy()
    if BETA < 0:
        arc_local[:, 1] = -arc_local[:, 1]
        arc_local[:, 2] = -arc_local[:, 2]
    xy_world  = (R_mat @ arc_local[:, :2].T).T + arc_entry
    hdg_world = arc_local[:, 2] + HEADING_IN
    arc_world = np.column_stack([xy_world, hdg_world])
    arc_exit  = arc_world[-1, :2].copy()

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
    return [
        [arc_entry[0], arc_entry[1], HEADING_IN],
        [WP_CORNER[0], WP_CORNER[1], HEADING_IN],
        [arc_exit[0],  arc_exit[1],  HEADING_OUT],
    ]


def _solve_corner_tt(corner_wps, w_energy, warm_start=None,
                     v_entry=None, v_exit=None,
                     a_entry=0.0, alpha_entry=0.0):
    v_e = float(v_entry) if v_entry is not None else V_HANDOFF
    v_x = float(v_exit)  if v_exit  is not None else V_HANDOFF
    return BSplineEnergyTractorTrailerCoverage(
        waypoints=corner_wps,
        w_time=1.0,
        w_energy=float(w_energy),
        v_entry=v_e,
        v_exit=v_x,
        a_entry=a_entry,
        alpha_entry=alpha_entry,
        **_make_bspline_tt(v_e),
    ).generate_trajectory(warm_start=warm_start)


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

    corner_wps = _build_corner_waypoints(arc_entry_ext, arc_exit_ext)

    # ------------------------------------------------------------------
    # Step 2a: EulerJLAP -- segment 1 (start -> corner entry)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Step 2a: EulerJLAPCoverage - segment 1 (0 -> {V_HANDOFF:.3f} m/s)")
    print("=" * 60)
    res_s1 = _run_jlap_seg(WP_START, arc_entry_ext.tolist(),
                            initial_vel=0.0, final_vel=V_HANDOFF)
    a_s1_exit     = float(res_s1['acc_path'][-1])
    alpha_s1_exit = float(res_s1['alpha'][-1])
    print(f"  T = {res_s1['time'][-1]:.3f} s   "
          f"v_exit = {res_s1['v'][-1]:.3f} m/s   "
          f"a_exit = {a_s1_exit:.4f} m/s²")

    # ------------------------------------------------------------------
    # Step 3a: Corner -- time-optimal (w_energy = 0)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3a: BSplineEnergyTractorTrailerCoverage - time-optimal (w_energy=0)")
    print("=" * 60)
    res_corner_time = _solve_corner_tt(
        corner_wps, w_energy=0.0,
        v_entry=V_HANDOFF, v_exit=V_HANDOFF,
        a_entry=a_s1_exit, alpha_entry=alpha_s1_exit,
    )
    mA = _corner_metrics(res_corner_time)
    print(f"  T = {mA['total_time']:.3f} s   "
          f"E = {mA['energy']:.3f} J   "
          f"P_peak = {mA['peak_power']:.2f} W")

    # ------------------------------------------------------------------
    # Step 3b: Corner -- energy-aware (w_energy = W_ENERGY, warm-started)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Step 3b: BSplineEnergyTractorTrailerCoverage - energy-aware "
          f"(w_energy={W_ENERGY}, warm-started)")
    print("=" * 60)
    res_corner_opt = _solve_corner_tt(
        corner_wps, w_energy=W_ENERGY,
        warm_start=res_corner_time,
        v_entry=V_HANDOFF, v_exit=V_HANDOFF,
        a_entry=a_s1_exit, alpha_entry=alpha_s1_exit,
    )
    mB = _corner_metrics(res_corner_opt)
    print(f"  T = {mB['total_time']:.3f} s   "
          f"E = {mB['energy']:.3f} J   "
          f"P_peak = {mB['peak_power']:.2f} W")

    _print_comparison(mA, mB)

    # ------------------------------------------------------------------
    # Step 2b: EulerJLAP -- segment 2 (tractor corner exit -> tractor end)
    # ------------------------------------------------------------------
    # After the corner (gamma_exit=0) the tractor is aligned with the trailer
    # and offset by (LF + LB) in HEADING_OUT.  JLAP plans the TRACTOR path.
    _st_exit = res_corner_opt['states'][-1:]
    _xt_exit, _yt_exit = tractor_xy(_st_exit)
    wp_s2_start = [float(_xt_exit[0]), float(_yt_exit[0])]
    wp_s2_end   = [
        WP_END[0] + (LF + LB) * np.cos(HEADING_OUT),
        WP_END[1] + (LF + LB) * np.sin(HEADING_OUT),
    ]
    v_corner_exit = float(max(0.0, res_corner_opt['v'][-1]))
    print()
    print("=" * 60)
    print(f"Step 2b: EulerJLAPCoverage - segment 2 (tractor) "
          f"(V_entry={v_corner_exit:.3f} -> 0)")
    print(f"  start = ({wp_s2_start[0]:.3f}, {wp_s2_start[1]:.3f})  "
          f"end = ({wp_s2_end[0]:.3f}, {wp_s2_end[1]:.3f})")
    print("=" * 60)
    res_s2 = _run_jlap_seg(wp_s2_start, wp_s2_end,
                            initial_vel=v_corner_exit, final_vel=0.0)
    print(f"  T = {res_s2['time'][-1]:.3f} s   "
          f"v_entry = {v_corner_exit:.3f} m/s")

    # ------------------------------------------------------------------
    # Step 4: PurePursuit closed-loop tracking (tractor reference)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 4: PurePursuit - closed-loop tracking (energy-opt corner)")
    print("=" * 60)
    ref_traj = _build_reference_trajectory(res_s1, res_s2, res_corner_opt)
    sim_pp   = _run_purepursuit(ref_traj)
    trk_end  = sim_pp.x_out[:2, -1]
    ref_end  = ref_traj.x[-1, :2]
    print(f"  Reference waypoints : {len(ref_traj.x)}")
    print(f"  Simulation steps    : {sim_pp.x_out.shape[1]}")
    print(f"  Final position error: {np.hypot(*(trk_end - ref_end)) * 1e3:.1f} mm")
    _print_tracking_summary(ref_traj, sim_pp, mB)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    _fig1_segmented_path(seg_info, res_s1, res_s2, res_corner_opt)
    _fig2_jlap_profiles(res_s1, res_s2)
    _fig3_corner_comparison(seg_info, res_corner_time, res_corner_opt, mA, mB)
    _fig4_stitched_trajectory(res_s1, res_s2, res_corner_time, res_corner_opt)
    _fig5_purepursuit(ref_traj, sim_pp)

    plt.show()
    plt.close('all')


# =============================================================================
# FIGURE 1 - Segmented path overview
# =============================================================================
def _draw_coverage_background(ax):
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


def _draw_tt_body(ax, states, n_poses=6, color='dimgray', alpha=0.5):
    """Draw tractor-trailer body sketches at evenly spaced poses."""
    step = max(1, len(states) // n_poses)
    for s in states[::step]:
        xt, yt = tractor_xy(s[np.newaxis, :])[0][0], tractor_xy(s[np.newaxis, :])[1][0]
        x_tr, y_tr, θ_tr, γ = s[0], s[1], s[2], s[3]
        # hitch link
        ax.plot([x_tr, xt], [y_tr, yt], '-', color=color, lw=0.8, alpha=alpha, zorder=3)


def _fig1_segmented_path(seg_info, res_s1, res_s2, res_corner_opt):
    fig, ax = plt.subplots(figsize=(8, 8),
                           num='Figure 1 - Segmented Path (Tractor-Trailer)')
    ax.set_aspect('equal')

    _draw_coverage_background(ax)

    ref = np.array([WP_START, WP_CORNER, WP_END])
    ax.plot(ref[:, 0], ref[:, 1], 'o', color='black', markersize=7, zorder=6)

    s1  = res_s1['states']
    sc  = res_corner_opt['states']
    s2  = res_s2['states']
    arc = seg_info['arc_world']

    # Trailer path
    ax.plot(s1[:, 0], s1[:, 1], '-', color=COL_S1, lw=2.0,
            label='Segment 1 (JLAP, trailer)', zorder=4)
    ax.plot(sc[:, 0], sc[:, 1], '-', color=COL_C_B, lw=2.0,
            label='Corner (B-spline energy-opt, trailer)', zorder=4)
    ax.plot(s2[:, 0], s2[:, 1], '-', color=COL_S2, lw=2.0,
            label='Segment 2 (JLAP, tractor)', zorder=4)

    # Tractor path for corner
    xt_c, yt_c = tractor_xy(sc)
    ax.plot(xt_c, yt_c, '--', color=COL_C_B, lw=1.5, alpha=0.7,
            label='Corner (tractor)', zorder=4)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)

    # Inset: corner detail
    xi1, xi2 = -0.8, 2.5
    yi1, yi2 = 7.5, 10.5
    axins = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
    axins.set_xlim(xi1, xi2)
    axins.set_ylim(yi1, yi2)
    axins.set_aspect('equal')

    ae, ax_e       = seg_info['arc_entry_world'],     seg_info['arc_exit_world']
    ae_ext, ax_ext = seg_info['arc_entry_ext_world'], seg_info['arc_exit_ext_world']

    axins.plot(arc[:, 0], arc[:, 1], '-.', color='purple', lw=1.8, zorder=3,
               label=f'PathSeg arc  R={seg_info["R"]:.2f} m')
    axins.plot(*ae,     's', color='purple', ms=6, zorder=7)
    axins.plot(*ax_e,   's', color='purple', ms=6, zorder=7)
    axins.plot(*ae_ext, 'D', color='dimgray', ms=5, zorder=7, label='Handoff pts')
    axins.plot(*ax_ext, 'D', color='dimgray', ms=5, zorder=7)

    mask_s1 = s1[:, 1] >= yi1
    if mask_s1.any():
        axins.plot(s1[mask_s1, 0], s1[mask_s1, 1], '-', color=COL_S1, lw=2.0, zorder=4)

    axins.plot(sc[:, 0], sc[:, 1], '-', color=COL_C_B, lw=2.0, zorder=4)
    axins.plot(xt_c, yt_c, '--', color=COL_C_B, lw=1.2, alpha=0.7, zorder=4)
    # hitch links at a few poses
    _draw_tt_body(axins, sc, n_poses=5, color='dimgray', alpha=0.4)

    mask_s2 = s2[:, 0] <= xi2
    if mask_s2.any():
        axins.plot(s2[mask_s2, 0], s2[mask_s2, 1], '-', color=COL_S2, lw=2.0, zorder=4)

    axins.set_xlabel('x [m]', fontsize=9)
    axins.set_ylabel('y [m]', fontsize=9)
    axins.tick_params(labelsize=8)
    axins.legend(fontsize=7, loc='lower right')

    ax.indicate_inset_zoom(axins, edgecolor='black', lw=1.2)

    fig.tight_layout()
    _savefig(fig, 'tt_fig_overview.png')


# =============================================================================
# FIGURE 2 - JLAP kinematic profiles
# =============================================================================
def _compute_wheel_kinematics_jlap(res):
    r = JLAP_ROBOT_PARAMS['wheel_radius']
    l = 0.5 * JLAP_ROBOT_PARAMS['robot_width']
    t = res['time']
    v, omega = res['v'], res['omega']
    omega_r = (v + l * omega) / r
    omega_l = (v - l * omega) / r
    alpha_r = (res['acc_path'] + l * res['alpha']) / r
    alpha_l = (res['acc_path'] - l * res['alpha']) / r
    jerk_r  = np.gradient(alpha_r, JLAP_DT) * r
    jerk_l  = np.gradient(alpha_l, JLAP_DT) * r
    return dict(time=t, omega_r=omega_r, omega_l=omega_l,
                alpha_r=alpha_r, alpha_l=alpha_l, jerk_r=jerk_r, jerk_l=jerk_l)


def _fig2_jlap_profiles(res_s1, res_s2):
    r = JLAP_ROBOT_PARAMS['wheel_radius']
    wk1 = _compute_wheel_kinematics_jlap(res_s1)
    wk2 = _compute_wheel_kinematics_jlap(res_s2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             num='Figure 2 - JLAP Kinematic Profiles')

    for col, (wk, color, title) in enumerate([
        (wk1, COL_S1, 'Segment 1 - Left Wheel'),
        (wk2, COL_S2, 'Segment 2 - Left Wheel'),
    ]):
        t = wk['time']
        axes[0, col].plot(t, wk['omega_l'] * r, '-', color=color, lw=1.8, label='v [m/s]')
        axes[0, col].plot(t, wk['alpha_l'] * r, '--', color=color, lw=1.8, label='a [m/s²]')
        axes[0, col].plot(t, wk['jerk_l'],      ':',  color=color, lw=1.8, label='j [m/s³]')
        axes[0, col].axhline(0, color='lightgray', lw=0.8)
        axes[0, col].set_xlabel('time [s]')
        axes[0, col].set_ylabel('m/s  /  m/s²  /  m/s³')
        axes[0, col].set_title(title)
        axes[0, col].legend(fontsize=9)

    for col, (wk, color, title) in enumerate([
        (wk1, COL_S1, 'Segment 1 - Right Wheel'),
        (wk2, COL_S2, 'Segment 2 - Right Wheel'),
    ]):
        t = wk['time']
        axes[1, col].plot(t, wk['omega_r'] * r, '-', color=color, lw=1.8, label='v [m/s]')
        axes[1, col].plot(t, wk['alpha_r'] * r, '--', color=color, lw=1.8, label='a [m/s²]')
        axes[1, col].plot(t, wk['jerk_r'],      ':',  color=color, lw=1.8, label='j [m/s³]')
        axes[1, col].axhline(0, color='lightgray', lw=0.8)
        axes[1, col].set_xlabel('time [s]')
        axes[1, col].set_ylabel('m/s  /  m/s²  /  m/s³')
        axes[1, col].set_title(title)
        axes[1, col].legend(fontsize=9)

    fig.tight_layout()
    _savefig(fig, 'tt_fig_jlap.png')


# =============================================================================
# FIGURE 3 - Corner comparison: time-opt vs energy-opt
# =============================================================================
def _fig3_corner_comparison(seg_info, res_time, res_opt, mA, mB):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10),
                             num='Figure 3 - Corner Comparison (TT)')

    arc = seg_info['arc_world']
    ae_ext = seg_info['arc_entry_ext_world']
    ax_ext = seg_info['arc_exit_ext_world']
    corner_wps = np.array(_build_corner_waypoints(ae_ext, ax_ext))

    xt_A, yt_A = tractor_xy(res_time['states'])
    xt_B, yt_B = tractor_xy(res_opt['states'])

    # ---- [0,0] Time-optimal XY ----
    ax = axes[0, 0]
    ax.set_aspect('equal')
    ax.plot(arc[:, 0], arc[:, 1], '-.', color='purple', lw=1.5,
            label=f'PathSeg arc  R={seg_info["R"]:.2f} m', zorder=2)
    ax.plot(corner_wps[:, 0], corner_wps[:, 1], 'o--', color=COL_REF,
            ms=6, lw=1.0, label='Waypoints', zorder=3)
    s_A = res_time['states']
    ax.plot(s_A[:, 0], s_A[:, 1], '-', color=COL_C_A, lw=2.0,
            label='Trailer', zorder=4)
    ax.plot(xt_A, yt_A, '--', color=COL_C_A, lw=1.5, alpha=0.75,
            label='Tractor', zorder=4)
    _draw_tt_body(ax, s_A, n_poses=6, color=COL_C_A, alpha=0.4)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'Time-optimal  (w_e=0)\n'
                 f'T={mA["total_time"]:.2f} s   E={mA["energy"]:.2f} J')
    ax.legend(fontsize=9)

    # ---- [0,1] Energy-optimal XY ----
    ax = axes[0, 1]
    ax.set_aspect('equal')
    ax.plot(arc[:, 0], arc[:, 1], '-.', color='purple', lw=1.5,
            label=f'PathSeg arc', zorder=2)
    ax.plot(corner_wps[:, 0], corner_wps[:, 1], 'o--', color=COL_REF,
            ms=6, lw=1.0, label='Waypoints', zorder=3)
    s_B = res_opt['states']
    ax.plot(s_B[:, 0], s_B[:, 1], '-', color=COL_C_B, lw=2.0,
            label='Trailer', zorder=4)
    ax.plot(xt_B, yt_B, '--', color=COL_C_B, lw=1.5, alpha=0.75,
            label='Tractor', zorder=4)
    _draw_tt_body(ax, s_B, n_poses=6, color=COL_C_B, alpha=0.4)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'Energy-aware  (w_e={W_ENERGY})\n'
                 f'T={mB["total_time"]:.2f} s   E={mB["energy"]:.2f} J')
    ax.legend(fontsize=9)

    # ---- [1,0] Power profiles ----
    ax = axes[1, 0]
    ax.plot(mA['time_ocp'], mA['P_total'], '-', color=COL_C_A, lw=1.8,
            label=f"Time-opt   peak={mA['peak_power']:.1f} W")
    ax.plot(mB['time_ocp'], mB['P_total'], '-', color=COL_C_B, lw=1.8,
            label=f"Energy-opt peak={mB['peak_power']:.1f} W")
    ax.axhline(2.0, color=COL_REF, ls=':', lw=1.0, label='P_elec = 2.0 W')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('Power [W]')
    ax.set_title('Corner Power Profile')
    ax.legend(fontsize=9)

    # ---- [1,1] Hitch angle ----
    ax = axes[1, 1]
    ax.plot(res_time['time'], np.rad2deg(res_time['states'][:, 3]),
            '-', color=COL_C_A, lw=1.8, label='Time-optimal')
    ax.plot(res_opt['time'],  np.rad2deg(res_opt['states'][:, 3]),
            '-', color=COL_C_B, lw=1.8, label='Energy-aware')
    ax.axhline( np.rad2deg(GAMMA_MAX), color='red', ls=':', lw=1.0)
    ax.axhline(-np.rad2deg(GAMMA_MAX), color='red', ls=':', lw=1.0,
               label=f'±γ_max = ±{np.rad2deg(GAMMA_MAX):.1f}°')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('Hitch angle γ [deg]')
    ax.set_title('Hitch Angle vs Time')
    ax.legend(fontsize=9)

    fig.tight_layout()
    _savefig(fig, 'tt_fig_corner.png')


# =============================================================================
# FIGURE 4 - Full stitched trajectory
# =============================================================================
def _fig4_stitched_trajectory(res_s1, res_s2, res_corner_time, res_corner_opt):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                             num='Figure 4 - Full Stitched Trajectory (TT)')

    T1 = float(res_s1['time'][-1])
    Tc = float(res_corner_opt['time'][-1])

    # Interpolate corner states onto IK time grid for smooth XY plotting
    t_ocp = res_corner_opt['time']
    t_ik  = res_corner_opt['time_ik']
    sc    = res_corner_opt['states']
    x_c   = np.interp(t_ik, t_ocp, sc[:, 0])
    y_c   = np.interp(t_ik, t_ocp, sc[:, 1])
    γ_c   = np.interp(t_ik, t_ocp, sc[:, 3])
    θ_c   = np.interp(t_ik, t_ocp, sc[:, 2])
    xt_c  = x_c + LF * np.cos(θ_c) + LB * np.cos(θ_c - γ_c)
    yt_c  = y_c + LF * np.sin(θ_c) + LB * np.sin(θ_c - γ_c)

    # ---- [0] Full XY (energy-opt) ----
    ax = axes[0]
    ax.set_aspect('equal')
    ref = np.array([WP_START, WP_CORNER, WP_END])
    ax.plot(ref[:, 0], ref[:, 1], '--', color=COL_REF, lw=1.2,
            label='Reference L-path', zorder=2)
    s1 = res_s1['states']
    s2 = res_s2['states']
    ax.plot(s1[:, 0], s1[:, 1], '-', color=COL_S1, lw=2.0,
            label='Segment 1 (JLAP)', zorder=4)
    ax.plot(x_c, y_c, '-', color=COL_C_B, lw=2.0,
            label='Corner (B-spline, trailer)', zorder=4)
    ax.plot(xt_c, yt_c, '--', color=COL_C_B, lw=1.5, alpha=0.7,
            label='Corner (tractor)', zorder=4)
    ax.plot(s2[:, 0], s2[:, 1], '-', color=COL_S2, lw=2.0,
            label='Segment 2 (JLAP, tractor)', zorder=4)
    ae, ax_e = res_s1['states'][-1, :2], res_s2['states'][0, :2]
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Stitched Trajectory (energy-opt corner)')
    ax.legend(fontsize=9)

    # ---- [1] Stitched velocity ----
    ax = axes[1]
    corner_v = res_corner_opt['v']
    ax.plot(res_s1['time'], res_s1['v'],
            '-', color=COL_S1, lw=1.8, label='Segment 1 (JLAP)')
    ax.plot(t_ik + T1, corner_v,
            '-', color=COL_C_B, lw=1.8, label='Corner opt (tractor v)')
    ax.plot(res_corner_time['time_ik'] + T1,
            res_corner_time['v'], '--', color=COL_C_A, lw=1.4,
            label='Corner time-opt (tractor v)')
    ax.plot(res_s2['time'] + T1 + Tc, res_s2['v'],
            '-', color=COL_S2, lw=1.8, label='Segment 2 (JLAP)')
    ax.axvline(T1,      color='black', ls=':', lw=1.0)
    ax.axvline(T1 + Tc, color='black', ls=':', lw=1.0)
    ax.set_xlabel('time [s]  (segments stitched)')
    ax.set_ylabel('v [m/s]')
    ax.set_title('Stitched Velocity Profile')
    ax.legend(fontsize=9)

    # ---- [2] Hitch angle comparison ----
    ax = axes[2]
    ax.plot(res_corner_time['time'] + T1,
            np.rad2deg(res_corner_time['states'][:, 3]),
            '-', color=COL_C_A, lw=1.8, label=f'Time-opt (w_e=0)')
    ax.plot(res_corner_opt['time'] + T1,
            np.rad2deg(res_corner_opt['states'][:, 3]),
            '-', color=COL_C_B, lw=1.8, label=f'Energy-opt (w_e={W_ENERGY})')
    ax.axhline( np.rad2deg(GAMMA_MAX), color='red', ls=':', lw=1.0)
    ax.axhline(-np.rad2deg(GAMMA_MAX), color='red', ls=':', lw=1.0,
               label=f'±γ_max = ±{np.rad2deg(GAMMA_MAX):.1f}°')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('Hitch angle γ [deg]')
    ax.set_title('Hitch Angle (corner phase)')
    ax.legend(fontsize=9)

    fig.tight_layout()
    _savefig(fig, 'tt_fig_stitched.png')


# =============================================================================
# PURE PURSUIT TRACKING
# =============================================================================
def _build_reference_trajectory(res_s1, res_s2, res_corner_opt, dt=0.05):
    """Stitch segments into a uniform-dt TRACTOR reference trajectory.

    Segment 1 (JLAP, trailer path) is offset by (LF+LB) in HEADING_IN so
    the tractor trajectory is continuous at all segment junctions.
    Tractor heading = theta_trailer - gamma (derived from the hitch angle).
    """
    T1    = float(res_s1['time'][-1])
    t_ocp = res_corner_opt['time']
    t_ik  = res_corner_opt['time_ik']
    sc    = res_corner_opt['states']
    Tc    = float(t_ocp[-1])

    # S1: tractor offset from trailer by (LF+LB) in HEADING_IN direction
    off_x = (LF + LB) * np.cos(HEADING_IN)
    off_y = (LF + LB) * np.sin(HEADING_IN)
    s1_x  = res_s1['states'][:, 0] + off_x
    s1_y  = res_s1['states'][:, 1] + off_y
    s1_th = res_s1['states'][:, 2]

    # Corner: tractor XY + heading interpolated onto IK grid
    x_tr = np.interp(t_ik, t_ocp, sc[:, 0])
    y_tr = np.interp(t_ik, t_ocp, sc[:, 1])
    θ_tr = np.interp(t_ik, t_ocp, sc[:, 2])
    γ_tr = np.interp(t_ik, t_ocp, sc[:, 3])
    xt_c = x_tr + LF * np.cos(θ_tr) + LB * np.cos(θ_tr - γ_tr)
    yt_c = y_tr + LF * np.sin(θ_tr) + LB * np.sin(θ_tr - γ_tr)
    th_c = θ_tr - γ_tr   # tractor heading

    # S2: JLAP was already planned for the tractor
    s2_x  = res_s2['states'][:, 0]
    s2_y  = res_s2['states'][:, 1]
    s2_th = res_s2['states'][:, 2]

    t_raw  = np.concatenate([res_s1['time'], t_ik + T1, res_s2['time'][1:] + T1 + Tc])
    x_raw  = np.concatenate([s1_x,  xt_c,  s2_x[1:]])
    y_raw  = np.concatenate([s1_y,  yt_c,  s2_y[1:]])
    th_raw = np.concatenate([s1_th, th_c,  s2_th[1:]])
    v_raw  = np.concatenate([res_s1['v'],     res_corner_opt['v'],     res_s2['v'][1:]])
    w_raw  = np.concatenate([res_s1['omega'], res_corner_opt['omega'], res_s2['omega'][1:]])

    t_uni  = np.arange(t_raw[0], t_raw[-1], dt)
    x_uni  = np.interp(t_uni, t_raw, x_raw)
    y_uni  = np.interp(t_uni, t_raw, y_raw)
    th_uni = np.interp(t_uni, t_raw, th_raw)
    v_uni  = np.interp(t_uni, t_raw, v_raw)
    w_uni  = np.interp(t_uni, t_raw, w_raw)

    states   = np.column_stack([x_uni, y_uni, th_uni])
    controls = np.vstack([v_uni, w_uni])
    return Trajectory(x=states, u=controls, t=t_uni, sampling_time=dt)


def _run_purepursuit(ref_traj):
    model      = DifferentialDrive(wheel_base=TRACTOR_WHEEL_BASE)
    sim        = TimeStepping(model, float(ref_traj.t[-1]), ref_traj.sampling_time)
    controller = PurePursuit(model, ref_traj)
    sim.run_with_controller(list(ref_traj.x[0]), ref_traj, controller)
    return sim


# =============================================================================
# TRACKING SUMMARY TABLE
# =============================================================================
def _print_tracking_summary(ref_traj, sim_pp, corner_metrics):
    """Side-by-side table: reference trajectory metrics vs PP tracking metrics."""
    lbl_w, col_w = 25, 20
    sep = '+' + '-' * lbl_w + '+' + ('-' * col_w + '+') * 2

    def _cell(v, fmt):
        return '---'.rjust(col_w - 2) if v is None else fmt.format(v)

    def _row(lbl, vals, fmts):
        cells = ''.join(f'| {_cell(v, f)} ' for v, f in zip(vals, fmts))
        return f'| {lbl:<{lbl_w - 2}} {cells}|'

    def _hrow(lbl, vals):
        cells = ''.join(f'| {str(v).center(col_w - 2)} ' for v in vals)
        return f'| {lbl:<{lbl_w - 2}} {cells}|'

    ref_xy = ref_traj.x[:, :2]
    trk_xy = sim_pp.x_out[:2, :].T
    nt     = min(ref_traj.x.shape[0], sim_pp.x_out.shape[1])
    cte    = np.array([np.min(np.hypot(ref_xy[:, 0] - pt[0], ref_xy[:, 1] - pt[1]))
                       for pt in trk_xy])
    he     = sim_pp.x_out[2, :nt] - ref_traj.x[:nt, 2]
    he     = np.arctan2(np.sin(he), np.cos(he))
    v_err  = np.abs(ref_traj.u[0, :nt] - sim_pp.u_out[0, :nt])

    F3 = '{:>18.3f}'
    F2 = '{:>18.2f}'
    F4 = '{:>18.4f}'

    print()
    print(sep)
    print(_hrow('', ['Reference', 'Tracked (PP)']))
    print(sep)
    print(_row('mission time [s]',
               [float(ref_traj.t[-1]),         float(sim_pp.t_out[-1])], [F3, F3]))
    print(_row('total energy [J]',
               [corner_metrics['energy'],       None],                    [F2, F2]))
    print(_row('peak power [W]',
               [corner_metrics['peak_power'],   None],                    [F2, F2]))
    print(sep)
    print(_row('mean CTE [cm]',
               [None, float(cte.mean() * 1e2)],                          [F3, F3]))
    print(_row('mean heading err [deg]',
               [None, float(np.rad2deg(np.abs(he).mean()))],              [F3, F3]))
    print(_row('max vel error [m/s]',
               [None, float(v_err.max())],                                [F4, F4]))
    print(sep)


# =============================================================================
# FIGURE 5 - PurePursuit tracking
# =============================================================================
def _fig5_purepursuit(ref_traj, sim):
    t      = sim.t_out
    ref_xy = ref_traj.x[:, :2]
    trk_xy = sim.x_out[:2, :].T
    cte    = np.array([
        np.min(np.hypot(ref_xy[:, 0] - pt[0], ref_xy[:, 1] - pt[1]))
        for pt in trk_xy
    ])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             num='Figure 5 - PurePursuit Tracking (Tractor)')

    # ---- [0] XY ----
    ax = axes[0]
    ax.set_aspect('equal')
    ax.plot(ref_traj.x[:, 0], ref_traj.x[:, 1], '--', color=COL_REF, lw=1.5,
            label='Reference (tractor)', zorder=2)
    ax.plot(trk_xy[:, 0], trk_xy[:, 1], '-', color=COL_C_B, lw=2.0,
            label='PurePursuit tracked', zorder=3)
    ax.plot(*ref_traj.x[0, :2],  'o', color='green', ms=8, zorder=4, label='Start')
    ax.plot(*ref_traj.x[-1, :2], 's', color='red',   ms=8, zorder=4, label='Goal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('XY Tracking (Tractor Reference)')
    ax.legend(fontsize=9)

    # ---- [1] Velocity ----
    ax = axes[1]
    nt = min(ref_traj.x.shape[0], sim.u_out.shape[1])
    ax.plot(ref_traj.t[:nt], ref_traj.u[0, :nt], '--', color=COL_REF, lw=1.5,
            label='Reference v')
    ax.plot(t[:nt], sim.u_out[0, :nt], '-', color=COL_C_B, lw=1.8,
            label='Tracked v')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('v [m/s]')
    ax.set_title('Linear Velocity')
    ax.legend(fontsize=9)

    # ---- [2] Cross-track error ----
    ax = axes[2]
    ax.plot(t[:len(cte)], cte * 1e3, '-', color='tomato', lw=1.8)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('CTE [mm]')
    ax.set_title('Cross-Track Error')
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    _savefig(fig, 'tt_fig_purepursuit.png')


# =============================================================================
if __name__ == '__main__':
    main()
