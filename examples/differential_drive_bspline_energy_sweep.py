#!/usr/bin/env python3
##
# @file differential_drive_bspline_energy_sweep.py
#
# @brief Dense w_energy sweep and Pareto analysis for the differential drive
# B-spline energy coverage problem.
#
# Produces:
#   Figure 4 — Time–Energy Pareto Front
#               8-point (w_time, w_energy) sweep with E/m as marker colour
#   Figure 5 — Dense w_energy Sweep
#               Peak power suppression, total energy, mission time,
#               2nd-derivative elbow detection across 26 log-spaced w_e values
#   Figure 6 — Sweep: XY Trajectory Overlay + Heading Profile
#   Figure 7 — Sweep: Kinematic & Power Profile Overlay
#
# For the simple two-run comparison (time-optimal vs energy-aware), see
# differential_drive_bspline_energy_coverage.py.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/04/16

# Standard library
import sys
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Internal library
from trajectory_generators.bspline_energy_coverage import BSplineEnergyCoverage


# =============================================================================
# PAPER FIGURE EXPORT
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
# Shared configuration  (mirrors differential_drive_bspline_energy_coverage.py)
# =============================================================================
WAYPOINTS = [
    [0.0, 0.0,  0.0],
    [1.0, 0.0,  0.0],
    [1.0, 1.0,  np.pi / 2],
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

ROBOT_PARAMS = {'l': 0.53 / 2, 'r': 0.3}

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
# =============================================================================
def _draw_polyhedra_corridors(ax, waypoints, bound,
                               color='gold', alpha=0.18,
                               edgecolor='steelblue', lw=0.8):
    patches = []
    for i in range(len(waypoints) - 1):
        A = np.array(waypoints[i][:2], dtype=float)
        B = np.array(waypoints[i + 1][:2], dtype=float)

        d = B - A
        d = d / np.linalg.norm(d)
        n = np.array([-d[1], d[0]])

        corners = np.array([
            B + d * bound + n * bound,
            B + d * bound - n * bound,
            A - d * bound - n * bound,
            A - d * bound + n * bound,
        ])
        patches.append(MplPolygon(corners, closed=True))

    col = PatchCollection(patches, facecolor=color, alpha=alpha,
                          edgecolor=edgecolor, linewidth=lw, zorder=1)
    ax.add_collection(col)
    ax.fill([], [], color=color, alpha=alpha + 0.15, edgecolor=edgecolor,
            linewidth=lw, label=f'Corridor (bound={bound:.2f} m)')


# =============================================================================
# Power computation
# =============================================================================
def _compute_wheel_power(v_arr, omega_arr, robot_params, cr, cl, p_elec):
    """Evaluate the polynomial power model on IK-output (v, omega) arrays.

    @return (P_total, P_right, P_left, energy) all arrays + scalar [J].
    """
    l  = robot_params['l']
    dt = 0.01

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
    energy  = float(np.trapz(P_total, dx=dt))
    return P_total, P_r, P_l, energy


# =============================================================================
# Main
# =============================================================================
def main():
    _set_paper_style()

    opt_we = _fig5_we_sweep()   # Figures 5, 6, 7 — returns optimal w_energy
    _fig4_pareto()              # Figure 4

    print()
    print(f"Optimal w_energy identified by sweep: {opt_we:.4f}")
    print("Use this value as W_ENERGY in differential_drive_bspline_energy_coverage.py")

    plt.show()
    plt.close('all')


# =============================================================================
# Figure 4 — Time–Energy Pareto Front
# =============================================================================
def _fig4_pareto():
    """Sweep w_energy / w_time to trace the Pareto front.

    Each marker is coloured by energy-per-meter efficiency [J/m].
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
        T_total  = float(res['time'][-1])

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
        avg_p = energies[i] / times[i] if times[i] > 0 else float('nan')
        print(f"  {w_t:>6.2f}  {w_e:>8.2f}  "
              f"{times[i]:>10.3f}  {energies[i]:>10.3f}  "
              f"{eff[i]:>14.4f}  {avg_p:>10.3f}")
    print()

    fig, ax = plt.subplots(figsize=(7, 5),
                           num='Figure 4 — Time–Energy Pareto Front')

    sc = ax.scatter(times, energies, c=eff, cmap='viridis_r', s=90, zorder=5)
    ax.plot(times, energies, '-', color='gray', linewidth=1.2, zorder=4, alpha=0.6)

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
        energy  = float(res['energy'])

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
    we_values      = np.array(we_values)

    # --- Drop non-converged outliers: mission time > 5× median ---------------
    if len(mission_times) >= 3:
        t_med = np.median(mission_times)
        valid = mission_times <= 5.0 * t_med
        if valid.sum() < len(mission_times):
            n_drop = (~valid).sum()
            print(f"  NOTE: dropping {n_drop} non-converged sweep point(s) "
                  f"(mission time > 5× median={t_med:.2f} s).")
            idx_v = np.where(valid)[0]
            peak_powers    = peak_powers[valid]
            total_energies = total_energies[valid]
            mission_times  = mission_times[valid]
            we_values      = we_values[valid]
            all_states     = [all_states[i]     for i in idx_v]
            all_time_traj  = [all_time_traj[i]  for i in idx_v]
            all_time_ik    = [all_time_ik[i]    for i in idx_v]
            all_v          = [all_v[i]          for i in idx_v]
            all_omega      = [all_omega[i]       for i in idx_v]
            all_P_tot      = [all_P_tot[i]       for i in idx_v]
            all_P_r        = [all_P_r[i]         for i in idx_v]
            all_P_l        = [all_P_l[i]         for i in idx_v]

    # --- Drop power/energy outliers: > 1.5× median ---------------------------
    if len(peak_powers) >= 3:
        pp_med = np.median(peak_powers)
        te_med = np.median(total_energies)
        valid  = ~((peak_powers > 1.5 * pp_med) | (total_energies > 1.5 * te_med))
        if valid.sum() < len(peak_powers):
            n_drop = (~valid).sum()
            print(f"  NOTE: dropping {n_drop} outlier(s) "
                  f"(peak P > 1.5× median or energy > 1.5× median).")
            idx_v = np.where(valid)[0]
            peak_powers    = peak_powers[valid]
            total_energies = total_energies[valid]
            mission_times  = mission_times[valid]
            we_values      = we_values[valid]
            all_states     = [all_states[i]     for i in idx_v]
            all_time_traj  = [all_time_traj[i]  for i in idx_v]
            all_time_ik    = [all_time_ik[i]    for i in idx_v]
            all_v          = [all_v[i]          for i in idx_v]
            all_omega      = [all_omega[i]       for i in idx_v]
            all_P_tot      = [all_P_tot[i]       for i in idx_v]
            all_P_r        = [all_P_r[i]         for i in idx_v]
            all_P_l        = [all_P_l[i]         for i in idx_v]

    # --- Second derivative (informational, plotted in Panel 1,1) --------------
    d1_pp = np.gradient(peak_powers, we_values)
    d2_pp = np.gradient(d1_pp,       we_values)

    # --- Pareto knee: max perpendicular distance from chord -------------------
    _pf = _pareto_front_idx(total_energies, peak_powers)
    if len(_pf) >= 3:
        _te_n = (total_energies - total_energies.min()) / max(float(total_energies.max() - total_energies.min()), 1e-12)
        _pp_n = (peak_powers    - peak_powers.min())    / max(float(peak_powers.max()    - peak_powers.min()),    1e-12)
        _ax, _ay = _te_n[_pf[0]],  _pp_n[_pf[0]]
        _bx, _by = _te_n[_pf[-1]], _pp_n[_pf[-1]]
        _denom   = max(float(np.hypot(_bx - _ax, _by - _ay)), 1e-12)
        _pf_mid  = _pf[1:-1]
        _dist    = np.abs((_by - _ay) * (_te_n[_pf_mid] - _ax) - (_bx - _ax) * (_pp_n[_pf_mid] - _ay)) / _denom
        opt_idx  = int(_pf_mid[np.argmax(_dist)])
    elif len(_pf) >= 1:
        opt_idx  = int(_pf[len(_pf) // 2])
    else:
        opt_idx  = int(np.argmin(total_energies))
    if float(we_values[opt_idx]) == 0.0 and len(we_values) > 1:
        opt_idx  = 1
    opt_we = float(we_values[opt_idx])

    time_ref_idx   = int(np.argmin(mission_times))
    energy_opt_idx = int(np.argmin(total_energies))

    print()
    print(f"  Time-optimal   w_e = {float(we_values[time_ref_idx]):.6f}  "
          f"T = {mission_times[time_ref_idx]:.3f} s")
    print(f"  Pareto knee    w_e = {opt_we:.6f}  "
          f"Peak P = {peak_powers[opt_idx]:.3f} W  "
          f"E = {total_energies[opt_idx]:.3f} J")
    print(f"  Energy-optimal w_e = {float(we_values[energy_opt_idx]):.6f}  "
          f"E = {total_energies[energy_opt_idx]:.3f} J")
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
               label=f'Pareto knee w_e = {opt_we:.4f}')
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
               label=f'Pareto knee w_e = {opt_we:.4f}')
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
               label=f'Pareto knee w_e = {opt_we:.4f}')
    ax.scatter([opt_we], [mission_times[opt_idx]], color='black', s=80, zorder=6)
    ax.set_xlabel('w_energy (log scale)')
    ax.set_ylabel('Mission Time [s]')
    ax.set_title('Mission Time vs w_energy')
    ax.set_xscale('log')
    ax.legend(fontsize=8)

    # Panel (1,1): Second derivative of Peak Power curve (informational)
    ax = axes[1, 1]
    ax.plot(we_values, d2_pp, 'D-', color='darkorange', linewidth=1.8,
            markersize=4, label='d²(Peak P)/d(w_e)²')
    ax.axvline(opt_we, color='black', linestyle='--', linewidth=1.2,
               label=f'Pareto knee w_e = {opt_we:.4f}')
    ax.scatter([opt_we], [d2_pp[opt_idx]], color='black', s=80, zorder=6)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
    ax.set_xlabel('w_energy (log scale)')
    ax.set_ylabel('d²(Peak P) / d(w_e)²  [W]')
    ax.set_title('2nd Derivative (informational)')
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
    cmap   = plt.cm.viridis
    norm   = plt.Normalize(vmin=float(we_values[0]), vmax=float(we_values[-1]))
    wps    = np.array(WAYPOINTS)
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
    cmap   = plt.cm.viridis
    norm   = plt.Normalize(vmin=float(we_values[0]), vmax=float(we_values[-1]))
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
    t_opt  = all_time_ik[opt_idx]
    v_opt  = all_v[opt_idx]
    w_opt  = all_omega[opt_idx]
    P_opt  = all_P_tot[opt_idx]
    Pr_opt = all_P_r[opt_idx]
    Pl_opt = all_P_l[opt_idx]
    E_opt  = np.cumsum(P_opt) * 0.01

    lbl = f'Optimal  w_e = {opt_we:.3f}'
    ax_v.plot(t_opt, v_opt, '-', color='red', lw=2.5, zorder=6, label=lbl)
    ax_w.plot(t_opt, w_opt, '-', color='red', lw=2.5, zorder=6, label=lbl)
    ax_p.plot(t_opt, P_opt, '-', color='red', lw=2.5, zorder=6, label=lbl)
    ax_p.plot(t_opt, Pr_opt, '--', color='red', lw=1.4, zorder=6,
              label='P_right (optimal)')
    ax_p.plot(t_opt, Pl_opt, ':',  color='red', lw=1.4, zorder=6,
              label='P_left  (optimal)')
    ax_e.plot(t_opt, E_opt, '-', color='red', lw=2.5, zorder=6, label=lbl)

    ax_v.set(xlabel='Time [s]', ylabel='v [m/s]',   title='Forward Velocity v(t)')
    ax_w.set(xlabel='Time [s]', ylabel='ω [rad/s]', title='Angular Velocity ω(t)')
    ax_p.set(xlabel='Time [s]', ylabel='P [W]',     title='Total Electrical Power P(t)')
    ax_e.set(xlabel='Time [s]', ylabel='E [J]',     title='Cumulative Energy E(t)')

    for ax in axes.flat:
        ax.legend(fontsize=7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), label='w_energy',
                 fraction=0.02, pad=0.04)

    fig.tight_layout()


# =============================================================================
if __name__ == '__main__':
    main()
