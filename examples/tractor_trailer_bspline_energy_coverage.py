#!/usr/bin/env python3
##
# @file tractor_trailer_bspline_energy_coverage.py
#
# @brief Energy-aware B-Spline coverage for a tractor-trailer robot.
#
# Compares a time-optimal trajectory against an energy-aware one for the same
# L-shaped coverage path.  The optimal w_energy is found automatically by
# sweeping log-spaced values and selecting the Pareto knee in
# (total energy, peak motor power) space — the same pipeline used in
# differential_drive_path_segment_combined.py.
#
# Pipeline
# --------
#   1. Sweep w_energy log-spaced [0, 0.01 ... 10] high→low with warm-starting.
#   2. Apply outlier filtering (5× median mission time; 1.5× median peak power
#      or energy) to discard non-converged solves.
#   3. Select Pareto knee (max perpendicular distance from chord) as the
#      energy-aware operating point.
#   4. Reuse sweep-cached results — no re-solve needed for the comparison.
#
# Figures
# -------
#   Figure 1 -- XY trajectories: trailer + tractor (side-by-side comparison)
#   Figure 2 -- Tractor velocity and angular velocity
#   Figure 3 -- Power profile + energy/time bar chart
#   Figure 4 -- Hitch angle vs time
#   Figure 5 -- w_energy sweep statistics (peak power, energy, time, d2)
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/06/22

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trajectory_generators.bspline_energy_tractor_trailer_coverage import (
    BSplineEnergyTractorTrailerCoverage,
)

# ---------------------------------------------------------------------------
# Waypoints for the TRAILER rear axle  [x, y, theta_trailer]
# ---------------------------------------------------------------------------
WAYPOINTS = [
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [2.0, 2.0, np.pi / 2],
]

LB = 0.2   # tractor rear axle -> hitch [m]
LF = 0.8   # hitch -> trailer rear axle [m]

COMMON = dict(
    waypoints=WAYPOINTS,
    bound=0.17,
    n_ctrl_pts=5,
    spline_order=3,
    n_sampling=20,
    vel_max=[0.2, 0.2, 0.196],
    vel_min_lin=0.01,
    eps_nonh=0.001,
    eps_hitch=0.05,
    length_back=LB,
    length_front=LF,
    gamma_max=0.785,
    gamma_entry=0.0,
    gamma_exit=0.0,
    acc_max=[5.0, 5.0, 4.0],
    jerk_max=[50.0, 50.0, 20.0],
    p_electronics=2.0,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tractor_xy(states):
    """Derive tractor XY position from trailer states and hitch angle."""
    gamma = states[:, 3]
    theta = states[:, 2]
    xt = states[:, 0] + LF * np.cos(theta) + LB * np.cos(theta - gamma)
    yt = states[:, 1] + LF * np.sin(theta) + LB * np.sin(theta - gamma)
    return xt, yt


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


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def _sweep_we():
    """Sweep w_energy log-spaced [0, 0.01 ... 10] and identify Pareto knee.

    Sweeps high to low for warm-starting.  Applies two-pass outlier filtering
    and selects the Pareto knee via max perpendicular distance from the chord
    connecting the two extreme Pareto-optimal endpoints.

    @return dict with sweep arrays, opt_we, we_time_ref, and res_by_we cache.
    """
    we_all   = np.concatenate([[0.0], np.logspace(-2, 1, 12)])
    we_sweep = we_all[::-1]   # high -> low for warm-starting

    peak_powers    = []
    total_energies = []
    mission_times  = []
    we_valid       = []
    res_by_we      = {}

    print()
    print("=" * 60)
    print(f"w_energy sweep  ({len(we_all)} IPOPT solves, "
          f"w_e in [0, 0.01 ... 10]) ...")
    print("=" * 60)
    print(f"  {'w_e':>10}  {'Time [s]':>10}  {'Energy [J]':>10}  "
          f"{'Peak P [W]':>10}")
    print("  " + "-" * 49)

    prev_res = None
    for w_e in we_sweep:
        try:
            gen = BSplineEnergyTractorTrailerCoverage(
                **COMMON, w_time=1.0, w_energy=float(w_e))
            res    = gen.generate_trajectory(warm_start=prev_res)
            peak_p = float(np.max(res['power']))
            energy = float(res['energy'])
            T      = float(res['time'][-1])
            if not (np.isfinite(peak_p) and np.isfinite(energy)):
                raise ValueError("non-finite result")
            prev_res = res
            res_by_we[float(w_e)] = res
            peak_powers.append(peak_p)
            total_energies.append(energy)
            mission_times.append(T)
            we_valid.append(float(w_e))
            print(f"  {w_e:>10.6f}  {T:>10.3f}  {energy:>10.3f}  {peak_p:>10.3f}")
        except Exception as exc:
            print(f"  {w_e:>10.6f}  skipped ({exc})")
            prev_res = None

    if len(peak_powers) < 2:
        raise RuntimeError(
            f"Sweep produced {len(peak_powers)} feasible result(s); "
            "need at least 2 to select an optimal w_e."
        )

    # Sort ascending by w_e for gradient analysis and plotting.
    sort_idx       = np.argsort(we_valid)
    peak_powers    = np.array(peak_powers)[sort_idx]
    total_energies = np.array(total_energies)[sort_idx]
    mission_times  = np.array(mission_times)[sort_idx]
    we_values      = np.array(we_valid)[sort_idx]

    # --- Drop non-converged outliers: mission time > 5× median ---------------
    if len(mission_times) >= 3:
        t_med = np.median(mission_times)
        valid = mission_times <= 5.0 * t_med
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

    # --- Drop power/energy outliers: > 1.5× median ---------------------------
    if len(peak_powers) >= 3:
        pp_med = np.median(peak_powers)
        te_med = np.median(total_energies)
        valid  = ~((peak_powers > 1.5 * pp_med) | (total_energies > 1.5 * te_med))
        if valid.sum() < len(peak_powers):
            n_drop = (~valid).sum()
            print(f"  NOTE: dropping {n_drop} outlier(s) "
                  f"(peak P > 1.5× median or energy > 1.5× median).")
            peak_powers    = peak_powers[valid]
            total_energies = total_energies[valid]
            mission_times  = mission_times[valid]
            we_values      = we_values[valid]
            for w_e in list(res_by_we):
                if w_e not in we_values:
                    del res_by_we[w_e]

    d1 = np.gradient(peak_powers, we_values)
    d2 = np.gradient(d1, we_values)

    # --- Time-optimal and energy-optimal anchors -----------------------------
    time_ref_idx   = int(np.argmin(mission_times))
    we_time_ref    = float(we_values[time_ref_idx])
    energy_opt_idx = int(np.argmin(total_energies))

    # --- Pareto knee: max perpendicular distance from chord ------------------
    _pf = _pareto_front_idx(total_energies, peak_powers)
    if len(_pf) >= 3:
        _te_n = ((total_energies - total_energies.min())
                 / max(float(total_energies.max() - total_energies.min()), 1e-12))
        _pp_n = ((peak_powers - peak_powers.min())
                 / max(float(peak_powers.max() - peak_powers.min()), 1e-12))
        _ax, _ay = _te_n[_pf[0]],  _pp_n[_pf[0]]
        _bx, _by = _te_n[_pf[-1]], _pp_n[_pf[-1]]
        _denom   = max(float(np.hypot(_bx - _ax, _by - _ay)), 1e-12)
        _pf_mid  = _pf[1:-1]
        _dist    = (np.abs((_by - _ay) * (_te_n[_pf_mid] - _ax)
                           - (_bx - _ax) * (_pp_n[_pf_mid] - _ay)) / _denom)
        opt_idx  = int(_pf_mid[np.argmax(_dist)])
    elif len(_pf) >= 1:
        opt_idx  = int(_pf[len(_pf) // 2])
    else:
        opt_idx  = energy_opt_idx
    if float(we_values[opt_idx]) == 0.0 and len(we_values) > 1:
        opt_idx  = 1
    opt_we = float(we_values[opt_idx])

    if we_time_ref != 0.0:
        print(f"  NOTE: w_e=0.0 did not yield minimum time; "
              f"using w_e={we_time_ref:.4f} as time reference.")
    print()
    print(f"  Time-optimal   w_e = {we_time_ref:.6f}  "
          f"T = {mission_times[time_ref_idx]:.3f} s")
    print(f"  Pareto knee    w_e = {opt_we:.6f}  "
          f"Peak P = {peak_powers[opt_idx]:.3f} W  "
          f"E = {total_energies[opt_idx]:.3f} J")
    print(f"  Energy-optimal w_e = {float(we_values[energy_opt_idx]):.6f}  "
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
        'res_by_we':      res_by_we,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _fig1_trajectories(res_time, res_ener, xt_time, yt_time,
                        xt_ener, yt_ener, we_time_ref, opt_we):
    """XY trajectories: time-optimal vs energy-aware."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True,
                             num='Figure 1 - Tractor-Trailer Trajectories')
    BOUND = float(COMMON['bound'])
    wps   = np.array(WAYPOINTS)

    for ax, res, xt, yt, label in [
        (axes[0], res_time, xt_time, yt_time,
         f'Time-Optimal  (w_e={we_time_ref:.4f})'),
        (axes[1], res_ener, xt_ener, yt_ener,
         f'Energy-Aware  (w_e={opt_we:.4f})'),
    ]:
        for seg in range(len(wps) - 1):
            A = wps[seg, :2]; B = wps[seg + 1, :2]
            d = (B - A) / np.linalg.norm(B - A)
            n = np.array([-d[1], d[0]])
            corners = np.array([
                A - d*BOUND + n*BOUND, A - d*BOUND - n*BOUND,
                B + d*BOUND - n*BOUND, B + d*BOUND + n*BOUND,
                A - d*BOUND + n*BOUND,
            ])
            ax.fill(corners[:, 0], corners[:, 1], alpha=0.12, color='steelblue')
            ax.plot(corners[:-1, 0], corners[:-1, 1], 'b--', lw=0.7)

        s     = res['states']
        t_ocp = res['time']
        ax.plot(s[:, 0], s[:, 1], 'k-',  lw=2,   label='Trailer')
        ax.plot(xt, yt,            'r--', lw=1.5, label='Tractor')
        ax.plot(wps[:, 0], wps[:, 1], 'ko', ms=5)
        for i in range(0, len(t_ocp), max(1, len(t_ocp) // 10)):
            ax.plot([s[i, 0], xt[i]], [s[i, 1], yt[i]], 'g-', lw=0.7, alpha=0.5)

        ax.set_aspect('equal')
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.set_title(f'{label}\nT={t_ocp[-1]:.2f} s  E={res["energy"]:.2f} J')
        ax.legend(fontsize=8)
        ax.grid(True, ls=':', alpha=0.5)

    fig.suptitle('Tractor-Trailer B-Spline Coverage Comparison')
    fig.tight_layout()


def _fig2_velocity_profiles(res_time, res_ener):
    """Tractor velocity and angular velocity."""
    fig, (ax_v, ax_w) = plt.subplots(2, 1, figsize=(8, 5),
                                      num='Figure 2 - Tractor Velocity Profiles')
    for res, lbl, ls in [(res_time, 'Time-opt', '-'),
                          (res_ener, 'Energy',   '--')]:
        t = res['time_ik']
        ax_v.plot(t, res['v'],                 ls, label=f'v_trac ({lbl})')
        ax_w.plot(t, np.rad2deg(res['omega']), ls, label=f'omega_trac ({lbl})')
    ax_v.set_ylabel('v_trac [m/s]'); ax_v.legend(); ax_v.grid(True, ls=':')
    ax_w.set_ylabel('omega_trac [deg/s]'); ax_w.set_xlabel('time [s]')
    ax_w.legend(); ax_w.grid(True, ls=':')
    fig.suptitle('Tractor Velocity Profiles'); fig.tight_layout()


def _fig3_power_energy(res_time, res_ener):
    """Power profile and mission metrics bar chart."""
    fig, (ax_p, ax_b) = plt.subplots(2, 1, figsize=(8, 5),
                                      num='Figure 3 - Power and Energy')
    for res, lbl, ls in [(res_time, 'Time-opt', '-'),
                          (res_ener, 'Energy',   '--')]:
        ax_p.plot(res['time'], res['power'], ls, label=lbl)
    ax_p.set_ylabel('Power [W]'); ax_p.set_xlabel('time [s]')
    ax_p.legend(); ax_p.grid(True, ls=':')

    cats     = ['Time-optimal', 'Energy-aware']
    energies = [res_time['energy'],   res_ener['energy']]
    times    = [res_time['time'][-1], res_ener['time'][-1]]
    x = np.arange(2)
    ax_b.bar(x - 0.2, energies, 0.35, label='Energy [J]', color='steelblue')
    ax_b.bar(x + 0.2, times,    0.35, label='Time [s]',   color='coral')
    ax_b.set_xticks(x); ax_b.set_xticklabels(cats)
    ax_b.legend(); ax_b.grid(True, ls=':', axis='y')
    fig.suptitle('Power Profile and Mission Metrics'); fig.tight_layout()


def _fig4_hitch_angle(res_time, res_ener):
    """Hitch angle vs time for both solutions."""
    fig, ax = plt.subplots(figsize=(8, 3), num='Figure 4 - Hitch Angle')
    for res, lbl, ls in [(res_time, 'Time-opt', '-'),
                          (res_ener, 'Energy',   '--')]:
        ax.plot(res['time'], np.rad2deg(res['states'][:, 3]), ls, label=lbl)
    gmax_deg = np.rad2deg(COMMON['gamma_max'])
    ax.axhline( gmax_deg, color='r', ls=':', lw=1)
    ax.axhline(-gmax_deg, color='r', ls=':', lw=1,
               label=f'±gamma_max = {gmax_deg:.1f} deg')
    ax.set_xlabel('time [s]'); ax.set_ylabel('gamma [deg]')
    ax.set_title('Hitch Angle vs Time'); ax.legend(); ax.grid(True, ls=':')
    fig.tight_layout()


def _fig5_sweep(sweep):
    """Four-panel w_energy sweep statistics."""
    we     = sweep['we_values']
    pp     = sweep['peak_powers']
    te     = sweep['total_energies']
    mt     = sweep['mission_times']
    d2     = sweep['d2_pp']
    oi     = sweep['opt_idx']
    opt_we = sweep['opt_we']
    tri    = sweep['time_ref_idx']
    we_ref = sweep['we_time_ref']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             num='Figure 5 - w_energy Sweep')
    fig.suptitle('Figure 5 — w_energy Sweep  (w_time = 1.0)', fontsize=13)

    specs = [
        (axes[0, 0], pp, 'o-', 'tomato',     'Peak Power [W]',
         'Peak Power Suppression'),
        (axes[0, 1], te, 's-', 'steelblue',  'Total Energy [J]',
         'Total Energy vs w_energy'),
        (axes[1, 0], mt, '^-', 'seagreen',   'Mission Time [s]',
         'Mission Time vs w_energy'),
        (axes[1, 1], d2, 'D-', 'darkorange', 'd2(Peak P)/d(w_e)2',
         '2nd Derivative (informational)'),
    ]

    for ax, data, marker, color, ylabel, title in specs:
        ax.plot(we, data, marker, color=color, linewidth=1.8, markersize=4,
                label=ylabel)
        ax.axvline(opt_we, color='black',     ls='--', lw=1.2,
                   label=f'Pareto knee w_e={opt_we:.4f}')
        ax.axvline(we_ref, color='steelblue', ls=':',  lw=1.2,
                   label=f'Time-ref w_e={we_ref:.4f}')
        ax.scatter([opt_we], [data[oi]],  color='black',     s=80, zorder=6)
        ax.scatter([we_ref], [data[tri]], color='steelblue',  s=60,
                   marker='^', zorder=5)
        if ax is axes[1, 1]:
            ax.axhline(0, color='gray', lw=0.8, ls=':')
        ax.set_xscale('symlog', linthresh=1e-2)
        ax.set_xlabel('w_energy (symlog scale)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.tight_layout()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Step 1: sweep to find Pareto knee -----------------------------------
    sweep = _sweep_we()
    opt_we      = sweep['opt_we']
    we_time_ref = sweep['we_time_ref']
    res_by_we   = sweep['res_by_we']

    # --- Step 2: extract both solutions from sweep cache (no re-solve) -------
    res_time = res_by_we[we_time_ref]
    if opt_we in res_by_we:
        res_ener = res_by_we[opt_we]
    else:
        print(f"  Re-solving energy-aware (w_e={opt_we:.4f}, warm from time-opt) ...")
        gen_ener = BSplineEnergyTractorTrailerCoverage(
            **COMMON, w_time=1.0, w_energy=opt_we)
        res_ener = gen_ener.generate_trajectory(warm_start=res_time)

    xt_time, yt_time = tractor_xy(res_time['states'])
    xt_ener, yt_ener = tractor_xy(res_ener['states'])

    # --- Print comparison table ----------------------------------------------
    print()
    print(f"{'':20s} {'Time [s]':>10s} {'Energy [J]':>12s} {'Peak P [W]':>12s}")
    print(f"{'Time-optimal':20s} {res_time['time'][-1]:10.2f} "
          f"{res_time['energy']:12.3f} "
          f"{float(np.max(res_time['power'])):12.3f}")
    print(f"{'Energy-aware':20s} {res_ener['time'][-1]:10.2f} "
          f"{res_ener['energy']:12.3f} "
          f"{float(np.max(res_ener['power'])):12.3f}")

    # --- Figures -------------------------------------------------------------
    _fig1_trajectories(res_time, res_ener,
                       xt_time, yt_time, xt_ener, yt_ener,
                       we_time_ref, opt_we)
    _fig2_velocity_profiles(res_time, res_ener)
    _fig3_power_energy(res_time, res_ener)
    _fig4_hitch_angle(res_time, res_ener)
    _fig5_sweep(sweep)

    plt.show()


if __name__ == '__main__':
    main()
