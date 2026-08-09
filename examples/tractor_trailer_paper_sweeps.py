#!/usr/bin/env python3
##
# @file tractor_trailer_paper_sweeps.py
#
# @brief Simulation sweeps for the tractor-trailer energy-aware paper.
#
# Produces the data/figures/tables that the single fixed-scenario example
# scripts (tractor_trailer_bspline_coverage.py,
# tractor_trailer_bspline_energy_coverage.py) do not: a time/energy Pareto
# sweep over w_energy (with knee-point selection), a sweep over hitch/
# wheelbase geometry (lb, lf), and a sweep over corner angle. All three reuse
# BSplineEnergyTractorTrailerCoverage directly (bypassing the PathSegment/
# JLAP straight-segment stages, which are unchanged from the companion
# single-body paper and not the subject of this script) on a single isolated
# corner.
#
# Every solve is a multi-start solve (cold start + jittered restarts +
# neighbor-continuation warm starts where a neighboring configuration has
# already been solved), with the lowest-objective candidate kept and the
# full candidate spread recorded, addressing the single-start local-optimum
# sensitivity documented in earlier runs of this script.
#
# Robot parameters are reconciled with the companion single-body paper's
# actual identified-motor platform (half-wheelbase l=0.265 m, wheel radius
# r=0.15 m) rather than the smaller placeholder values used in some of the
# original example scripts, since the six-coefficient power-model
# coefficients were identified on that specific platform.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/08/08

import sys
import os
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trajectory_generators.bspline_energy_tractor_trailer_coverage import (
    BSplineEnergyTractorTrailerCoverage,
)

# =============================================================================
# OUTPUT LOCATION
# =============================================================================
OUT_DIR = (pathlib.Path(__file__).resolve().parent.parent.parent
           / 'Writting' / 'energy_aware_trailer_tractor' / 'figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Full trajectory solutions (control points, states, commanded v/omega, wheel
# jerk, power) for every solve kept as the "best" multi-start candidate,
# saved separately from the scalar summary metrics in sweep_*.json so a
# physical trial (or an offline path-library deployment) can replay the
# exact planned trajectory later without re-solving the OCP.
TRAJ_DIR = OUT_DIR / 'trajectories'
TRAJ_DIR.mkdir(parents=True, exist_ok=True)


def _savefig(fig, filename):
    out = OUT_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"  [paper] Saved {filename} -> {out}")


def _save_json(data, filename):
    out = OUT_DIR / filename
    with open(out, 'w') as f:
        json.dump(data, f, indent=2, default=float)
    print(f"  [paper] Saved {filename} -> {out}")


def _save_trajectory(res, name, extra_meta=None):
    """Persist a full solved trajectory (control points, states, time,
    tractor v/omega/alpha, per-wheel angular velocity and jerk, power) as a
    .npz file, keyed by a descriptive name, for later experiment playback.
    Scalar/metadata fields (e.g. config parameters) can be attached via
    extra_meta and are stored as 0-d object arrays.
    """
    out = TRAJ_DIR / f'{name}.npz'
    payload = {k: v for k, v in res.items()}
    if extra_meta:
        payload['meta'] = json.dumps(extra_meta)
    np.savez(out, **payload)
    print(f"  [paper] Saved trajectory {name} -> {out}")


# =============================================================================
# RECONCILED ROBOT PARAMETERS (same physical platform as the companion paper)
# =============================================================================
ROBOT_PARAMS = {'l': 0.265, 'r': 0.15}   # half-wheelbase, wheel radius [m]

BASE_LB = 0.2   # tractor rear axle -> hitch [m]
BASE_LF = 0.8   # hitch -> trailer rear axle [m]
GAMMA_MAX = 0.785   # 45 deg jackknife limit [rad]

L1 = 2.0   # entry leg length [m]
L2 = 2.0   # exit leg length [m]


def make_waypoints(beta_rad):
    """Corner waypoints: entry leg along +x, exit leg at heading beta."""
    p0 = [0.0, 0.0, 0.0]
    p1 = [L1, 0.0, 0.0]
    p2 = [L1 + L2 * np.cos(beta_rad), L2 * np.sin(beta_rad), beta_rad]
    return [p0, p1, p2]


### -----------------------------------------------------------------------
### Discretization / solver budget
###
### The energy-aware OCP's wheel-jerk actuation constraint routes third-
### order trailer-spline derivatives through the hitch inverse-kinematics
### chain at every sample node, which makes IPOPT convergence markedly
### slower than the time-optimal (energy-free) OCP. A single cold solve at
### n_sampling=20 (the value used by the repository's own single-scenario
### example scripts) did not converge within a 10000-iteration cap
### (diagnosed separately). n_sampling=10, n_ctrl_pts=5, max_iter=1500 was
### verified to converge to a physically sane solution in ~99 s and is used
### for every solve below. The jackknife bound is now enforced via the
### B-spline convex-hull property (bounding control points directly, see
### BSplineTractorTrailerCoverage._add_jackknife_constraints) rather than
### pointwise sampling, so it holds over the whole spline arc, not just at
### these nt sample nodes.
### -----------------------------------------------------------------------
N_SAMPLING = 10
N_CTRL_PTS = 5
MAX_ITER = 1500
N_JITTER = 1          # jittered restarts per solve, in addition to cold start
JITTER_FRAC = 0.12     # relative perturbation scale for jittered restarts


def common_kwargs(waypoints, lb=BASE_LB, lf=BASE_LF, gamma_max=GAMMA_MAX):
    return dict(
        waypoints=waypoints,
        bound=0.17,
        n_ctrl_pts=N_CTRL_PTS,
        spline_order=3,
        n_sampling=N_SAMPLING,
        vel_max=[0.2, 0.2, 0.196],
        vel_min_lin=0.01,
        eps_nonh=0.001,
        eps_hitch=0.05,
        length_back=lb,
        length_front=lf,
        gamma_max=gamma_max,
        gamma_entry=0.0,
        gamma_exit=0.0,
        acc_max=[5.0, 5.0, 4.0],
        jerk_max=[50.0, 50.0, 20.0],
        p_electronics=2.0,
        robot_params=ROBOT_PARAMS,
    )


def _make_gen(kw, w_energy, w_time=1.0):
    gen = BSplineEnergyTractorTrailerCoverage(**kw, w_time=w_time, w_energy=w_energy)
    gen._optimizer.solver(
        'ipopt', {'print_time': False},
        {'max_iter': MAX_ITER, 'print_level': 0, 'tol': 1e-5,
         'acceptable_tol': 5e-3, 'acceptable_iter': 15,
         'constr_viol_tol': 1e-4, 'hessian_approximation': 'limited-memory'})
    return gen


def _proxy_obj(res, w_time, w_energy):
    """Objective value used to rank multi-start candidates against each
    other. Uses total (not motor-only) energy, which differs from the
    solver's own internal objective by the constant electronics load
    integrated over mission time -- close enough for ranking candidates of
    the SAME configuration/weights against each other."""
    return w_time * res['time'][-1] + w_energy * res['energy']


def multi_start_solve(kw, w_time, w_energy, n_jitter=N_JITTER,
                       extra_warm_starts=None, seed=0, tag=''):
    """Cold start + jittered restarts + any supplied neighbor-continuation
    warm starts; return the lowest-objective candidate plus a diagnostic
    dict describing the full candidate spread, so local-optimum sensitivity
    is visible rather than hidden behind a single silently-chosen result.
    """
    rng = np.random.default_rng(seed)
    candidates = []   # list of (obj, res, tag)
    n_failed = 0

    def _try(warm_start, ctag):
        nonlocal n_failed
        gen = _make_gen(kw, w_energy, w_time=w_time)
        try:
            res = gen.generate_trajectory(warm_start=warm_start)
            candidates.append((_proxy_obj(res, w_time, w_energy), res, ctag))
        except Exception:
            n_failed += 1

    _try(None, f'{tag}cold')

    base = candidates[0][1] if candidates else None
    if base is not None:
        for k in range(n_jitter):
            scale = JITTER_FRAC * (np.abs(base['ctrl_pts']).mean(axis=0,
                                    keepdims=True) + 0.05)
            cp_jit = base['ctrl_pts'] + rng.normal(scale=scale,
                                                    size=base['ctrl_pts'].shape)
            ws = {'ctrl_pts': cp_jit, 'time': base['time']}
            _try(ws, f'{tag}jitter{k}')

    for i, ews in enumerate(extra_warm_starts or []):
        _try(ews, f'{tag}neighbor{i}')

    if not candidates:
        raise RuntimeError(f"all multi-start attempts failed for {tag!r} "
                            f"({n_failed} failures)")

    candidates.sort(key=lambda c: c[0])
    best_obj, best_res, best_tag = candidates[0]
    objs = [c[0] for c in candidates]
    diag = {
        'n_attempts': len(candidates) + n_failed,
        'n_failed': n_failed,
        'n_valid': len(candidates),
        'best_tag': best_tag,
        'best_obj': float(best_obj),
        'worst_obj': float(max(objs)),
        'spread_pct': float((max(objs) - best_obj) / best_obj * 100)
                      if best_obj > 1e-9 else 0.0,
    }
    return best_res, diag


def metrics(res, diag=None):
    st = res['states']
    plen = float(np.sum(np.sqrt(np.diff(st[:, 0]) ** 2 + np.diff(st[:, 1]) ** 2)))
    gamma = st[:, 3]
    m = {
        'total_time':  float(res['time'][-1]),
        'path_length': plen,
        'energy':      float(res['energy']),
        'energy_per_m': float(res['energy']) / plen if plen > 1e-9 else float('inf'),
        'peak_power':  float(np.max(res['power'])),
        'avg_power':   float(np.mean(res['power'])),
        'peak_gamma_deg': float(np.max(np.abs(gamma)) * 180.0 / np.pi),
        'jackknife_margin': float(1.0 - np.max(np.abs(gamma)) / GAMMA_MAX),
    }
    if diag is not None:
        m['multistart'] = diag
    return m


def solve_pair(waypoints, lb=BASE_LB, lf=BASE_LF, gamma_max=GAMMA_MAX,
               w_energy=1.0, neighbor_pair=None):
    """Time-optimal then energy-aware (warm-started), each via multi-start.
    neighbor_pair, if given, is a (res_t, res_e) tuple from an already-
    solved nearby configuration, supplied as an extra continuation
    warm-start candidate for both legs.
    """
    kw = common_kwargs(waypoints, lb, lf, gamma_max)

    extra_t = [neighbor_pair[0]] if neighbor_pair is not None else None
    res_t, diag_t = multi_start_solve(kw, 1.0, 0.0,
                                       extra_warm_starts=extra_t, tag='t-')

    extra_e = [res_t]
    if neighbor_pair is not None:
        extra_e.append(neighbor_pair[1])
    res_e, diag_e = multi_start_solve(kw, 1.0, w_energy,
                                       extra_warm_starts=extra_e, tag='e-')
    return (res_t, diag_t), (res_e, diag_e)


# =============================================================================
# SWEEP 1 -- TIME/ENERGY PARETO (w_energy sweep) + KNEE-POINT SELECTION
# =============================================================================
def _knee_point(energies, peaks):
    """Geometric knee-point detection (Das, 1999): the point on the Pareto
    front with maximum perpendicular distance from the line connecting its
    two extreme points, in min-max-normalized objective space."""
    e = np.asarray(energies, dtype=float)
    p = np.asarray(peaks, dtype=float)
    e_n = (e - e.min()) / (e.max() - e.min() + 1e-12)
    p_n = (p - p.min()) / (p.max() - p.min() + 1e-12)
    x0, y0 = e_n[0], p_n[0]
    x1, y1 = e_n[-1], p_n[-1]
    seg = np.array([x1 - x0, y1 - y0])
    seg_len = np.linalg.norm(seg) + 1e-12
    seg_unit = seg / seg_len
    dists = np.zeros(len(e_n))
    for i in range(len(e_n)):
        pt = np.array([e_n[i] - x0, p_n[i] - y0])
        proj = np.dot(pt, seg_unit) * seg_unit
        perp = pt - proj
        dists[i] = np.linalg.norm(perp)
    return int(np.argmax(dists)), dists


def sweep_pareto():
    print("\n" + "=" * 60)
    print("Sweep 1: time/energy Pareto (w_energy sweep, multi-start)")
    print("=" * 60)
    waypoints = make_waypoints(np.pi / 2)
    kw = common_kwargs(waypoints)

    w_grid = np.linspace(0.0, 1.0, 11)
    results = []
    prev_res = None
    for w in w_grid:
        extra = [prev_res] if prev_res is not None else None
        res, diag = multi_start_solve(kw, 1.0, float(w),
                                       extra_warm_starts=extra, tag=f'w{w:.2f}-')
        prev_res = res
        m = metrics(res, diag)
        m['w_energy'] = float(w)
        results.append(m)
        _save_trajectory(res, f'traj_pareto_we{w:.2f}',
                          extra_meta={'w_energy': float(w), 'lb': BASE_LB,
                                      'lf': BASE_LF, 'beta_deg': 90.0})
        print(f"  w_e={w:5.3f}  T={m['total_time']:6.3f}s  "
              f"E={m['energy']:7.3f}J  Ppeak={m['peak_power']:6.3f}W  "
              f"|gamma|max={m['peak_gamma_deg']:5.2f} deg  "
              f"[{diag['n_valid']}/{diag['n_attempts']} valid, "
              f"spread={diag['spread_pct']:.1f}%, best={diag['best_tag']}]")

    energies = [r['energy'] for r in results]
    peaks = [r['peak_power'] for r in results]
    knee_idx, knee_dists = _knee_point(energies, peaks)
    for i, r in enumerate(results):
        r['knee_distance'] = float(knee_dists[i])
    print(f"\n  Knee point (Das 1999 max-perpendicular-distance): "
          f"w_e={results[knee_idx]['w_energy']:.3f}  "
          f"(E={energies[knee_idx]:.2f} J, Ppeak={peaks[knee_idx]:.2f} W)")

    _save_json({'results': results, 'knee_index': knee_idx}, 'sweep_pareto.json')

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(energies, peaks, c=w_grid, cmap='viridis', s=40, zorder=3)
    ax.plot(energies, peaks, '-', color='gray', lw=0.8, alpha=0.6, zorder=2)
    ax.scatter([energies[knee_idx]], [peaks[knee_idx]], s=180,
               facecolors='none', edgecolors='red', linewidths=2, zorder=4,
               label=f'knee ($w_e$={w_grid[knee_idx]:.2f})')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r'$w_e$')
    ax.set_xlabel('Total motor energy [J]')
    ax.set_ylabel('Peak motor power [W]')
    ax.set_title('Time/Energy Pareto Sweep (90-deg corner, multi-start)')
    ax.legend()
    ax.grid(True, ls=':', alpha=0.5)
    fig.tight_layout()
    _savefig(fig, 'tt_fig_pareto.png')
    plt.close(fig)
    return results, knee_idx


# =============================================================================
# SWEEP 2 -- MULTI HITCH/WHEELBASE (lb, lf) GEOMETRY
# =============================================================================
def sweep_hitch_geometry(w_energy_knee):
    print("\n" + "=" * 60)
    print("Sweep 2: hitch/wheelbase geometry (lb, lf), multi-start")
    print("=" * 60)
    configs = [
        ('short',    0.15, 0.40),
        ('baseline', 0.20, 0.80),
        ('long',     0.30, 1.20),
    ]
    waypoints = make_waypoints(np.pi / 2)
    results = []
    prev_pair = None
    for name, lb, lf in configs:
        (res_t, diag_t), (res_e, diag_e) = solve_pair(
            waypoints, lb=lb, lf=lf, w_energy=w_energy_knee,
            neighbor_pair=prev_pair)
        prev_pair = (res_t, res_e)
        mt, me = metrics(res_t, diag_t), metrics(res_e, diag_e)
        results.append({'config': name, 'lb': lb, 'lf': lf,
                         'time_optimal': mt, 'energy_aware': me})
        meta = {'config': name, 'lb': lb, 'lf': lf, 'beta_deg': 90.0}
        _save_trajectory(res_t, f'traj_hitch_{name}_timeopt', extra_meta=meta)
        _save_trajectory(res_e, f'traj_hitch_{name}_energy',
                          extra_meta={**meta, 'w_energy': w_energy_knee})
        print(f"  [{name:8s}] lb={lb:.2f} lf={lf:.2f}  "
              f"time-opt |g|max={mt['peak_gamma_deg']:5.2f} deg "
              f"(margin {mt['jackknife_margin']*100:5.1f}%, "
              f"spread {diag_t['spread_pct']:.1f}%)  "
              f"energy-aware |g|max={me['peak_gamma_deg']:5.2f} deg "
              f"(margin {me['jackknife_margin']*100:5.1f}%, "
              f"spread {diag_e['spread_pct']:.1f}%)  "
              f"E: {mt['energy']:.2f}->{me['energy']:.2f} J")
    _save_json(results, 'sweep_hitch_geometry.json')
    return results


# =============================================================================
# SWEEP 3 -- MULTI CORNER ANGLE
# =============================================================================
def sweep_corner_angle(w_energy_knee):
    print("\n" + "=" * 60)
    print("Sweep 3: corner angle (45/90/135 deg), multi-start")
    print("=" * 60)
    angles_deg = [45.0, 90.0, 135.0]
    results = []
    prev_pair = None
    for beta_deg in angles_deg:
        waypoints = make_waypoints(np.deg2rad(beta_deg))
        (res_t, diag_t), (res_e, diag_e) = solve_pair(
            waypoints, w_energy=w_energy_knee, neighbor_pair=prev_pair)
        prev_pair = (res_t, res_e)
        mt, me = metrics(res_t, diag_t), metrics(res_e, diag_e)
        results.append({'beta_deg': beta_deg,
                         'time_optimal': mt, 'energy_aware': me})
        meta = {'beta_deg': beta_deg, 'lb': BASE_LB, 'lf': BASE_LF}
        _save_trajectory(res_t, f'traj_angle_{beta_deg:.0f}_timeopt', extra_meta=meta)
        _save_trajectory(res_e, f'traj_angle_{beta_deg:.0f}_energy',
                          extra_meta={**meta, 'w_energy': w_energy_knee})
        print(f"  [beta={beta_deg:5.1f} deg] "
              f"time-opt |g|max={mt['peak_gamma_deg']:5.2f} deg "
              f"(margin {mt['jackknife_margin']*100:5.1f}%, "
              f"spread {diag_t['spread_pct']:.1f}%)  "
              f"energy-aware |g|max={me['peak_gamma_deg']:5.2f} deg "
              f"(margin {me['jackknife_margin']*100:5.1f}%, "
              f"spread {diag_e['spread_pct']:.1f}%)  "
              f"E: {mt['energy']:.2f}->{me['energy']:.2f} J")
    _save_json(results, 'sweep_corner_angle.json')
    return results, angles_deg


# =============================================================================
# JACKKNIFE-MARGIN SUMMARY FIGURE (derived from sweeps 2 and 3)
# =============================================================================
def jackknife_margin_figure(hitch_results, angle_results, angles_deg):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    names = [r['config'] for r in hitch_results]
    m_t = [r['time_optimal']['jackknife_margin'] * 100 for r in hitch_results]
    m_e = [r['energy_aware']['jackknife_margin'] * 100 for r in hitch_results]
    x = np.arange(len(names))
    ax1.bar(x - 0.18, m_t, 0.35, label='Time-optimal', color='tomato')
    ax1.bar(x + 0.18, m_e, 0.35, label='Energy-aware (knee)', color='darkorange')
    ax1.set_xticks(x); ax1.set_xticklabels(names)
    ax1.set_ylabel('Jackknife margin [%]')
    ax1.set_title('Margin vs. hitch/wheelbase geometry (90-deg corner)')
    ax1.legend(); ax1.grid(True, ls=':', axis='y', alpha=0.5)

    m_t2 = [r['time_optimal']['jackknife_margin'] * 100 for r in angle_results]
    m_e2 = [r['energy_aware']['jackknife_margin'] * 100 for r in angle_results]
    ax2.plot(angles_deg, m_t2, 'o-', color='tomato', label='Time-optimal')
    ax2.plot(angles_deg, m_e2, 's--', color='darkorange', label='Energy-aware (knee)')
    ax2.set_xlabel('Corner angle [deg]')
    ax2.set_ylabel('Jackknife margin [%]')
    ax2.set_title('Margin vs. corner angle (baseline geometry)')
    ax2.legend(); ax2.grid(True, ls=':', alpha=0.5)

    fig.tight_layout()
    _savefig(fig, 'tt_fig_jackknife_margin.png')
    plt.close(fig)


if __name__ == '__main__':
    pareto_results, knee_idx = sweep_pareto()
    w_energy_knee = pareto_results[knee_idx]['w_energy']
    print(f"\nUsing knee-selected w_e={w_energy_knee:.3f} for Sweeps 2-3.\n")
    hitch_results = sweep_hitch_geometry(w_energy_knee)
    angle_results, angles_deg = sweep_corner_angle(w_energy_knee)
    jackknife_margin_figure(hitch_results, angle_results, angles_deg)
    print("\nAll sweeps complete.")
