#!/usr/bin/env python3
##
# @file tractor_trailer_hitch_angle_repair.py
#
# @brief Extends the Pareto-sweep dominance/consistency validation
# (tractor_trailer_pareto_repair.py) to the hitch-geometry and corner-angle
# sweeps.
#
# Unlike the Pareto sweep, where every point shares the same feasible
# region (only the objective weight changes), each hitch geometry and each
# corner angle is a DISTINCT feasible region (different hitch-coupling ODE
# coefficients, or different waypoints/corridor), so trajectories cannot be
# cross-pollinated between configs the way they were between Pareto weights.
# Two checks are still meaningful and applied per config:
#
#   1. A broader multi-start budget (cold + 3 jittered restarts, plus the
#      previously-saved trajectory as a candidate) replaces the original
#      1-jitter budget, giving more chances to escape a poor local optimum.
#   2. A within-config weighted-sum consistency check: since the
#      time-optimal and energy-aware legs of the SAME config share the same
#      feasible region (only w_e differs, exactly as in the Pareto sweep),
#      does the time-optimal leg's own trajectory score better under the
#      energy-aware weight than the officially reported energy-aware
#      result, or vice versa? A direct warm-start cross-attempt (energy-
#      aware solve warm-started from the time-optimal leg) is also tried.
#      A sanity check flags the impossible case of the energy-aware leg's
#      mission time being LOWER than the time-optimal leg's own -- if that
#      happens, "time-optimal" was not actually time-optimal.
#
# The baseline (90 deg, lb=0.2/lf=0.8) rows of both sweeps are NOT re-solved
# here: they are the literal same feasible region and weight as points
# already validated by the Pareto-sweep repair, so they are reconciled
# directly against that result instead.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/08/08

import sys
import os
import json
import pathlib
import hashlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tractor_trailer_paper_sweeps import (
    common_kwargs, make_waypoints, _make_gen, BASE_LB, BASE_LF, GAMMA_MAX,
    OUT_DIR, TRAJ_DIR,
)

W_ENERGY = 0.5
N_JITTER_BROAD = 3


def _stable_seed(*parts):
    """Deterministic seed independent of Python's per-process string-hash
    randomization. hash(tuple_containing_str) is NOT reproducible run-to-run
    unless PYTHONHASHSEED is fixed externally; this is."""
    h = hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31)


def _load_traj(name):
    d = np.load(TRAJ_DIR / f'{name}.npz', allow_pickle=True)
    return {k: d[k] for k in d.files if k != 'meta'}


def _obj(traj, w_time, w_energy):
    return w_time * float(traj['time'][-1]) + w_energy * float(traj['energy'])


def _broad_multistart(kw, w_time, w_energy, existing_traj=None,
                       n_jitter=N_JITTER_BROAD, seed=0, tag=''):
    rng = np.random.default_rng(seed)
    candidates = []

    def _try(warm, ctag):
        gen = _make_gen(kw, w_energy, w_time=w_time)
        try:
            res = gen.generate_trajectory(warm_start=warm)
            candidates.append((_obj(res, w_time, w_energy), res, ctag))
        except Exception:
            pass

    _try(None, f'{tag}cold')
    base = candidates[0][1] if candidates else existing_traj
    if base is not None:
        for k in range(n_jitter):
            scale = 0.15 * (np.abs(base['ctrl_pts']).mean(axis=0, keepdims=True) + 0.05)
            cp = base['ctrl_pts'] + rng.normal(scale=scale, size=base['ctrl_pts'].shape)
            _try({'ctrl_pts': cp, 'time': base['time']}, f'{tag}jitter{k}')
    if existing_traj is not None:
        candidates.append((_obj(existing_traj, w_time, w_energy), existing_traj, f'{tag}existing'))

    if not candidates:
        raise RuntimeError(f"all attempts failed for {tag!r}")
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1], candidates[0][0], candidates[0][2], len(candidates)


def process_config(name, waypoints, lb, lf, existing_t_name, existing_e_name):
    print(f"\n=== Config: {name} (lb={lb}, lf={lf}) ===")
    kw = common_kwargs(waypoints, lb, lf, GAMMA_MAX)

    existing_t = _load_traj(existing_t_name)
    existing_e = _load_traj(existing_e_name)

    print(f"  Broad multi-start: time-optimal leg")
    res_t, obj_t, tag_t, n_t = _broad_multistart(
        kw, 1.0, 0.0, existing_traj=existing_t, seed=_stable_seed(name, 't'), tag='t-')
    print(f"    -> best={obj_t:.4f} ({tag_t}), {n_t} candidates tried "
          f"(was T={existing_t['time'][-1]:.3f} -> now T={res_t['time'][-1]:.3f})")

    print(f"  Broad multi-start: energy-aware leg (w_e={W_ENERGY})")
    res_e, obj_e, tag_e, n_e = _broad_multistart(
        kw, 1.0, W_ENERGY, existing_traj=existing_e, seed=_stable_seed(name, 'e'), tag='e-')
    print(f"    -> best={obj_e:.4f} ({tag_e}), {n_e} candidates tried "
          f"(was E={existing_e['energy']:.3f} -> now E={res_e['energy']:.3f})")

    # Explicit cross-attempt: energy-aware warm-started directly from the
    # (possibly improved) time-optimal leg.
    gen_cross = _make_gen(kw, W_ENERGY, w_time=1.0)
    try:
        res_cross = gen_cross.generate_trajectory(warm_start=res_t)
        obj_cross = _obj(res_cross, 1.0, W_ENERGY)
        if obj_cross < obj_e - 1e-6:
            print(f"  Cross warm-start from time-optimal leg improved energy-aware: "
                  f"{obj_e:.4f} -> {obj_cross:.4f}")
            res_e, obj_e, tag_e = res_cross, obj_cross, 't-crosswarm'
    except Exception:
        pass

    # Within-config weighted-sum consistency check (same idea as the
    # Pareto-sweep repair, but only 2 points sharing this feasible region).
    inconsistent = False
    t_score_under_e_weight = _obj(res_t, 1.0, W_ENERGY)
    if t_score_under_e_weight < obj_e - 1e-6:
        print(f"  INCONSISTENT: time-opt trajectory scores {t_score_under_e_weight:.4f} "
              f"under w_e={W_ENERGY}, beats energy-aware's own {obj_e:.4f} -- adopting it")
        res_e, obj_e = res_t, t_score_under_e_weight
        inconsistent = True

    e_score_under_t_weight = float(res_e['time'][-1])  # w_e=0 objective is just T
    if e_score_under_t_weight < float(res_t['time'][-1]) - 1e-6:
        print(f"  INCONSISTENT: energy-aware trajectory has LOWER mission time "
              f"({e_score_under_t_weight:.3f}) than the time-optimal leg's own "
              f"({res_t['time'][-1]:.3f}) -- time-optimal was not actually time-optimal; adopting it")
        res_t = res_e
        inconsistent = True

    return res_t, res_e, inconsistent


def metrics(traj):
    st = traj['states']
    plen = float(np.sum(np.sqrt(np.diff(st[:, 0]) ** 2 + np.diff(st[:, 1]) ** 2)))
    gamma = st[:, 3]
    return {
        'total_time': float(traj['time'][-1]),
        'path_length': plen,
        'energy': float(traj['energy']),
        'energy_per_m': float(traj['energy']) / plen if plen > 1e-9 else float('inf'),
        'peak_power': float(np.max(traj['power'])),
        'avg_power': float(np.mean(traj['power'])),
        'peak_gamma_deg': float(np.max(np.abs(gamma)) * 180.0 / np.pi),
        'jackknife_margin': float(1.0 - np.max(np.abs(gamma)) / GAMMA_MAX),
    }


if __name__ == '__main__':
    configs = [
        ('short', make_waypoints(np.pi / 2), 0.15, 0.40,
         'traj_hitch_short_timeopt', 'traj_hitch_short_energy'),
        ('long', make_waypoints(np.pi / 2), 0.30, 1.20,
         'traj_hitch_long_timeopt', 'traj_hitch_long_energy'),
        ('angle45', make_waypoints(np.deg2rad(45.0)), BASE_LB, BASE_LF,
         'traj_angle_45_timeopt', 'traj_angle_45_energy'),
        ('angle135', make_waypoints(np.deg2rad(135.0)), BASE_LB, BASE_LF,
         'traj_angle_135_timeopt', 'traj_angle_135_energy'),
    ]

    out = {}
    any_inconsistent = False
    for name, waypoints, lb, lf, tname, ename in configs:
        res_t, res_e, inconsistent = process_config(name, waypoints, lb, lf, tname, ename)
        any_inconsistent = any_inconsistent or inconsistent
        mt, me = metrics(res_t), metrics(res_e)
        out[name] = {'time_optimal': mt, 'energy_aware': me, 'inconsistent': inconsistent}
        np.savez(TRAJ_DIR / f'traj_{name}_timeopt_repaired.npz', **res_t)
        np.savez(TRAJ_DIR / f'traj_{name}_energy_repaired.npz', **res_e)
        print(f"  FINAL [{name}] time-opt: T={mt['total_time']:.3f} E={mt['energy']:.3f} "
              f"|g|max={mt['peak_gamma_deg']:.2f} margin={mt['jackknife_margin']*100:.1f}%")
        print(f"  FINAL [{name}] energy-aware: T={me['total_time']:.3f} E={me['energy']:.3f} "
              f"|g|max={me['peak_gamma_deg']:.2f} margin={me['jackknife_margin']*100:.1f}%")

    out['any_inconsistent'] = any_inconsistent
    with open(OUT_DIR / 'sweep_hitch_angle_repaired.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved -> {OUT_DIR / 'sweep_hitch_angle_repaired.json'}")
    print(f"Any inconsistency found and fixed: {any_inconsistent}")
