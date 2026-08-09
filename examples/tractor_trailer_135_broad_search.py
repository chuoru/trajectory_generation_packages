#!/usr/bin/env python3
##
# @file tractor_trailer_135_broad_search.py
#
# @brief Genuinely broader multi-start search for the 135-deg corner's
# energy-aware (w_e=0.5) trajectory.
#
# The dominance/consistency check in tractor_trailer_hitch_angle_repair.py
# found that the 135-deg corner's reported energy-aware result was beaten
# by its own time-optimal trajectory evaluated under the energy-aware
# weight -- i.e. no distinct energy-reducing trajectory had actually been
# found for this corner angle. That check used a modest budget (cold +
# 3 jittered restarts + the existing candidate). This script tries a
# substantially larger and more diverse candidate pool to test whether a
# genuinely better trajectory exists but was simply not found yet, or
# whether the flat result reflects real problem structure at this angle:
#
#   - 1 standard cold start
#   - 10 independent jittered restarts from the cold start, with jitter
#     magnitude randomized per restart (not a single fixed scale) to widen
#     the region explored
#   - cross-warm-starts from the already-solved 45-deg and 90-deg
#     energy-aware trajectories' control points (dimensionally compatible,
#     since every angle config uses the same spline discretization; using
#     them as a warm start for 135-deg is a deliberately different, if
#     geometrically mismatched, starting guess intended to land in a
#     different attraction basin)
#   - the current best (time-optimal-collapsed) result itself, for
#     reference
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/08/09

import sys
import os
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tractor_trailer_paper_sweeps import (
    common_kwargs, make_waypoints, _make_gen, BASE_LB, BASE_LF, GAMMA_MAX,
    OUT_DIR, TRAJ_DIR,
)

W_ENERGY = 0.5
N_RANDOM_JITTER = 10


def _load_traj(name):
    d = np.load(TRAJ_DIR / f'{name}.npz', allow_pickle=True)
    return {k: d[k] for k in d.files if k != 'meta'}


def _obj(traj, w_time, w_energy):
    return w_time * float(traj['time'][-1]) + w_energy * float(traj['energy'])


def main():
    waypoints = make_waypoints(np.deg2rad(135.0))
    kw = common_kwargs(waypoints, BASE_LB, BASE_LF, GAMMA_MAX)

    candidates = []  # (obj, res, tag)

    def _try(warm, tag):
        gen = _make_gen(kw, W_ENERGY, w_time=1.0)
        try:
            res = gen.generate_trajectory(warm_start=warm)
            o = _obj(res, 1.0, W_ENERGY)
            candidates.append((o, res, tag))
            print(f"  {tag:24s} -> obj={o:.4f}  T={res['time'][-1]:.3f}  E={res['energy']:.3f}")
        except Exception as e:
            print(f"  {tag:24s} -> FAILED ({e})")

    print("=== 135-deg energy-aware (w_e=0.5) broad search ===\n")

    print("Cold start:")
    _try(None, 'cold')
    base = candidates[0][1] if candidates else None

    print(f"\n{N_RANDOM_JITTER} randomized jittered restarts:")
    rng = np.random.default_rng(42)
    if base is not None:
        for k in range(N_RANDOM_JITTER):
            jitter_frac = rng.uniform(0.05, 0.40)
            scale = jitter_frac * (np.abs(base['ctrl_pts']).mean(axis=0, keepdims=True) + 0.05)
            cp = base['ctrl_pts'] + rng.normal(scale=scale, size=base['ctrl_pts'].shape)
            _try({'ctrl_pts': cp, 'time': base['time']}, f'jitter{k}(frac={jitter_frac:.2f})')

    print("\nCross-warm-starts from other angles' energy-aware trajectories:")
    for src_name in ['traj_angle_45_energy', 'traj_angle_90_energy']:
        try:
            src = _load_traj(src_name)
        except FileNotFoundError:
            print(f"  {src_name}: not found, skipping")
            continue
        _try({'ctrl_pts': src['ctrl_pts'], 'time': src['time']}, f'cross-{src_name}')

    print("\nExisting best-known (post dominance-check) result:")
    try:
        existing = _load_traj('traj_angle135_energy_repaired')
    except FileNotFoundError:
        existing = _load_traj('traj_angle_135_energy')
    candidates.append((_obj(existing, 1.0, W_ENERGY), existing, 'existing'))
    print(f"  existing                 -> obj={candidates[-1][0]:.4f}  "
          f"T={existing['time'][-1]:.3f}  E={existing['energy']:.3f}")

    candidates.sort(key=lambda c: c[0])
    best_obj, best_res, best_tag = candidates[0]

    print(f"\n=== Result: {len(candidates)} candidates tried ===")
    print(f"Best: {best_tag}, obj={best_obj:.4f}, T={best_res['time'][-1]:.3f}, "
          f"E={best_res['energy']:.3f}")

    time_opt = _load_traj('traj_angle_135_timeopt')
    time_opt_obj = _obj(time_opt, 1.0, W_ENERGY)
    print(f"Time-optimal trajectory scored under w_e=0.5: {time_opt_obj:.4f} "
          f"(T={time_opt['time'][-1]:.3f}, E={time_opt['energy']:.3f})")

    if best_obj < time_opt_obj - 1e-6:
        print(f"\n*** FOUND a genuinely better energy-aware trajectory: "
              f"{time_opt_obj:.4f} -> {best_obj:.4f} ***")
        found_better = True
    else:
        print(f"\nNo candidate among {len(candidates)} tried beat the time-optimal "
              f"trajectory's own score under w_e=0.5. This strengthens (but does not "
              f"prove) the case that the flat result reflects real problem structure "
              f"at this corner angle rather than search inadequacy.")
        found_better = False

    st = best_res['states']
    plen = float(np.sum(np.sqrt(np.diff(st[:, 0]) ** 2 + np.diff(st[:, 1]) ** 2)))
    gamma = st[:, 3]
    out = {
        'n_candidates': len(candidates),
        'found_better': found_better,
        'best_tag': best_tag,
        'best_obj': float(best_obj),
        'time_optimal_obj_under_we': float(time_opt_obj),
        'result': {
            'total_time': float(best_res['time'][-1]),
            'path_length': plen,
            'energy': float(best_res['energy']),
            'energy_per_m': float(best_res['energy']) / plen if plen > 1e-9 else float('inf'),
            'peak_power': float(np.max(best_res['power'])),
            'avg_power': float(np.mean(best_res['power'])),
            'peak_gamma_deg': float(np.max(np.abs(gamma)) * 180.0 / np.pi),
            'jackknife_margin': float(1.0 - np.max(np.abs(gamma)) / GAMMA_MAX),
        },
        'all_candidates': [{'tag': t, 'obj': float(o)} for o, _, t in candidates],
    }
    with open(OUT_DIR / 'sweep_angle135_broad_search.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    if found_better:
        np.savez(TRAJ_DIR / 'traj_angle135_energy_broadsearch.npz', **best_res)
    print(f"\nSaved -> {OUT_DIR / 'sweep_angle135_broad_search.json'}")


if __name__ == '__main__':
    main()
