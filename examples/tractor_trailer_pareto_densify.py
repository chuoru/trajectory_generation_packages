#!/usr/bin/env python3
##
# @file tractor_trailer_pareto_densify.py
#
# @brief Densifies the Pareto sweep grid in the previously-unsampled
# w_e in (0, 0.1) interval, to test whether the validated front's collapse
# to two points (Section IV-B/IV-C of the paper) is better explained by
# under-sampled grid resolution than by weighted-sum scalarization's
# inability to reach non-convex regions of the true front.
#
# The original 11-point sweep tested w_e in {0, 0.1, 0.2, ..., 1.0} with no
# points between the w_e=0 (time-optimal, T=20.77, E=117.39) and w_e=0.1
# (dominant regime, T=21.03, E=113.17) results. This adds 9 more points at
# w_e in {0.01, ..., 0.09}, each solved with multi-start (cold + jitter +
# continuation from BOTH the w_e=0 and w_e=0.1 trajectories, since either
# neighbor could plausibly be the better warm start for an intermediate
# weight), then re-runs the full dominance/weighted-sum-consistency check
# against the union of the original 11 points and these 9 new ones.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/08/09

import sys
import os
import json
import hashlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tractor_trailer_paper_sweeps import (
    common_kwargs, make_waypoints, _make_gen, OUT_DIR, TRAJ_DIR,
)

N_JITTER = 2


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


def _multistart(kw, w_time, w_energy, extra_warm_starts, seed, tag):
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
    base = candidates[0][1] if candidates else (extra_warm_starts[0] if extra_warm_starts else None)
    if base is not None:
        for k in range(N_JITTER):
            scale = 0.15 * (np.abs(base['ctrl_pts']).mean(axis=0, keepdims=True) + 0.05)
            cp = base['ctrl_pts'] + rng.normal(scale=scale, size=base['ctrl_pts'].shape)
            _try({'ctrl_pts': cp, 'time': base['time']}, f'{tag}jitter{k}')
    for i, ews in enumerate(extra_warm_starts or []):
        _try(ews, f'{tag}neighbor{i}')

    if not candidates:
        raise RuntimeError(f"all attempts failed for {tag!r}")
    candidates.sort(key=lambda c: c[0])
    best_obj, best_res, best_tag = candidates[0]
    objs = [c[0] for c in candidates]
    diag = {'n_attempts': len(candidates), 'best_tag': best_tag,
            'spread_pct': float((max(objs) - best_obj) / best_obj * 100) if best_obj > 1e-9 else 0.0}
    return best_res, best_obj, diag


def main():
    waypoints = make_waypoints(np.pi / 2)
    kw = common_kwargs(waypoints)

    we0 = _load_traj('traj_pareto_we0.00')
    we1 = _load_traj('traj_pareto_we0.10')

    new_grid = [round(x, 2) for x in np.arange(0.01, 0.10, 0.01)]
    results = []
    print("=== Densified Pareto sweep: w_e in (0, 0.1) ===\n")
    for w in new_grid:
        res, obj, diag = _multistart(
            kw, 1.0, w, extra_warm_starts=[we0, we1], seed=_stable_seed('densify', w),
            tag=f'w{w:.2f}-')
        results.append({'w_energy': w, 'total_time': float(res['time'][-1]),
                         'energy': float(res['energy']), 'obj': float(obj)})
        print(f"  w_e={w:.2f}  T={res['time'][-1]:6.3f}  E={res['energy']:7.3f}  "
              f"obj={obj:.4f}  [{diag['n_attempts']} tried, spread={diag['spread_pct']:.2f}%, "
              f"best={diag['best_tag']}]")
        np.savez(TRAJ_DIR / f'traj_pareto_densify_we{w:.2f}.npz', **res)

    # Build the full pool: original 11 + these 9 new ones, for dominance/
    # consistency checking against everything, not just the new points.
    with open(OUT_DIR / 'sweep_pareto.json') as f:
        original = json.load(f)['results']

    pool = [{'w_energy': r['w_energy'], 'total_time': r['total_time'], 'energy': r['energy']}
            for r in original] + results

    print(f"\n=== Dominance check across full pool ({len(pool)} points) ===")
    dominated = []
    for i, pi in enumerate(pool):
        for j, pj in enumerate(pool):
            if i == j:
                continue
            if (pj['total_time'] <= pi['total_time'] and pj['energy'] <= pi['energy']
                    and (pj['total_time'] < pi['total_time'] or pj['energy'] < pi['energy'])):
                dominated.append((pi['w_energy'], pj['w_energy']))
    if dominated:
        for wi, wj in dominated:
            print(f"  w_e={wi:.2f} DOMINATED by w_e={wj:.2f}")
    else:
        print("  No dominated points in the combined pool.")

    print(f"\n=== Weighted-sum consistency check (new points only, against full pool) ===")
    any_new_better = False
    for r in results:
        we = r['w_energy']
        own = r['total_time'] + we * r['energy']
        best_alt = None
        for p in pool:
            if p['w_energy'] == we and p['total_time'] == r['total_time']:
                continue
            alt = p['total_time'] + we * p['energy']
            if alt < own - 1e-6 and (best_alt is None or alt < best_alt[1]):
                best_alt = (p['w_energy'], alt)
        if best_alt is not None:
            print(f"  w_e={we:.2f}: own={own:.4f}, beaten by w_e={best_alt[0]:.2f} ({best_alt[1]:.4f})")

    # Does any NEW point score better than the established w_e=0.1+ dominant
    # trajectory or the w_e=0 time-optimal one, under ITS OWN weight?
    dominant = next(r for r in original if r['w_energy'] == 0.1)
    time_opt = next(r for r in original if r['w_energy'] == 0.0)
    print(f"\n=== Does any new point beat the two known trajectories under its own weight? ===")
    found_new_distinct = False
    for r in results:
        we = r['w_energy']
        own = r['total_time'] + we * r['energy']
        dom_score = dominant['total_time'] + we * dominant['energy']
        topt_score = time_opt['total_time'] + we * time_opt['energy']
        beats_both = own < dom_score - 1e-6 and own < topt_score - 1e-6
        print(f"  w_e={we:.2f}: own={own:.4f}  vs dominant-traj={dom_score:.4f}  "
              f"vs time-opt-traj={topt_score:.4f}  {'*** NEW DISTINCT POINT ***' if beats_both else ''}")
        if beats_both:
            found_new_distinct = True

    print(f"\nfound_new_distinct_point = {found_new_distinct}")
    out = {'new_grid_results': results, 'found_new_distinct_point': found_new_distinct,
           'n_dominated_in_full_pool': len(dominated)}
    with open(OUT_DIR / 'sweep_pareto_densify.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved -> {OUT_DIR / 'sweep_pareto_densify.json'}")


if __name__ == '__main__':
    main()
