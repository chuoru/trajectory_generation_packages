#!/usr/bin/env python3
##
# @file tractor_trailer_flat_region_densify.py
#
# @brief Applies the paper's OWN stated densification protocol (Conclusion,
# ssec:res_pareto practical-guidance paragraph: re-solve a collapsed
# interval at roughly ten times the original grid density, stop once two
# consecutive tenfold refinements add nothing) to the highest-priority
# sub-interval of the "flat region," w_e in (0.1, 0.2), instead of only the
# three hand-picked spot-check probes of tractor_trailer_flat_region_check.py.
#
# A round-5 Devil's Advocate review correctly pointed out that the 3-probe
# spot-check does not meet the bar the paper's own Conclusion sets for
# "accepting a flat region as genuine": that bar was met for (0, 0.1) (9
# points at Delta=0.01) but not for any part of (0.1, 1.0], despite the
# guidance paragraph explicitly prioritizing intervals flanking an
# already-known collapse point -- which w_e in (0.1, 0.2) is, immediately
# adjacent to the validated curve's own endpoint at w_e=0.1.
#
# This densifies w_e in {0.11, ..., 0.19} at exactly the same resolution
# and multi-start budget as the original (0, 0.1) densify
# (tractor_trailer_pareto_densify.py), warm-started from both the w_e=0.1
# curve endpoint and the established w_e>=0.1 dominant trajectory, and
# re-runs the full dominance/weighted-sum-consistency check against the
# combined pool (original 11 + first densify's 9 + this densify's 9).
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

    we01 = _load_traj('traj_pareto_we0.10')   # curve endpoint / flat-region entry
    we02 = _load_traj('traj_pareto_we0.20')   # flat-region interior (already confirmed identical to we0.10)

    new_grid = [round(x, 2) for x in np.arange(0.11, 0.20, 0.01)]
    results = []
    print("=== Flat-region densify: w_e in (0.1, 0.2), 10x original grid density ===\n")
    for w in new_grid:
        res, obj, diag = _multistart(
            kw, 1.0, w, extra_warm_starts=[we01, we02], seed=_stable_seed('flatdensify', w),
            tag=f'w{w:.2f}-')
        results.append({'w_energy': w, 'total_time': float(res['time'][-1]),
                         'energy': float(res['energy']), 'obj': float(obj)})
        print(f"  w_e={w:.2f}  T={res['time'][-1]:6.3f}  E={res['energy']:7.3f}  "
              f"obj={obj:.4f}  [{diag['n_attempts']} tried, spread={diag['spread_pct']:.2f}%, "
              f"best={diag['best_tag']}]")
        np.savez(TRAJ_DIR / f'traj_pareto_flatdensify_we{w:.2f}.npz', **res)

    with open(OUT_DIR / 'sweep_pareto.json') as f:
        original = json.load(f)['results']
    with open(OUT_DIR / 'sweep_pareto_densify.json') as f:
        densify1 = json.load(f)['new_grid_results']

    pool = ([{'w_energy': r['w_energy'], 'total_time': r['total_time'], 'energy': r['energy']}
             for r in original] + densify1)

    print(f"\n=== Dominance check across full pool ({len(pool) + len(results)} points) ===")
    full_pool = pool + results
    dominated = []
    for i, pi in enumerate(full_pool):
        for j, pj in enumerate(full_pool):
            if i == j:
                continue
            if (pj['total_time'] <= pi['total_time'] and pj['energy'] <= pi['energy']
                    and (pj['total_time'] < pi['total_time'] or pj['energy'] < pi['energy'])):
                dominated.append((pi['w_energy'], pj['w_energy']))
    new_dominated = [d for d in dominated if round(d[0], 2) in new_grid]
    if new_dominated:
        for wi, wj in new_dominated:
            print(f"  w_e={wi:.2f} DOMINATED by w_e={wj:.2f}")
    else:
        print("  No new-grid point is dominated by anything in the full pool.")

    print(f"\n=== Does any new point beat the established w_e=0.1 dominant trajectory under its own weight? ===")
    any_new_better = False
    same_as_dominant = True
    for r in results:
        we = r['w_energy']
        own = r['total_time'] + we * r['energy']
        dom_score = we01['time'][-1] + we * we01['energy']
        beats = own < float(dom_score) - 1e-6
        same = (abs(r['total_time'] - float(we01['time'][-1])) < 1e-3
                and abs(r['energy'] - float(we01['energy'])) < 1e-2)
        if not same:
            same_as_dominant = False
        print(f"  w_e={we:.2f}: own={own:.4f}  vs dominant-traj={float(dom_score):.4f}  "
              f"same_as_dominant={same}  {'*** NEW DISTINCT / BETTER POINT ***' if beats else ''}")
        if beats:
            any_new_better = True

    print(f"\nfound_new_distinct_point = {any_new_better}")
    print(f"all_nine_points_match_dominant_exactly = {same_as_dominant}")
    out = {'new_grid_results': results, 'found_new_distinct_point': any_new_better,
           'all_match_dominant': same_as_dominant, 'n_dominated_in_full_pool': len(new_dominated)}
    with open(OUT_DIR / 'sweep_pareto_flat_densify.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved -> {OUT_DIR / 'sweep_pareto_flat_densify.json'}")


if __name__ == '__main__':
    main()
