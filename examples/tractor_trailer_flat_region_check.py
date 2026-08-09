#!/usr/bin/env python3
##
# @file tractor_trailer_flat_region_check.py
#
# @brief Spot-checks whether the "flat region" w_e in (0.1, 1.0] of the
# Pareto sweep (Section IV-B of the paper) is genuinely flat, or conceals
# the same kind of under-sampled-grid structure that densifying w_e in
# (0, 0.1) revealed (tractor_trailer_pareto_densify.py). That earlier
# densify used a Delta w_e = 0.1 grid; the flat region has never been
# tested at any finer resolution than that same Delta w_e = 0.1 spacing,
# despite the fact that this exact spacing was just shown, one interval to
# the left, to hide a genuine trade-off curve.
#
# Rather than re-running the full 9-point densify (expensive) across the
# whole flat region, this probes three widely-separated midpoints inside
# it -- w_e = 0.15, 0.55, 0.95 -- each solved with the same multi-start
# procedure used throughout the paper, plus continuation warm starts from
# both of its immediate Delta=0.1 neighbors (already-solved, both known to
# collapse to the single dominant trajectory). If all three midpoints also
# converge to that same dominant trajectory (to a tight tolerance), that is
# positive evidence the flat region is genuinely flat rather than merely
# under-sampled. If any midpoint finds something distinct and better under
# its own weight, that is direct evidence of concealed structure, exactly
# analogous to what was found in (0, 0.1).
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
    randomization (unlike hash(tuple_of_str), which is NOT reproducible
    across runs unless PYTHONHASHSEED is fixed externally)."""
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

    we_dominant = _load_traj('traj_pareto_we0.10')  # the flat-region trajectory

    with open(OUT_DIR / 'sweep_pareto.json') as f:
        original = json.load(f)['results']
    with open(OUT_DIR / 'sweep_pareto_densify.json') as f:
        densify = json.load(f)['new_grid_results']

    pool = [{'w_energy': r['w_energy'], 'total_time': r['total_time'], 'energy': r['energy']}
            for r in original] + densify

    probes = [0.15, 0.55, 0.95]
    results = []
    print("=== Flat-region spot check: w_e in {0.15, 0.55, 0.95} ===\n")
    for w in probes:
        res, obj, diag = _multistart(
            kw, 1.0, w, extra_warm_starts=[we_dominant],
            seed=_stable_seed('flatcheck', w), tag=f'w{w:.2f}-')
        results.append({'w_energy': w, 'total_time': float(res['time'][-1]),
                         'energy': float(res['energy']), 'obj': float(obj)})
        print(f"  w_e={w:.2f}  T={res['time'][-1]:6.3f}  E={res['energy']:7.3f}  "
              f"obj={obj:.4f}  [{diag['n_attempts']} tried, spread={diag['spread_pct']:.2f}%, "
              f"best={diag['best_tag']}]")
        np.savez(TRAJ_DIR / f'traj_pareto_flatcheck_we{w:.2f}.npz', **res)

    dominant_obj_at_w = {w: (we_dominant['time'][-1] + w * we_dominant['energy']) for w in probes}
    print("\n=== Does any probe beat the dominant flat-region trajectory under its own weight? ===")
    any_better = False
    for r in results:
        w = r['w_energy']
        own = r['total_time'] + w * r['energy']
        dom = float(dominant_obj_at_w[w])
        beats = own < dom - 1e-6
        same_traj = (abs(r['total_time'] - float(we_dominant['time'][-1])) < 1e-3
                     and abs(r['energy'] - float(we_dominant['energy'])) < 1e-2)
        print(f"  w_e={w:.2f}: own={own:.4f}  vs dominant-traj={dom:.4f}  "
              f"same_as_dominant={same_traj}  {'*** BEATS DOMINANT ***' if beats else ''}")
        if beats:
            any_better = True

    print(f"\n=== Full-pool dominance check (pool + 3 probes) ===")
    full_pool = pool + results
    dominated = []
    for i, pi in enumerate(full_pool):
        for j, pj in enumerate(full_pool):
            if i == j:
                continue
            if (pj['total_time'] <= pi['total_time'] and pj['energy'] <= pi['energy']
                    and (pj['total_time'] < pi['total_time'] or pj['energy'] < pi['energy'])):
                dominated.append((pi['w_energy'], pj['w_energy']))
    new_dominated = [d for d in dominated if d[0] in probes]
    if new_dominated:
        for wi, wj in new_dominated:
            print(f"  w_e={wi:.2f} DOMINATED by w_e={wj:.2f}")
    else:
        print("  No probe point is dominated by anything in the full pool.")

    print(f"\nany_probe_beats_dominant = {any_better}")
    out = {'probe_results': results, 'any_probe_beats_dominant': any_better,
           'probe_dominated': new_dominated}
    with open(OUT_DIR / 'sweep_pareto_flatcheck.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved -> {OUT_DIR / 'sweep_pareto_flatcheck.json'}")


if __name__ == '__main__':
    main()
