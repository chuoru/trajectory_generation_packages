#!/usr/bin/env python3
##
# @file tractor_trailer_pareto_repair.py
#
# @brief Cross-pollination repair pass for the Pareto sweep in
# tractor_trailer_paper_sweeps.py.
#
# Validation of the original 11-point multi-start Pareto sweep found that
# 5 of 11 points were Pareto-dominated in (T, E) space, and that EVERY
# point from w_e=0.2 onward (including the w_e=0.5 knee point) was
# provably suboptimal for its own weighted-sum objective: some other
# already-solved trajectory in the sweep, evaluated under that point's own
# weight, scored strictly better than what the solver reported as the best
# result for that weight. The original multi-start budget (cold start + 1
# jittered restart + 1 immediate-neighbor continuation) was not thorough
# enough to escape this.
#
# This script repairs it cheaply by reusing the 11 already-solved
# trajectories (saved as .npz in figures/trajectories/) as a shared warm-
# start pool: for every target w_e, every other point's trajectory is
# evaluated under that w_e's own weight; if any beats the current best, a
# fresh solve is warm-started from it (plus one jittered variant) and kept
# if it improves on the incumbent. This is iterated across the whole sweep
# for two rounds, since fixing one point can unlock a better warm start for
# its neighbors, then Pareto-dominance and weighted-sum consistency are
# re-verified.
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

from trajectory_generators.bspline_energy_tractor_trailer_coverage import (
    BSplineEnergyTractorTrailerCoverage,
)

from tractor_trailer_paper_sweeps import (
    common_kwargs, make_waypoints, _make_gen, N_SAMPLING, N_CTRL_PTS,
    MAX_ITER, OUT_DIR, TRAJ_DIR,
)

W_GRID = [round(x, 2) for x in np.linspace(0.0, 1.0, 11)]


def _stable_seed(*parts):
    """Deterministic seed independent of Python's per-process string-hash
    randomization. hash(tuple_containing_str) is NOT reproducible run-to-run
    unless PYTHONHASHSEED is fixed externally; this is. (hash((rnd, w)) with
    int/float-only tuples was accidentally stable already, but this keeps
    every sweep/repair script on one deterministic seeding convention.)"""
    h = hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31)


def _load_traj(name):
    d = np.load(TRAJ_DIR / f'{name}.npz', allow_pickle=True)
    return {k: d[k] for k in d.files if k != 'meta'}


def _obj(traj, w_time, w_energy):
    return w_time * float(traj['time'][-1]) + w_energy * float(traj['energy'])


def _resolve_from(kw, w_time, w_energy, ctrl_pts, time_arr, seed, jitter=True):
    """Warm-start from a given (ctrl_pts, time) pair, optionally trying one
    jittered variant too; return the better of the two as (res, obj)."""
    rng = np.random.default_rng(seed)
    candidates = []
    ws = {'ctrl_pts': ctrl_pts, 'time': time_arr}
    for tag, warm in [('direct', ws)]:
        gen = _make_gen(kw, w_energy, w_time=w_time)
        try:
            res = gen.generate_trajectory(warm_start=warm)
            candidates.append((_obj(res, w_time, w_energy), res, tag))
        except Exception:
            pass
    if jitter and candidates:
        base = candidates[0][1]
        scale = 0.08 * (np.abs(base['ctrl_pts']).mean(axis=0, keepdims=True) + 0.05)
        cp_jit = base['ctrl_pts'] + rng.normal(scale=scale, size=base['ctrl_pts'].shape)
        gen2 = _make_gen(kw, w_energy, w_time=w_time)
        try:
            res2 = gen2.generate_trajectory(warm_start={'ctrl_pts': cp_jit, 'time': base['time']})
            candidates.append((_obj(res2, w_time, w_energy), res2, 'jitter'))
        except Exception:
            pass
    if not candidates:
        return None, None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1], candidates[0][0]


def repair_pareto(n_rounds=2):
    waypoints = make_waypoints(np.pi / 2)
    kw = common_kwargs(waypoints)

    pool = {}   # w_e -> trajectory dict
    for w in W_GRID:
        pool[w] = _load_traj(f'traj_pareto_we{w:.2f}')

    incumbent_obj = {w: _obj(pool[w], 1.0, w) for w in W_GRID}

    print("=== Initial state ===")
    for w in W_GRID:
        print(f"  w_e={w:.2f}  T={pool[w]['time'][-1]:.3f}  "
              f"E={pool[w]['energy']:.3f}  obj={incumbent_obj[w]:.4f}")

    for rnd in range(n_rounds):
        print(f"\n=== Repair round {rnd+1}/{n_rounds} ===")
        improved_any = False
        for w in W_GRID:
            # Find the best candidate trajectory currently in the pool
            # (including w's own) under w's own weight.
            best_donor_w = min(pool.keys(), key=lambda wj: _obj(pool[wj], 1.0, w))
            donor_obj = _obj(pool[best_donor_w], 1.0, w)
            if donor_obj < incumbent_obj[w] - 1e-6 and best_donor_w != w:
                print(f"  w_e={w:.2f}: donor w_e={best_donor_w:.2f} scores "
                      f"{donor_obj:.4f} vs incumbent {incumbent_obj[w]:.4f} -- re-solving")
                donor = pool[best_donor_w]
                res, obj = _resolve_from(kw, 1.0, w, donor['ctrl_pts'], donor['time'],
                                          seed=_stable_seed(rnd, w))
                # Compare three candidates: the current incumbent, whatever
                # IPOPT converges to when warm-started from the donor (which
                # can drift to something WORSE than the donor itself, since
                # the donor is not a stationary point for w's own weight),
                # and the raw donor trajectory adopted as-is (fully feasible
                # regardless of weight, since w_e only changes the
                # objective, not the constraint set).
                candidates = [(incumbent_obj[w], pool[w], 'incumbent'),
                              (donor_obj, donor, 'raw_donor')]
                if res is not None:
                    candidates.append((obj, res, 'resolved'))
                candidates.sort(key=lambda c: c[0])
                best_obj_final, best_res_final, best_tag_final = candidates[0]
                if best_obj_final < incumbent_obj[w] - 1e-6:
                    print(f"    -> improved w_e={w:.2f} via {best_tag_final}: "
                          f"{incumbent_obj[w]:.4f} -> {best_obj_final:.4f}")
                    pool[w] = best_res_final
                    incumbent_obj[w] = best_obj_final
                    improved_any = True
                else:
                    print(f"    -> neither resolve nor raw donor beat incumbent; keeping incumbent")
        if not improved_any:
            print("  No further improvements found this round; stopping early.")
            break

    print("\n=== Final state after repair ===")
    results = []
    for w in W_GRID:
        traj = pool[w]
        st = traj['states']
        plen = float(np.sum(np.sqrt(np.diff(st[:, 0]) ** 2 + np.diff(st[:, 1]) ** 2)))
        gamma = st[:, 3]
        m = {
            'w_energy': w,
            'total_time': float(traj['time'][-1]),
            'path_length': plen,
            'energy': float(traj['energy']),
            'energy_per_m': float(traj['energy']) / plen if plen > 1e-9 else float('inf'),
            'peak_power': float(np.max(traj['power'])),
            'avg_power': float(np.mean(traj['power'])),
            'peak_gamma_deg': float(np.max(np.abs(gamma)) * 180.0 / np.pi),
            'jackknife_margin': float(1.0 - np.max(np.abs(gamma)) / 0.785),
        }
        results.append(m)
        print(f"  w_e={w:.2f}  T={m['total_time']:.3f}  E={m['energy']:.3f}  "
              f"Ppeak={m['peak_power']:.3f}  |gamma|max={m['peak_gamma_deg']:.2f}")
        np.savez(TRAJ_DIR / f'traj_pareto_we{w:.2f}_repaired.npz', **traj)

    return results, pool


def verify(results):
    print("\n=== Post-repair Pareto dominance check (T, E) ===")
    n = len(results)
    dominated = []
    for i in range(n):
        Ti, Ei = results[i]['total_time'], results[i]['energy']
        for j in range(n):
            if i == j:
                continue
            Tj, Ej = results[j]['total_time'], results[j]['energy']
            if Tj <= Ti and Ej <= Ei and (Tj < Ti or Ej < Ei):
                dominated.append((i, j))
    if dominated:
        for i, j in dominated:
            print(f"  w_e={results[i]['w_energy']:.2f} DOMINATED by w_e={results[j]['w_energy']:.2f}")
    else:
        print("  No dominated points. Front is valid.")

    print("\n=== Post-repair weighted-sum consistency check ===")
    any_subopt = False
    for i in range(n):
        we_i = results[i]['w_energy']
        own_obj = 1.0 * results[i]['total_time'] + we_i * results[i]['energy']
        best_alt = None
        for j in range(n):
            if i == j:
                continue
            alt_obj = 1.0 * results[j]['total_time'] + we_i * results[j]['energy']
            if alt_obj < own_obj - 1e-6:
                if best_alt is None or alt_obj < best_alt[1]:
                    best_alt = (j, alt_obj)
        if best_alt is not None:
            any_subopt = True
            j, alt_obj = best_alt
            print(f"  w_e={we_i:.2f}: own={own_obj:.4f}, beaten by w_e="
                  f"{results[j]['w_energy']:.2f} ({alt_obj:.4f})")
    if not any_subopt:
        print("  No inconsistencies. Every point is the best available for its own weight.")
    return dominated, any_subopt


def knee_point(results):
    e = np.array([r['energy'] for r in results])
    p = np.array([r['peak_power'] for r in results])
    e_n = (e - e.min()) / (e.max() - e.min() + 1e-12)
    p_n = (p - p.min()) / (p.max() - p.min() + 1e-12)
    x0, y0 = e_n[0], p_n[0]
    x1, y1 = e_n[-1], p_n[-1]
    seg = np.array([x1 - x0, y1 - y0])
    seg_unit = seg / (np.linalg.norm(seg) + 1e-12)
    dists = np.zeros(len(e_n))
    for i in range(len(e_n)):
        pt = np.array([e_n[i] - x0, p_n[i] - y0])
        proj = np.dot(pt, seg_unit) * seg_unit
        dists[i] = np.linalg.norm(pt - proj)
    idx = int(np.argmax(dists))
    print(f"\nKnee point: w_e={results[idx]['w_energy']:.2f} "
          f"(E={results[idx]['energy']:.2f} J, P_peak={results[idx]['peak_power']:.2f} W)")
    return idx


if __name__ == '__main__':
    results, pool = repair_pareto(n_rounds=2)
    dominated, any_subopt = verify(results)
    idx = knee_point(results)
    out = {'results': results, 'knee_index': idx,
           'still_dominated': len(dominated), 'still_inconsistent': any_subopt}
    with open(OUT_DIR / 'sweep_pareto_repaired.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved -> {OUT_DIR / 'sweep_pareto_repaired.json'}")
