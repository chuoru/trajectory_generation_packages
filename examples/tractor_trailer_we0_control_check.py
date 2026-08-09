#!/usr/bin/env python3
##
# @file tractor_trailer_we0_control_check.py
#
# @brief Diagnoses the mechanism behind the w_e=0 "dominated" finding of
# tractor_trailer_pareto_densify.py: the original w_e=0 point (T=20.77s,
# E=117.39J) is dominated by four of the newly densified points
# (w_e=0.01-0.04). That finding was reported as a local-optimum artifact
# of the pure-time objective, but the w_e=0 trajectory it was compared
# against was never itself re-solved with access to the same expanded
# warm-start pool the new points had (it was reused verbatim from the
# original repaired 11-point sweep). This leaves two live hypotheses
# unresolved: (a) w_e=0 truly is a harder/more degenerate NLP for IPOPT
# (flat objective gradient near a tied-time manifold), so a small energy
# regularizer helps conditioning and finds a better point that a pure
# T-only search, given the SAME warm-start budget, could not; or (b) the
# original w_e=0 solve was simply warm-start-starved relative to the later
# points and an ordinary re-solve with the richer pool closes the gap.
#
# This re-solves w_e=0 (pure time-minimization) warm-started from every one
# of the nine densified trajectories (w_e=0.01-0.09) plus the w_e=0.1
# trajectory, in addition to the usual cold start and jittered restarts. If
# the best result still cannot reach T<=20.76s (matching the dominating
# points), hypothesis (a) is supported: pure time-minimization is
# genuinely harder to solve to the same quality even with identical warm
# starts. If it does reach T<=20.76s, hypothesis (b) is supported: the
# original finding was a warm-start artifact, not a landscape feature.
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


def _stable_seed(*parts):
    h = hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31)


def _load_traj(name):
    d = np.load(TRAJ_DIR / f'{name}.npz', allow_pickle=True)
    return {k: d[k] for k in d.files if k != 'meta'}


def main():
    waypoints = make_waypoints(np.pi / 2)
    kw = common_kwargs(waypoints)

    densify_names = [f'traj_pareto_densify_we0.0{k}' for k in range(1, 10)]
    warm_starts = [_load_traj(n) for n in densify_names]
    warm_starts.append(_load_traj('traj_pareto_we0.10'))

    orig_we0 = _load_traj('traj_pareto_we0.00')
    print("=== w_e=0 control re-solve, warm-started from all 10 densified/flat points ===")
    print(f"  original w_e=0: T={float(orig_we0['time'][-1]):.4f}  E={float(orig_we0['energy']):.4f}\n")

    rng = np.random.default_rng(_stable_seed('we0control'))
    candidates = []

    def _try(warm, ctag):
        gen = _make_gen(kw, 0.0, w_time=1.0)
        try:
            res = gen.generate_trajectory(warm_start=warm)
            candidates.append((float(res['time'][-1]), res, ctag))
            print(f"  [{ctag:20s}] T={float(res['time'][-1]):.4f}  E={float(res['energy']):.4f}")
        except Exception as e:
            print(f"  [{ctag:20s}] FAILED: {e}")

    _try(None, 'cold')
    base = candidates[0][1] if candidates else warm_starts[0]
    for k in range(2):
        scale = 0.15 * (np.abs(base['ctrl_pts']).mean(axis=0, keepdims=True) + 0.05)
        cp = base['ctrl_pts'] + rng.normal(scale=scale, size=base['ctrl_pts'].shape)
        _try({'ctrl_pts': cp, 'time': base['time']}, f'jitter{k}')

    for name, ws in zip(densify_names + ['traj_pareto_we0.10'], warm_starts):
        _try({'ctrl_pts': ws['ctrl_pts'], 'time': ws['time']}, f'warmstart-{name}')

    candidates.sort(key=lambda c: c[0])
    best_T, best_res, best_tag = candidates[0]
    print(f"\n=== Best of {len(candidates)} candidates ===")
    print(f"  best: T={best_T:.4f}  E={float(best_res['energy']):.4f}  tag={best_tag}")
    print(f"  original w_e=0 T={float(orig_we0['time'][-1]):.4f}")

    matches_dominating = best_T <= 20.760 + 1e-3
    improves_original = best_T < float(orig_we0['time'][-1]) - 1e-6

    print(f"\n  reaches T<=20.76 (matches w_e=0.01-0.04 cluster)? {matches_dominating}")
    print(f"  improves on original w_e=0 result?                 {improves_original}")

    if matches_dominating:
        print("\n  => Hypothesis (b) supported: original w_e=0 finding was warm-start-")
        print("     starved, not a genuine landscape/conditioning difference.")
    else:
        print("\n  => Hypothesis (a) supported: pure time-minimization remains harder")
        print("     to solve to the same (T,E) quality even given the identical warm-start")
        print("     pool the dominating points had access to.")

    out = {
        'original_we0': {'time': float(orig_we0['time'][-1]), 'energy': float(orig_we0['energy'])},
        'best_control': {'time': best_T, 'energy': float(best_res['energy']), 'tag': best_tag},
        'n_candidates': len(candidates),
        'matches_dominating_cluster': bool(matches_dominating),
        'improves_on_original': bool(improves_original),
    }
    with open(OUT_DIR / 'sweep_pareto_we0_control.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved -> {OUT_DIR / 'sweep_pareto_we0_control.json'}")


if __name__ == '__main__':
    main()
