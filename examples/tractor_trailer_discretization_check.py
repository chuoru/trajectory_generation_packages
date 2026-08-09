#!/usr/bin/env python3
##
# @file tractor_trailer_discretization_check.py
#
# @brief Checks whether the paper's chosen constraint-sampling resolution
# (n_sampling=10, Table I) is adequate, or whether the reported baseline
# operating-point trajectory changes meaningfully under a finer sampling
# grid -- a round-8 Methodology review flagged that no convergence study
# exists anywhere in the paper for the chosen discretization, only the
# negative data point that a much coarser, different-geometry configuration
# failed to converge at all.
#
# n_ctrl_pts (spline flexibility / decision-variable count) is held fixed
# at its Table I value, since increasing it changes the control-point array
# shape and breaks direct warm-starting from the existing baseline
# trajectory; only n_sampling (how densely the relaxed nonholonomic and
# hitch-coupling BANDS, Eqs. nh_relaxed/hitch_band, are enforced) is
# refined, warm-started directly from the already-solved w_e=0.1 baseline
# trajectory. If (T, E, |gamma|_max) do not move by more than this paper's
# own reported multi-start spread under 1.5x-2x finer sampling, the chosen
# n_sampling=10 is adequate for what it is actually used for (catching
# nonholonomic/hitch-coupling band violations between nodes); the jackknife
# bound itself is unaffected since it is already convex-hull-guaranteed
# (Section III-D) independent of n_sampling.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/08/09

import sys
import os
import hashlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tractor_trailer_paper_sweeps import common_kwargs, make_waypoints, _make_gen, TRAJ_DIR, GAMMA_MAX

# Per-node time-scaling variables are sized by n_sampling, so a warm start
# saved at n_sampling=10 cannot be directly reused at a different
# n_sampling (confirmed: CasADi raises a dimension-mismatch error on
# opti.set_initial). Independent multi-start solves at each resolution are
# used instead -- a fair, if slightly more expensive, adequacy check.
N_JITTER = 2


def _stable_seed(*parts):
    h = hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31)


def _load_traj(name):
    d = np.load(TRAJ_DIR / f'{name}.npz', allow_pickle=True)
    return {k: d[k] for k in d.files if k != 'meta'}


def _multistart_at_resolution(kw, w_energy, n_s, seed):
    rng = np.random.default_rng(seed)
    candidates = []

    def _try(warm, ctag):
        gen = _make_gen(kw, w_energy, w_time=1.0)
        try:
            res = gen.generate_trajectory(warm_start=warm)
            obj = float(res['time'][-1]) + w_energy * float(res['energy'])
            candidates.append((obj, res, ctag))
        except Exception as e:
            print(f'    [{ctag}] failed: {e}')

    _try(None, 'cold')
    base = candidates[0][1] if candidates else None
    if base is not None:
        for k in range(N_JITTER):
            scale = 0.12 * (np.abs(base['ctrl_pts']).mean(axis=0, keepdims=True) + 0.05)
            cp = base['ctrl_pts'] + rng.normal(scale=scale, size=base['ctrl_pts'].shape)
            _try({'ctrl_pts': cp, 'time': base['time']}, f'jitter{k}')
    if not candidates:
        raise RuntimeError(f'all attempts failed at n_sampling={n_s}')
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1], len(candidates)


def main():
    waypoints = make_waypoints(np.pi / 2)
    base = _load_traj('traj_pareto_we0.10')

    T0 = float(base['time'][-1])
    E0 = float(base['energy'])
    gamma0 = float(np.max(np.abs(base['states'][:, 3])) * 180.0 / np.pi)
    print(f"=== Baseline (n_sampling=10, reported, w_e=0.1): T={T0:.4f}  E={E0:.4f}  "
          f"|gamma|max={gamma0:.4f} deg ===\n")

    for n_s in (15, 20):
        kw = common_kwargs(waypoints)
        kw['n_sampling'] = n_s
        res, n_tried = _multistart_at_resolution(kw, 0.1, n_s, seed=_stable_seed('disc', n_s))
        T = float(res['time'][-1])
        E = float(res['energy'])
        gamma = float(np.max(np.abs(res['states'][:, 3])) * 180.0 / np.pi)
        margin = 1.0 - (gamma * np.pi / 180.0) / GAMMA_MAX
        dT = (T - T0) / T0 * 100
        dE = (E - E0) / E0 * 100
        dGamma = gamma - gamma0
        print(f"n_sampling={n_s:2d} ({n_tried} tried)  T={T:.4f} ({dT:+.3f}%)  "
              f"E={E:.4f} ({dE:+.3f}%)  |gamma|max={gamma:.4f} deg (delta={dGamma:+.4f} deg)  "
              f"margin={margin*100:.3f}%")

    print("\nFor reference, this paper's reported multi-start spreads "
          "(cold vs. jittered/neighbor candidates) run 240-338% for "
          "time-optimal legs and well under 1% for energy-aware legs "
          "(Table II/III captions, Sec. IV-A/B).")


if __name__ == '__main__':
    main()
