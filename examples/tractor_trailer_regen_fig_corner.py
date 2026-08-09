#!/usr/bin/env python3
##
# @file tractor_trailer_regen_fig_corner.py
#
# @brief Regenerates tt_fig_corner.png (Fig. 3 of the paper) from the
# actual validated baseline trajectories (traj_pareto_we0.00.npz,
# traj_pareto_we0.10.npz) rather than the stale figure previously produced
# by tractor_trailer_path_segment_combined.py's earlier, pre-reconciliation
# parameters -- a round-9 domain-expert review found the figure's annotated
# T/E/peak-power numbers did not match Table II's validated baseline row
# for the same nominal configuration.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/08/09

import sys
import os
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

OUT_DIR = (pathlib.Path(__file__).resolve().parent.parent.parent
           / 'Writting' / 'energy_aware_trailer_tractor' / 'figures')
TRAJ_DIR = OUT_DIR / 'trajectories'

LB, LF = 0.2, 0.8  # baseline hitch geometry


def _load(name):
    d = np.load(TRAJ_DIR / f'{name}.npz', allow_pickle=True)
    return {k: d[k] for k in d.files if k != 'meta'}


def _tractor_xy(states):
    x, y, theta, gamma = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    xt = x + LF * np.cos(theta) + LB * np.cos(theta - gamma)
    yt = y + LF * np.sin(theta) + LB * np.sin(theta - gamma)
    return xt, yt


def main():
    t_opt = _load('traj_pareto_we0.00')
    e_aware = _load('traj_pareto_we0.10')

    print(f"time-optimal:  T={float(t_opt['time'][-1]):.3f}  E={float(t_opt['energy']):.3f}  "
          f"Ppeak={float(np.max(t_opt['power'])):.3f}")
    print(f"energy-aware:  T={float(e_aware['time'][-1]):.3f}  E={float(e_aware['energy']):.3f}  "
          f"Ppeak={float(np.max(e_aware['power'])):.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    for res, label, color in [(t_opt, 'time-optimal ($w_e=0$)', 'tomato'),
                               (e_aware, 'energy-aware ($w_e=0.1$)', 'seagreen')]:
        st = res['states']
        xt, yt = _tractor_xy(st)
        axes[0].plot(st[:, 0], st[:, 1], '-', color=color, label=f'{label}, trailer')
        axes[0].plot(xt, yt, '--', color=color, alpha=0.6, label=f'{label}, tractor')
        axes[1].plot(res['time'], res['power'], '-', color=color, label=label)
        axes[2].plot(res['time'], np.rad2deg(st[:, 3]), '-', color=color, label=label)

    axes[0].set_xlabel('x [m]'); axes[0].set_ylabel('y [m]')
    axes[0].set_title('XY path'); axes[0].legend(fontsize=7); axes[0].axis('equal')
    axes[0].grid(True, ls=':', alpha=0.5)

    axes[1].set_xlabel('t [s]'); axes[1].set_ylabel('Motor power [W]')
    axes[1].set_title('Tracked power'); axes[1].legend(fontsize=8)
    axes[1].grid(True, ls=':', alpha=0.5)

    axes[2].set_xlabel('t [s]'); axes[2].set_ylabel('Hitch angle [deg]')
    axes[2].axhline(44.98, color='gray', ls='--', lw=0.8)
    axes[2].axhline(-44.98, color='gray', ls='--', lw=0.8)
    axes[2].set_title('Hitch angle'); axes[2].legend(fontsize=8)
    axes[2].grid(True, ls=':', alpha=0.5)

    fig.suptitle('Baseline $90^{\\circ}$ corner: time-optimal vs. energy-aware '
                  '(validated $w_e=0.1$) trailer/tractor path, power, hitch angle')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'tt_fig_corner.png', dpi=300, bbox_inches='tight')
    print(f"Saved -> {OUT_DIR / 'tt_fig_corner.png'}")


if __name__ == '__main__':
    main()
