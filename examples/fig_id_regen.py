#!/usr/bin/env python3
##
# @file fig_id_regen.py
#
# @brief Regenerates the power-model identification figures:
#        fig_id_excitation.png (wheel acceleration / velocity excitation
#        commands) and fig_id_fit.png (measured vs. fitted wheel power).
#
# Data and fitted coefficients come from the original identification run in
# mobile_robot_packages/trajectory_generators/test (test_excite_generator.py,
# test_lse_minization.py). Nothing is re-fitted here: the saved coefficients
# are the ones listed in the paper's power-model table (and in
# differential_drive_comparison.ENERGY_COEFFS_*). Both figures show the same
# ID_WINDOW_S-second window, since the full ~466 s record is unreadable at
# the printed 9 cm width.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/09/24

import sys
import os
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import differential_drive_comparison as dc

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_DIR = (ROOT / 'mobile_robot_packages' / 'trajectory_generators'
            / 'test' / 'data')
FIG_OUT_DIR = ROOT / 'Writting' / 'energy_aware'

ID_WINDOW_S = 40.0

COL_LEFT, COL_RIGHT = 'black', 'tab:orange'
COL_MEAS, COL_FIT = '0.45', 'tab:red'


def _power_model(c, v, a):
    """Wheel power model of the paper, on |v|, |a| (test_lse_minization.py)."""
    v, a = np.abs(v), np.abs(a)
    return c[0]*a**2 + c[1]*v**2 + c[2]*a + c[3]*v + c[4]*v*a + c[5]


def _save(fig, name):
    out = FIG_OUT_DIR / name
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'[paper] Saved {name} -> {out}')


def fig_excitation(traj):
    w = traj['timestamp'] <= ID_WINDOW_S
    t = traj['timestamp'][w]

    fig, (ax_a, ax_v) = plt.subplots(2, 1, figsize=(dc.FIG_W_9CM, 3.0),
                                     sharex=True, constrained_layout=True)
    for ax, key, ylabel in [(ax_a, 'accel', r'Acc. [m/s$^2$]'),
                            (ax_v, 'vel',   'Vel. [m/s]')]:
        ax.plot(t, traj[f'left_{key}'][w], color=COL_LEFT, lw=0.9,
                label='Left wheel')
        # The right-wheel command equals the left one; dashed so both show.
        ax.plot(t, traj[f'right_{key}'][w], color=COL_RIGHT, lw=0.9,
                ls=(0, (3, 3)), label='Right wheel')
        ax.set_ylabel(ylabel)
        ax.grid(True, color='0.9', lw=0.5)
    ax_v.set_xlabel('Time [s]')
    ax_v.set_xlim(0, ID_WINDOW_S)
    fig.legend(*ax_a.get_legend_handles_labels(), loc='outside upper center',
               ncol=2, fontsize=7)
    _save(fig, 'fig_id_excitation.png')


def fig_fit(traj, power, c_right, c_left):
    w = (traj['timestamp'] <= ID_WINDOW_S).to_numpy()
    t = traj['timestamp'].to_numpy()[w]

    fig, axes = plt.subplots(2, 1, figsize=(dc.FIG_W_9CM, 3.0),
                             sharex=True, constrained_layout=True)
    for ax, side, meas_col, c in [(axes[0], 'right', 'P1', c_right),
                                  (axes[1], 'left',  'P2', c_left)]:
        p_fit = _power_model(c, traj[f'{side}_vel'].to_numpy(),
                             traj[f'{side}_accel'].to_numpy())
        ax.plot(t, power[meas_col].to_numpy()[w], color=COL_MEAS, lw=0.8,
                label='Measured')
        ax.plot(t, p_fit[w], color=COL_FIT, lw=0.9, label='Fitted model')
        ax.set_ylabel(f'{side.capitalize()} wheel\npower [W]')
        ax.grid(True, color='0.9', lw=0.5)
    axes[1].set_xlabel('Time [s]')
    axes[1].set_xlim(0, ID_WINDOW_S)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center',
               ncol=2, fontsize=7)
    _save(fig, 'fig_id_fit.png')


def main():
    dc._set_print_style()
    traj = pd.read_csv(DATA_DIR / 'differential_drive_trajectory.csv')
    power = pd.read_csv(DATA_DIR / 'power.csv')
    assert len(traj) == len(power)
    with open(DATA_DIR / 'optimized_params_right.json') as f:
        c_right = json.load(f)
    with open(DATA_DIR / 'optimized_params_left.json') as f:
        c_left = json.load(f)

    fig_excitation(traj)
    fig_fit(traj, power, c_right, c_left)


if __name__ == '__main__':
    main()
