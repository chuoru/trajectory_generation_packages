#!/usr/bin/env python3
##
# @file fig_jlap_schematic.py
#
# @brief Regenerates jlap.png (paper Fig. "jlap"): symbolic seven-phase
#        jerk-limited acceleration profile (JLAP) showing jerk, acceleration
#        and velocity over t_0..t_7. The original was a MATLAB export with
#        no surviving source; this rebuilds it from the piecewise-constant
#        jerk profile, integrated numerically. Axes carry symbolic ticks
#        only (J_lim, a_0, v_0), so each curve is normalised to its own
#        symbolic level.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/09/24

import sys
import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import differential_drive_comparison as dc

FIG_OUT_DIR = (pathlib.Path(__file__).resolve().parent.parent.parent
               / 'Writting' / 'energy_aware')

# Phase boundaries t_0..t_7 (arbitrary units, proportions as in the paper's
# original figure) and the jerk sign in each of the seven phases.
T_KNOTS = np.array([0.0, 1.0, 3.2, 4.2, 6.6, 7.6, 9.8, 10.8])
JERK_SIGN = [+1, 0, -1, 0, -1, 0, +1]

# Plot levels of the symbolic ticks.
LVL_J, LVL_V, LVL_A = 1.0, 0.8, 0.5


def main():
    dc._set_print_style()

    t = np.linspace(T_KNOTS[0], T_KNOTS[-1], 4001)
    idx = np.clip(np.searchsorted(T_KNOTS, t, side='right') - 1, 0, 6)
    jerk = np.array(JERK_SIGN, dtype=float)[idx]
    dt = t[1] - t[0]
    acc = np.concatenate([[0.0], np.cumsum(0.5 * (jerk[1:] + jerk[:-1]) * dt)])
    vel = np.concatenate([[0.0], np.cumsum(0.5 * (acc[1:] + acc[:-1]) * dt)])

    jerk_n = LVL_J * jerk
    acc_n = LVL_A * acc / np.max(np.abs(acc))
    vel_n = LVL_V * vel / np.max(vel)

    fig, ax = plt.subplots(figsize=(dc.FIG_W_9CM, 2.5), constrained_layout=True)
    ax.plot(t, jerk_n, '-', color='red', lw=1.1,
            label=r'$\dddot{x}_k$ [m/s$^3$]')
    ax.plot(t, acc_n, '--', color='blue', lw=1.1,
            label=r'$\ddot{x}_k$ [m/s$^2$]')
    ax.plot(t, vel_n, '-.', color='green', lw=1.1,
            label=r'$\dot{x}_k$ [m/s]')

    ax.set_xticks(T_KNOTS)
    ax.set_xticklabels([f'$t_{i}$' for i in range(len(T_KNOTS))])
    ax.set_yticks([-LVL_J, -LVL_A, 0.0, LVL_A, LVL_V, LVL_J])
    ax.set_yticklabels([r'$-J_{\lim}$', r'$-a_0$', '0', r'$a_0$', r'$v_0$',
                        r'$J_{\lim}$'])
    ax.tick_params(labelsize=8)
    ax.grid(True, color='0.88', lw=0.5)
    ax.set_xlim(T_KNOTS[0], T_KNOTS[-1])
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('Time [s]')
    fig.legend(*ax.get_legend_handles_labels(), loc='outside right center',
               fontsize=8, handlelength=2.2)

    out = FIG_OUT_DIR / 'jlap.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'[paper] Saved jlap.png -> {out}')


if __name__ == '__main__':
    main()
