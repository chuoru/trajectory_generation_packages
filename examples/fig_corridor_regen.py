#!/usr/bin/env python3
##
# @file fig_corridor_regen.py
#
# @brief Regenerates fig_corridor.png for the paper: time-optimal (Method B),
#        knee (Method D) and energy-optimal (Method C) corner trajectories,
#        overlaid on the actual OCP corridor constraint (d_max=0.25 m
#        half-width around the two-segment corner reference polyline).
#
# The original fig_corridor.png (source script lost, not tracked anywhere in
# git history) plotted identical T/s/E annotations for two curves -- a
# plotting bug, not a real result. This script independently solves and plots
# Methods B, D, C from the same canonical sweep used for Table "metrics" in
# differential_drive_comparison.py, so all curves and their annotations are
# guaranteed distinct and numerically consistent with the rest of the paper.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/08/05

import sys
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import differential_drive_comparison as dc

SAVE_FIGS = True
FIG_OUT_DIR = (pathlib.Path(__file__).resolve().parent.parent.parent
               / 'Writting' / 'energy_aware')

COL_B, COL_D, COL_C = 'tomato', 'darkorange', 'seagreen'


def _corridor_band(wps, bound, ax):
    """Shade a half-width `bound` band around each segment of the polyline
    defined by 3-point `wps` (x, y, heading)."""
    pts = np.array([[p[0], p[1]] for p in wps])
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        d = p1 - p0
        L = np.hypot(*d)
        n = np.array([-d[1], d[0]]) / L
        quad = np.array([p0 + bound * n, p1 + bound * n,
                          p1 - bound * n, p0 - bound * n])
        ax.fill(quad[:, 0], quad[:, 1], color='khaki', alpha=0.4,
                 label='Corridor ($d_{max}$=%.2f m)' % bound if i == 0 else None)


def main():
    seg = dc._run_segmented_pipeline()
    we_b, we_d, we_c = 0.0, 0.0769, 0.4872
    res_b = dc._build_segmented_result(seg, we_b)
    res_d = dc._build_segmented_result(seg, we_d)
    res_c = dc._build_segmented_result(seg, we_c)

    seg_info = seg['seg_info']
    corner_wps = dc._build_corner_waypoints(
        seg_info['arc_entry_ext_world'], seg_info['arc_exit_ext_world'])

    bound = 0.25  # matches _make_bspline_common's 'bound' kwarg

    def _metrics(res_seg):
        T = res_seg['T_s1'] + res_seg['T_corner'] + res_seg['T_s2']
        states = dc._stitch_states(res_seg)
        dx, dy = np.diff(states[:, 0]), np.diff(states[:, 1])
        s = float(np.sum(np.sqrt(dx**2 + dy**2)))
        pm = dc._power_for_segmented(res_seg)
        return T, s, pm['energy'], states

    T_b, s_b, E_b, states_b = _metrics(res_b)
    T_d, s_d, E_d, states_d = _metrics(res_d)
    T_c, s_c, E_c, states_c = _metrics(res_c)

    fig, ax = plt.subplots(figsize=(6, 6))
    _corridor_band(corner_wps, bound, ax)
    wp_xy = np.array([[p[0], p[1]] for p in corner_wps])
    ax.plot(wp_xy[:, 0], wp_xy[:, 1], '--', color='gray', lw=1.2, label='Reference path')
    ax.plot(states_b[:, 0], states_b[:, 1], color=COL_B, lw=2.2,
             label=f'Time-optimal (B, $w_e$={we_b:.3f})')
    ax.plot(states_d[:, 0], states_d[:, 1], color=COL_D, lw=2.2, ls='-.',
             label=f'Knee (D, $w_e$={we_d:.3f})')
    ax.plot(states_c[:, 0], states_c[:, 1], color=COL_C, lw=2.2, ls='--',
             label=f'Energy-optimal (C, $w_e$={we_c:.3f})')
    ax.plot(*wp_xy[0, :2], 'o', color='black', ms=6, zorder=5)
    ax.plot(*wp_xy[-1, :2], 's', color='black', ms=6, zorder=5)

    txt = (f"Time-optimal (B):   T={T_b:.2f} s | s={s_b:.3f} m | E={E_b:.2f} J\n"
           f"Knee (D):           T={T_d:.2f} s | s={s_d:.3f} m | E={E_d:.2f} J\n"
           f"Energy-optimal (C): T={T_c:.2f} s | s={s_c:.3f} m | E={E_c:.2f} J")
    ax.text(0.02, 0.55, txt, transform=ax.transAxes, fontsize=9,
             va='top', ha='left',
             bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.9))

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title('Corner corridor: Method B vs. D vs. C')
    fig.tight_layout()

    if SAVE_FIGS:
        out = FIG_OUT_DIR / 'fig_corridor.png'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'[paper] Saved fig_corridor.png -> {out}')

    print(f'\n  Time-optimal (B):   T={T_b:.3f} s  s={s_b:.4f} m  E={E_b:.3f} J')
    print(f'  Knee (D):           T={T_d:.3f} s  s={s_d:.4f} m  E={E_d:.3f} J')
    print(f'  Energy-optimal (C): T={T_c:.3f} s  s={s_c:.4f} m  E={E_c:.3f} J')


if __name__ == '__main__':
    main()
