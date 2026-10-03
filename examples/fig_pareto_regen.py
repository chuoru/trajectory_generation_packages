#!/usr/bin/env python3
##
# @file fig_pareto_regen.py
#
# @brief Regenerates energy_aware_fine_sweep/fig_pareto.png (paper Fig.
#        "pareto"): corner-only peak motor power vs. total energy over the
#        40-point fine w_e sweep, coloured by log10(w_e), with the
#        time-optimal, saturation-selected (Method D) and energy-optimal
#        points marked and the JLAP-segment ceiling drawn.
#
# The script that produced the previous version of this figure is lost; this
# one reads the sweep's own results from
# csv_output_fine_sweep/sweep_statistics.csv (written by
# differential_drive_path_segment_fine_sweep.py), so no solves are run.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/09/24

import sys
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import differential_drive_comparison as dc

HERE = pathlib.Path(__file__).resolve().parent
SWEEP_CSV = HERE / 'csv_output_fine_sweep' / 'sweep_statistics.csv'
FIG_OUT_DIR = (HERE.parent.parent / 'Writting' / 'energy_aware'
               / 'energy_aware_fine_sweep')

# Whole-mission peak-power ceiling set by the flanking JLAP segments
# (Table "peak_breakdown"; independent of w_e).
P_JLAP_CEILING = 19.23

COL_TIME, COL_SAT, COL_ENE = 'steelblue', 'darkorange', 'mediumorchid'


def main():
    dc._set_print_style()
    sw = pd.read_csv(SWEEP_CSV)
    we = sw['w_e'].to_numpy()
    pp = sw['peak_power_W'].to_numpy()
    te = sw['total_energy_J'].to_numpy()

    i_time = int(np.argmin(we))
    i_sat = int(np.argmin(np.abs(we - dc.WE_D)))
    i_ene = int(np.argmin(te))

    fig, ax = plt.subplots(figsize=(dc.FIG_W_9CM, 3.0), constrained_layout=True)

    log_we = np.log10(np.where(we > 0, we, 1e-5))
    sc = ax.scatter(te, pp, c=log_we, cmap='viridis', s=12, zorder=3,
                    linewidths=0)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r'$\log_{10} w_e$')
    cbar.ax.tick_params(labelsize=7)

    ax.axhline(P_JLAP_CEILING, color='gray', ls=':', lw=0.9, zorder=2,
               label=f'JLAP segment ceiling ({P_JLAP_CEILING:.2f} W)')
    ax.scatter([te[i_time]], [pp[i_time]], marker='^', s=45, color=COL_TIME,
               edgecolor='white', linewidth=0.5, zorder=6,
               label=f'Time-opt. ($w_e$={we[i_time]:g})')
    ax.scatter([te[i_sat]], [pp[i_sat]], marker='o', s=45, color=COL_SAT,
               edgecolor='black', linewidth=0.6, zorder=6,
               label=f'Sat.-selected (D) ($w_e$={we[i_sat]:.4f})')
    ax.scatter([te[i_ene]], [pp[i_ene]], marker='s', s=40, color=COL_ENE,
               edgecolor='white', linewidth=0.5, zorder=6,
               label=f'Energy-opt. ($w_e$={we[i_ene]:.4f})')

    ax.set_xlabel('Total energy [J]')
    ax.set_ylabel('Corner peak motor power [W]')
    fig.legend(*ax.get_legend_handles_labels(), loc='outside upper center',
               ncol=2, fontsize=6.5, handlelength=1.5, columnspacing=0.8)

    out = FIG_OUT_DIR / 'fig_pareto.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'[paper] Saved fig_pareto.png -> {out}')
    print(f'  time-opt  w_e={we[i_time]:.4f}  E={te[i_time]:.2f} J  P={pp[i_time]:.2f} W')
    print(f'  sat.-sel. w_e={we[i_sat]:.4f}  E={te[i_sat]:.2f} J  P={pp[i_sat]:.2f} W')
    print(f'  ene-opt   w_e={we[i_ene]:.4f}  E={te[i_ene]:.2f} J  P={pp[i_ene]:.2f} W')


if __name__ == '__main__':
    main()
