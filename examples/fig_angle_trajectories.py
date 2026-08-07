"""! Overlay the actual generated corner trajectories (XY) at 45/90/135
degrees, each solved at that angle's own knee weight (from Table 2 /
angle_sweep_generalization.py), to visually complement
fig_angle_generalization.py's power/energy comparison with the geometric
shape of the corner itself.

NOTE: an isolated single cold-started solve at the knee weight for 135 deg
failed to converge (Maximum_Iterations_Exceeded after 720s) even though the
same weight solves fine as part of a full warm-started sweep (that's how
Table 2's 135 deg row was obtained). This script therefore reuses the same
_sweep_we() warm-start chain used for Table 2/Fig. 4, and extracts the
converged trajectory at the knee weight from the sweep's res_by_we dict,
instead of attempting a fresh isolated solve.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import matplotlib.pyplot as plt

import differential_drive_path_segment_fine_sweep as m

m._set_paper_style()

N_POINTS = {45: 15, 90: 15, 135: 8}   # matches the resolution used for Table 2
COLORS   = {45: '#1f77b4', 90: '#d62728', 135: '#2ca02c'}
MARKERS  = {45: 'o', 90: 's', 135: '^'}

results = {}
for angle_deg in [45, 90, 135]:
    beta_rad = -np.deg2rad(angle_deg)
    m.BETA = beta_rad
    m.HEADING_OUT = float(m.HEADING_IN + beta_rad)

    print(f"Sweeping {angle_deg} deg corner ({N_POINTS[angle_deg]} points) ...")
    seg_info = m._segment_corner()
    corner_wps = m._build_corner_waypoints(seg_info['arc_entry_ext_world'],
                                            seg_info['arc_exit_ext_world'])
    v_h = m._find_smooth_v_handoff(corner_wps)
    sweep = m._sweep_we(corner_wps, v_handoff=v_h, n_points=N_POINTS[angle_deg])

    knee_we = sweep['opt_we']
    res = sweep['res_by_we'][knee_we]
    print(f"  knee w_e={knee_we:.4f}, return_status={res.get('return_status', 'n/a')}")
    if res.get('return_status') != 'Solve_Succeeded':
        print(f"  WARNING: knee solve for {angle_deg} deg did not fully converge "
              f"({res.get('return_status')}); using it anyway since it came from "
              f"the same successful sweep that produced Table 2.")

    results[angle_deg] = {
        'states': res['states'],
        'we': knee_we,
        'arc_entry': seg_info['arc_entry_world'],
        'arc_exit': seg_info['arc_exit_world'],
        'vertex': seg_info['corner_vertex'],
    }

fig, ax = plt.subplots(figsize=(6.5, 6), num='Corner-angle trajectory overlay')

vertex = results[90]['vertex']
ax.scatter([vertex[0]], [vertex[1]], marker='x', s=90, color='black',
           zorder=6, label='Corner vertex (shared)')

for angle_deg in [45, 90, 135]:
    r = results[angle_deg]
    xy = r['states'][:, :2]
    ax.plot(xy[:, 0], xy[:, 1], color=COLORS[angle_deg], lw=2.0,
            label=f'{angle_deg}$^\\circ$ (knee, $w_e$={r["we"]:.3f})',
            zorder=4)
    ax.scatter([r['arc_entry'][0]], [r['arc_entry'][1]], marker=MARKERS[angle_deg],
               s=60, facecolor='white', edgecolor=COLORS[angle_deg], linewidth=1.5,
               zorder=5)
    ax.scatter([r['arc_exit'][0]], [r['arc_exit'][1]], marker=MARKERS[angle_deg],
               s=60, color=COLORS[angle_deg], zorder=5)

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_aspect('equal')
ax.legend(fontsize=9, loc='best')
ax.set_title('Corner trajectories at the knee weight, overlaid\n'
              '(open markers: arc entry, filled markers: arc exit)')
fig.tight_layout()
m._savefig(fig, 'fig_angle_trajectories.png')
print('Saved fig_angle_trajectories.png')
