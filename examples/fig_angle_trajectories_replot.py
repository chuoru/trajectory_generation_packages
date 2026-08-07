"""! Rebuild fig_angle_trajectories.png with the legend moved to lower right
(matching Fig. 6's style request). Re-solves only 45/90 deg (fast, reliable)
and reuses the already-saved, independently-verified 135 deg trajectory from
traj_135_knee.csv (see fig_angle_traj_135_only.py) instead of re-solving the
slow/flaky 135 deg case again.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import matplotlib.pyplot as plt

import differential_drive_path_segment_fine_sweep as m

m._set_paper_style()

COLORS  = {45: '#1f77b4', 90: '#d62728', 135: '#2ca02c'}
MARKERS = {45: 'o', 90: 's', 135: '^'}

cache_dir = os.path.join(os.path.dirname(__file__), 'csv_output_fine_sweep')
os.makedirs(cache_dir, exist_ok=True)


def _swap(pt):
    """Plot-orientation transform (x'=y, y'=x): converts this script's
    north-to-east turn into an east-to-north turn, matching fig_corridor.png
    (Fig. 6)'s approach-from-the-left, turn-toward-bottom-right, exit-upward
    convention (a coordinate swap, not a rotation, since the two turns are
    mirror images of each other in handedness)."""
    pt = np.asarray(pt)
    if pt.ndim == 1:
        return np.array([pt[1], pt[0]])
    return np.column_stack([pt[:, 1], pt[:, 0]])


results = {}
for angle_deg in [45, 90]:
    beta_rad = -np.deg2rad(angle_deg)
    m.BETA = beta_rad
    m.HEADING_OUT = float(m.HEADING_IN + beta_rad)

    print(f"Sweeping {angle_deg} deg corner (15 points) ...")
    seg_info = m._segment_corner()
    corner_wps = m._build_corner_waypoints(seg_info['arc_entry_ext_world'],
                                            seg_info['arc_exit_ext_world'])
    v_h = m._find_smooth_v_handoff(corner_wps)
    sweep = m._sweep_we(corner_wps, v_handoff=v_h, n_points=15)
    knee_we = sweep['opt_we']
    res = sweep['res_by_we'][knee_we]
    print(f"  knee w_e={knee_we:.4f}, status={res.get('return_status')}")
    results[angle_deg] = {
        'states': res['states'], 'we': knee_we,
        'arc_entry': seg_info['arc_entry_world'],
        'arc_exit': seg_info['arc_exit_world'],
        'vertex': seg_info['corner_vertex'],
    }
    # Cache so any further plot-only tweak never needs to re-solve again.
    np.savetxt(os.path.join(cache_dir, f'traj_{angle_deg}_knee_cache.csv'),
               res['states'][:, :2], delimiter=',', header='x,y', comments='')
    with open(os.path.join(cache_dir, f'traj_{angle_deg}_anchors_cache.csv'), 'w') as f:
        f.write('name,x,y\n')
        f.write(f"we,{knee_we},0\n")
        f.write(f"arc_entry,{seg_info['arc_entry_world'][0]},{seg_info['arc_entry_world'][1]}\n")
        f.write(f"arc_exit,{seg_info['arc_exit_world'][0]},{seg_info['arc_exit_world'][1]}\n")
        f.write(f"vertex,{seg_info['corner_vertex'][0]},{seg_info['corner_vertex'][1]}\n")

# Reuse the already-verified 135 deg trajectory (matches Table 2 exactly;
# see fig_angle_traj_135_only.py / traj_135_knee.csv).
xy_135 = np.loadtxt(os.path.join(cache_dir, 'traj_135_knee.csv'),
                     delimiter=',', skiprows=1)
results[135] = {
    'states': xy_135, 'we': 0.142857,
    'arc_entry': np.array([-6.123234e-17, 9.000000e+00]),
    'arc_exit': np.array([0.70710678, 9.29289322]),
    'vertex': np.array([0.0, 10.0]),
}

# Apply the plot-orientation swap to everything before drawing.
for angle_deg in [45, 90, 135]:
    r = results[angle_deg]
    r['states'] = _swap(r['states'])
    r['arc_entry'] = _swap(r['arc_entry'])
    r['arc_exit'] = _swap(r['arc_exit'])
    r['vertex'] = _swap(r['vertex'])

fig, ax = plt.subplots(figsize=(6.5, 6), num='Corner-angle trajectory overlay')

vertex = results[90]['vertex']
ax.scatter([vertex[0]], [vertex[1]], marker='x', s=90, color='black',
           zorder=6, label='Corner vertex (shared)')

for angle_deg in [45, 90, 135]:
    r = results[angle_deg]
    xy = r['states'][:, :2] if r['states'].ndim == 2 and r['states'].shape[1] >= 2 else r['states']
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
ax.legend(fontsize=9, loc='lower right')
ax.set_title('Corner trajectories at the knee weight, overlaid\n'
              '(open markers: arc entry, filled markers: arc exit)')
fig.tight_layout()
m._savefig(fig, 'fig_angle_trajectories.png')
print('Saved fig_angle_trajectories.png')
