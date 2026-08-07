"""! Isolated 135-degree sweep (matching angle_sweep_135_only.py's proven-
convergent conditions exactly) to get a properly Solve_Succeeded trajectory
at the knee weight for fig_angle_trajectories.png, since running it back-
to-back with 45/90 deg in one process was less reliable.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import differential_drive_path_segment_fine_sweep as m

beta_deg = 135.0
beta_rad = -np.deg2rad(beta_deg)
m.BETA = beta_rad
m.HEADING_OUT = float(m.HEADING_IN + beta_rad)

seg_info = m._segment_corner()
corner_wps = m._build_corner_waypoints(seg_info['arc_entry_ext_world'],
                                        seg_info['arc_exit_ext_world'])
v_h = m._find_smooth_v_handoff(corner_wps)
sweep = m._sweep_we(corner_wps, v_handoff=v_h, n_points=8)

knee_we = sweep['opt_we']
res = sweep['res_by_we'][knee_we]
print(f"RESULT_135_TRAJ: we={knee_we} status={res.get('return_status')}")

out_path = os.path.join(os.path.dirname(__file__), 'csv_output_fine_sweep',
                        'traj_135_knee.csv')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.savetxt(out_path, res['states'][:, :2], delimiter=',', header='x,y', comments='')
print(f"Saved {out_path}")
print(f"arc_entry={seg_info['arc_entry_world']} arc_exit={seg_info['arc_exit_world']} "
      f"vertex={seg_info['corner_vertex']}")
