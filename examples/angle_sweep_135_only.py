"""! Single-angle (135 deg) retry of angle_sweep_generalization.py at reduced
sweep resolution, since the full 3-angle run showed 135 deg converges far
slower than 45/90 deg at n_points=15 (individual solves ~400s vs ~30-60s).
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import differential_drive_path_segment_fine_sweep as m

N_SWEEP_POINTS = 8
beta_deg = 135.0
beta_rad = -np.deg2rad(beta_deg)
m.BETA = beta_rad
m.HEADING_OUT = float(m.HEADING_IN + beta_rad)

print(f"Angle = {beta_deg:.0f} deg (BETA={np.rad2deg(m.BETA):.1f}, "
      f"HEADING_OUT={np.rad2deg(m.HEADING_OUT):.1f})")
seg_info = m._segment_corner()
print(f"  Step-1 arc feasible = {seg_info['feasible']}")
corner_wps = m._build_corner_waypoints(seg_info['arc_entry_ext_world'],
                                        seg_info['arc_exit_ext_world'])
v_h = m._find_smooth_v_handoff(corner_wps)
sweep = m._sweep_we(corner_wps, v_handoff=v_h, n_points=N_SWEEP_POINTS)

oi, tri, eoi = sweep['opt_idx'], sweep['time_ref_idx'], sweep['energy_opt_idx']
print("\nRESULT_135:")
print(f"  feasible={seg_info['feasible']} v_handoff={v_h}")
print(f"  we_knee={sweep['opt_we']} we_energy={sweep['opt_we_energy']}")
print(f"  t_time_opt={sweep['mission_times'][tri]} pp_time_opt={sweep['peak_powers'][tri]} e_time_opt={sweep['total_energies'][tri]}")
print(f"  t_knee={sweep['mission_times'][oi]} pp_knee={sweep['peak_powers'][oi]} e_knee={sweep['total_energies'][oi]}")
print(f"  t_energy_opt={sweep['mission_times'][eoi]} pp_energy_opt={sweep['peak_powers'][eoi]} e_energy_opt={sweep['total_energies'][eoi]}")
