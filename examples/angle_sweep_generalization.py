"""! Corner-angle generalisation check: does the paper's energy-aware pipeline
and automated knee-selection criterion still behave sensibly at turning
angles other than the single 90-degree corner used throughout the paper?

Peer review (methodology, cross-disciplinary, devil's advocate) flagged that
every quantitative result in the paper comes from one 90-degree corner. This
script reuses the exact same segmented pipeline
(differential_drive_path_segment_fine_sweep.py) at two additional angles,
45 degrees (shallow) and 135 degrees (sharp), by overriding its module-level
BETA/HEADING_OUT constants -- the 90-degree pipeline module and its cached
results/paper figures are NOT modified.

Sweep resolution is reduced (n_points=15 instead of the paper's 40) purely
for wall-clock cost; the saturation-based knee criterion and adaptive
handoff-velocity search are otherwise identical to the paper's methodology.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

import differential_drive_path_segment_fine_sweep as m

N_SWEEP_POINTS = 15

ANGLES_DEG = [45.0, 90.0, 135.0]


def _run_for_angle(beta_deg):
    """Override BETA/HEADING_OUT, rebuild geometry, and run a reduced sweep."""
    beta_rad = -np.deg2rad(beta_deg)   # right turn, same sign convention as the module default
    m.BETA = beta_rad
    m.HEADING_OUT = float(m.HEADING_IN + beta_rad)

    print(f"\n{'='*70}\nAngle = {beta_deg:.0f} deg  "
          f"(BETA={np.rad2deg(m.BETA):.1f} deg, HEADING_OUT={np.rad2deg(m.HEADING_OUT):.1f} deg)\n{'='*70}")

    seg_info = m._segment_corner()
    print(f"  Step-1 arc feasible = {seg_info['feasible']}")
    corner_wps = m._build_corner_waypoints(seg_info['arc_entry_ext_world'],
                                            seg_info['arc_exit_ext_world'])

    v_h = m._find_smooth_v_handoff(corner_wps)
    sweep = m._sweep_we(corner_wps, v_handoff=v_h, n_points=N_SWEEP_POINTS)

    oi, tri, eoi = sweep['opt_idx'], sweep['time_ref_idx'], sweep['energy_opt_idx']
    return {
        'angle_deg':    beta_deg,
        'feasible':     bool(seg_info['feasible']),
        'v_handoff':    v_h,
        'we_knee':      sweep['opt_we'],
        'we_energy':    sweep['opt_we_energy'],
        't_time_opt':   sweep['mission_times'][tri],
        'pp_time_opt':  sweep['peak_powers'][tri],
        'e_time_opt':   sweep['total_energies'][tri],
        't_knee':       sweep['mission_times'][oi],
        'pp_knee':      sweep['peak_powers'][oi],
        'e_knee':       sweep['total_energies'][oi],
        't_energy_opt': sweep['mission_times'][eoi],
        'pp_energy_opt':sweep['peak_powers'][eoi],
        'e_energy_opt': sweep['total_energies'][eoi],
    }


def main():
    results = [_run_for_angle(a) for a in ANGLES_DEG]

    print(f"\n\n{'='*100}")
    print("SUMMARY: corner-angle generalisation")
    print(f"{'='*100}")
    hdr = (f"  {'Angle':>6} {'Feas.':>6} {'v_h':>6} {'we*knee':>8} {'we*egy':>8} | "
           f"{'T_B':>6} {'Pk_B':>6} {'E_B':>7} | {'T_D':>6} {'Pk_D':>6} {'E_D':>7} | "
           f"{'T_C':>6} {'Pk_C':>6} {'E_C':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        print(f"  {r['angle_deg']:>5.0f}d {str(r['feasible']):>6} {r['v_handoff']:>6.3f} "
              f"{r['we_knee']:>8.4f} {r['we_energy']:>8.4f} | "
              f"{r['t_time_opt']:>6.2f} {r['pp_time_opt']:>6.2f} {r['e_time_opt']:>7.2f} | "
              f"{r['t_knee']:>6.2f} {r['pp_knee']:>6.2f} {r['e_knee']:>7.2f} | "
              f"{r['t_energy_opt']:>6.2f} {r['pp_energy_opt']:>6.2f} {r['e_energy_opt']:>7.2f}")

    out_dir = os.path.join(os.path.dirname(__file__), 'csv_output_fine_sweep')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'angle_sweep_generalization.csv')
    with open(out_path, 'w') as f:
        f.write("angle_deg,feasible,v_handoff,we_knee,we_energy,"
                "t_time_opt,pp_time_opt,e_time_opt,"
                "t_knee,pp_knee,e_knee,"
                "t_energy_opt,pp_energy_opt,e_energy_opt\n")
        for r in results:
            f.write(f"{r['angle_deg']},{r['feasible']},{r['v_handoff']},"
                     f"{r['we_knee']},{r['we_energy']},"
                     f"{r['t_time_opt']},{r['pp_time_opt']},{r['e_time_opt']},"
                     f"{r['t_knee']},{r['pp_knee']},{r['e_knee']},"
                     f"{r['t_energy_opt']},{r['pp_energy_opt']},{r['e_energy_opt']}\n")
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
