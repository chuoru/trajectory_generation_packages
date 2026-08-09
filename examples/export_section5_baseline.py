"""! Export the full Section V (Experimental Results) simulated baseline
dataset to disk, for later comparison against real-hardware validation runs.

Saves, per method (A, B, D, C):
  - raw per-seed tracking/energy metrics (one row per seed, N=25 seeds) --
    NOT just the mean/std summary already printed in the paper's Table 6/7,
    so a future real-hardware dataset can be compared with the same or
    additional statistical tests (paired or unpaired) against the full
    simulated distribution, not just its first two moments.
  - the seed-0 representative time-series (tracked x,y,theta,v,omega vs.
    reference), matching what Fig. 9/10 plot.
  - summary statistics (mean/std/min/max), matching Table 6/7 exactly, as a
    sanity-checkable cross-reference to the paper.
  - the sensor-noise-model and pure-pursuit controller parameters actually
    used, and basic run metadata, so the baseline is fully reproducible/
    comparable even if this codebase changes later.

Output directory: csv_output_fine_sweep/section5_baseline_export/
"""
import os
import json
import datetime

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

import differential_drive_comparison as dc
import section5_purepursuit as s5
import purepursuit_fine_sweep as pps

OUT_DIR = os.path.join(os.path.dirname(__file__), 'csv_output_fine_sweep',
                       'section5_baseline_export')
os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS = s5.N_SEEDS
METHOD_WE = {'B': 0.0, 'D': 0.0769, 'C': 0.4872}

print("Building Method A (EulerJLAP full path) ...")
a_dict, T_a = s5._method_a_dict()

print("Building Methods B, D & C (segmented pipeline, shared sweep, cached) ...")
seg = dc._run_segmented_pipeline()
res_b = dc._build_segmented_result(seg, METHOD_WE['B'])
res_d = dc._build_segmented_result(seg, METHOD_WE['D'])
res_c = dc._build_segmented_result(seg, METHOD_WE['C'])
T_b = res_b['T_s1'] + res_b['T_corner'] + res_b['T_s2']
T_d = res_d['T_s1'] + res_d['T_corner'] + res_d['T_s2']
T_c = res_c['T_s1'] + res_c['T_corner'] + res_c['T_s2']

METHODS = {
    'A': (a_dict, T_a),
    'B': (s5._segmented_dict(res_b), T_b),
    'D': (s5._segmented_dict(res_d), T_d),
    'C': (s5._segmented_dict(res_c), T_c),
}

METRIC_KEYS = ['mission_time', 'energy_sim', 'peak_power_sim', 'energy_meas',
              'peak_power_meas', 'model_rmse', 'max_cte_cm', 'mean_cte_cm',
              'rms_cte_cm', 'final_err_cm', 'max_he_deg', 'mean_he_deg',
              'max_verr_ms', 'track_dur_s']

summary = {}
for label, (d, mission_time) in METHODS.items():
    print(f"\nTracking Method {label}: {N_SEEDS} seeds ...")
    raw_rows = []
    seed0_traj = seed0_sim = None
    for seed in range(N_SEEDS):
        metrics, traj, sim = s5._track_one_seed(d, mission_time, seed)
        row = {'seed': seed}
        row.update(metrics)
        raw_rows.append(row)
        if seed == 0:
            seed0_traj, seed0_sim = traj, sim

    # --- raw per-seed CSV -------------------------------------------------
    raw_path = os.path.join(OUT_DIR, f'method_{label}_raw_per_seed.csv')
    header = ['seed'] + METRIC_KEYS
    with open(raw_path, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in raw_rows:
            f.write(','.join(str(row[k]) for k in header) + '\n')
    print(f"  Saved {raw_path}")

    # --- summary stats (matches Table 6/7) --------------------------------
    stats = {}
    for k in METRIC_KEYS:
        vals = np.array([row[k] for row in raw_rows])
        stats[k] = {'mean': float(vals.mean()), 'std': float(vals.std()),
                    'min': float(vals.min()), 'max': float(vals.max())}
    summary[label] = stats

    # --- seed-0 representative time series (matches Fig. 9/10) ------------
    nt = min(seed0_traj.x.shape[0], seed0_sim.x_out.shape[1])
    ts_path = os.path.join(OUT_DIR, f'method_{label}_seed0_timeseries.csv')
    with open(ts_path, 'w') as f:
        f.write('t,x_ref,y_ref,theta_ref,v_ref,omega_ref,'
                'x_trk,y_trk,theta_trk,v_trk,omega_trk\n')
        for i in range(nt):
            f.write(f"{seed0_sim.t_out[i]},"
                     f"{seed0_traj.x[i,0]},{seed0_traj.x[i,1]},{seed0_traj.x[i,2]},"
                     f"{seed0_traj.u[0,i]},{seed0_traj.u[1,i]},"
                     f"{seed0_sim.x_out[0,i]},{seed0_sim.x_out[1,i]},{seed0_sim.x_out[2,i]},"
                     f"{seed0_sim.u_out[0,i]},{seed0_sim.u_out[1,i]}\n")
    print(f"  Saved {ts_path}")

# --- summary stats JSON (all methods) -------------------------------------
summary_path = os.path.join(OUT_DIR, 'summary_stats.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved {summary_path}")

# --- metadata: sensor model, controller params, run info -----------------
metadata = {
    'export_timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
    'n_seeds': N_SEEDS,
    'reference_scenario': {
        'waypoints_m': [[0, 0], [5, 0], [5, 5]],
        'corridor_half_width_m': 0.25,
        'corner_angle_deg': 90,
    },
    'method_energy_weights': METHOD_WE,
    'sensor_noise_model': {
        'gps_rate_hz': pps.GPS_RATE_HZ,
        'gps_std_xy_m': pps.GPS_STD_XY,
        'imu_control_rate_hz': 1.0 / pps.SIM_DT,
        'imu_heading_std_rad': pps.IMU_STD_THETA,
        'imu_velocity_std_mps': pps.IMU_STD_V,
    },
    'pure_pursuit_controller': {
        'lookahead_gain_kv_s': 0.4,
        'lookahead_gain_kv_note': 'INACTIVE in this dataset: forward-velocity '
            'estimate fed into the lookahead search is bugged (wired to 0), '
            'so L_d is held constant at L_d_min throughout every run below. '
            'Fix this before comparing against real hardware where the '
            'lookahead law is presumably speed-adaptive as designed.',
        'min_lookahead_Ld_min_m': 0.25,
        'speed_kp_per_s': 6.0,
        'speed_ki_per_s2': 0.1,
        'feedforward_gain_kff': 1.0,
        'max_linear_accel_mps2': 1.0,
        'max_angular_accel_radps2': 4.0,
        'control_rate_hz': 20,
    },
    'motor_power_model': 'polynomial P_w(v_w, a_w) per wheel, coefficients '
        'in Table 1 of the paper / ENERGY_COEFFS_RIGHT/LEFT in '
        'differential_drive_comparison.py; P_elec=2.0W constant hotel load '
        'added for total system power.',
    'robot_params': {'mass_kg': 50.4, 'wheelbase_m': 0.53, 'wheel_radius_m': 0.15},
    'metric_definitions': {
        'mission_time': 'planned trajectory duration [s]',
        'energy_sim': 'open-loop (planned) total energy [J], from acc_path/alpha fields',
        'peak_power_sim': 'open-loop (planned) peak total power [W]',
        'energy_meas': 'closed-loop (tracked) total energy [J], integrated from tracked v/omega',
        'peak_power_meas': 'closed-loop (tracked) peak total power [W]',
        'model_rmse': 'RMSE [W] between open-loop reference power and tracked power',
        'max_cte_cm/mean_cte_cm/rms_cte_cm': 'cross-track error [cm], point-to-polyline vs reference path',
        'final_err_cm': 'final position error [cm]',
        'max_he_deg/mean_he_deg': 'heading error [deg], wrap-corrected tracked-minus-reference',
        'max_verr_ms': 'max absolute forward-velocity tracking error [m/s]',
        'track_dur_s': 'actual simulated tracking duration [s]',
    },
    'source_paper': 'Energy-Aware Function-Parameterized Optimal Control for '
        'Mobile Robot Trajectory Generation (IEEE Access submission), Section V',
    'note': 'ALL data in this export is SIMULATED (GPS/IMU noise model + '
        'simulated differential-drive plant), not from physical hardware. '
        'Intended as a baseline for later comparison against real-hardware '
        'validation of the same four methods (A/B/D/C) on the same reference '
        'scenario.',
}
meta_path = os.path.join(OUT_DIR, 'metadata.json')
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Saved {meta_path}")

print(f"\nAll Section V baseline data exported to: {OUT_DIR}")
