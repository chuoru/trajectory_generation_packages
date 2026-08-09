"""! Export closed-loop tracking data for an ALTERNATE Method D, solved at
w_e=0.0513 instead of the paper's official knee weight w_e=0.0769.

w_e=0.0513 is not one of the cached sweep's kept target weights (0.0, 0.0769,
0.4872 -- see differential_drive_comparison.py's _sweep_we), so this script
solves it fresh, warm-started from the already-converged w_e=0.0769 corner
result, then runs the same N=25-seed closed-loop pure-pursuit tracking used
for the paper's Method D and exports the same file set as
export_section5_baseline.py, under a separate 'D_we0513' label so it can be
compared directly against the existing w_e=0.0769 baseline export without
overwriting it.

Output directory: csv_output_fine_sweep/section5_D_we0513_export/
"""
import os
import json
import datetime

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

import differential_drive_comparison as dc
import section5_purepursuit as s5

OUT_DIR = os.path.join(os.path.dirname(__file__), 'csv_output_fine_sweep',
                       'section5_D_we0513_export')
os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS = s5.N_SEEDS
WE_ALT = 0.0513
WE_WARM_START_FROM = 0.0769

print("Loading cached segmented pipeline (for res_s1/res_s2/v_handoff/corner_wps) ...")
seg = dc._run_segmented_pipeline()

seg_info = seg['seg_info']
corner_wps = dc._build_corner_waypoints(seg_info['arc_entry_ext_world'],
                                        seg_info['arc_exit_ext_world'])
v_handoff = seg['v_handoff']
res_s1 = seg['res_s1']
a_s1_exit = float(res_s1['acc_path'][-1])
alpha_s1_exit = float(res_s1['alpha'][-1])

warm_start_res = seg['sweep']['res_by_we'][WE_WARM_START_FROM]
print(f"Solving corner at w_e={WE_ALT} (warm-started from w_e={WE_WARM_START_FROM}) ...")
res_corner_alt = dc._solve_corner(
    corner_wps, WE_ALT, warm_start=warm_start_res,
    v_entry=v_handoff, v_exit=v_handoff,
    a_entry=a_s1_exit, alpha_entry=alpha_s1_exit,
)
status = res_corner_alt.get('return_status', 'n/a')
print(f"  return_status={status}")
if status != 'Solve_Succeeded':
    print(f"  WARNING: corner solve at w_e={WE_ALT} did not fully converge "
          f"({status}). Proceeding anyway but flagging this in the metadata.")

res_seg_alt = dict(
    seg_info=seg_info, res_s1=seg['res_s1'], res_s2=seg['res_s2'],
    res_corner=res_corner_alt, v_handoff=v_handoff,
    T_s1=seg['T_s1'], T_corner=float(res_corner_alt['time'][-1]), T_s2=seg['T_s2'],
)
d_alt_dict = s5._segmented_dict(res_seg_alt)
T_d_alt = res_seg_alt['T_s1'] + res_seg_alt['T_corner'] + res_seg_alt['T_s2']

print(f"\nTracking Method D (w_e={WE_ALT}): {N_SEEDS} seeds ...")
METRIC_KEYS = ['mission_time', 'energy_sim', 'peak_power_sim', 'energy_meas',
              'peak_power_meas', 'model_rmse', 'max_cte_cm', 'mean_cte_cm',
              'rms_cte_cm', 'final_err_cm', 'max_he_deg', 'mean_he_deg',
              'max_verr_ms', 'track_dur_s']

raw_rows = []
seed0_traj = seed0_sim = None
for seed in range(N_SEEDS):
    metrics, traj, sim = s5._track_one_seed(d_alt_dict, T_d_alt, seed)
    row = {'seed': seed}
    row.update(metrics)
    raw_rows.append(row)
    if seed == 0:
        seed0_traj, seed0_sim = traj, sim

label = 'D_we0513'

raw_path = os.path.join(OUT_DIR, f'method_{label}_raw_per_seed.csv')
header = ['seed'] + METRIC_KEYS
with open(raw_path, 'w') as f:
    f.write(','.join(header) + '\n')
    for row in raw_rows:
        f.write(','.join(str(row[k]) for k in header) + '\n')
print(f"  Saved {raw_path}")

stats = {}
for k in METRIC_KEYS:
    vals = np.array([row[k] for row in raw_rows])
    stats[k] = {'mean': float(vals.mean()), 'std': float(vals.std()),
                'min': float(vals.min()), 'max': float(vals.max())}

summary_path = os.path.join(OUT_DIR, 'summary_stats.json')
with open(summary_path, 'w') as f:
    json.dump({label: stats}, f, indent=2)
print(f"  Saved {summary_path}")

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

metadata = {
    'export_timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
    'n_seeds': N_SEEDS,
    'method_label': label,
    'w_e': WE_ALT,
    'warm_started_from_we': WE_WARM_START_FROM,
    'corner_solve_return_status': status,
    'note': 'This is an ALTERNATE Method D at w_e=0.0513, NOT the paper\'s '
        'official knee weight (w_e=0.0769). w_e=0.0513 was, earlier in this '
        'project, the "robust knee" candidate when the epsilon-tolerance '
        'search was restricted to the peak-power curve\'s not-yet-saturated '
        'region; the paper ultimately uses w_e=0.0769 (see Section III, '
        'eq:knee) as the primary automated criterion\'s selection. This '
        'export exists for direct comparison against the existing '
        'section5_baseline_export (w_e=0.0769) data, both tracked under the '
        'SAME simulated GPS/IMU noise model and pure-pursuit controller '
        '(with the same fixed, not speed-adaptive, lookahead-distance '
        'defect -- see Table 5/Section V of the paper).',
    'sibling_export_for_comparison': 'section5_baseline_export/ '
        '(contains Methods A, B, D [w_e=0.0769], C)',
}
meta_path = os.path.join(OUT_DIR, 'metadata.json')
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  Saved {meta_path}")

print(f"\nAll w_e={WE_ALT} Method D data exported to: {OUT_DIR}")
