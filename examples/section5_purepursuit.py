#!/usr/bin/env python3
##
# @file section5_purepursuit.py
#
# @brief Closed-loop Pure Pursuit + GPS/IMU-noise tracking validation for
#        Methods A, B, D, C -- built from the SAME full trajectories as
#        differential_drive_comparison.py (Section IV / Table "metrics"),
#        so mission times/energies here are directly consistent with that
#        table. Runs each method under N_SEEDS independent noise
#        realisations and reports mean/min/max per metric, so Section V's
#        tracking-robustness claims are backed by repeated trials rather
#        than a single noise draw.
#
# Reuses:
#   - differential_drive_comparison.py: _run_method_a, _run_segmented_pipeline,
#     _build_segmented_result  (builds A/B/D/C exactly like Section IV)
#   - purepursuit_fine_sweep.py: corner_to_trajectory, run_purepursuit,
#     cross_track_error, heading_error, compute_motor_power  (same simulator,
#     same GPS/IMU noise model, same metric definitions as the rest of the
#     paper's pure-pursuit validation)
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/08/04
# - Extended to Method D + multi-seed statistics on 2026/08/05

import sys
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator as _Pchip

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import differential_drive_comparison as dc
import purepursuit_fine_sweep as pps

SAVE_FIGS = True
FIG_OUT_DIR = (pathlib.Path(__file__).resolve().parent.parent.parent
               / 'Writting' / 'energy_aware')

N_SEEDS = 25   # independent GPS/IMU noise realisations per method

COL_A, COL_B, COL_D, COL_C, COL_REF = (
    'steelblue', 'tomato', 'darkorange', 'seagreen', '#888888')


def _savefig(fig, name):
    if SAVE_FIGS:
        out = FIG_OUT_DIR / name
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f"  [paper] Saved {name} -> {out}")


# =============================================================================
# BUILD FULL A/B/D/C REFERENCE TRAJECTORIES (uniform-time, x/y/theta/v/omega)
# =============================================================================
def _method_a_dict():
    res = dc._run_method_a()
    return {
        'time':     res['time'],
        'x':        res['states'][:, 0],
        'y':        res['states'][:, 1],
        'theta':    res['states'][:, 2],
        'v':        res['v'],
        'omega':    res['omega'],
        'acc_path': res['acc_path'],
        'alpha':    res['alpha'],
    }, float(res['time'][-1])


def _segmented_dict(res_seg):
    """Stitch res_s1 + res_corner + res_s2 onto one absolute-time axis with
    x/y/theta/v/omega all sampled at the SAME points (corner's x/y/theta are
    interpolated from the coarse OCP-node grid onto its fine time_ik grid,
    matching how v/omega are already represented there)."""
    res_s1, res_s2, res_corner = (res_seg['res_s1'], res_seg['res_s2'],
                                   res_seg['res_corner'])
    T_s1, T_corner = res_seg['T_s1'], res_seg['T_corner']

    t_ik = res_corner['time_ik']
    x_c = _Pchip(res_corner['time'], res_corner['states'][:, 0])(t_ik)
    y_c = _Pchip(res_corner['time'], res_corner['states'][:, 1])(t_ik)
    th_c = _Pchip(res_corner['time'], res_corner['states'][:, 2])(t_ik)

    time_full = np.concatenate([
        res_s1['time'],
        t_ik + T_s1,
        res_s2['time'][1:] + T_s1 + T_corner,
    ])
    x_full = np.concatenate([res_s1['states'][:, 0], x_c, res_s2['states'][1:, 0]])
    y_full = np.concatenate([res_s1['states'][:, 1], y_c, res_s2['states'][1:, 1]])
    th_full = np.concatenate([res_s1['states'][:, 2], th_c, res_s2['states'][1:, 2]])
    v_full = np.concatenate([res_s1['v'], res_corner['v'], res_s2['v'][1:]])
    om_full = np.concatenate([res_s1['omega'], res_corner['omega'], res_s2['omega'][1:]])
    # acc_path/alpha: Pchip-interpolated directly from each stage's own
    # analytically-correct acceleration (JLAP's own profile; the corner's
    # already available on time_ik) -- NOT re-derived from v/omega, which
    # would reintroduce the differentiation artifact _compute_power_uniform
    # was found to have (see differential_drive_comparison.py).
    acc_full = np.concatenate([res_s1['acc_path'], res_corner['acc_path'],
                               res_s2['acc_path'][1:]])
    alpha_full = np.concatenate([res_s1['alpha'], res_corner['alpha'],
                                 res_s2['alpha'][1:]])

    return {
        'time': time_full, 'x': x_full, 'y': y_full, 'theta': th_full,
        'v': v_full, 'omega': om_full,
        'acc_path': acc_full, 'alpha': alpha_full,
    }


# =============================================================================
# CLOSED-LOOP TRACKING + METRICS  (identical definitions to purepursuit_fine_sweep.py)
# =============================================================================
def _track_one_seed(d, mission_time, seed):
    traj = pps.corner_to_trajectory(d, dt=pps.SIM_DT)
    sim = pps.run_purepursuit(traj, seed=seed)

    nt = min(traj.x.shape[0], sim.x_out.shape[1])
    trk_xy = sim.x_out[:2, :].T
    ref_xy = traj.x[:, :2]
    cte = pps.cross_track_error(ref_xy, trk_xy)
    he = pps.heading_error(traj.x[:nt, 2], sim.x_out[2, :nt])
    v_err = np.abs(traj.u[0, :nt] - sim.u_out[0, :nt])
    final_err = np.hypot(sim.x_out[0, -1] - traj.x[-1, 0],
                          sim.x_out[1, -1] - traj.x[-1, 1])

    # Uses the analytically-correct acc_path/alpha fields, not np.gradient
    # re-differentiation (see dc._compute_power_uniform's docstring).
    power_ref = dc._compute_power_from_accel(d['v'], d['omega'],
                                             d['acc_path'], d['alpha'])
    total_energy_ref = float(np.trapz(power_ref, d['time']))
    peak_power_ref = float(power_ref.max())

    v_trk = sim.u_out[0, :nt]
    om_trk = sim.u_out[1, :nt]
    t_trk = sim.t_out[:nt]
    power_trk = pps.compute_motor_power(v_trk, om_trk, t_trk)
    total_energy_trk = float(np.trapz(power_trk, t_trk))
    peak_power_trk = float(power_trk.max())

    metrics = {
        'mission_time':     mission_time,
        'energy_sim':       total_energy_ref,
        'peak_power_sim':   peak_power_ref,
        'energy_meas':      total_energy_trk,
        'peak_power_meas':  peak_power_trk,
        'model_rmse':       float(np.sqrt(np.mean(
            (np.interp(t_trk, d['time'], power_ref) - power_trk) ** 2))),
        'max_cte_cm':       float(cte.max()  * 1e2),
        'mean_cte_cm':      float(cte.mean() * 1e2),
        'rms_cte_cm':       float(np.sqrt((cte ** 2).mean()) * 1e2),
        'final_err_cm':     float(final_err  * 1e2),
        'max_he_deg':       float(np.rad2deg(np.abs(he).max())),
        'mean_he_deg':      float(np.rad2deg(np.abs(he).mean())),
        'max_verr_ms':      float(v_err.max()),
        'track_dur_s':      float(sim.t_out[-1]),
    }
    return metrics, traj, sim


def _track_multi_seed(label, d, mission_time, n_seeds=N_SEEDS):
    """Run n_seeds independent noise realisations; return (stats, traj0, sim0)
    where stats[key] = {'mean':, 'std':, 'min':, 'max':} and traj0/sim0 are
    the seed=0 run, kept as the representative trajectory for XY/power plots.
    """
    all_metrics = []
    traj0 = sim0 = None
    for seed in range(n_seeds):
        m, traj, sim = _track_one_seed(d, mission_time, seed)
        all_metrics.append(m)
        if seed == 0:
            traj0, sim0 = traj, sim
    print(f"  Method {label}: {n_seeds} seeds tracked.")

    keys = all_metrics[0].keys()
    stats = {}
    for k in keys:
        vals = np.array([m[k] for m in all_metrics])
        stats[k] = {
            'mean': float(vals.mean()),
            'std':  float(vals.std()),
            'min':  float(vals.min()),
            'max':  float(vals.max()),
        }
    return stats, traj0, sim0


def main():
    print("Building Method A (EulerJLAP full path) ...")
    a_dict, T_a = _method_a_dict()

    print("Building Methods B, D & C (segmented pipeline, shared sweep) ...")
    seg = dc._run_segmented_pipeline()
    we_b, we_d, we_c = 0.0, 0.0769, 0.4872
    res_b = dc._build_segmented_result(seg, we_b)
    res_d = dc._build_segmented_result(seg, we_d)
    res_c = dc._build_segmented_result(seg, we_c)
    T_b = res_b['T_s1'] + res_b['T_corner'] + res_b['T_s2']
    T_d = res_d['T_s1'] + res_d['T_corner'] + res_d['T_s2']
    T_c = res_c['T_s1'] + res_c['T_corner'] + res_c['T_s2']
    b_dict = _segmented_dict(res_b)
    d_dict = _segmented_dict(res_d)
    c_dict = _segmented_dict(res_c)

    print(f"\nRunning closed-loop Pure Pursuit ({N_SEEDS} seeds/method, "
          f"GPS/IMU noise) for A, B, D, C ...")
    s_a, traj_a, sim_a = _track_multi_seed('A', a_dict, T_a)
    s_b, traj_b, sim_b = _track_multi_seed('B', b_dict, T_b)
    s_d, traj_d, sim_d = _track_multi_seed('D', d_dict, T_d)
    s_c, traj_c, sim_c = _track_multi_seed('C', c_dict, T_c)

    print(f"\n  {'Metric':22s} {'Method A':>18s} {'Method B':>18s} "
          f"{'Method D':>18s} {'Method C':>18s}")
    print("  " + "-" * 98)
    rows = [
        ('Mission time [s]',      'mission_time'),
        ('Peak power sim. [W]',   'peak_power_sim'),
        ('Peak power meas. [W]',  'peak_power_meas'),
        ('Total energy sim. [J]', 'energy_sim'),
        ('Total energy meas. [J]','energy_meas'),
        ('Model error RMSE [W]',  'model_rmse'),
        ('Max CTE [cm]',          'max_cte_cm'),
        ('Mean CTE [cm]',         'mean_cte_cm'),
        ('RMS CTE [cm]',          'rms_cte_cm'),
        ('Final pos error [cm]',  'final_err_cm'),
        ('Max heading err [deg]', 'max_he_deg'),
        ('Mean heading err [deg]','mean_he_deg'),
        ('Max vel error [m/s]',   'max_verr_ms'),
    ]

    def _fmt(st):
        return f"{st['mean']:.3f}±{st['std']:.3f}"

    for lbl, key in rows:
        print(f"  {lbl:22s} {_fmt(s_a[key]):>18s} {_fmt(s_b[key]):>18s} "
              f"{_fmt(s_d[key]):>18s} {_fmt(s_c[key]):>18s}")

    print(f"\n  {'Metric':22s} {'A [min,max]':>24s} {'B [min,max]':>24s} "
          f"{'D [min,max]':>24s} {'C [min,max]':>24s}")
    print("  " + "-" * 118)
    for lbl, key in rows:
        def _mm(st):
            return f"[{st['min']:.3f}, {st['max']:.3f}]"
        print(f"  {lbl:22s} {_mm(s_a[key]):>24s} {_mm(s_b[key]):>24s} "
              f"{_mm(s_d[key]):>24s} {_mm(s_c[key]):>24s}")

    # ------------------------------------------------------------------
    # Figures for the paper: fig_exp_xy.png, fig_exp_power.png
    # (representative seed=0 run; tables above carry the multi-seed stats)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    for lbl, traj, sim, col in [('A', traj_a, sim_a, COL_A),
                                 ('B', traj_b, sim_b, COL_B),
                                 ('D', traj_d, sim_d, COL_D),
                                 ('C', traj_c, sim_c, COL_C)]:
        ax.plot(traj.x[:, 0], traj.x[:, 1], color=col, ls='--', lw=1.2, alpha=0.6)
        ax.plot(sim.x_out[0, :], sim.x_out[1, :], color=col, ls='-', lw=2.0,
                label=f'Method {lbl} (tracked)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.set_title('Closed-loop tracking under GPS/IMU noise (seed 0 of '
                 f'{N_SEEDS})\n(dashed: reference, solid: tracked)')
    fig.tight_layout()
    _savefig(fig, 'fig_exp_xy.png')

    fig2, ax2 = plt.subplots(figsize=(7, 4.2))
    for lbl, traj, sim, col in [('A', traj_a, sim_a, COL_A),
                                 ('B', traj_b, sim_b, COL_B),
                                 ('D', traj_d, sim_d, COL_D),
                                 ('C', traj_c, sim_c, COL_C)]:
        nt = sim.t_out.shape[0]
        P_trk = pps.compute_motor_power(sim.u_out[0, :nt], sim.u_out[1, :nt], sim.t_out[:nt])
        ax2.plot(sim.t_out[:nt], P_trk, color=col, lw=1.8, label=f'Method {lbl}')
    ax2.axhline(pps.P_ELECTRONICS, color='k', ls=':', lw=1, label='Hotel load')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Total motor power [W]')
    ax2.legend(fontsize=9)
    ax2.set_title(f'Tracked total electrical power P(t) (seed 0 of {N_SEEDS})')
    fig2.tight_layout()
    _savefig(fig2, 'fig_exp_power.png')

    plt.close('all')
    print("\nDone.")


if __name__ == '__main__':
    main()
