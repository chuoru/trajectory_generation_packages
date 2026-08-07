"""! Paired significance testing for the 25-seed closed-loop tracking comparison.

Peer review (methodology) flagged that Methods A/B/D/C are tracked under the
SAME 25 seeds (a paired design) in section5_purepursuit.py, but only
mean +/- std is reported -- no paired significance test. This script reuses
the same cached segmented pipeline and _track_one_seed() logic to collect
RAW per-seed metrics (not just the reduced mean/std/min/max already printed
by section5_purepursuit.py), then runs a Wilcoxon signed-rank test (paired,
distribution-free -- appropriate here since normality across only 25 seeds
is not established) on the key metrics discussed in the paper: mean CTE,
tracked peak power, and tracked total energy, for the D-vs-B and D-vs-C
comparisons that the paper's narrative leans on.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
from scipy.stats import wilcoxon

import differential_drive_comparison as dc
import section5_purepursuit as s5

N_SEEDS = s5.N_SEEDS


def _collect_raw(label, d, mission_time, n_seeds=N_SEEDS):
    """Like s5._track_multi_seed, but keeps the raw per-seed metric arrays."""
    all_metrics = []
    for seed in range(n_seeds):
        m, _, _ = s5._track_one_seed(d, mission_time, seed)
        all_metrics.append(m)
    print(f"  Method {label}: {n_seeds} seeds tracked.")
    keys = all_metrics[0].keys()
    return {k: np.array([m[k] for m in all_metrics]) for k in keys}


def main():
    print("Building Methods B, D & C (segmented pipeline, shared sweep, cached) ...")
    seg = dc._run_segmented_pipeline()
    we_b, we_d, we_c = 0.0, 0.0769, 0.4872
    res_b = dc._build_segmented_result(seg, we_b)
    res_d = dc._build_segmented_result(seg, we_d)
    res_c = dc._build_segmented_result(seg, we_c)
    T_b = res_b['T_s1'] + res_b['T_corner'] + res_b['T_s2']
    T_d = res_d['T_s1'] + res_d['T_corner'] + res_d['T_s2']
    T_c = res_c['T_s1'] + res_c['T_corner'] + res_c['T_s2']
    b_dict = s5._segmented_dict(res_b)
    d_dict = s5._segmented_dict(res_d)
    c_dict = s5._segmented_dict(res_c)

    print(f"\nRunning closed-loop Pure Pursuit ({N_SEEDS} seeds/method, "
          f"GPS/IMU noise) for B, D, C (raw per-seed capture) ...")
    raw_b = _collect_raw('B', b_dict, T_b)
    raw_d = _collect_raw('D', d_dict, T_d)
    raw_c = _collect_raw('C', c_dict, T_c)

    metrics = [
        ('mean_cte_cm',     'Mean CTE [cm]'),
        ('peak_power_meas', 'Peak power tracked [W]'),
        ('energy_meas',     'Total energy tracked [J]'),
        ('mean_he_deg',     'Mean heading err [deg]'),
    ]

    print(f"\n  {'Metric':28s} {'D mean':>10s} {'B mean':>10s} "
          f"{'D-vs-B p':>10s} {'D mean':>10s} {'C mean':>10s} {'D-vs-C p':>10s}")
    print("  " + "-" * 92)
    for key, label in metrics:
        d_vals, b_vals, c_vals = raw_d[key], raw_b[key], raw_c[key]
        # Wilcoxon requires at least one non-zero paired difference.
        diff_db = d_vals - b_vals
        diff_dc = d_vals - c_vals
        p_db = wilcoxon(diff_db).pvalue if np.any(diff_db != 0) else float('nan')
        p_dc = wilcoxon(diff_dc).pvalue if np.any(diff_dc != 0) else float('nan')
        print(f"  {label:28s} {d_vals.mean():10.4f} {b_vals.mean():10.4f} "
              f"{p_db:10.2e} {d_vals.mean():10.4f} {c_vals.mean():10.4f} {p_dc:10.2e}")

    print("\nDone. p < 0.05 (Wilcoxon signed-rank, paired across the same 25 seeds) "
          "indicates the median paired difference is statistically significant.")


if __name__ == '__main__':
    main()
