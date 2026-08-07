#!/usr/bin/env python3
##
# @file test_peak_power_constraint.py
#
# @brief Validates the new p_peak_max constraint in BSplineEnergyCoverage:
#        re-solves the corner OCP at Method C's weight (w_e=0.4872, the
#        weight that produced a 139.8 W open-loop power spike) with an
#        explicit peak-power cap, to confirm the spike is preventable by
#        directly constraining P(t) rather than only weighting its integral.
#
# Uses the cached segmented-pipeline result (corner geometry, JLAP boundary
# conditions) from differential_drive_comparison.py's csv_output cache, so no
# new PathSegment/JLAP solve is needed -- only the corner OCP is re-solved,
# once unconstrained (baseline, should reproduce the 139.8 W spike) and once
# with p_peak_max=20.0 W (a target close to Method B's 19.3 W peak).
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/08/06

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import differential_drive_comparison as dc
from trajectory_generators.bspline_energy_coverage import BSplineEnergyCoverage

WE_TARGET = 0.4872
P_PEAK_TARGET = 22.0  # W


def _solve(corner_wps, v_h, a_entry, alpha_entry, w_e, p_peak_max, warm_start):
    return BSplineEnergyCoverage(
        waypoints=corner_wps,
        robot_params=dc.ROBOT_PARAMS_BSPLINE,
        energy_coeffs_right=dc.ENERGY_COEFFS_RIGHT,
        energy_coeffs_left=dc.ENERGY_COEFFS_LEFT,
        w_time=1.0,
        w_energy=w_e,
        p_electronics=dc.P_ELECTRONICS,
        v_entry=v_h, v_exit=v_h,
        a_entry=a_entry, a_exit=0.0,
        alpha_entry=alpha_entry, alpha_exit=0.0,
        max_iter=15000,
        p_peak_max=p_peak_max,
        **dc._make_bspline_common(v_h),
    ).generate_trajectory(warm_start=warm_start)


def main():
    seg = dc._run_segmented_pipeline()  # loads from cache
    seg_info = seg['seg_info']
    corner_wps = dc._build_corner_waypoints(
        seg_info['arc_entry_ext_world'], seg_info['arc_exit_ext_world'])
    v_h = seg['v_handoff']
    res_s1 = seg['res_s1']
    a_entry = float(res_s1['acc_path'][-1])
    alpha_entry = float(res_s1['alpha'][-1])

    print(f"v_handoff={v_h:.4f}  a_entry={a_entry:.4f}  alpha_entry={alpha_entry:.4f}")

    print("\n=== Baseline: w_e=0.4872, unconstrained peak power "
          "(reusing cached Method C solution) ===")
    res_base = seg['sweep']['res_by_we'][WE_TARGET]
    print("return_status:", res_base['return_status'])
    P_base = dc._compute_power_uniform(res_base['v'], res_base['omega'], 0.01)
    T_base = float(res_base['time'][-1])
    E_base = float(res_base['energy'])
    print(f"T={T_base:.3f} s  E(model)={E_base:.3f} J  "
          f"peak_P(uniform-model)={float(np.max(P_base)):.3f} W  "
          f"peak_P(internal power_sym)={float(np.max(res_base['power'])):.3f} W")

    print(f"\n=== Constrained: w_e=0.4872, p_peak_max={P_PEAK_TARGET} W "
          f"(warm-started from Method B's low-power solution, w_e=0) ===")
    res_time_opt = seg['sweep']['res_by_we'][0.0]
    res_cap = _solve(corner_wps, v_h, a_entry, alpha_entry,
                      WE_TARGET, P_PEAK_TARGET, warm_start=res_time_opt)
    print("return_status:", res_cap['return_status'])
    P_cap = dc._compute_power_uniform(res_cap['v'], res_cap['omega'], 0.01)
    T_cap = float(res_cap['time'][-1])
    E_cap = float(res_cap['energy'])
    print(f"T={T_cap:.3f} s  E(model)={E_cap:.3f} J  "
          f"peak_P(uniform-model)={float(np.max(P_cap)):.3f} W  "
          f"peak_P(internal power_sym)={float(np.max(res_cap['power'])):.3f} W")

    print("\n=== Comparison ===")
    print(f"{'':20s} {'Baseline':>14s} {'Capped':>14s}")
    print(f"{'Mission time [s]':20s} {T_base:14.3f} {T_cap:14.3f}")
    print(f"{'Energy [J]':20s} {E_base:14.3f} {E_cap:14.3f}")
    print(f"{'Peak power [W]':20s} {float(np.max(P_base)):14.3f} {float(np.max(P_cap)):14.3f}")


if __name__ == '__main__':
    main()
