#!/usr/bin/env python3
##
# @file verify_fine_resolution_power.py
#
# @brief Rigorously checks whether Method C's apparent 139.8 W power spike is
#        a genuine feature of the solved B-spline trajectory or a numerical
#        artifact of the Pchip-interpolate-then-np.gradient recomputation
#        pipeline used by differential_drive_comparison.py's
#        _compute_power_uniform.
#
# Does NOT re-solve the OCP (expensive). Instead, reuses the ALREADY-SOLVED
# control points and knot vector from the cached corner solution, and
# evaluates the B-spline's position/derivatives ANALYTICALLY (exact basis
# function algorithm, Piegl & Tiller NURBS Book Algorithm A2.3 -- the same
# code the OCP itself uses internally) at a much finer tau grid than the 441
# native collocation nodes. The per-sample time-scaling T(tau) is smoothly
# interpolated between its 441 solved values onto this finer grid. This
# gives the true continuous v(t)/a(t)/P(t) profile implied by the solved
# spline, independent of both the coarse-grid sampling and the separate
# Pchip+np.gradient recomputation pipeline -- resolving which of the two
# prior estimates (17.9 W vs 139.8 W) is closer to the truth.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/08/07

import sys
import os
import pickle
import numpy as np
from scipy.interpolate import PchipInterpolator as _Pchip

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import differential_drive_comparison as dc
from trajectory_generators.bspline_energy_coverage import BSplineEnergyCoverage

N_FINE = 20000  # tau samples for the dense analytic re-evaluation


def check_one(seg, w_e):
    res_c = seg['sweep']['res_by_we'][w_e]
    seg_info = seg['seg_info']
    corner_wps = dc._build_corner_waypoints(
        seg_info['arc_entry_ext_world'], seg_info['arc_exit_ext_world'])
    v_h = seg['v_handoff']
    res_s1 = seg['res_s1']
    a_entry = float(res_s1['acc_path'][-1])
    alpha_entry = float(res_s1['alpha'][-1])

    # Build an identically-configured (but unsolved) instance purely to get
    # its knot vector / basis-function machinery -- no optimization run.
    ocp = BSplineEnergyCoverage(
        waypoints=corner_wps,
        robot_params=dc.ROBOT_PARAMS_BSPLINE,
        energy_coeffs_right=dc.ENERGY_COEFFS_RIGHT,
        energy_coeffs_left=dc.ENERGY_COEFFS_LEFT,
        w_time=1.0, w_energy=w_e,
        p_electronics=dc.P_ELECTRONICS,
        v_entry=v_h, v_exit=v_h,
        a_entry=a_entry, a_exit=0.0,
        alpha_entry=alpha_entry, alpha_exit=0.0,
        **dc._make_bspline_common(v_h),
    )

    ctrl_pts = res_c['ctrl_pts']            # (n_Q, 3) solved control points
    t_real = res_c['time']                  # (nt,) real time at each node
    nt = len(t_real)
    # Recover the per-node T design variable from its cumulative sum.
    T_val = np.empty(nt)
    T_val[0] = t_real[0] * nt
    T_val[1:] = (t_real[1:] - t_real[:-1]) * nt

    knot = ocp._build_knot_vector()
    n_Q = ocp._n_Q

    # Native (coarse) tau grid: uniform in [0, 1], nt points (matches how
    # the OCP itself samples -- see BSplineCoverage.generate_trajectory).
    tau_coarse = np.linspace(0.0, 1.0, nt)
    tau_fine = np.linspace(0.0, 1.0, N_FINE)

    print(f"nt (native) = {nt}, N_FINE = {N_FINE}  "
          f"(native tau spacing={1.0/(nt-1):.6f}, fine={1.0/(N_FINE-1):.6f})")

    B, dB, ddB, _ = ocp._build_basis_matrices(tau_fine, knot)
    s_fine = B @ ctrl_pts
    ds_fine = dB @ ctrl_pts
    dds_fine = ddB @ ctrl_pts

    # Smoothly interpolate T(tau) from its nt native values onto tau_fine.
    T_fine = _Pchip(tau_coarse, T_val)(tau_fine)

    theta = s_fine[:, 2]
    cos_th, sin_th = np.cos(theta), np.sin(theta)
    v_fwd = (cos_th * ds_fine[:, 0] + sin_th * ds_fine[:, 1]) / T_fine
    a_fwd = (cos_th * dds_fine[:, 0] + sin_th * dds_fine[:, 1]) / T_fine**2
    omega_robot = ds_fine[:, 2] / T_fine
    alpha_robot = dds_fine[:, 2] / T_fine**2

    l = dc.ROBOT_PARAMS_BSPLINE['l']
    v_r, v_l = v_fwd + l * omega_robot, v_fwd - l * omega_robot
    a_r, a_l = a_fwd + l * alpha_robot, a_fwd - l * alpha_robot

    def _p(vw, aw, c):
        return np.maximum(c[0]*aw**2 + c[1]*vw**2 + np.abs(c[2]*aw)
                           + np.abs(c[3]*vw) + np.abs(c[4]*vw*aw) + c[5], 0.0)

    P = (_p(v_r, a_r, dc.ENERGY_COEFFS_RIGHT)
         + _p(v_l, a_l, dc.ENERGY_COEFFS_LEFT) + dc.P_ELECTRONICS)

    idx = int(np.argmax(P))
    print(f"\n=== w_e = {w_e} ===")
    print(f"nt (native) = {nt}, N_FINE = {N_FINE}")
    print(f"  Analytic dense peak power = {P[idx]:.3f} W  at tau={tau_fine[idx]:.6f}, "
          f"t~{np.interp(tau_fine[idx], tau_coarse, t_real):.4f}s")
    print(f"  a_fwd there = {a_fwd[idx]:.4f} m/s^2,  v_fwd there = {v_fwd[idx]:.4f} m/s")
    print(f"  max |a_fwd| over whole corner = {np.max(np.abs(a_fwd)):.4f} m/s^2")
    print(f"  max |alpha_robot| over whole corner = {np.max(np.abs(alpha_robot)):.4f} rad/s^2")
    print(f"  coarse power_sym (441 native nodes)   peak = {float(np.max(res_c['power'])):.3f} W")
    P_pchip_gradient = dc._compute_power_uniform(res_c['v'], res_c['omega'], 0.01)
    print(f"  Pchip+np.gradient recomputation        peak = {float(np.max(P_pchip_gradient)):.3f} W")
    return float(P[idx]), float(np.max(res_c['power'])), float(np.max(P_pchip_gradient))


def main():
    with open(dc._CACHE_PATH, 'rb') as f:
        seg = pickle.load(f)
    print(f"{'w_e':>8s} {'analytic-dense':>16s} {'coarse-native':>16s} {'pchip+gradient':>16s}")
    for w_e in [0.0, 0.0769, 0.4872]:
        p_dense, p_coarse, p_pchip = check_one(seg, w_e)
        print(f"{w_e:>8.4f} {p_dense:>16.3f} {p_coarse:>16.3f} {p_pchip:>16.3f}")


if __name__ == '__main__':
    main()
