#!/usr/bin/env python3
##
# @file tractor_trailer_bspline_energy_coverage.py
#
# @brief Example: Energy-aware B-Spline coverage for a tractor-trailer robot.
#
# Compares a time-optimal trajectory (w_energy = 0) against an energy-aware
# one (w_energy > 0) for the same L-shaped coverage path. Energy is computed
# from the TRACTOR's wheel-level power model; the trailer sweeps the coverage
# area.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/06/22

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trajectory_generators.bspline_energy_tractor_trailer_coverage import (
    BSplineEnergyTractorTrailerCoverage,
)

# ---------------------------------------------------------------------------
# Waypoints for the TRAILER rear axle  [x, y, theta_trailer]
# ---------------------------------------------------------------------------
WAYPOINTS = [
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [2.0, 2.0, np.pi / 2],
]

LB = 0.2   # tractor rear axle → hitch [m]
LF = 0.8   # hitch → trailer rear axle [m]

COMMON = dict(
    waypoints=WAYPOINTS,
    bound=0.17,
    n_ctrl_pts=5,
    spline_order=3,
    n_sampling=20,
    vel_max=[0.2, 0.2, 0.196],
    vel_min_lin=0.01,
    eps_nonh=0.001,
    eps_hitch=0.05,
    length_back=LB,
    length_front=LF,
    gamma_max=0.785,
    gamma_entry=0.0,
    gamma_exit=0.0,
    acc_max=[5.0, 5.0, 4.0],
    jerk_max=[50.0, 50.0, 20.0],
    p_electronics=2.0,
)

# ---------------------------------------------------------------------------
# Solve: time-optimal  (w_energy = 0)
# ---------------------------------------------------------------------------
print("Solving time-optimal trajectory ...")
gen_time = BSplineEnergyTractorTrailerCoverage(
    **COMMON, w_time=1.0, w_energy=0.0)
res_time = gen_time.generate_trajectory()

# ---------------------------------------------------------------------------
# Solve: energy-aware  (w_time = 1, w_energy = 1)
# ---------------------------------------------------------------------------
print("Solving energy-aware trajectory (warm-started from time-optimal) ...")
gen_ener = BSplineEnergyTractorTrailerCoverage(
    **COMMON, w_time=1.0, w_energy=1.0)
res_ener = gen_ener.generate_trajectory(warm_start=res_time)

# ---------------------------------------------------------------------------
# Helper: derive tractor XY from trailer state + hitch angle
# ---------------------------------------------------------------------------
def tractor_xy(states):
    γ = states[:, 3]
    θ = states[:, 2]
    xt = states[:, 0] + LF * np.cos(θ) + LB * np.cos(θ - γ)
    yt = states[:, 1] + LF * np.sin(θ) + LB * np.sin(θ - γ)
    return xt, yt

xt_time, yt_time = tractor_xy(res_time['states'])
xt_ener, yt_ener = tractor_xy(res_ener['states'])

# ---------------------------------------------------------------------------
# Figure 1 – XY trajectories
# ---------------------------------------------------------------------------
fig1, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
BOUND = 0.17
wps = np.array(WAYPOINTS)

for ax, res, xt, yt, label in [
        (axes[0], res_time, xt_time, yt_time, 'Time-Optimal'),
        (axes[1], res_ener, xt_ener, yt_ener, 'Energy-Aware'),
]:
    for seg in range(len(wps) - 1):
        A = wps[seg, :2]; B = wps[seg + 1, :2]
        d = (B - A) / np.linalg.norm(B - A)
        n = np.array([-d[1], d[0]])
        corners = np.array([
            A - d*BOUND + n*BOUND, A - d*BOUND - n*BOUND,
            B + d*BOUND - n*BOUND, B + d*BOUND + n*BOUND,
            A - d*BOUND + n*BOUND,
        ])
        ax.fill(corners[:, 0], corners[:, 1], alpha=0.12, color='steelblue')
        ax.plot(corners[:-1, 0], corners[:-1, 1], 'b--', lw=0.7)

    s = res['states']
    ax.plot(s[:, 0], s[:, 1], 'k-', lw=2, label='Trailer')
    ax.plot(xt, yt, 'r--', lw=1.5, label='Tractor')
    ax.plot(wps[:, 0], wps[:, 1], 'ko', ms=5)
    t_ocp = res['time']
    for i in range(0, len(t_ocp), max(1, len(t_ocp) // 10)):
        ax.plot([s[i, 0], xt[i]], [s[i, 1], yt[i]], 'g-', lw=0.7, alpha=0.5)

    ax.set_aspect('equal')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title(f'{label}\n'
                 f'T={t_ocp[-1]:.2f} s  E={res["energy"]:.2f} J')
    ax.legend(fontsize=8)
    ax.grid(True, ls=':', alpha=0.5)

fig1.suptitle('Tractor-Trailer B-Spline Coverage Comparison')
fig1.tight_layout()

# ---------------------------------------------------------------------------
# Figure 2 – Tractor velocity and angular velocity
# ---------------------------------------------------------------------------
fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(8, 5), sharex=False)
for res, lbl, ls in [(res_time, 'Time-opt', '-'), (res_ener, 'Energy', '--')]:
    t = res['time_ik']
    ax2a.plot(t, res['v'],     ls, label=f'v_trac ({lbl})')
    ax2b.plot(t, np.rad2deg(res['omega']), ls, label=f'ω_trac ({lbl})')
ax2a.set_ylabel('v_trac [m/s]'); ax2a.legend(); ax2a.grid(True, ls=':')
ax2b.set_ylabel('ω_trac [deg/s]'); ax2b.set_xlabel('time [s]')
ax2b.legend(); ax2b.grid(True, ls=':')
fig2.suptitle('Tractor Velocity Profiles'); fig2.tight_layout()

# ---------------------------------------------------------------------------
# Figure 3 – Power and energy
# ---------------------------------------------------------------------------
fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(8, 5), sharex=False)
for res, lbl, ls in [(res_time, 'Time-opt', '-'), (res_ener, 'Energy', '--')]:
    t = res['time']
    ax3a.plot(t, res['power'], ls, label=lbl)
ax3a.set_ylabel('Power [W]'); ax3a.set_xlabel('time [s]')
ax3a.legend(); ax3a.grid(True, ls=':')

cats = ['Time-optimal', 'Energy-aware']
energies = [res_time['energy'], res_ener['energy']]
times    = [res_time['time'][-1], res_ener['time'][-1]]
x = np.arange(2)
ax3b.bar(x - 0.2, energies, 0.35, label='Energy [J]', color='steelblue')
ax3b.bar(x + 0.2, times,    0.35, label='Time [s]', color='coral')
ax3b.set_xticks(x); ax3b.set_xticklabels(cats)
ax3b.legend(); ax3b.grid(True, ls=':', axis='y')
fig3.suptitle('Power Profile and Mission Metrics'); fig3.tight_layout()

# ---------------------------------------------------------------------------
# Figure 4 – Hitch angle
# ---------------------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(8, 3))
for res, lbl, ls in [(res_time, 'Time-opt', '-'), (res_ener, 'Energy', '--')]:
    ax4.plot(res['time'], np.rad2deg(res['states'][:, 3]), ls, label=lbl)
ax4.axhline( np.rad2deg(0.785), color='r', ls=':', lw=1)
ax4.axhline(-np.rad2deg(0.785), color='r', ls=':', lw=1, label='±γ_max')
ax4.set_xlabel('time [s]'); ax4.set_ylabel('γ [deg]')
ax4.set_title('Hitch Angle vs Time'); ax4.legend(); ax4.grid(True, ls=':')
fig4.tight_layout()

print(f"\n{'':20s} {'Time [s]':>10s} {'Energy [J]':>12s}")
print(f"{'Time-optimal':20s} {res_time['time'][-1]:10.2f} {res_time['energy']:12.3f}")
print(f"{'Energy-aware':20s} {res_ener['time'][-1]:10.2f} {res_ener['energy']:12.3f}")

plt.show()
