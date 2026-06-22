#!/usr/bin/env python3
##
# @file tractor_trailer_bspline_coverage.py
#
# @brief Example: B-Spline coverage trajectory for a tractor-trailer robot.
#
# The trailer rear axle follows an L-shaped coverage path while the tractor
# pulls it via a rigid hitch. The OCP minimizes total traversal time of the
# trailer subject to:
#   - Nonholonomic trailer kinematics
#   - Hitch angle coupling and bounds
#   - Velocity / acceleration / jerk limits
#   - Rectangular corridor constraints
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/06/22

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trajectory_generators.bspline_tractor_trailer_coverage import (
    BSplineTractorTrailerCoverage,
)

# ---------------------------------------------------------------------------
# Waypoints for the TRAILER rear axle  [x, y, theta_trailer]
# An L-shaped turn: go right, then turn north
# ---------------------------------------------------------------------------
WAYPOINTS = [
    [0.0,  0.0,  0.0],
    [2.0,  0.0,  0.0],
    [2.0,  2.0,  np.pi / 2],
]

# Tractor-trailer geometry (metres)
LB = 0.2   # tractor rear axle → hitch
LF = 0.8   # hitch → trailer rear axle

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
gen = BSplineTractorTrailerCoverage(
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
)

result = gen.generate_trajectory()

states    = result['states']       # (nt, 4)
t_ocp     = result['time']         # (nt,)
t_ik      = result['time_ik']      # (M,)
v_trac    = result['v']            # tractor linear vel (M,)
w_trac    = result['omega']        # tractor angular vel (M,)
v_trailer = result['v_trailer']    # trailer forward speed (M,)
gamma_ocp = states[:, 3]           # hitch angle at OCP nodes

# ---------------------------------------------------------------------------
# Derive tractor pose for visualisation
#   x_tractor = x_trailer + lf·cos(θ_t) + lb·cos(θ_t − γ)
#   y_tractor = y_trailer + lf·sin(θ_t) + lb·sin(θ_t − γ)
# ---------------------------------------------------------------------------
x_t = states[:, 0] + LF * np.cos(states[:, 2]) + LB * np.cos(states[:, 2] - gamma_ocp)
y_t = states[:, 1] + LF * np.sin(states[:, 2]) + LB * np.sin(states[:, 2] - gamma_ocp)

# ---------------------------------------------------------------------------
# Figure 1 – XY trajectories + corridors
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(6, 6))

BOUND = 0.17
wps = np.array(WAYPOINTS)
for seg in range(len(wps) - 1):
    A = wps[seg, :2]
    B = wps[seg + 1, :2]
    d = B - A
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    corners = np.array([
        A - d * BOUND + n * BOUND,
        A - d * BOUND - n * BOUND,
        B + d * BOUND - n * BOUND,
        B + d * BOUND + n * BOUND,
        A - d * BOUND + n * BOUND,
    ])
    ax1.fill(corners[:, 0], corners[:, 1], alpha=0.15, color='steelblue')
    ax1.plot(corners[:-1, 0], corners[:-1, 1], 'b--', lw=0.8)

ax1.plot(states[:, 0], states[:, 1], 'k-', lw=2, label='Trailer rear axle')
ax1.plot(x_t, y_t, 'r--', lw=1.5, label='Tractor rear axle')
ax1.plot(wps[:, 0], wps[:, 1], 'ko', ms=6, zorder=5)

for i in range(0, len(t_ocp), max(1, len(t_ocp) // 12)):
    xb, yb = states[i, 0], states[i, 1]
    xt, yt = x_t[i], y_t[i]
    ax1.plot([xb, xt], [yb, yt], 'g-', lw=0.8, alpha=0.6)

ax1.set_aspect('equal')
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')
ax1.set_title('Tractor-Trailer B-Spline Coverage — XY')
ax1.legend()
ax1.grid(True, ls=':', alpha=0.5)
fig1.tight_layout()

# ---------------------------------------------------------------------------
# Figure 2 – Hitch angle γ over time
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(7, 3))
ax2.plot(t_ocp, np.rad2deg(gamma_ocp), 'b-', lw=1.8, label='γ (hitch angle)')
ax2.axhline(np.rad2deg(0.785),  color='r', ls='--', lw=1, label='±γ_max')
ax2.axhline(np.rad2deg(-0.785), color='r', ls='--', lw=1)
ax2.set_xlabel('time [s]')
ax2.set_ylabel('γ [deg]')
ax2.set_title('Hitch Angle vs Time')
ax2.legend()
ax2.grid(True, ls=':', alpha=0.5)
fig2.tight_layout()

# ---------------------------------------------------------------------------
# Figure 3 – Tractor inputs v, ω over time
# ---------------------------------------------------------------------------
fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
ax3a.plot(t_ik, v_trac,    'b-', lw=1.5, label='v_tractor')
ax3a.plot(t_ik, v_trailer, 'k--', lw=1.2, label='v_trailer')
ax3a.set_ylabel('Linear vel [m/s]')
ax3a.legend()
ax3a.grid(True, ls=':', alpha=0.5)

ax3b.plot(t_ik, np.rad2deg(w_trac), 'r-', lw=1.5, label='ω_tractor')
ax3b.set_xlabel('time [s]')
ax3b.set_ylabel('Angular vel [deg/s]')
ax3b.legend()
ax3b.grid(True, ls=':', alpha=0.5)

fig3.suptitle('Tractor Inputs vs Time')
fig3.tight_layout()

# ---------------------------------------------------------------------------
# Figure 4 – Trailer heading and hitch angle vs time (combined)
# ---------------------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(7, 3))
ax4.plot(t_ocp, np.rad2deg(states[:, 2]), 'b-', lw=1.8, label='θ_trailer')
ax4.plot(t_ocp, np.rad2deg(gamma_ocp),   'g--', lw=1.5, label='γ (hitch)')
ax4.set_xlabel('time [s]')
ax4.set_ylabel('[deg]')
ax4.set_title('Trailer Heading and Hitch Angle vs Time')
ax4.legend()
ax4.grid(True, ls=':', alpha=0.5)
fig4.tight_layout()

print(f"Total time: {t_ocp[-1]:.2f} s")
print(f"Max hitch angle: {np.rad2deg(np.max(np.abs(gamma_ocp))):.1f} deg")
print(f"Max tractor speed: {np.max(np.abs(v_trac)):.3f} m/s")
print(f"Max tractor omega: {np.rad2deg(np.max(np.abs(w_trac))):.1f} deg/s")

plt.show()
