#!/usr/bin/env python3
##
# @file differential_drive_bspline_coverage.py
#
# @brief B-Spline parameterized OCP trajectory generation for a differential
# drive robot navigating a simple L-shaped corner.
#
# Via-points:  (0,0,0) → (1,0,0) → (1,1,pi/2)
# The optimizer finds control points for (x, y, theta) and per-sample time
# variables T that minimize total traversal time while satisfying velocity,
# acceleration, jerk, nonholonomic, and corridor constraints.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/04/15

# Standard library
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Internal library
from trajectory_generators.bspline_coverage import BSplineCoverage


def main():
    # L-shaped path: start at origin heading east, turn north at corner (1, 0)
    waypoints = [
        [0.0, 0.0,  0.0],          # start: (x=0, y=0, theta=0)
        [1.0, 0.0,  0.0],          # corner
        [1.0, 1.0,  np.pi / 2],    # end: heading north
    ]

    generator = BSplineCoverage(
        waypoints=waypoints,
        bound=0.17,          # corridor half-width [m]
        n_ctrl_pts=6,        # control points per segment
        spline_order=3,      # cubic B-spline
        n_sampling=30,       # samples per segment (keep small for speed)
        vel_max=[0.2, 0.2, 0.196],
        vel_min_lin=0.01,
        eps_nonh=0.001,
    )

    result = generator.generate_trajectory()

    states = result['states']       # (nt, 3): x, y, theta along spline
    time = result['time']           # (nt,):   real time at each sample
    ctrl_pts = result['ctrl_pts']   # (n_Q, 3): optimized control points
    v = result['v']                 # (M,): linear velocity
    omega = result['omega']         # (M,): angular velocity
    time_ik = result['time_ik']     # (M,): time for v/omega

    _plot(waypoints, states, time, ctrl_pts, v, omega, time_ik)


def _plot(waypoints, states, time, ctrl_pts, v, omega, time_ik):
    wps = np.array(waypoints)

    # --- Trajectory in XY ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')

    # Reference path (piecewise linear)
    ax.plot(wps[:, 0], wps[:, 1], '--', color='gray',
            linewidth=1.5, label='Reference path')

    # Optimized spline
    ax.plot(states[:, 0], states[:, 1], '-', color='steelblue',
            linewidth=2, label='B-Spline trajectory')

    # Control points
    ax.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], 'x', color='tomato',
            markersize=8, markeredgewidth=2, label='Control points')

    # Via-points
    ax.plot(wps[:, 0], wps[:, 1], 'o', color='green',
            markersize=8, label='Waypoints')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('B-Spline OCP — L-shape trajectory')
    ax.legend()
    ax.grid(True)

    # --- States over time ---
    fig2, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    labels = ['x [m]', 'y [m]', 'θ [rad]']
    for i, (lbl, ref_val) in enumerate(zip(labels,
                                           [wps[-1, 0], wps[-1, 1], wps[-1, 2]])):
        axs[i].plot(time, states[:, i], linewidth=2)
        axs[i].axhline(ref_val, linestyle='--', color='gray', linewidth=1)
        axs[i].set_ylabel(lbl)
        axs[i].grid(True)
    axs[-1].set_xlabel('time [s]')
    fig2.suptitle('State trajectory over time')

    # --- Control inputs ---
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax3a.plot(time_ik, v, linewidth=2, color='steelblue')
    ax3a.set_ylabel('v [m/s]')
    ax3a.set_title('Differential drive inputs (inverse kinematics)')
    ax3a.grid(True)

    ax3b.plot(time_ik, omega, linewidth=2, color='tomato')
    ax3b.set_ylabel('ω [rad/s]')
    ax3b.set_xlabel('time [s]')
    ax3b.grid(True)

    plt.tight_layout()
    plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
