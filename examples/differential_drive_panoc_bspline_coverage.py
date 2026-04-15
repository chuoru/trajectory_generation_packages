#!/usr/bin/env python3
##
# @file differential_drive_panoc_bspline_coverage.py
#
# @brief PANOC B-Spline trajectory generation for a differential drive robot
# navigating a simple L-shaped corner.
#
# Via-points:  (0,0,0) → (1,0,0) → (1,1,pi/2)
# On first run the Rust PANOC solver is compiled (~30-60 s). Subsequent runs
# reuse the compiled binary and finish in milliseconds.
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
from trajectory_generators.panoc.bspline_coverage import BSplineCoverage


def main():
    # L-shaped path: origin → corner → goal
    waypoints = [
        [0.0, 0.0,  0.0],
        [1.0, 0.0,  0.0],
        [1.0, 1.0,  np.pi / 2],
    ]

    # Build (or reuse) the PANOC solver, then solve
    generator = BSplineCoverage(
        n_waypoints=3,
        n_ctrl_pts=6,
        spline_order=3,
        n_sampling=30,
        bound=0.17,
        vel_max=0.2,
        w_nonh=10.0,     # nonholonomic penalty weight
        w_jerk=1e-3,     # jerk smoothness weight
        w_corr=50.0,     # corridor soft-penalty weight
    )

    result = generator.generate_trajectory(
        waypoints=waypoints,
        total_time=10.0,    # fixed total time [s] — not optimized
    )

    states = result['states']       # (nt, 3): x, y, theta
    time = result['time']           # (nt,):   time vector
    ctrl_pts = result['ctrl_pts']   # (n_Q, 3): optimized control points

    # Quick diagnostics
    lateral = (-np.diff(states[:, 0]) * np.sin(states[:-1, 2])
               + np.diff(states[:, 1]) * np.cos(states[:-1, 2]))
    print(f"Final state  : x={states[-1,0]:.3f}  y={states[-1,1]:.3f}"
          f"  θ={states[-1,2]:.3f} rad")
    print(f"Max |lateral|: {np.max(np.abs(lateral)):.4f} m (should be ≈ 0)")

    _plot(waypoints, states, time, ctrl_pts)


def _plot(waypoints, states, time, ctrl_pts):
    wps = np.array(waypoints)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.plot(wps[:, 0], wps[:, 1], '--', color='gray',
            linewidth=1.5, label='Reference path')
    ax.plot(states[:, 0], states[:, 1], '-', color='steelblue',
            linewidth=2, label='PANOC B-Spline trajectory')
    ax.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], 'x', color='tomato',
            markersize=8, markeredgewidth=2, label='Control points')
    ax.plot(wps[:, 0], wps[:, 1], 'o', color='green',
            markersize=8, label='Waypoints')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('PANOC B-Spline OCP — L-shape trajectory')
    ax.legend()
    ax.grid(True)

    fig2, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    labels = ['x [m]', 'y [m]', 'θ [rad]']
    finals = [wps[-1, 0], wps[-1, 1], wps[-1, 2]]
    for i, (lbl, ref) in enumerate(zip(labels, finals)):
        axs[i].plot(time, states[:, i], linewidth=2)
        axs[i].axhline(ref, linestyle='--', color='gray', linewidth=1)
        axs[i].set_ylabel(lbl)
        axs[i].grid(True)
    axs[-1].set_xlabel('time [s]')
    fig2.suptitle('State trajectory over time')

    plt.tight_layout()
    plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
