#!/usr/bin/env python3
##
# @file differential_drive_direct_coverage.py
#
# @brief Direct collocation OCP trajectory generation for a differential
# drive robot navigating a simple L-shaped corner.
#
# Via-points:  (0,0,0) → (1,0,0) → (1,1,pi/2)
# The optimizer finds the state trajectory, control inputs, and free terminal
# time that minimize total traversal time while keeping the robot within a
# corridor of width `path_bound` around the piecewise-linear reference.
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
from trajectory_generators.direct_coverage import DirectCoverage


def main():
    # L-shaped path: start at origin heading east, turn north at corner (1, 0)
    waypoints = [
        [0.0, 0.0,  0.0],          # start: (x=0, y=0, theta=0)
        [1.0, 0.0,  0.0],          # corner
        [1.0, 1.0,  np.pi / 2],    # end: heading north
    ]

    generator = DirectCoverage(
        waypoints=waypoints,
        n_intervals=200,            # collocation intervals
        path_bound=0.1,             # corridor half-width [m]
        v_max=0.2,                  # maximum forward speed [m/s]
        v_min=0.05,                 # minimum forward speed (no reversal)
        omega_max=0.196,            # maximum angular speed [rad/s]
        acc_input_max=0.9,          # maximum input rate of change
        acc_max=20.0,               # maximum state acceleration
        jerk_max=1e3,               # maximum state jerk
        tf_init=10.0,               # initial guess for total time [s]
        tf_max=60.0,
    )

    result = generator.generate_trajectory()

    states = result['states']       # (N+1, 3): x, y, theta
    inputs = result['inputs']       # (N, 2):   v, omega
    time = result['time']           # (N+1,):   time vector
    tf = result['tf']               # float:    optimal total time
    x_ref = result['x_ref']        # (N, 2):   reference xy

    print(f"Optimal total time: {tf:.3f} s")
    print(f"Final state: x={states[-1,0]:.3f}  y={states[-1,1]:.3f}"
          f"  theta={states[-1,2]:.3f} rad")

    _plot(waypoints, states, inputs, time, x_ref)


def _plot(waypoints, states, inputs, time, x_ref):
    wps = np.array(waypoints)

    # --- Trajectory in XY ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')

    ax.plot(x_ref[:, 0], x_ref[:, 1], '--', color='gray',
            linewidth=1.5, label='Reference path')
    ax.plot(states[:, 0], states[:, 1], '-', color='steelblue',
            linewidth=2, label='Optimized trajectory')
    ax.plot(wps[:, 0], wps[:, 1], 'o', color='green',
            markersize=8, label='Waypoints')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Direct Collocation OCP — L-shape trajectory')
    ax.legend()
    ax.grid(True)

    # --- States over time ---
    fig2, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    state_labels = ['x [m]', 'y [m]', 'θ [rad]']
    ref_finals = [wps[-1, 0], wps[-1, 1], wps[-1, 2]]
    for i, (lbl, ref) in enumerate(zip(state_labels, ref_finals)):
        axs[i].plot(time, states[:, i], linewidth=2)
        axs[i].axhline(ref, linestyle='--', color='gray', linewidth=1)
        axs[i].set_ylabel(lbl)
        axs[i].grid(True)
    axs[-1].set_xlabel('time [s]')
    fig2.suptitle('State trajectory over time')

    # --- Control inputs ---
    t_u = time[:-1]         # inputs defined on intervals, not knots
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax3a.plot(t_u, inputs[:, 0], linewidth=2, color='steelblue')
    ax3a.set_ylabel('v [m/s]')
    ax3a.set_title('Control inputs')
    ax3a.grid(True)

    ax3b.plot(t_u, inputs[:, 1], linewidth=2, color='tomato')
    ax3b.set_ylabel('ω [rad/s]')
    ax3b.set_xlabel('time [s]')
    ax3b.grid(True)

    plt.tight_layout()
    plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
