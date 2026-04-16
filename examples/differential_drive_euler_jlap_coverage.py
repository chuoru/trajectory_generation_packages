#!/usr/bin/env python3
##
# @file differential_drive_euler_jlap_coverage.py
#
# @brief Example: Euler-spiral corner smoothing + JLAP velocity profiling
# for a differential-drive robot navigating an L-shaped path.
#
# Via-points:  (0,0) → (3,0) → (3,2)
#
# The generator:
#   1. Fits an Euler spiral (clothoid) at the interior waypoint.
#   2. Profiles velocity on each straight segment with jerk-limited
#      acceleration (JLAP, 7-segment profile).
#   3. Samples the composite path at dt=0.05 s.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/04/16

# Standard library
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Internal library
from trajectory_generators.euler_jlap_coverage import EulerJLAPCoverage


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
WAYPOINTS = [
    [0.0, 0.0],
    [3.0, 0.0],
    [3.0, 2.0],
]

ROBOT_PARAMS = {
    'robot_mass':         120.4,
    'robot_width':        0.510,
    'wheel_radius':       0.075,
    'gear_ratio':         40.0,
    'rated_motor_torque': 1.3,
    'rated_motor_speed':  3500.0,
    'motor_inertia':      0.66e-4,
    'path_vel_lim':       0.5,
}

SAMPLING_TIME = 0.05   # [s]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    generator = EulerJLAPCoverage(
        waypoints=WAYPOINTS,
        sampling_time=SAMPLING_TIME,
        robot_params=ROBOT_PARAMS,
        path_vel_step=0.01,
        epsilon_offset=0.1,
        lc_scale=0.4,
        initial_vel=0.0,
    )

    result = generator.generate_trajectory()

    states   = result['states']      # (N, 3): x, y, phi
    time     = result['time']        # (N,)
    v        = result['v']           # (N,) path velocity
    acc_path = result['acc_path']    # (N,) path acceleration
    omega    = result['omega']       # (N,) yaw rate
    corners  = result['corner_info'] # list of corner dicts

    _print_summary(time, v, acc_path, corners)
    _plot(WAYPOINTS, states, time, v, acc_path, omega, corners)


# ---------------------------------------------------------------------------
# TEXT OUTPUT
# ---------------------------------------------------------------------------
def _print_summary(time, v, acc_path, corners):
    T_total = time[-1] if len(time) > 0 else 0.0
    N       = len(time)
    print("=" * 56)
    print("  Euler-JLAP Coverage - Trajectory Summary")
    print("=" * 56)
    print(f"  Samples          : {N}")
    print(f"  Mission time     : {T_total:.3f} s")
    print(f"  Peak path vel    : {np.max(v):.3f} m/s")
    print(f"  Avg  path vel    : {np.mean(v):.3f} m/s")
    print(f"  Peak path acc    : {np.max(np.abs(acc_path)):.3f} m/s^2")
    print(f"  Corners          : {len(corners)}")
    for k, c in enumerate(corners):
        print(f"    corner {k}: a_euler={c['a_euler']:.4f} m  "
              f"eps={c['epsilon']:.4f} m  "
              f"v={c['vel']:.3f} m/s  "
              f"T={c['T_corner']:.3f} s  "
              f"dir={'CCW' if c['turn_dir'] == 1 else 'CW'}")
    print("=" * 56)


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def _plot(waypoints, states, time, v, acc_path, omega, corners):
    wps = np.array(waypoints)

    # ------------------------------------------------------------------ Fig 1
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    fig1.canvas.manager.set_window_title("Figure 1 - Euler-JLAP Trajectory")

    ax1.set_aspect('equal')
    ax1.plot(wps[:, 0], wps[:, 1], '--', color='gray',
             linewidth=1.5, label='Reference path')
    ax1.plot(states[:, 0], states[:, 1], '-', color='steelblue',
             linewidth=2, label='Euler-JLAP trajectory')
    ax1.plot(wps[:, 0], wps[:, 1], 'o', color='green',
             markersize=8, zorder=5, label='Waypoints')

    # Mark corner geometry
    for k, c in enumerate(corners):
        ax1.plot(*c['pos_start'], 's', color='tomato', markersize=7,
                 zorder=6, label='Corner start/end' if k == 0 else '')
        ax1.plot(*c['pos_end'],   's', color='tomato', markersize=7, zorder=6)
        ax1.plot(*c['pos_mid'],   '^', color='purple', markersize=8,
                 zorder=6, label='Corner mid' if k == 0 else '')

    # Heading arrows (every 10th sample)
    step = max(1, len(states) // 20)
    for s in states[::step]:
        dx = 0.08 * np.cos(s[2])
        dy = 0.08 * np.sin(s[2])
        ax1.annotate('', xy=(s[0]+dx, s[1]+dy), xytext=(s[0], s[1]),
                     arrowprops=dict(arrowstyle='->', color='navy',
                                     lw=1.2))

    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('Euler-JLAP — XY trajectory')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True)

    # ------------------------------------------------------------------ Fig 2
    fig2, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    fig2.canvas.manager.set_window_title("Figure 2 - Kinematic Profiles")

    axs[0].plot(time, v, color='steelblue', linewidth=1.8)
    axs[0].set_ylabel('v [m/s]')
    axs[0].set_title('Path velocity')
    axs[0].grid(True)

    axs[1].plot(time, acc_path, color='tomato', linewidth=1.8)
    axs[1].set_ylabel('a [m/s²]')
    axs[1].set_title('Path acceleration')
    axs[1].grid(True)

    axs[2].plot(time, omega, color='darkorange', linewidth=1.8)
    axs[2].set_ylabel('ω [rad/s]')
    axs[2].set_xlabel('time [s]')
    axs[2].set_title('Yaw rate')
    axs[2].grid(True)

    fig2.tight_layout()

    # ------------------------------------------------------------------ Fig 3
    fig3, axs3 = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    fig3.canvas.manager.set_window_title("Figure 3 - State Profiles")

    labels = ['x [m]', 'y [m]', 'φ [rad]']
    colors = ['steelblue', 'seagreen', 'darkorchid']
    for i in range(3):
        axs3[i].plot(time, states[:, i], color=colors[i], linewidth=1.8)
        axs3[i].set_ylabel(labels[i])
        axs3[i].grid(True)
    axs3[-1].set_xlabel('time [s]')
    fig3.suptitle('State trajectory over time')
    fig3.tight_layout()

    plt.show()
    plt.close('all')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
