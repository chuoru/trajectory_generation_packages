#!/usr/bin/env python3
##
# @file method_a_purepursuit.py
#
# @brief Closed-loop Pure Pursuit tracking (GPS-IMU noise) of Method A
#        (EulerJLAPCoverage, full 3-waypoint path) so it can be compared
#        against the Method B / C closed-loop results already produced by
#        purepursuit_fine_sweep.py.
#
# Reuses the exact same simulator, noise model, and metric definitions as
# purepursuit_fine_sweep.py (imported directly) so all three methods are
# scored identically. Method A's full-path trajectory is built with the
# same parameters as differential_drive_comparison.py's _run_method_a().
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/08/04

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trajectory_generators.euler_jlap_coverage import EulerJLAPCoverage
import purepursuit_fine_sweep as pps

WP_START, WP_CORNER, WP_END = [0.0, 0.0], [5.0, 0.0], [5.0, 5.0]
JLAP_DT = 0.05
JLAP_ROBOT_PARAMS = {
    'robot_mass':         50.4,
    'robot_width':        0.53,
    'wheel_radius':       0.15,
    'gear_ratio':         40.0,
    'rated_motor_torque': 1.3,
    'rated_motor_speed':  3500.0,
    'motor_inertia':      0.66e-4,
    'path_vel_lim':       0.5,
}


def main():
    res = EulerJLAPCoverage(
        waypoints=[WP_START, WP_CORNER, WP_END],
        sampling_time=JLAP_DT,
        robot_params=JLAP_ROBOT_PARAMS,
        path_vel_step=0.01,
        epsilon_offset=0.25,
        lc_scale=0.4,
        initial_vel=0.0,
        final_vel=0.0,
    ).generate_trajectory()

    a_dict = {
        'time':  res['time'],
        'x':     res['states'][:, 0],
        'y':     res['states'][:, 1],
        'theta': res['states'][:, 2],
        'v':     res['v'],
        'omega': res['omega'],
    }

    traj = pps.corner_to_trajectory(a_dict, dt=pps.SIM_DT)
    sim = pps.run_purepursuit(traj, seed=0)

    nt = min(traj.x.shape[0], sim.x_out.shape[1])
    trk_xy = sim.x_out[:2, :].T
    ref_xy = traj.x[:, :2]
    cte = pps.cross_track_error(ref_xy, trk_xy)
    he = pps.heading_error(traj.x[:nt, 2], sim.x_out[2, :nt])
    v_err = np.abs(traj.u[0, :nt] - sim.u_out[0, :nt])
    final_err = np.hypot(
        sim.x_out[0, -1] - traj.x[-1, 0],
        sim.x_out[1, -1] - traj.x[-1, 1],
    )

    power = pps.compute_motor_power(res['v'], res['omega'], res['time'])
    mission_time = float(res['time'][-1] - res['time'][0])
    total_energy = float(np.trapz(power, res['time']))
    peak_power = float(power.max())

    metrics = {
        'label':        'Method A',
        'mission_time': mission_time,
        'total_energy': total_energy,
        'peak_power':   peak_power,
        'max_cte':      float(cte.max()  * 1e2),
        'mean_cte':     float(cte.mean() * 1e2),
        'rms_cte':      float(np.sqrt((cte**2).mean()) * 1e2),
        'final_err':    float(final_err  * 1e2),
        'max_he':       float(np.rad2deg(np.abs(he).max())),
        'mean_he':      float(np.rad2deg(np.abs(he).mean())),
        'max_verr':     float(v_err.max()),
        'track_dur':    float(sim.t_out[-1]),
    }

    print("\n  Method A closed-loop Pure Pursuit tracking (GPS/IMU noise, seed=0)")
    print("  " + "-" * 70)
    for k, v in metrics.items():
        print(f"    {k:15s} = {v}")


if __name__ == '__main__':
    main()
