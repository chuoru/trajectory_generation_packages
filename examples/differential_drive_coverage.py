#!/usr/bin/env python3
##
# @file differential_drive_coverage.py
#
# @brief Provide execution of coverage trajectory generation
#        for a differential drive robot using cross-track error minimization.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/04/14

# Standard library
import sys
import os

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Internal library
from visualizers.plotter import Plotter
from simulators.time_stepping import TimeStepping
from models.differential_drive import DifferentialDrive
from trajectory_generators.simple_coverage import SimpleCoverage


def main():
    T = 4

    dt = 0.1

    N = int(T / dt)

    model = DifferentialDrive(0.1)

    trajectory_generator = SimpleCoverage(model, T, dt)

    simulator = TimeStepping(model, T, dt)

    visualizer = Plotter(simulator)

    initial_position = [0, 0, 0]

    # Reference path: L-shaped route with N waypoints
    # First half goes along x-axis, second half turns along y-axis
    reference_paths = []

    half = N // 2

    for i in range(half):
        x = i * 2.0 / (half - 1)
        reference_paths.append([x, 0.0, 0.0])

    for i in range(N - half):
        x = 2.0
        y = i * 2.0 / (N - half - 1)
        reference_paths.append([x, y, np.pi / 2])

    u = trajectory_generator.generate_trajectory(
        initial_position, reference_paths)

    simulator.run(initial_position, u)

    visualizer.plot(reference_paths=reference_paths)


if __name__ == '__main__':
    main()
