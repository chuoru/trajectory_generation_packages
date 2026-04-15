#!/usr/bin/env python3
##
# @file run.py
#
# @brief Provide executation of the program.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2024/08/01

# Internal library
from matplotlib import pyplot as plt

# External library

# Internal library


class Plotter:
    def __init__(self, simulator):
        """! The constructor of the class.
        @param simulators: The simulator of the robot.
        """
        self._simulator = simulator

    def plot(self, reference_paths=None):
        """! The function to plot the result of the simulation.
        @param reference_paths<list>: Optional list of waypoints [[x, y, ...], ...]
            to overlay on the trajectory plot.
        """
        _, ax1 = plt.subplots(1, 1)

        ax1.set_box_aspect(1)

        if reference_paths is not None:
            ref_x = [wp[0] for wp in reference_paths]
            ref_y = [wp[1] for wp in reference_paths]
            ax1.plot(ref_x, ref_y, "--", color="gray", label="Reference path")

        ax1.plot(self._simulator.x_out[0, :],
                 self._simulator.x_out[1, :],
                 "-o", label="Robot trajectory")

        ax1.set_title("Trajectory of the robot")

        ax1.legend()

        _, ax = plt.subplots(3, 1)

        ax[0].plot(self._simulator.t_out[:len(self._simulator.x_out[0, :])],
                   self._simulator.x_out[0, :])

        ax[1].plot(self._simulator.t_out[:len(self._simulator.x_out[1, :])],
                   self._simulator.x_out[1, :])

        ax[2].plot(self._simulator.t_out[:len(self._simulator.x_out[2, :])],
                   self._simulator.x_out[2, :])

        ax[0].set_title("Position x of the robot")

        ax[1].set_title("Position y of the robot")

        ax[2].set_title("Orientation of the robot")

        _, (ax2, ax3) = plt.subplots(2, 1)

        ax2.plot(self._simulator.t_out[:len(self._simulator.u_out[0, :])],
                 self._simulator.u_out[0, :])

        ax2.set_title("Velocity of the robot")

        ax3.plot(self._simulator.t_out[:len(self._simulator.u_out[0, :])],
                 self._simulator.u_out[1, :])

        ax3.set_title("Angular velocity of the robot")

        _, (ax4, ax5) = plt.subplots(2, 1)

        ax4.plot(self._simulator.t_out, self._simulator.dudt_out[0, :])

        ax4.set_title("Acceleration of the robot")

        ax5.plot(self._simulator.t_out, self._simulator.dudt_out[1, :])

        ax5.set_title("Angular acceleration of the robot")

        plt.show()
