#!/usr/bin/env python3
##
# @file time_stepping.py
#
# @brief Provide implementation of time stepping for autonomous driving.
# autonomous driving.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 03/08/2024.
#
# Copyright (c) 2024 System Engineering Laboratory.  All rights reserved.

# Standard library
import numpy as np


class TimeStepping:
    # ==================================================================
    # PUBLIC METHODS
    # ==================================================================
    def __init__(self, model, T, dt):
        """! Constructor
        @param model<instance>: The vehicle model
        @param T<float>: The time to reach the final position
        @param dt<float>: The time step of the optimization problem
        """
        self.t_out = []

        self.x_out = []

        self.y_out = []

        self.u_out = []

        self.dudt_out = []

        self.ddudt_out = []

        self._model = model

        self._goal_tolerance = 0.1

        self._T = T

        self._dt = dt

    def run(self, inital_position, u):
        """! Run the time stepping
        @param initial_position<list>: The initial position of the vehicle
        @param u<list>: List of control inputs
        """
        self.t_out = np.linspace(0, self._T, int((1 / self._dt) * self._T))

        nt = self.t_out.shape[-1]

        self.x_out = np.zeros([self._model.nx, nt])

        self.y_out = np.zeros([self._model.nx, nt])

        self.x_out[:, 0] = inital_position

        self.y_out[:, 0] = inital_position

        self.u_out = np.zeros([self._model.nu, nt])

        self.dudt_out = np.zeros([self._model.nu, nt])

        self.ddudt_out = np.zeros([self._model.nu, nt])

        for index in range(nt):
            y_m = self.y_out[:, index]

            u_m = u[:, index]

            self.dudt_out[:, index] = (
                u_m - self.u_out[:, index]) / self._dt

            x_m = self._model.function(
                self.x_out[:, index], self.u_out[:, index], self._dt
            )

            x_m = np.reshape(x_m, (self._model.nx,))

            y_m = x_m

            if index < nt - 1:
                self.x_out[:, index + 1] = x_m

                self.y_out[:, index + 1] = y_m

                self.u_out[:, index + 1] = u_m

        self.u_out = self.u_out[:, : index + 1]

        self.x_out = self.x_out[:, : index + 1]

        self.y_out = self.y_out[:, : index + 1]

    def run_with_controller(self, initial_position, trajectory, controller):
        """! Run the simulation in closed-loop using a feedback controller.
        @param initial_position<list>: The initial state of the vehicle
        @param trajectory<instance>: Trajectory object with .x, .u, .t, .sampling_time
        @param controller<instance>: Controller with initialize() and execute() methods
        """
        self.t_out = trajectory.t

        nt = len(self.t_out)

        self.x_out = np.zeros([self._model.nx, nt])

        self.y_out = np.zeros([self._model.nx, nt])

        self.u_out = np.zeros([self._model.nu, nt])

        self.dudt_out = np.zeros([self._model.nu, nt])

        self.ddudt_out = np.zeros([self._model.nu, nt])

        self.x_out[:, 0] = initial_position

        self.y_out[:, 0] = initial_position

        controller.initialize()

        for index in range(nt):
            status, u_m = controller.execute(
                self.x_out[:, index], self.u_out[:, index], index
            )

            if not status:
                u_m = [0.0, 0.0]

            u_m = np.array(u_m)

            self.dudt_out[:, index] = (u_m - self.u_out[:, index]) / self._dt

            self.u_out[:, index] = u_m

            x_m = self._model.function(
                self.x_out[:, index], u_m, self._dt
            )

            x_m = np.reshape(x_m, (self._model.nx,))

            if index < nt - 1:
                self.x_out[:, index + 1] = x_m

                self.y_out[:, index + 1] = x_m
