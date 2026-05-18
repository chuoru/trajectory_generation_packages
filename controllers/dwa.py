#!/usr/bin/env python3
# Standard library
import math
import numpy as np


class DynamicWindowApproach:
    """! Dynamic Window Approach (DWA) controller
    The class provides implementation of Dynamic Window Approach (DWA)
    controller for autonomous driving.
    @note The controller is used to follow a trajectory.
    """

    def __init__(self, model, trajectory):
        """! Constructor
        @param model<instance>: The vehicle model
        @param trajectory<instance>: The trajectory
        @note The trajectory is a set of point to point.
        """
        self.model = model

        self.trajectory = trajectory

        self._max_speed = 0.27

        self._min_speed = -0.27

        self._max_yaw_rate = self._convert_degrees_to_radians(40.0)

        self.max_accel = 0.27

        self.max_delta_yaw_rate = self._convert_degrees_to_radians(40.0)

        self.v_resolution = 0.03

        self.yaw_rate_resolution = self._convert_degrees_to_radians(1.0)

        self._dt = 0.1

        self._predict_time = 2.0

        self._to_goal_cost_gain = 0.15

        self._speed_cost_gain = 1.0

        self._goal_index = 0

        self._goal = self.trajectory.x[self._goal_index]

        self._goal_tolerance = 0.75

    # ==================================================================================================
    # PUBLIC METHODS
    # =================================================================================================

    def initialize(self):
        """! Initialize the controller
        @note The method is used to initialize the controller.
        """
        pass

    def execute(self, state, input, index):
        """! Execute the controller
        @param state<list>: The state of the vehicle
        @param input<list>: The input of the vehicle
        @param index<int>: The previous index
        @return<tuple>: The status and control
        """
        status = True

        goal_distance = math.sqrt(
            (self._goal[0] - state[0])**2 + (self._goal[1] - state[1])**2)

        if goal_distance <= self._goal_tolerance:

            self._goal_index += 1

        if self._goal_index >= len(self.trajectory.x):

            u = [0.0, 0.0]

            return status, u

        self._goal = self.trajectory.x[self._goal_index]

        u = self._execute_dwa_control(state, self._goal, input)

        return status, u

    # ==================================================================================================
    # PRIVATE METHODS
    # ==================================================================================================
    def _get_nearest_waypoint(self, x, y, update_prev_idx=False):
        """! Get the nearest waypoint
        @param x<float>: The x-coordinate of the vehicle
        @param y<float>: The y-coordinate of the vehicle
        @param update_prev_idx<bool>: The flag to update the previous index
        """
        SEARCH_IDX_LEN = 200

        prev_idx = self.prev_waypoints_idx

        dx = [x - ref_x for ref_x in self.trajectory.x[
            prev_idx:(prev_idx + SEARCH_IDX_LEN), 0]]

        dy = [y - ref_y for ref_y in self.trajectory.x[
            prev_idx:(prev_idx + SEARCH_IDX_LEN), 1]]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        min_d = min(d)

        nearest_idx = d.index(min_d) + prev_idx

        if update_prev_idx:
            self.prev_waypoints_idx = nearest_idx

        return nearest_idx

    def _convert_degrees_to_radians(self, degrees):
        """! Convert degrees to radians
        @param degrees<float>: The angle in degrees
        """
        radians = degrees * (math.pi / 180)

        return radians

    def _execute_dwa_control(self, state, goal, input):
        """! This function is used to calculate the control.
        @param state<list>: The state of the vehicle.
        @param goal<list>: The goal of current step.
        @param input<list>: The input of the vehicle.
        """
        dw = self._calculate_dynamic_window(input)

        u = self._calculate_control_and_trajectory(state, dw, goal)

        return u

    def _calculate_dynamic_window(self, input):
        """! This function is used to calculate the dynamic window.
        @param state<list>: The state of the vehicle.
        @param input<list>: The input of the vehicle.
        """
        Vs = [self._min_speed, self._max_speed,
              -self._max_yaw_rate, self._max_yaw_rate]

        Vd = [input[0] - self.max_accel * self._dt,
              input[0] + self.max_accel * self._dt,
              input[1] - self.max_delta_yaw_rate * self._dt,
              input[1] + self.max_delta_yaw_rate * self._dt]

        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw

    def _predict_trajectory(self, state, v, y):
        """! This function uses the kinematic model.
        Predict the state of the robot in self._predict_time
        using the control [v, y].
        @param state<list>: The state of the vehicle.
        @param v<float>: The speed of the vehicle.
        @param y<float>: The yaw rate of the vehicle.
        """
        trajectory = np.array(state)

        time = 0

        while time <= self._predict_time:

            state = self.model.function(state, [v, y], self._dt)

            trajectory = np.vstack((trajectory, state))

            time += self._dt

        return trajectory

    def _calculate_goal_cost(self, trajectory, goal):
        """! This function is used to calculate the goal cost.
        @param trajectory<list>: The trajectory of the vehicle.
        @param goal<list>: The goal of current step.
        """
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

        return cost

    def _calculate_control_and_trajectory(self, state, dw, goal):
        """! This function is used to calculate the control and trajectory.
        The best control of all sample is the one that minimize the cost
        function.
        Cost function = goal_cost + speed_cost

        @param state<list>: The state of the vehicle.
        @param dw<list>: The dynamic window.
        @param goal<list>: The goal of current step.
        """
        min_cost = float("inf")

        best_u = [0.0, 0.0]

        for v in np.arange(dw[0], dw[1], self.v_resolution):

            for y in np.arange(dw[2], dw[3], self.yaw_rate_resolution):

                trajectory = self._predict_trajectory(state, v, y)

                to_goal_cost = self._to_goal_cost_gain * \
                    self._calculate_goal_cost(trajectory, goal)

                speed_cost = self._speed_cost_gain * (self._max_speed - v)

                final_cost = to_goal_cost + speed_cost

                if min_cost >= final_cost:

                    min_cost = final_cost

                    best_u = [v, y]

        return best_u
