#!/usr/bin/env python3
# Standard library
import math
import numpy as np


class PurePursuit:
    """! Pure Pursuit controller

    The class provides implementation of pure pursuit controller for
    autonomous driving.
    @note The controller is used to follow a trajectory.
    """
    lookahead_distance = 0.3

    lookahead_gain = 0.0

    k = 0.6

    k_i = 0.0

    k_ff = 1.0

    # ==================================================================================================
    # PUBLIC METHODS
    # ==================================================================================================
    def __init__(self, model, trajectory):
        """! Constructor
        @param model<instance>: The vehicle model
        @param trajectory<instance>: The trajectory
        @note The trajectory is a set of waypoints.
        """
        self.model = model

        self.trajectory = trajectory

        self.old_nearest_point_index = None

        self._v = 0.0

        self._integral_v = 0.0

        self._max_acceleration = 1.0

        self._max_angular_acceleration = 4.0

        self._w = 0.0

    def initialize(self):
        """! Initialize the controller
        @note The method is used to initialize the controller.
        """
        self._v = 0.0
        self._integral_v = 0.0
        self._w = 0.0

    def execute(self, state, input, previous_index):
        """! Execute the controller
        @param state<list>: The state of the vehicle
        @param input<list>: The input of the vehicle
        @param previous_index<int>: The previous index
        @return<tuple>: The status and control
        """
        status = True

        index, lookahead_distance = self._search_target_index(state, input)

        if previous_index >= index:
            index = previous_index

        if self._is_terminated(state):
            status = False

            return status, [0.0, 0.0]

        if index < len(self.trajectory.x):
            trajectory_x = self.trajectory.x[index, 0]

            trajectory_y = self.trajectory.x[index, 1]

        else:
            trajectory_x = self.trajectory.x[-1, 0]

            trajectory_y = self.trajectory.x[-1, 1]

            index = len(self.trajectory.x) - 1

        alpha = (
            math.atan2(
                trajectory_y - state[1],
                trajectory_x - state[0],
            )
            - state[2]
        )

        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        n_pts = len(self.trajectory.u[0])
        v_ref = self.trajectory.u[0, index]
        v_next = self.trajectory.u[0, min(index + 1, n_pts - 1)]
        a_ff = PurePursuit.k_ff * (v_next - v_ref) / self.trajectory.sampling_time

        e_v = v_ref - self._v
        self._integral_v += e_v * self.trajectory.sampling_time
        self._integral_v = np.clip(self._integral_v, -1.0, 1.0)
        a = a_ff + PurePursuit.k * e_v + PurePursuit.k_i * self._integral_v

        a = self._max_acceleration if a > self._max_acceleration else a

        a = -self._max_acceleration if a < -self._max_acceleration else a

        self._v = self._v + a * self.trajectory.sampling_time

        w = self._v * 2.0 * alpha / lookahead_distance

        dwdt = (w - self._w) / self.trajectory.sampling_time

        if dwdt > self._max_angular_acceleration:
            w = self._w + self._max_angular_acceleration * self.trajectory.sampling_time
        elif dwdt < -self._max_angular_acceleration:
            w = self._w - self._max_angular_acceleration * self.trajectory.sampling_time
        else:
            w = self._w + dwdt * self.trajectory.sampling_time
        self._w = w

        return status, [self._v, w]

    # ==================================================================================================
    # STATIC METHODS
    # ==================================================================================================
    def _is_terminated(self, state):
        """! Check if the controller is terminated
        @param state<list>: The state of the vehicle
        @return<bool>: The status
        """
        terminal_point = self.trajectory.x[-1]
        distance = self._calculate_distance(terminal_point, state)
        if distance < 0.1:
            return True

        return False

    @staticmethod
    def _apply_proportional_control(k_p, target, current):
        """! Apply proportional control
        @param k_p<float>: The proportional gain
        @param target<list>: The target
        @param current<list>: The current
        @return<float>: The acceleration
        """
        a = k_p * (target - current)

        return a

    @staticmethod
    def _calculate_distance(reference_x, current_x):
        """! Calculate the distance
        @param reference_x<list>: The reference x
        @param current_x<list>: The current x
        @return<float>: The distance
        """
        distance = current_x - reference_x

        x = distance[:, 0] if distance.ndim == 2 else distance[0]

        y = distance[:, 1] if distance.ndim == 2 else distance[1]

        return np.hypot(x, y)

    # ==================================================================================================
    # PRIVATE METHODS
    # ==================================================================================================
    def _search_target_index(self, state, input):
        """! Search the target index
        @param state<list>: The state of the vehicle
        @param input<list>: The input of the vehicle
        @return<tuple>: The index and lookahead distance
        """
        if self.old_nearest_point_index is None:
            all_distance = self._calculate_distance(self.trajectory.x, state)

            index = np.argmin(all_distance)

        else:
            index = self.old_nearest_point_index

            this_distance = self._calculate_distance(
                self.trajectory.x[index], state)

            while True:
                next_distance = self._calculate_distance(
                    self.trajectory.x[index + 1], state
                )

                if this_distance < next_distance:
                    break

                if (index + 1) < len(self.trajectory.x):
                    index += 1

                this_distance = next_distance

            self.old_nearest_point_index = index

        v = (input[0] + input[1]) / 2

        lookahead_distance = (
            PurePursuit.lookahead_gain * v + PurePursuit.lookahead_distance
        )

        distance = self._calculate_distance(self.trajectory.x[index], state)

        while lookahead_distance > distance:
            if index + 1 >= len(self.trajectory.x):
                break

            index += 1

            distance = self._calculate_distance(
                self.trajectory.x[index], state)

        return index, lookahead_distance
