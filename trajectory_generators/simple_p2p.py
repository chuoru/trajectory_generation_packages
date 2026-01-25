#!/usr/bin/env python3
##
# @file navigation.py
#
# @brief Provide the navigation program for the robot.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2024/08/01

# Standard library
import numpy as np

# External library
import casadi as cs

# Internal library


class SimpleP2P:
    """! The class to generate the trajectory for the robot.

    The class provides the method to generate the trajectory for the robot.
    In this case, the robot moves from the initial position to the final
    position.
    """
    # ==================================================================================================
    # PUBLIC METHODS
    # ==================================================================================================
    def __init__(self, model, T, dt):
        """! The constructor of the class.
        @param model: The model of the robot.
        @param T: The time to reach the final position.
        @param dt: The time step of the optimization problem.
        @note The constructor initializes the model of the robot, the time to
        reach the final position, and the time step of the optimization
        problem. For instance:
        - self._N: The number of time steps.
        - self._dt: The time step of the optimization problem.
        - self._weights: The weights of the optimization problem.
        [q, qtheta, r, qN, qthetaN]
        """
        self._model = model

        self._optimizer = cs.Opti()

        self._optimizer.solver('ipopt', {'print_time': False},
                               {'max_iter': 3000, 'acceptable_iter': 1000,
                                'print_level': 5, 'tol': 0.0001,
                                'acceptable_tol': 0.1, 'mu_init': 0.05,
                                'warm_start_init_point': 'yes'})

        self._N = int(T / dt)

        self._dt = dt

        self._weight_values = [10, 0.1, 1, 200, 2]

        self._debug = True

    def generate_trajectory(self, initial_position, final_position,
                            is_plot=True):
        """! Generate the trajectory for the robot.
        @param initial_position: The initial position of the robot.
        @param final_position: The final position of the robot.
        @param is_plot: The flag to plot the solution.
        """
        self._define_problem()

        self._optimizer.set_value(self._weights, self._weight_values)

        self._optimizer.set_value(self._final_position, final_position)

        self._optimizer.set_value(self._inital_position, initial_position)

        self._optimizer.set_initial(self._u, np.zeros([self._model.nu,
                                                       self._N]))

        try:
            solution = self._optimizer.solve()

        except Exception:
            import traceback
            traceback.print_exc()
            solution = self._optimizer.debug

        return solution.value(self._u)

    # ==================================================================================================
    # PRIVATE METHODS
    # ==================================================================================================

    def _define_problem(self):
        """! Define the optimization problem.
        """
        self._u = self._optimizer.variable(self._model.nu, self._N)

        self._weights = self._optimizer.parameter(len(self._weight_values))

        self._weights = self._weight_values

        self._final_position = self._optimizer.parameter(self._model.nx)

        self._inital_position = self._optimizer.parameter(self._model.nx)

        x = cs.horzcat(self._inital_position)

        cost = 0

        for t in range(0, self._N):
            cost += self._stage_cost(x,
                                     self._u[:, t], self._final_position)

            x = self._model.function(x, self._u[:, t], self._dt)

            self._constraints(self._u[:, t])

        cost += self._terminal_cost(x, self._final_position)

        self._optimizer.minimize(cost)

    def _stage_cost(self, x, u, final_position):
        """! The stage cost of the optimization problem. 
        The distance between \f$(x_1,y_1)\f$ and \f$(x_2,y_2)\f$ is 
        \f$\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}\f$.
        @param x: The state of the robot.
        @param u: The input of the robot.
        @param final_position: The final position of the robot.
        """
        position_error = x - final_position

        cost = cs.dot(self._weights[:3], cs.power(position_error, 2))

        cost += self._weights[3] * cs.dot(u, u)

        return cost

    def _terminal_cost(self, x, final_position):
        """! The terminal cost of the optimization problem.
        @param x: The state of the robot.
        @param final_position: The final position of the robot.
        """
        position_error = x - final_position

        cost = self._weights[4] * cs.dot(position_error, position_error)

        return cost

    def _constraints(self, u):
        """! The constraints of the optimization problem.
        @param u: The input of the robot.
        """
        self._optimizer.subject_to(u[0]**2 < self._model.velocity_max**2)

        self._optimizer.subject_to(u[1]**2 < self._model.velocity_max**2)