#!/usr/bin/env python3
##
# @file direct_coverage.py
#
# @brief Direct collocation OCP for trajectory generation on a differential
# drive robot following a piecewise-linear reference path.
#
# Based on the work of Fabian Friz, Toyohashi University of Technology (08/2024).
# Discretizes the unicycle dynamics with the implicit midpoint rule. The free
# terminal time tf and the state/input trajectories are optimized jointly by
# IPOPT. A path-corridor constraint keeps the trajectory close to the reference.
# Finite differences of the dynamics enforce acceleration and jerk limits.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/04/15

# Standard library
import numpy as np

# External library
import casadi as cs


class DirectCoverage:
    """! Direct collocation OCP for differential drive coverage.

    Discretizes the unicycle model xdot = [cos(theta)*v, sin(theta)*v, omega]
    with the implicit midpoint rule over N intervals. Minimizes total time tf
    subject to path-corridor, velocity, acceleration, jerk, and input-rate
    constraints.
    """

    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    def __init__(self, waypoints, n_intervals=200, path_bound=0.1,
                 v_max=0.2, v_min=0.05, omega_max=0.196,
                 acc_input_max=0.9, acc_max=20.0, jerk_max=1e3,
                 tf_init=10.0, tf_max=60.0):
        """! Constructor.
        @param waypoints<list>: Via-points [[x, y, theta], ...]. At least 2.
        @param n_intervals<int>: Number of collocation intervals N.
        @param path_bound<float>: Corridor half-width around reference [m].
        @param v_max<float>: Maximum forward speed [m/s].
        @param v_min<float>: Minimum forward speed [m/s] (> 0 prevents reversal).
        @param omega_max<float>: Maximum angular speed [rad/s].
        @param acc_input_max<float>: Maximum input rate of change [1/s].
        @param acc_max<float>: Maximum state acceleration [m/s^2 or rad/s^2].
        @param jerk_max<float>: Maximum state jerk [m/s^3 or rad/s^3].
        @param tf_init<float>: Initial guess for terminal time [s].
        @param tf_max<float>: Upper bound for terminal time [s].
        """
        self._waypoints = np.array(waypoints, dtype=float)
        self._N = n_intervals
        self._path_bound = path_bound
        self._v_max = v_max
        self._v_min = v_min
        self._omega_max = omega_max
        self._acc_input_max = acc_input_max
        self._acc_max = acc_max
        self._jerk_max = jerk_max
        self._tf_init = tf_init
        self._tf_max = tf_max

        self._nx = 3    # [x, y, theta]
        self._nu = 2    # [v, omega]

        self._optimizer = cs.Opti()
        self._optimizer.solver(
            'ipopt',
            {'print_time': False},
            {
                'max_iter': 10000,
                'print_level': 5,
                'tol': 1e-5,
                'acceptable_tol': 1e-4,
                'acceptable_iter': 25,
                'constr_viol_tol': 1e-5,
            }
        )

    def generate_trajectory(self):
        """! Build and solve the direct collocation OCP.
        @return dict with keys:
            - 'states'   : (N+1, 3) array [x, y, theta]
            - 'inputs'   : (N, 2)   array [v, omega]
            - 'time'     : (N+1,)   time vector [s]
            - 'tf'       : float    optimal total time [s]
            - 'x_ref'    : (N, 2)   reference xy path
        """
        N = self._N
        nx, nu = self._nx, self._nu

        # Build reference trajectory (N points, one per interval start)
        x_ref, y_ref, theta_ref = self._build_reference(N)

        opti = self._optimizer

        # --- Decision variables ---
        tf = opti.variable()
        opti.subject_to(tf >= 0.1)
        opti.subject_to(tf <= self._tf_max)
        opti.set_initial(tf, self._tf_init)

        # States at N+1 knot points (X[0] = initial, X[N] = final)
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)

        # --- Warm start ---
        self._set_initial_conditions(opti, X, U, x_ref, y_ref, theta_ref)

        # --- Objective: minimize time + small control effort ---
        h = 1.0 / N     # normalized step size (T=1 internally)
        cost = tf + cs.sum2(cs.sum1(U**2)) * (1e-2 * tf * h)
        opti.minimize(cost)

        # --- Boundary conditions ---
        x0 = self._waypoints[0]
        xf = self._waypoints[-1]
        opti.subject_to(X[:, 0] == x0)
        opti.subject_to(X[:2, N] == xf[:2])
        opti.subject_to(X[2, N] == xf[2])

        # --- Dynamics and path constraints per interval ---
        second_deriv_prev = None
        Uk_prev = None

        for k in range(N):
            Xk = X[:, k]
            Xk1 = X[:, k + 1]
            Uk = U[:, k]

            # Implicit midpoint collocation
            X_mid = (Xk + Xk1) / 2.0
            f_mid = self._dynamics(X_mid, Uk, tf)
            opti.subject_to(Xk1 - Xk == h * f_mid)

            # Path corridor: |X_k[0:2] - ref[k]| <= path_bound
            opti.subject_to(
                X[:2, k] - cs.vertcat(x_ref[k], y_ref[k])
                <= self._path_bound)
            opti.subject_to(
                -(self._path_bound)
                <= X[:2, k] - cs.vertcat(x_ref[k], y_ref[k]))

            # Input bounds
            opti.subject_to(self._v_min <= Uk[0])
            opti.subject_to(Uk[0] <= self._v_max)
            opti.subject_to(-self._omega_max <= Uk[1])
            opti.subject_to(Uk[1] <= self._omega_max)

            # Acceleration: finite difference of dynamics
            f_prev = self._dynamics(Xk, Uk, tf)
            second_deriv = (f_mid - f_prev) / h

            opti.subject_to(-self._acc_max <= second_deriv)
            opti.subject_to(second_deriv <= self._acc_max)

            # Jerk and input rate (need previous step)
            if k > 0:
                third_deriv = (second_deriv - second_deriv_prev) / h
                opti.subject_to(-self._jerk_max <= third_deriv)
                opti.subject_to(third_deriv <= self._jerk_max)

                du = (Uk - Uk_prev) / h
                opti.subject_to(-self._acc_input_max <= du)
                opti.subject_to(du <= self._acc_input_max)

            second_deriv_prev = second_deriv
            Uk_prev = Uk

        # --- Solve ---
        try:
            sol = opti.solve()
            dbg = sol
        except Exception:
            dbg = opti.debug

        # --- Extract solution ---
        tf_opt = float(dbg.value(tf))
        X_opt = dbg.value(X).T          # (N+1, 3)
        U_opt = dbg.value(U).T          # (N, 2)
        t_opt = np.linspace(0.0, tf_opt, N + 1)

        return {
            'states': X_opt,
            'inputs': U_opt,
            'time': t_opt,
            'tf': tf_opt,
            'x_ref': np.column_stack([x_ref, y_ref]),
        }

    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    def _dynamics(self, x, u, tf):
        """! Unicycle kinematics scaled by free terminal time tf.
        @param x: State [x, y, theta].
        @param u: Input [v, omega].
        @param tf: Free terminal time (scales dynamics to normalized time).
        @return xdot * tf (CasADi expression).
        """
        return cs.vertcat(
            cs.cos(x[2]) * u[0],
            cs.sin(x[2]) * u[0],
            u[1]
        ) * tf

    def _build_reference(self, N):
        """! Build a piecewise-linear reference trajectory of N points.

        Points are distributed proportionally to segment lengths so that
        the robot travels at roughly uniform speed along the reference.

        @param N: Number of reference points (one per collocation interval).
        @return Tuple (x_ref, y_ref, theta_ref) each of length N.
        """
        wps = self._waypoints
        n_seg = len(wps) - 1

        # Segment lengths
        seg_len = [np.linalg.norm(wps[i + 1, :2] - wps[i, :2])
                   for i in range(n_seg)]
        total_len = sum(seg_len)

        # Distribute N points proportionally
        seg_pts = [max(1, round(N * seg_len[i] / total_len))
                   for i in range(n_seg)]
        # Adjust rounding to exactly N total
        diff = N - sum(seg_pts)
        seg_pts[-1] += diff

        x_ref, y_ref, theta_ref = [], [], []
        for i in range(n_seg):
            n = seg_pts[i]
            alphas = np.linspace(0.0, 1.0, n, endpoint=(i == n_seg - 1))
            p0, p1 = wps[i], wps[i + 1]
            x_ref.extend((1 - alphas) * p0[0] + alphas * p1[0])
            y_ref.extend((1 - alphas) * p0[1] + alphas * p1[1])
            theta_ref.extend((1 - alphas) * p0[2] + alphas * p1[2])

        return (np.array(x_ref[:N]),
                np.array(y_ref[:N]),
                np.array(theta_ref[:N]))

    def _set_initial_conditions(self, opti, X, U, x_ref, y_ref, theta_ref):
        """! Warm-start states along reference, inputs at v_max/2."""
        N = self._N
        wps = self._waypoints

        # States: follow reference path
        for k in range(N):
            opti.set_initial(X[0, k], x_ref[k])
            opti.set_initial(X[1, k], y_ref[k])
            opti.set_initial(X[2, k], theta_ref[k])

        opti.set_initial(X[0, N], wps[-1, 0])
        opti.set_initial(X[1, N], wps[-1, 1])
        opti.set_initial(X[2, N], wps[-1, 2])

        # Inputs: constant forward at half max speed
        opti.set_initial(U[0, :], self._v_max / 2.0)
        opti.set_initial(U[1, :], 0.0)
