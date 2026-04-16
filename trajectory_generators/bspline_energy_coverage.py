#!/usr/bin/env python3
##
# @file bspline_energy_coverage.py
#
# @brief Energy-aware B-spline OCP for differential drive coverage trajectory
# generation, driven by the TJ108 physics-based wheel-level power model.
#
# Inherits the full B-spline parameterization, knot vectors, basis matrices,
# kinodynamic constraint builders, and inverse kinematics from BSplineCoverage.
# The objective is replaced by a weighted combination of total mission time and
# total electrical energy consumed, enabling time-energy Pareto trade-offs.
#
# TJ108 Power Model (per wheel motor):
#   P(omega_w, omega_w_dot) = c1
#                             + c2 * omega_w
#                             + c3 * omega_w^2
#                             + c4 * omega_w^3
#                             + c6 * omega_w_dot
#                             + c7 * omega_w_dot^2
#                             + c8 * omega_w * omega_w_dot
#
# State-to-wheel mapping for a differential drive with half-wheelbase l and
# wheel radius r:
#   omega_r     = (v + l * Omega) / r
#   omega_l     = (v - l * Omega) / r
#   omega_r_dot = (a + l * alpha) / r
#   omega_l_dot = (a - l * alpha) / r
#
# where v, a are body-frame forward velocity and acceleration (projected along
# heading), and Omega, alpha are robot angular velocity and acceleration.
#
# Objective:
#   J = w_time * T_mission + w_energy * E_total
#     = w_time * (sum(T) / nt) + w_energy * (sum(P_total_i * T_i) / nt)
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/04/16

# Standard library
import numpy as np

# External library
import casadi as cs

# Internal library
from .bspline_coverage import BSplineCoverage


class BSplineEnergyCoverage(BSplineCoverage):
    """! Energy-aware B-spline OCP using the TJ108 physics-based power model.

    Extends BSplineCoverage by replacing the time-minimization objective with
    a weighted sum of mission time and total electrical energy. The energy is
    computed by integrating a polynomial motor power model over the trajectory,
    using wheel-level angular velocities and accelerations derived from the
    B-spline derivatives.

    The Pareto front between minimum-time and minimum-energy trajectories can
    be explored by sweeping the ratio w_time / w_energy.
    """

    # TJ108 polynomial power coefficients (per motor), identified from bench tests.
    # Each is a 6-element list [c1, c2, c3, c4, c5, c6] for the model:
    #   P(omega_w, omega_w_dot) = c1
    #                           + c2  * omega_w
    #                           + c3  * omega_w^2
    #                           + c4  * omega_w^3
    #                           + c5  * omega_w_dot
    #                           + c6  * omega_w_dot^2
    # Right and left motors have SEPARATE sets: each physical motor has
    # distinct friction, wear, and inertia characteristics.
    _DEFAULT_ENERGY_COEFFS_RIGHT = [
        0.302433145557389,       # c1 – static / base load           [W]
        31.887262598534413,      # c2 – viscous friction (linear)    [W·s/rad]
        2.4140287888312457,      # c3 – viscous friction (quadratic) [W·s²/rad²]
        0.9658866923308425,      # c4 – viscous friction (cubic)     [W·s³/rad³]
        0.8260871406535432,      # c5 – inertial (linear in α)       [W·s²/rad]
        2.37456174658809e-08,    # c6 – inertial (quadratic in α)    [W·s⁴/rad²]
    ]

    _DEFAULT_ENERGY_COEFFS_LEFT = [
        0.33789198669595977,     # c1
        28.204019732889346,      # c2
        2.5903002025839688,      # c3
        0.00847962183165042,     # c4
        6.412423386896174e-09,   # c5
        0.3614761744737831,      # c6
    ]

    # Robot physical parameters.
    _DEFAULT_ROBOT_PARAMS = {
        'l': 0.175,     # half-wheelbase [m]  (full track = 2l = 0.35 m)
        'r': 0.050,     # wheel radius   [m]
    }

    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    def __init__(self, waypoints, bound=0.1, n_ctrl_pts=6, spline_order=3,
                 n_sampling=50, vel_max=None, vel_min_lin=0.01,
                 eps_nonh=0.001, robot_params=None,
                 energy_coeffs_right=None, energy_coeffs_left=None,
                 w_time=1.0, w_energy=1.0, e_max=None, p_electronics=2.0):
        """! Constructor.

        All parameters of BSplineCoverage are accepted unchanged.

        @param waypoints<list>: Via-points [[x, y, theta], ...].
        @param bound<float>: Corridor half-width [m].
        @param n_ctrl_pts<int>: Control points per segment.
        @param spline_order<int>: B-spline degree.
        @param n_sampling<int>: Discrete samples per segment.
        @param vel_max<list|None>: [vx_max, vy_max, omega_max].
        @param vel_min_lin<float>: Minimum feedrate [m/s].
        @param eps_nonh<float>: Nonholonomic constraint tolerance.
        @param robot_params<dict|None>: Physical robot parameters.
            Keys: 'l' (half-wheelbase [m]), 'r' (wheel radius [m]).
            Missing keys fall back to _DEFAULT_ROBOT_PARAMS.
        @param energy_coeffs_right<list|None>: 6-element list [c1..c6] for the
            RIGHT motor. None uses _DEFAULT_ENERGY_COEFFS_RIGHT.
        @param energy_coeffs_left<list|None>: 6-element list [c1..c6] for the
            LEFT motor. None uses _DEFAULT_ENERGY_COEFFS_LEFT.
        @param w_time<float>: Weight on total traversal time in the objective.
        @param w_energy<float>: Weight on total energy consumption.
        @param e_max<float|None>: Hard upper bound on energy [J]. None = none.
        @param p_electronics<float>: Constant hotel load (sensors, computer) [W].
        """
        super().__init__(waypoints, bound, n_ctrl_pts, spline_order,
                         n_sampling, vel_max, vel_min_lin, eps_nonh)

        self._robot = {**self._DEFAULT_ROBOT_PARAMS, **(robot_params or {})}
        self._e_coeffs_right = list(
            energy_coeffs_right if energy_coeffs_right is not None
            else self._DEFAULT_ENERGY_COEFFS_RIGHT
        )
        self._e_coeffs_left = list(
            energy_coeffs_left if energy_coeffs_left is not None
            else self._DEFAULT_ENERGY_COEFFS_LEFT
        )
        if len(self._e_coeffs_right) != 6 or len(self._e_coeffs_left) != 6:
            raise ValueError("energy_coeffs_right and energy_coeffs_left "
                             "must each have exactly 6 elements [c1..c6].")
        self._w_time = float(w_time)
        self._w_energy = float(w_energy)
        self._e_max = e_max
        self._p_elec = float(p_electronics)

    def generate_trajectory(self):
        """! Build and solve the energy-aware OCP.

        @return dict with all keys from BSplineCoverage, plus:
            - 'power'  : (nt,) total power signal P(t) over the spline [W]
            - 'energy' : float total mission energy E_total [J]
        """
        knot = self._build_knot_vector()
        tau = np.linspace(0.0, 1.0, self._nt)

        basis, dot_basis, ddot_basis, dddot_basis = \
            self._build_basis_matrices(tau, knot)

        opti = self._optimizer

        # Decision variables: control points + per-sample time scaling.
        px = opti.variable(self._n_Q)
        py = opti.variable(self._n_Q)
        pt = opti.variable(self._n_Q)
        T = opti.variable(self._nt)

        P_ctrl = cs.horzcat(px, py, pt)     # (n_Q, 3) control-point matrix

        s    = basis        @ P_ctrl        # (nt, 3) spline values
        ds   = dot_basis    @ P_ctrl        # (nt, 3) first derivatives / dtau
        dds  = ddot_basis   @ P_ctrl        # (nt, 3) second derivatives / dtau^2
        ddds = dddot_basis  @ P_ctrl        # (nt, 3) third derivatives / dtau^3

        self._set_initial_conditions(opti, px, py, pt, T, basis)

        # --- Energy-aware objective ------------------------------------------
        energy_sym, power_sym = self._build_energy_terms(s, ds, dds, T)
        T_mission = cs.sum1(T) / self._nt
        opti.minimize(self._w_time * T_mission
                      + self._w_energy * energy_sym)

        # --- Kinodynamic + path constraints (all inherited) ------------------
        self._add_dynamic_constraints(opti, s, ds, dds, ddds, T)
        self._add_boundary_constraints(opti, px, py, pt)
        self._add_corridor_constraints(opti, px, py)

        # --- Optional hard energy-budget constraint --------------------------
        if self._e_max is not None:
            opti.subject_to(energy_sym <= self._e_max)

        # --- Solve -----------------------------------------------------------
        try:
            sol = opti.solve()
            dbg = sol
        except Exception:
            dbg = opti.debug

        T_val     = dbg.value(T)
        s_val     = dbg.value(s)
        ctrl_pts  = dbg.value(P_ctrl)
        power_val = np.array(dbg.value(power_sym)).flatten()
        energy_val = float(dbg.value(energy_sym))

        # Real time axis: t[i] = cumsum(T)[i] / nt
        t_real = np.cumsum(T_val) / self._nt

        # Inverse kinematics to recover (v, omega) control inputs
        ts_des = 0.01
        t_interp = np.arange(ts_des, t_real[-1], ts_des)
        states_interp = np.column_stack([
            np.interp(t_interp, t_real, s_val[:, 0]),
            np.interp(t_interp, t_real, s_val[:, 1]),
            np.interp(t_interp, t_real, s_val[:, 2]),
        ])
        v_arr, omega_arr = self._inverse_kinematics(states_interp, ts_des)

        return {
            'states':   s_val,
            'time':     t_real,
            'ctrl_pts': ctrl_pts,
            'v':        v_arr,
            'omega':    omega_arr,
            'time_ik':  t_interp[:-1],
            'power':    power_val,
            'energy':   energy_val,
        }

    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    def _build_energy_terms(self, s, ds, dds, T):
        """! Construct symbolic total energy and per-sample power expressions.

        Computes, in vectorized CasADi form, the robot body-frame forward
        velocity/acceleration and angular velocity/acceleration at every
        spline sample. Maps these to individual wheel dynamics via the
        differential-drive kinematic model, then evaluates the TJ108 power
        polynomial for each wheel. Finally integrates power over time.

        @param s:   (nt, 3) CasADi MX spline values [x, y, theta].
        @param ds:  (nt, 3) first  B-spline derivatives w.r.t. tau.
        @param dds: (nt, 3) second B-spline derivatives w.r.t. tau.
        @param T:   (nt, 1) CasADi MX per-sample time-scaling variables.
        @return Tuple (energy_scalar, power_column):
            - energy_scalar: CasADi scalar expression for E_total [J].
            - power_column:  (nt, 1) CasADi expression for P_total(t) [W].
        """
        l = self._robot['l']
        r = self._robot['r']
        nt = self._nt

        # --- Body-frame projections ------------------------------------------
        # Heading angle at each sample
        theta  = s[:, 2]                        # (nt, 1)
        cos_th = cs.cos(theta)
        sin_th = cs.sin(theta)

        # Physical forward velocity: v = (ds_x*cos + ds_y*sin) / T
        # (quasi-static approximation; exact under the nonholonomic constraint)
        v_fwd  = (cos_th * ds[:, 0] + sin_th * ds[:, 1]) / T       # [m/s]

        # Physical forward acceleration: a ≈ (dds_x*cos + dds_y*sin) / T^2
        a_fwd  = (cos_th * dds[:, 0] + sin_th * dds[:, 1]) / T**2  # [m/s²]

        # Robot yaw rate and angular acceleration
        omega_robot = ds[:, 2]  / T                                  # [rad/s]
        alpha_robot = dds[:, 2] / T**2                               # [rad/s²]

        # --- Wheel-level dynamics --------------------------------------------
        omega_r, omega_l, omega_r_dot, omega_l_dot = \
            self._calculate_wheel_dynamics(
                v_fwd, omega_robot, a_fwd, alpha_robot, l, r)

        # --- TJ108 motor power per wheel (separate coefficient sets) ---------
        P_r = self._tj108_power(omega_r, omega_r_dot, self._e_coeffs_right)
        P_l = self._tj108_power(omega_l, omega_l_dot, self._e_coeffs_left)

        # Total electrical power at each sample (both motors + hotel load)
        P_total = P_r + P_l + self._p_elec                          # (nt, 1)

        # --- Energy integral -------------------------------------------------
        # E = integral P(t) dt ≈ sum_i P_i * delta_t_i
        # Time step at sample i: delta_t_i = T_i / nt
        energy = cs.sum1(P_total * T) / nt                          # scalar

        return energy, P_total

    def _calculate_wheel_dynamics(self, v_fwd, omega_robot,
                                  a_fwd, alpha_robot, l, r):
        """! Map robot body-frame kinematics to individual wheel dynamics.

        Uses the standard differential-drive inverse kinematic model:
            omega_r     = (v + l * Omega) / r
            omega_l     = (v - l * Omega) / r
            omega_r_dot = (a + l * alpha) / r
            omega_l_dot = (a - l * alpha) / r

        All inputs and outputs are CasADi (nt, 1) column vectors.

        @param v_fwd:        Forward velocity [m/s].
        @param omega_robot:  Robot yaw rate   [rad/s].
        @param a_fwd:        Forward acceleration [m/s²].
        @param alpha_robot:  Robot angular acceleration [rad/s²].
        @param l:            Half-wheelbase [m].
        @param r:            Wheel radius   [m].
        @return Tuple (omega_r, omega_l, omega_r_dot, omega_l_dot), each (nt,1).
        """
        omega_r     = (v_fwd + l * omega_robot) / r
        omega_l     = (v_fwd - l * omega_robot) / r
        omega_r_dot = (a_fwd + l * alpha_robot) / r
        omega_l_dot = (a_fwd - l * alpha_robot) / r
        return omega_r, omega_l, omega_r_dot, omega_l_dot

    def _tj108_power(self, omega_w, omega_w_dot, coeffs):
        """! TJ108 polynomial power model for one wheel motor.

        P(omega_w, omega_w_dot) =
            c[0]
          + c[1] * omega_w
          + c[2] * omega_w^2
          + c[3] * omega_w^3
          + c[4] * omega_w_dot
          + c[5] * omega_w_dot^2

        Captures static friction (c[0]), viscous losses (c[1]..c[3]), and
        the inertial work required for acceleration (c[4]..c[5]).

        Right and left motors use separate 6-element coefficient lists
        because each physical motor has distinct friction and inertia.

        @param omega_w:     Wheel angular velocity    [rad/s], CasADi MX.
        @param omega_w_dot: Wheel angular acceleration [rad/s²], CasADi MX.
        @param coeffs<list>: 6-element list [c1..c6] for this motor.
        @return CasADi expression for motor power [W], same shape as inputs.
        """
        return (coeffs[0]
                + coeffs[1] * omega_w
                + coeffs[2] * omega_w**2
                + coeffs[3] * omega_w**3
                + coeffs[4] * omega_w_dot
                + coeffs[5] * omega_w_dot**2)
