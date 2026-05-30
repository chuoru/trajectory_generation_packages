#!/usr/bin/env python3
##
# @file bspline_energy_coverage.py
#
# @brief Energy-aware B-spline OCP for differential drive coverage trajectory
# generation, driven by a physics-based wheel-level power model.
#
# Inherits the full B-spline parameterization, knot vectors, basis matrices,
# kinodynamic constraint builders, and inverse kinematics from BSplineCoverage.
# The objective is replaced by a weighted combination of total mission time and
# total electrical energy consumed, enabling time-energy Pareto trade-offs.
#
# Power Model (per wheel motor):
#   P(v_w, a_w) = p[0] * a_w^2
#                + p[1] * v_w^2
#                + |p[2] * a_w|
#                + |p[3] * v_w|
#                + |p[4] * v_w * a_w|
#                + p[5]
#
# State-to-wheel mapping for a differential drive with half-wheelbase l:
#   v_r = v + l * Omega      (right wheel linear velocity [m/s])
#   v_l = v - l * Omega      (left wheel linear velocity [m/s])
#   a_r = a + l * alpha      (right wheel linear acceleration [m/s²])
#   a_l = a - l * alpha      (left wheel linear acceleration [m/s²])
#
# where v, a are body-frame forward velocity and acceleration (projected along
# heading), and Omega, alpha are robot angular velocity and acceleration.
#
# Objective:
#   J = w_time * T_mission + w_energy * E_total
#     = w_time * (sum(T) / nt)
#       + w_energy * (sum((P_i+P_{i+1})/2 * (T_i+T_{i+1})/2) / nt)  [trapezoid]
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/04/16

# Standard library
import numpy as np

# External library
import casadi as cs
from scipy.interpolate import PchipInterpolator as _Pchip

# Internal library
from .bspline_coverage import BSplineCoverage


class BSplineEnergyCoverage(BSplineCoverage):
    """! Energy-aware B-spline OCP using a physics-based wheel-level power model.

    Extends BSplineCoverage by replacing the time-minimization objective with
    a weighted sum of mission time and total electrical energy. The energy is
    computed by integrating a polynomial motor power model over the trajectory,
    using wheel-level linear velocities and accelerations derived from the
    B-spline derivatives.

    The Pareto front between minimum-time and minimum-energy trajectories can
    be explored by sweeping the ratio w_time / w_energy.
    """

    # Polynomial power coefficients (per motor), identified from bench tests.
    # Each is a 6-element list [p0..p5] for the model:
    #   P(v_w, a_w) = p[0]*a_w^2 + p[1]*v_w^2
    #               + |p[2]*a_w| + |p[3]*v_w| + |p[4]*v_w*a_w|
    #               + p[5]
    # Right and left motors have SEPARATE sets: each physical motor has
    # distinct friction, wear, and inertia characteristics.
    _DEFAULT_ENERGY_COEFFS_RIGHT = [
        0.302433145557389,       # p[0] – inertial quadratic  a²    [W·s⁴/m²]
        31.887262598534413,      # p[1] – viscous quadratic   v²    [W·s²/m²]
        2.4140287888312457,      # p[2] – inertial linear    |a|    [W·s²/m]
        0.9658866923308425,      # p[3] – viscous linear     |v|    [W·s/m]
        0.8260871406535432,      # p[4] – cross term       |v·a|    [W·s³/m²]
        2.37456174658809e-08,    # p[5] – static / base load        [W]
    ]

    _DEFAULT_ENERGY_COEFFS_LEFT = [
        0.33789198669595977,     # p[0]
        28.204019732889346,      # p[1]
        2.5903002025839688,      # p[2]
        0.00847962183165042,     # p[3]
        6.412423386896174e-09,   # p[4]
        0.3614761744737831,      # p[5]
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
                 eps_nonh=0.001, v_entry=None, v_exit=None,
                 a_entry=None, a_exit=None,
                 omega_entry=None, omega_exit=None,
                 alpha_entry=None, alpha_exit=None, robot_params=None,
                 energy_coeffs_right=None, energy_coeffs_left=None,
                 w_time=1.0, w_energy=1.0, e_max=None, p_electronics=2.0,
                 acc_max=None, jerk_max=None):
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
        @param v_entry<float|None>: Exact entry speed [m/s]. None = free.
        @param v_exit<float|None>: Exact exit speed [m/s]. None = free.
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
        @param acc_max<list|None>: [ax_max, ay_max, alpha_max] physical acceleration
            limits [m/s², m/s², rad/s²]. None keeps BSplineCoverage defaults.
        @param jerk_max<list|None>: [jx_max, jy_max, jalpha_max] physical jerk
            limits [m/s³, m/s³, rad/s³]. None keeps BSplineCoverage defaults.
        """
        super().__init__(waypoints, bound, n_ctrl_pts, spline_order,
                         n_sampling, vel_max, vel_min_lin, eps_nonh,
                         v_entry, v_exit, a_entry, a_exit,
                         omega_entry, omega_exit,
                         alpha_entry, alpha_exit,
                         acc_max, jerk_max)

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

    def generate_trajectory(self, warm_start=None):
        """! Build and solve the energy-aware OCP.

        @param warm_start<dict|None>: Result dict from a previous call to
            generate_trajectory().  When provided, the solver is warm-started
            from that solution instead of the default linear initialisation.
            Useful for continuation sweeps (Pareto front tracing).
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

        if warm_start is not None:
            opti.set_initial(px, warm_start['ctrl_pts'][:, 0])
            opti.set_initial(py, warm_start['ctrl_pts'][:, 1])
            opti.set_initial(pt, warm_start['ctrl_pts'][:, 2])
            # Recover per-sample T from the previous real-time axis.
            t_prev = warm_start['time']
            T_prev = np.diff(t_prev) * len(t_prev)
            T_prev = np.append(T_prev, T_prev[-1])
            opti.set_initial(T, T_prev[:self._nt])
        else:
            self._set_initial_conditions(opti, px, py, pt, T, basis,
                                         dot_basis)

        # --- Energy-aware objective ------------------------------------------
        energy_motor_sym, energy_sym, power_sym = \
            self._build_energy_terms(s, ds, dds, T)
        T_mission = cs.sum1(T) / self._nt
        opti.minimize(self._w_time * T_mission
                      + self._w_energy * energy_motor_sym)

        # --- Kinodynamic + path constraints (all inherited) ------------------
        self._add_dynamic_constraints(opti, s, ds, dds, ddds, T)
        self._add_boundary_constraints(opti, px, py, pt)
        self._add_corridor_constraints(opti, px, py)

        # --- Wheel-level angular jerk constraints ----------------------------
        # The Cartesian jerk constraints in _add_dynamic_constraints limit each
        # coordinate independently. For a differential drive in a corner, the
        # right / left wheel angular jerk combines both body and angular jerk:
        #   jerk_{r,l} * r = d(acc_path)/dt ± l * d(alpha)/dt
        # Bounding this directly ensures |jerk_{r,l}| ≤ J_LIM/r everywhere,
        # which the independent Cartesian constraints do not guarantee.
        jerk_whl = float(self._jerk_max[0])   # J_LIM [m/s³]
        l_w = self._robot['l']
        for i in range(self._nt):
            Ti    = T[i]
            cos_i = cs.cos(s[i, 2])
            sin_i = cs.sin(s[i, 2])
            body_j = cos_i * ddds[i, 0] + sin_i * ddds[i, 1]
            ang_j  = l_w * ddds[i, 2]
            lim    = jerk_whl * Ti**3
            opti.subject_to(body_j + ang_j <= lim)
            opti.subject_to(-lim <= body_j + ang_j)
            opti.subject_to(body_j - ang_j <= lim)
            opti.subject_to(-lim <= body_j - ang_j)

        # --- Optional hard energy-budget constraint --------------------------
        if self._e_max is not None:
            opti.subject_to(energy_sym <= self._e_max)

        # --- Solve -----------------------------------------------------------
        try:
            sol = opti.solve()
            dbg = sol
        except Exception:
            dbg = opti.debug

        T_val      = dbg.value(T)
        s_val      = dbg.value(s)
        ds_val     = np.array(dbg.value(ds))
        dds_val    = np.array(dbg.value(dds))
        ddds_val   = np.array(dbg.value(ddds))
        ctrl_pts   = dbg.value(P_ctrl)
        power_val  = np.array(dbg.value(power_sym)).flatten()
        energy_val = float(dbg.value(energy_sym))

        # Real time axis: t[i] = cumsum(T)[i] / nt
        t_real = np.cumsum(T_val) / self._nt

        # Velocity and angular velocity at OCP nodes from B-spline derivatives
        cos_th = np.cos(s_val[:, 2])
        sin_th = np.sin(s_val[:, 2])
        v_ocp  = (cos_th * ds_val[:, 0] + sin_th * ds_val[:, 1]) / T_val
        om_ocp = ds_val[:, 2] / T_val

        # Physical forward and angular acceleration at OCP nodes, derived from
        # the B-spline 2nd derivative.  These pass through the exact OCP-node
        # values (including the pinned a_entry / alpha_entry boundary values),
        # so interpolating them directly avoids the boundary mismatch that
        # occurs when using the CubicSpline 1st derivative for v/omega.
        a_ocp     = (cos_th * dds_val[:, 0] + sin_th * dds_val[:, 1]) / T_val**2
        alpha_ocp = dds_val[:, 2] / T_val**2

        ts_des   = 0.01
        t_inner  = np.arange(max(ts_des, t_real[0]), t_real[-1], ts_des)
        # Append the exact OCP endpoint so that omega_exit / v_exit boundary
        # constraints are reflected in the output arrays (not dropped by [:-1]).
        t_interp = np.append(t_inner, t_real[-1])
        # Pchip for all kinematic signals: monotonicity-preserving interpolation
        # avoids the overshoot that CubicSpline produces on the non-uniform OCP
        # time grid that results from energy-weighted objectives (w_e > 0).
        v_arr        = _Pchip(t_real, v_ocp)(t_interp)
        omega_arr    = _Pchip(t_real, om_ocp)(t_interp)
        acc_path_arr = _Pchip(t_real, a_ocp)(t_interp)
        alpha_arr    = _Pchip(t_real, alpha_ocp)(t_interp)

        l, r = self._robot['l'], self._robot['r']
        omega_r_arr = (v_arr + l * omega_arr) / r
        omega_l_arr = (v_arr - l * omega_arr) / r

        # Wheel linear jerk [m/s³] directly from OCP 3rd B-spline derivatives.
        # jerk_r = d(acc_path)/dt + l * d(alpha)/dt — this is exactly the
        # quantity bounded by the OCP wheel-jerk constraint (≤ J_LIM in m/s³),
        # so Pchip interpolation of these OCP-node values stays within ±J_LIM.
        body_jerk_ocp = (cos_th * ddds_val[:, 0] + sin_th * ddds_val[:, 1]) / T_val**3
        ang_jerk_ocp  = ddds_val[:, 2] / T_val**3
        jerk_r_ocp    = body_jerk_ocp + l * ang_jerk_ocp   # m/s³, no /r
        jerk_l_ocp    = body_jerk_ocp - l * ang_jerk_ocp   # m/s³, no /r
        jerk_r_arr    = _Pchip(t_real, jerk_r_ocp)(t_interp)
        jerk_l_arr    = _Pchip(t_real, jerk_l_ocp)(t_interp)

        return {
            'states':   s_val,
            'time':     t_real,
            'ctrl_pts': ctrl_pts,
            'v':        v_arr,
            'omega':    omega_arr,
            'acc_path':   acc_path_arr,
            'alpha':      alpha_arr,
            'omega_r':    omega_r_arr,
            'omega_l':  omega_l_arr,
            'jerk_r':   jerk_r_arr,
            'jerk_l':   jerk_l_arr,
            'time_ik':  t_interp,
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
        spline sample. Maps these to individual wheel linear dynamics via the
        differential-drive kinematic model, then evaluates the polynomial power
        model for each wheel. Finally integrates power over time.

        @param s:   (nt, 3) CasADi MX spline values [x, y, theta].
        @param ds:  (nt, 3) first  B-spline derivatives w.r.t. tau.
        @param dds: (nt, 3) second B-spline derivatives w.r.t. tau.
        @param T:   (nt, 1) CasADi MX per-sample time-scaling variables.
        @return Tuple (energy_motor, energy_total, power_column):
            - energy_motor: CasADi scalar, motor-only energy E_motor [J] — used in objective.
            - energy_total: CasADi scalar, total energy E_motor + P_elec·T [J] — for reporting.
            - power_column: (nt, 1) CasADi expression for P_total(t) [W].
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
        v_r, v_l, a_r, a_l = \
            self._calculate_wheel_dynamics(
                v_fwd, omega_robot, a_fwd, alpha_robot, l)

        # --- Motor power per wheel (separate coefficient sets) ---------------
        # cs.fmax floors at zero: the model has no regenerative braking term,
        # so negative power (from large deceleration) must not reduce energy.
        P_r = cs.fmax(self._calculate_power(v_r, a_r,
                                            self._e_coeffs_right), 0.0)
        P_l = cs.fmax(self._calculate_power(v_l, a_l,
                                            self._e_coeffs_left), 0.0)

        # Hotel load is a constant time-proportional overhead independent of
        # trajectory shape.  Including it in the objective shifts the Pareto
        # indifference point outside [0,1]; optimizing motor energy alone
        # keeps w_e* inside the sweep range.
        P_motor = P_r + P_l                                          # (nt, 1)
        P_total = P_motor + self._p_elec                             # for output

        T_mid = (T[:-1] + T[1:]) / 2

        # Motor energy → objective (trapezoidal rule, O(h²))
        P_mid_m = (P_motor[:-1] + P_motor[1:]) / 2
        energy_motor = cs.sum1(P_mid_m * T_mid) / nt                # scalar

        # Total energy → returned in result dict for reporting
        P_mid = (P_total[:-1] + P_total[1:]) / 2
        energy = cs.sum1(P_mid * T_mid) / nt                        # scalar

        return energy_motor, energy, P_total

    def _calculate_wheel_dynamics(self, v_fwd, omega_robot,
                                  a_fwd, alpha_robot, l):
        """! Map robot body-frame kinematics to individual wheel linear dynamics.

        Uses the differential-drive inverse kinematic model in linear units:
            v_r = v + l * Omega      [m/s]
            v_l = v - l * Omega      [m/s]
            a_r = a + l * alpha      [m/s²]
            a_l = a - l * alpha      [m/s²]

        All inputs and outputs are CasADi (nt, 1) column vectors.

        @param v_fwd:        Forward velocity [m/s].
        @param omega_robot:  Robot yaw rate   [rad/s].
        @param a_fwd:        Forward acceleration [m/s²].
        @param alpha_robot:  Robot angular acceleration [rad/s²].
        @param l:            Half-wheelbase [m].
        @return Tuple (v_r, v_l, a_r, a_l), each (nt,1) [m/s or m/s²].
        """
        v_r = v_fwd + l * omega_robot
        v_l = v_fwd - l * omega_robot
        a_r = a_fwd + l * alpha_robot
        a_l = a_fwd - l * alpha_robot
        return v_r, v_l, a_r, a_l

    def _calculate_power(self, v_wheel, a_wheel, coeffs):
        """! Polynomial power model for one wheel motor.

        P(v_w, a_w) =
            p[0] * a_w^2
          + p[1] * v_w^2
          + |p[2] * a_w|
          + |p[3] * v_w|
          + |p[4] * v_w * a_w|
          + p[5]

        Captures viscous losses (p[1], p[3]), inertial work (p[0], p[2]),
        a coupled velocity-acceleration term (p[4]), and static load (p[5]).

        Right and left motors use separate 6-element coefficient lists
        because each physical motor has distinct friction and inertia.

        @param v_wheel:      Wheel linear velocity    [m/s], CasADi MX.
        @param a_wheel:      Wheel linear acceleration [m/s²], CasADi MX.
        @param coeffs<list>: 6-element list [p0..p5] for this motor.
        @return CasADi expression for motor power [W], same shape as inputs.
        """
        return (coeffs[0] * a_wheel**2
                + coeffs[1] * v_wheel**2
                + cs.fabs(coeffs[2] * a_wheel)
                + cs.fabs(coeffs[3] * v_wheel)
                + cs.fabs(coeffs[4] * v_wheel * a_wheel)
                + coeffs[5])
