#!/usr/bin/env python3
##
# @file bspline_energy_tractor_trailer_coverage.py
#
# @brief Energy-aware B-Spline OCP for coverage trajectory generation on a
# tractor-trailer robot.
#
# Extends BSplineTractorTrailerCoverage by replacing the time-minimization
# objective with a weighted sum of mission time and total electrical energy,
# enabling time-energy Pareto trade-offs.
#
# The 4-state trailer B-spline [x, y, θ_trailer, γ] is unchanged from the
# base class. Energy is computed from the TRACTOR's wheel-level dynamics,
# since the tractor motor (not the passive trailer wheels) consumes power.
#
# Tractor kinematics (from trailer B-spline via IK):
#   v_trac = (V·cos γ − lf·θ̇·sin γ)        [m/s]
#   w_trac = −(lf·θ̇·cos γ + V·sin γ) / lb  [rad/s]
#
# Tractor acceleration (τ-derivative under lateral ≈ 0 approximation):
#   a_trac = [F_dot·cos γ − F·sin γ·γ' − lf·(θ''·sin γ + θ'·cos γ·γ')] / T²
#   α_trac = −[lf·(θ''·cos γ − θ'·sin γ·γ') + F_dot·sin γ + F·cos γ·γ'] / (lb·T²)
#
# where F = ds_x·cos(θ) + ds_y·sin(θ)  (= V·T),  F_dot = dds_x·cos(θ) + dds_y·sin(θ),
#           θ' = ds[:,2],  θ'' = dds[:,2],  γ = s[:,3],  γ' = ds[:,3].
#
# Power Model (per wheel motor, same as BSplineEnergyCoverage):
#   P(v_w, a_w) = p[0]*a_w² + p[1]*v_w²
#               + |p[2]*a_w| + |p[3]*v_w| + |p[4]*v_w·a_w| + p[5]
#
# Tractor wheel mapping (half-wheelbase l of the TRACTOR):
#   v_r = v_trac + l·w_trac,   a_r = a_trac + l·α_trac
#   v_l = v_trac − l·w_trac,   a_l = a_trac − l·α_trac
#
# Objective:
#   J = w_time · (sum(T) / nt) + w_energy · E_motor
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/06/22

# Standard library
import numpy as np

# External library
import casadi as cs
from scipy.interpolate import PchipInterpolator as _Pchip
from scipy.interpolate import CubicSpline as _CubicSpline

# Internal library
from .bspline_tractor_trailer_coverage import BSplineTractorTrailerCoverage


class BSplineEnergyTractorTrailerCoverage(BSplineTractorTrailerCoverage):
    """! Energy-aware B-Spline OCP for tractor-trailer coverage.

    Extends BSplineTractorTrailerCoverage with a physics-based wheel-level
    power model applied to the TRACTOR drivetrain. The tractor kinematics are
    derived from the trailer B-spline through the hitch angle inverse kinematics
    at every sample point, allowing the energy objective to penalize high-speed
    or sharp-turn tractor manoeuvres even though the optimized state trajectory
    describes the trailer.
    """

    # Polynomial power coefficients (per motor) — inherited from TJ108 bench tests.
    # P(v_w, a_w) = p[0]*a² + p[1]*v² + |p[2]*a| + |p[3]*v| + |p[4]*v*a| + p[5]
    _DEFAULT_ENERGY_COEFFS_RIGHT = [
        0.302433145557389,
        31.887262598534413,
        2.4140287888312457,
        0.9658866923308425,
        0.8260871406535432,
        2.37456174658809e-08,
    ]
    _DEFAULT_ENERGY_COEFFS_LEFT = [
        0.33789198669595977,
        28.204019732889346,
        2.5903002025839688,
        0.00847962183165042,
        6.412423386896174e-09,
        0.3614761744737831,
    ]

    # Tractor physical parameters (differential drive drivetrain).
    _DEFAULT_ROBOT_PARAMS = {
        'l': 0.175,     # half-wheelbase of tractor [m]
        'r': 0.050,     # wheel radius of tractor [m]
    }

    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    def __init__(self, waypoints, bound=0.1, n_ctrl_pts=6, spline_order=3,
                 n_sampling=50, vel_max=None, vel_min_lin=0.01,
                 eps_nonh=0.001, v_entry=None, v_exit=None,
                 a_entry=None, a_exit=None,
                 omega_entry=None, omega_exit=None,
                 alpha_entry=None, alpha_exit=None,
                 acc_max=None, jerk_max=None,
                 length_back=0.2, length_front=0.8,
                 gamma_max=0.785, gamma_entry=0.0, gamma_exit=0.0,
                 vel_tractor_max=None,
                 eps_hitch=0.05,
                 robot_params=None,
                 energy_coeffs_right=None, energy_coeffs_left=None,
                 w_time=1.0, w_energy=1.0,
                 e_max=None, p_electronics=2.0):
        """! Constructor.

        All parameters of BSplineTractorTrailerCoverage are accepted unchanged.

        @param eps_hitch<float>: Hitch coupling tolerance (see base class).
        @param robot_params<dict|None>: Tractor physical parameters.
            Keys: 'l' (half-wheelbase [m]), 'r' (wheel radius [m]).
            Missing keys fall back to _DEFAULT_ROBOT_PARAMS.
        @param energy_coeffs_right<list|None>: 6-element list [p0..p5] for the
            RIGHT tractor motor. None uses _DEFAULT_ENERGY_COEFFS_RIGHT.
        @param energy_coeffs_left<list|None>: 6-element list [p0..p5] for the
            LEFT tractor motor. None uses _DEFAULT_ENERGY_COEFFS_LEFT.
        @param w_time<float>: Weight on total traversal time.
        @param w_energy<float>: Weight on total energy consumption.
        @param e_max<float|None>: Hard upper bound on energy [J]. None = none.
        @param p_electronics<float>: Constant hotel load (sensors, computer) [W].
        """
        super().__init__(
            waypoints=waypoints, bound=bound, n_ctrl_pts=n_ctrl_pts,
            spline_order=spline_order, n_sampling=n_sampling,
            vel_max=vel_max, vel_min_lin=vel_min_lin, eps_nonh=eps_nonh,
            v_entry=v_entry, v_exit=v_exit,
            a_entry=a_entry, a_exit=a_exit,
            omega_entry=omega_entry, omega_exit=omega_exit,
            alpha_entry=alpha_entry, alpha_exit=alpha_exit,
            acc_max=acc_max, jerk_max=jerk_max,
            length_back=length_back, length_front=length_front,
            gamma_max=gamma_max, gamma_entry=gamma_entry,
            gamma_exit=gamma_exit, vel_tractor_max=vel_tractor_max,
            eps_hitch=eps_hitch,
        )
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
            raise ValueError("energy_coeffs must each have exactly 6 elements.")
        self._w_time = float(w_time)
        self._w_energy = float(w_energy)
        self._e_max = e_max
        self._p_elec = float(p_electronics)

    def generate_trajectory(self, warm_start=None):
        """! Build and solve the energy-aware OCP for the tractor-trailer system.

        @param warm_start<dict|None>: Result dict from a previous call.
            ctrl_pts must be (n_Q, 4): columns [x, y, theta, gamma].
        @return dict with keys:
            - 'states'    : (nt, 4)  [x_t, y_t, θ_t, γ]
            - 'time'      : (nt,)    real time at each sample [s]
            - 'ctrl_pts'  : (n_Q, 4) optimized control points
            - 'v'         : (M,)    tractor linear velocity [m/s]
            - 'omega'     : (M,)    tractor angular velocity [rad/s]
            - 'v_trailer' : (M,)    trailer forward speed [m/s]
            - 'acc_path'  : (M,)    tractor forward acceleration [m/s²]
            - 'alpha'     : (M,)    tractor angular acceleration [rad/s²]
            - 'omega_r'   : (M,)    right tractor wheel angular velocity [rad/s]
            - 'omega_l'   : (M,)    left tractor wheel angular velocity [rad/s]
            - 'jerk_r'    : (M,)    right wheel jerk [m/s³]
            - 'jerk_l'    : (M,)    left wheel jerk [m/s³]
            - 'time_ik'   : (M,)    time vector for kinematic signals
            - 'power'     : (nt,)   total power P(t) over spline nodes [W]
            - 'energy'    : float   total mission energy [J]
        """
        knot = self._build_knot_vector()
        tau  = np.linspace(0.0, 1.0, self._nt)

        basis, dot_basis, ddot_basis, dddot_basis = \
            self._build_basis_matrices(tau, knot)

        opti = self._optimizer

        # Decision variables (4-column: x, y, theta, gamma)
        px = opti.variable(self._n_Q)
        py = opti.variable(self._n_Q)
        pt = opti.variable(self._n_Q)
        pg = opti.variable(self._n_Q)
        T  = opti.variable(self._nt)

        P_ctrl = cs.horzcat(px, py, pt, pg)    # (n_Q, 4)

        s    = basis        @ P_ctrl            # (nt, 4)
        ds   = dot_basis    @ P_ctrl            # (nt, 4)
        dds  = ddot_basis   @ P_ctrl            # (nt, 4)
        ddds = dddot_basis  @ P_ctrl            # (nt, 4)

        if warm_start is not None:
            cp = warm_start['ctrl_pts']
            opti.set_initial(px, cp[:, 0])
            opti.set_initial(py, cp[:, 1])
            opti.set_initial(pt, cp[:, 2])
            opti.set_initial(pg, cp[:, 3])
            t_prev = warm_start['time']
            T_prev = np.diff(t_prev) * len(t_prev)
            T_prev = np.append(T_prev, T_prev[-1])
            opti.set_initial(T, T_prev[:self._nt])
        else:
            self._set_initial_conditions(opti, px, py, pt, pg, T,
                                         basis, dot_basis)

        # Energy-aware objective
        energy_motor_sym, energy_sym, power_sym = \
            self._build_energy_terms(s, ds, dds, T)
        T_mission = cs.sum1(T) / self._nt
        opti.minimize(self._w_time * T_mission
                      + self._w_energy * energy_motor_sym)

        # Kinodynamic + path constraints (tractor-trailer, inherited)
        self._add_dynamic_constraints(opti, s, ds, dds, ddds, T)
        self._add_jackknife_constraints(opti, pg)
        self._add_boundary_constraints(opti, px, py, pt, pg)
        self._add_corridor_constraints(opti, px, py)

        # Tractor wheel angular jerk constraint
        # jerk_{r,l} ≤ J_LIM ensures actuator-side smoothness beyond what
        # the body-frame Cartesian jerk bounds in _add_dynamic_constraints cover.
        jerk_whl = float(self._jerk_max[0])
        lb  = self._lb
        lf  = self._lf
        l_w = self._robot['l']
        for i in range(self._nt):
            Ti    = T[i]
            cos_i = cs.cos(s[i, 2])
            sin_i = cs.sin(s[i, 2])
            cos_g = cs.cos(s[i, 3])
            sin_g = cs.sin(s[i, 3])
            gp    = ds[i, 3]          # γ' = dγ/dτ

            F_i   = ds[i, 0] * cos_i + ds[i, 1] * sin_i
            Fd_i  = dds[i, 0] * cos_i + dds[i, 1] * sin_i
            Fdd_i = ddds[i, 0] * cos_i + ddds[i, 1] * sin_i
            tp    = ds[i, 2]          # θ'
            tpp   = dds[i, 2]         # θ''
            tppp  = ddds[i, 2]        # θ'''
            gpp   = dds[i, 3]         # γ''

            # τ-derivative of U (= v_trac · T)
            dU = Fd_i*cos_g - F_i*sin_g*gp - lf*(tpp*sin_g + tp*cos_g*gp)
            # τ-derivative of W (= w_trac · T · lb)
            dW = -(lf*(tpp*cos_g - tp*sin_g*gp) + Fd_i*sin_g + F_i*cos_g*gp)

            # Second τ-derivative of U (for jerk)
            # d²U/dτ² (body jerk * T³)
            d2U = (Fdd_i*cos_g
                   - 2*Fd_i*sin_g*gp
                   - F_i*cos_g*gp**2
                   - F_i*sin_g*gpp
                   - lf*(tppp*sin_g + 2*tpp*cos_g*gp - tp*sin_g*gp**2 + tp*cos_g*gpp))
            # d²W/dτ²
            d2W = -(lf*(tppp*cos_g - 2*tpp*sin_g*gp - tp*cos_g*gp**2 - tp*sin_g*gpp)
                    + Fdd_i*sin_g + 2*Fd_i*cos_g*gp - F_i*sin_g*gp**2 + F_i*cos_g*gpp)

            # Tractor wheel jerk (body_j = d²U/dτ² / T³, ang_j = d²W/(lb*T³)*l_w)
            body_j = d2U
            ang_j  = d2W / lb * l_w
            lim    = jerk_whl * Ti**3
            opti.subject_to(body_j + ang_j <= lim)
            opti.subject_to(-lim <= body_j + ang_j)
            opti.subject_to(body_j - ang_j <= lim)
            opti.subject_to(-lim <= body_j - ang_j)

        # Optional hard energy budget
        if self._e_max is not None:
            opti.subject_to(energy_sym <= self._e_max)

        # Solve
        try:
            sol = opti.solve()
            dbg = sol
        except Exception:
            dbg = opti.debug

        T_val    = dbg.value(T)
        s_val    = dbg.value(s)
        ds_val   = np.array(dbg.value(ds))
        dds_val  = np.array(dbg.value(dds))
        ddds_val = np.array(dbg.value(ddds))
        ctrl_pts = dbg.value(P_ctrl)
        power_val  = np.array(dbg.value(power_sym)).flatten()
        energy_val = float(dbg.value(energy_sym))

        t_real = np.cumsum(T_val) / self._nt

        # Tractor kinematics at OCP nodes
        cos_th  = np.cos(s_val[:, 2])
        sin_th  = np.sin(s_val[:, 2])
        cos_gam = np.cos(s_val[:, 3])
        sin_gam = np.sin(s_val[:, 3])

        F_val    = cos_th * ds_val[:, 0] + sin_th * ds_val[:, 1]  # V·T
        Fd_val   = cos_th * dds_val[:, 0] + sin_th * dds_val[:, 1]
        tp_val   = ds_val[:, 2]
        tpp_val  = dds_val[:, 2]
        gp_val   = ds_val[:, 3]

        v_trac_ocp = (F_val * cos_gam - lf * tp_val * sin_gam) / T_val
        w_trac_ocp = -(lf * tp_val * cos_gam + F_val * sin_gam) / (lb * T_val)
        v_trail_ocp = F_val / T_val

        dU_val = (Fd_val * cos_gam - F_val * sin_gam * gp_val
                  - lf * (tpp_val * sin_gam + tp_val * cos_gam * gp_val))
        dW_val = -(lf * (tpp_val * cos_gam - tp_val * sin_gam * gp_val)
                   + Fd_val * sin_gam + F_val * cos_gam * gp_val)

        a_trac_ocp = dU_val / T_val**2
        alpha_trac_ocp = dW_val / (lb * T_val**2)

        ts_des   = 0.01
        t_inner  = np.arange(max(ts_des, t_real[0]), t_real[-1], ts_des)
        t_interp = np.append(t_inner, t_real[-1])

        v_arr        = _Pchip(t_real, v_trac_ocp)(t_interp)
        omega_arr    = _Pchip(t_real, w_trac_ocp)(t_interp)
        v_trailer_arr = _CubicSpline(t_real, v_trail_ocp)(t_interp)
        acc_arr      = _Pchip(t_real, a_trac_ocp)(t_interp)
        alpha_arr    = _Pchip(t_real, alpha_trac_ocp)(t_interp)

        l_w, r_w = self._robot['l'], self._robot['r']
        omega_r_arr = (v_arr + l_w * omega_arr) / r_w
        omega_l_arr = (v_arr - l_w * omega_arr) / r_w

        # Tractor wheel jerk at OCP nodes
        gpp_val  = dds_val[:, 3]
        tppp_val = ddds_val[:, 2]
        Fdd_val  = cos_th * ddds_val[:, 0] + sin_th * ddds_val[:, 1]

        d2U_val = (Fdd_val * cos_gam
                   - 2 * Fd_val * sin_gam * gp_val
                   - F_val * cos_gam * gp_val**2
                   - F_val * sin_gam * gpp_val
                   - lf * (tppp_val * sin_gam + 2 * tpp_val * cos_gam * gp_val
                            - tp_val * sin_gam * gp_val**2 + tp_val * cos_gam * gpp_val))
        d2W_val = -(lf * (tppp_val * cos_gam - 2 * tpp_val * sin_gam * gp_val
                           - tp_val * cos_gam * gp_val**2 - tp_val * sin_gam * gpp_val)
                    + Fdd_val * sin_gam + 2 * Fd_val * cos_gam * gp_val
                    - F_val * sin_gam * gp_val**2 + F_val * cos_gam * gpp_val)

        jerk_r_ocp = d2U_val / T_val**3 + (d2W_val / lb) * l_w / T_val**3
        jerk_l_ocp = d2U_val / T_val**3 - (d2W_val / lb) * l_w / T_val**3
        jerk_r_arr = _Pchip(t_real, jerk_r_ocp)(t_interp)
        jerk_l_arr = _Pchip(t_real, jerk_l_ocp)(t_interp)

        return {
            'states':    s_val,
            'time':      t_real,
            'ctrl_pts':  ctrl_pts,
            'v':         v_arr,
            'omega':     omega_arr,
            'v_trailer': v_trailer_arr,
            'acc_path':  acc_arr,
            'alpha':     alpha_arr,
            'omega_r':   omega_r_arr,
            'omega_l':   omega_l_arr,
            'jerk_r':    jerk_r_arr,
            'jerk_l':    jerk_l_arr,
            'time_ik':   t_interp,
            'power':     power_val,
            'energy':    energy_val,
        }

    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    def _build_energy_terms(self, s, ds, dds, T):
        """! Construct symbolic energy and power expressions for TRACTOR wheels.

        Projects the 4D trailer B-spline [x, y, θ_trailer, γ] through the
        tractor inverse kinematics to obtain tractor body-frame velocity and
        acceleration, then maps to individual wheel dynamics via the
        differential-drive wheel model.

        @param s:   (nt, 4) CasADi MX spline values [x, y, theta, gamma].
        @param ds:  (nt, 4) first B-spline derivatives w.r.t. tau.
        @param dds: (nt, 4) second B-spline derivatives w.r.t. tau.
        @param T:   (nt, 1) per-sample time-scaling variables.
        @return Tuple (energy_motor, energy_total, power_column).
        """
        lb = self._lb
        lf = self._lf
        l  = self._robot['l']
        nt = self._nt

        cos_th = cs.cos(s[:, 2])
        sin_th = cs.sin(s[:, 2])
        cos_g  = cs.cos(s[:, 3])
        sin_g  = cs.sin(s[:, 3])

        # Forward feedrate of trailer (= V · T)
        F    = cos_th * ds[:, 0] + sin_th * ds[:, 1]     # (nt, 1)

        # τ-derivative of F (≈ dV/dτ · T, lateral terms ≈ 0)
        F_dot = cos_th * dds[:, 0] + sin_th * dds[:, 1]  # (nt, 1)

        tp   = ds[:, 2]     # θ' = dθ/dτ
        tpp  = dds[:, 2]    # θ'' = d²θ/dτ²
        gp   = ds[:, 3]     # γ' = dγ/dτ

        # Tractor velocity (× T)
        U = F * cos_g - lf * tp * sin_g               # v_trac · T
        W = -(lf * tp * cos_g + F * sin_g)            # w_trac · T · lb

        # Tractor body-frame velocity
        v_trac = U / T                                 # [m/s]
        w_trac = W / (lb * T)                          # [rad/s]

        # τ-derivative of U  (under lateral ≈ 0)
        dU = (F_dot * cos_g
              - F * sin_g * gp
              - lf * (tpp * sin_g + tp * cos_g * gp))  # dU/dτ

        # τ-derivative of W
        dW = -(lf * (tpp * cos_g - tp * sin_g * gp)
               + F_dot * sin_g
               + F * cos_g * gp)                       # dW/dτ

        # Tractor forward and angular acceleration
        a_trac = dU / T**2                             # [m/s²]
        alpha_trac = dW / (lb * T**2)                  # [rad/s²]

        # Tractor wheel velocities and accelerations
        v_r, v_l, a_r, a_l = self._calculate_wheel_dynamics(
            v_trac, w_trac, a_trac, alpha_trac, l)

        # Motor power per wheel (floor at 0 — no regenerative braking term)
        P_r = cs.fmax(self._calculate_power(v_r, a_r, self._e_coeffs_right), 0.0)
        P_l = cs.fmax(self._calculate_power(v_l, a_l, self._e_coeffs_left),  0.0)

        P_motor = P_r + P_l
        P_total = P_motor + self._p_elec

        T_mid = (T[:-1] + T[1:]) / 2

        P_mid_m = (P_motor[:-1] + P_motor[1:]) / 2
        energy_motor = cs.sum1(P_mid_m * T_mid) / nt

        P_mid = (P_total[:-1] + P_total[1:]) / 2
        energy = cs.sum1(P_mid * T_mid) / nt

        return energy_motor, energy, P_total

    def _calculate_wheel_dynamics(self, v_body, omega_body, a_body, alpha_body, l):
        """! Map body-frame kinematics to differential-drive wheel dynamics.
        @param v_body:     Body forward velocity [m/s].
        @param omega_body: Body yaw rate [rad/s].
        @param a_body:     Body forward acceleration [m/s²].
        @param alpha_body: Body angular acceleration [rad/s²].
        @param l:          Half-wheelbase [m].
        @return Tuple (v_r, v_l, a_r, a_l).
        """
        v_r = v_body + l * omega_body
        v_l = v_body - l * omega_body
        a_r = a_body + l * alpha_body
        a_l = a_body - l * alpha_body
        return v_r, v_l, a_r, a_l

    def _calculate_power(self, v_wheel, a_wheel, coeffs):
        """! Polynomial power model P(v_w, a_w) for one wheel motor.
        @param v_wheel: Wheel linear velocity [m/s].
        @param a_wheel: Wheel linear acceleration [m/s²].
        @param coeffs:  6-element list [p0..p5].
        @return CasADi expression for motor power [W].
        """
        return (coeffs[0] * a_wheel**2
                + coeffs[1] * v_wheel**2
                + cs.fabs(coeffs[2] * a_wheel)
                + cs.fabs(coeffs[3] * v_wheel)
                + cs.fabs(coeffs[4] * v_wheel * a_wheel)
                + coeffs[5])
