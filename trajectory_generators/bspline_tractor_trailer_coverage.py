#!/usr/bin/env python3
##
# @file bspline_tractor_trailer_coverage.py
#
# @brief B-Spline parameterized optimal control for coverage trajectory
# generation on a tractor-trailer robot.
#
# Extends BSplineCoverage to handle a 4-state tractor-trailer system where the
# TRAILER pose [x, y, θ] is what sweeps the coverage area. The tractor provides
# the actuated inputs [v, w] that indirectly drive the trailer through a rigid
# hitch.
#
# Kinematic model (trailer-centric, from nonlinear-mpc-tractor-trailer):
#   state = [x_trailer, y_trailer, θ_trailer, γ]
#   input = [v_tractor, w_tractor]
#
#   ẋ = cos(θ) · [v·cos(γ) − w·lb·sin(γ)]
#   ẏ = sin(θ) · [v·cos(γ) − w·lb·sin(γ)]
#   θ̇ = −v·sin(γ)/lf − w·lb·cos(γ)/lf
#   γ̇ = θ̇ − w
#
# where lb = length_back (tractor rear axle → hitch) and
#       lf = length_front (hitch → trailer rear axle).
#
# The OCP parameterizes [x, y, θ, γ] as a single 4-column B-spline and
# enforces:
#   1. Trailer lateral velocity = 0 (nonholonomic, same as diff-drive)
#   2. Hitch angle coupling ODE:  lb·dγ/dτ = dθ/dτ·(lb + lf·cos(γ)) + V·T·sin(γ)
#   3. γ ∈ [−γ_max, γ_max]
#   4. Standard feedrate, acceleration, jerk, and time-positivity bounds
#   5. Optional tractor velocity bounds
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/06/22

# Standard library
import numpy as np

# External library
import casadi as cs
from scipy.interpolate import CubicSpline as _CubicSpline

# Internal library
from .bspline_coverage import BSplineCoverage


class BSplineTractorTrailerCoverage(BSplineCoverage):
    """! B-Spline parameterized OCP for tractor-trailer coverage.

    Extends BSplineCoverage to handle the 4-state tractor-trailer system.
    The full robot state [x_trailer, y_trailer, θ_trailer, γ] is represented
    as a B-spline. The OCP minimizes total traversal time of the trailer while
    enforcing tractor-trailer kinematics, corridor constraints, and physical
    limits.

    Inherited unchanged: knot vector, basis matrices, corridor geometry.
    Overridden: state dimension (3→4), dynamic constraints, boundary conditions,
    inverse kinematics output.
    """

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
                 gamma_max=0.785,
                 gamma_entry=0.0, gamma_exit=0.0,
                 vel_tractor_max=None,
                 eps_hitch=0.05):
        """! Constructor.
        @param waypoints<list>: Via-points [[x, y, theta], ...] for the
            trailer rear axle. At least 2.
        @param bound<float>: Half-width of the trailer corridor [m].
        @param n_ctrl_pts<int>: B-spline control points per segment.
        @param spline_order<int>: B-spline degree (3 = cubic).
        @param n_sampling<int>: Discrete samples per segment.
        @param vel_max<list|None>: [vx_max, vy_max, omega_max] for the
            trailer. Defaults to [0.2, 0.2, 0.196] m/s and rad/s.
        @param vel_min_lin<float>: Minimum trailer feedrate [m/s].
        @param eps_nonh<float>: Nonholonomic constraint tolerance.
        @param v_entry<float|None>: Trailer speed at start [m/s].
        @param v_exit<float|None>: Trailer speed at end [m/s].
        @param a_entry<float|None>: Trailer forward acceleration at start.
        @param a_exit<float|None>: Trailer forward acceleration at end.
        @param omega_entry<float|None>: Trailer angular rate at start [rad/s].
        @param omega_exit<float|None>: Trailer angular rate at end [rad/s].
        @param alpha_entry<float|None>: Trailer angular accel at start.
        @param alpha_exit<float|None>: Trailer angular accel at end.
        @param acc_max<list|None>: [ax, ay, alpha] physical acc limits.
        @param jerk_max<list|None>: [jx, jy, jalpha] physical jerk limits.
        @param length_back<float>: lb — tractor rear axle to hitch [m].
        @param length_front<float>: lf — hitch to trailer rear axle [m].
        @param gamma_max<float>: Maximum hitch angle magnitude [rad].
        @param gamma_entry<float>: Hitch angle at trajectory start [rad].
        @param gamma_exit<float>: Hitch angle at trajectory end [rad].
        @param vel_tractor_max<list|None>: [v_max, w_max] tractor velocity
            limits [m/s, rad/s]. None skips tractor velocity constraints.
        @param eps_hitch<float>: Tolerance for the hitch angle coupling
            constraint [m/s in physical units]. The ODE lb·γ̇ = f(θ,γ,V)
            is enforced as an inequality band of width ±eps_hitch·T rather
            than a strict equality, matching the nonholonomic constraint style.
            Default 0.05 (50× looser than eps_nonh to allow the γ B-spline
            to approximate the coupling ODE).
        """
        super().__init__(
            waypoints=waypoints,
            bound=bound,
            n_ctrl_pts=n_ctrl_pts,
            spline_order=spline_order,
            n_sampling=n_sampling,
            vel_max=vel_max,
            vel_min_lin=vel_min_lin,
            eps_nonh=eps_nonh,
            v_entry=v_entry,
            v_exit=v_exit,
            a_entry=a_entry,
            a_exit=a_exit,
            omega_entry=omega_entry,
            omega_exit=omega_exit,
            alpha_entry=alpha_entry,
            alpha_exit=alpha_exit,
            acc_max=acc_max,
            jerk_max=jerk_max,
        )
        self._lb = float(length_back)
        self._lf = float(length_front)
        self._gamma_max = float(gamma_max)
        self._gamma_entry = float(gamma_entry)
        self._gamma_exit = float(gamma_exit)
        self._vel_tractor_max = (
            [float(x) for x in vel_tractor_max]
            if vel_tractor_max is not None else None
        )
        self._eps_hitch = float(eps_hitch)

    def generate_trajectory(self):
        """! Build and solve the OCP for the tractor-trailer system.
        @return dict with keys:
            - 'states'    : (nt, 4) array [x_t, y_t, theta_t, gamma]
            - 'time'      : (nt,)   real time at each sample [s]
            - 'ctrl_pts'  : (n_Q, 4) optimized control points
            - 'v'         : (M,)    tractor linear velocity [m/s]
            - 'omega'     : (M,)    tractor angular velocity [rad/s]
            - 'time_ik'   : (M,)    time vector for v/omega
            - 'v_trailer' : (M,)    trailer forward speed [m/s]
        """
        knot = self._build_knot_vector()
        tau = np.linspace(0.0, 1.0, self._nt)

        basis, dot_basis, ddot_basis, dddot_basis = \
            self._build_basis_matrices(tau, knot)

        opti = self._optimizer

        # Decision variables: 4-column control points + per-sample time
        px = opti.variable(self._n_Q)
        py = opti.variable(self._n_Q)
        pt = opti.variable(self._n_Q)     # theta_trailer control points
        pg = opti.variable(self._n_Q)     # gamma control points
        T  = opti.variable(self._nt)

        # 4-column control-point matrix
        P = cs.horzcat(px, py, pt, pg)    # (n_Q, 4)

        s    = basis      @ P             # (nt, 4): [x, y, theta, gamma]
        ds   = dot_basis  @ P             # (nt, 4): first derivatives
        dds  = ddot_basis @ P             # (nt, 4): second derivatives
        ddds = dddot_basis @ P            # (nt, 4): third derivatives

        self._set_initial_conditions(opti, px, py, pt, pg, T,
                                     basis, dot_basis)

        opti.minimize(cs.sum1(T))

        self._add_dynamic_constraints(opti, s, ds, dds, ddds, T)
        self._add_jackknife_constraints(opti, pg)
        self._add_boundary_constraints(opti, px, py, pt, pg)
        self._add_corridor_constraints(opti, px, py)

        try:
            sol = opti.solve()
            dbg = sol
        except Exception:
            dbg = opti.debug

        T_val    = dbg.value(T)
        s_val    = dbg.value(s)
        ds_val   = dbg.value(ds)
        ctrl_pts = dbg.value(P)

        t_real = np.cumsum(T_val) / self._nt

        # Tractor inverse kinematics at OCP nodes
        cos_th  = np.cos(s_val[:, 2])
        sin_th  = np.sin(s_val[:, 2])
        cos_gam = np.cos(s_val[:, 3])
        sin_gam = np.sin(s_val[:, 3])

        # V·T = forward feedrate of trailer (scaled by T)
        V_scaled = cos_th * ds_val[:, 0] + sin_th * ds_val[:, 1]

        v_trailer_ocp = V_scaled / T_val  # trailer forward speed

        # Tractor inputs from inverse kinematics
        v_trac_ocp = (V_scaled * cos_gam
                      - self._lf * ds_val[:, 2] * sin_gam) / T_val
        w_trac_ocp = -(self._lf * ds_val[:, 2] * cos_gam
                       + V_scaled * sin_gam) / (self._lb * T_val)

        ts_des   = 0.01
        t_interp = np.arange(ts_des, t_real[-1], ts_des)
        v_arr        = _CubicSpline(t_real, v_trac_ocp)(t_interp[:-1])
        omega_arr    = _CubicSpline(t_real, w_trac_ocp)(t_interp[:-1])
        v_trailer_arr = _CubicSpline(t_real, v_trailer_ocp)(t_interp[:-1])

        return {
            'states':    s_val,
            'time':      t_real,
            'ctrl_pts':  ctrl_pts,
            'v':         v_arr,
            'omega':     omega_arr,
            'time_ik':   t_interp[:-1],
            'v_trailer': v_trailer_arr,
        }

    # ==========================================================================
    # PRIVATE METHODS  (overrides)
    # ==========================================================================
    def _set_initial_conditions(self, opti, px, py, pt, pg, T,
                                basis, dot_basis=None):
        """! Warm-start: lay control points along piecewise-linear path,
        initialize T from feedrate, and initialize γ by forward-integrating
        the hitch coupling ODE along the initial (x, y, θ) trajectory.
        """
        wps    = self._waypoints
        n_ctrl = self._n_ctrl_pts
        n_pcs  = self._n_pieces

        cp0 = np.zeros((self._n_Q, 3))
        idx = 0
        for seg in range(n_pcs):
            p0 = wps[seg]
            p1 = wps[seg + 1]
            for j in range(n_ctrl):
                alpha = j / (n_ctrl - 1)
                cp0[idx] = (1 - alpha) * p0 + alpha * p1
                opti.set_initial(px[idx], cp0[idx, 0])
                opti.set_initial(py[idx], cp0[idx, 1])
                opti.set_initial(pt[idx], cp0[idx, 2])
                idx += 1

        if dot_basis is not None:
            ds0 = dot_basis @ cp0          # (nt, 3)
            ds0_xy = np.linalg.norm(ds0[:, :2], axis=1)
        else:
            ds0 = np.zeros((self._nt, 3))
            ds0_xy = np.linalg.norm((basis @ cp0)[:, :2], axis=1)

        v_max  = self._vel_max[0]
        T_init = np.where(ds0_xy > 1e-8, ds0_xy / v_max, 1.0 / v_max)

        if self._v_entry is not None and self._v_entry > 1e-6:
            v0 = min(float(self._v_entry), v_max)
            T_init[0] = ds0_xy[0] / v0 if ds0_xy[0] > 1e-8 else 1.0 / v0
        if self._v_exit is not None and self._v_exit > 1e-6:
            vn = min(float(self._v_exit), v_max)
            T_init[-1] = ds0_xy[-1] / vn if ds0_xy[-1] > 1e-8 else 1.0 / vn

        opti.set_initial(T, T_init)

        # γ warm-start: forward-integrate the coupling ODE along the initial
        # (x, y, θ) trajectory so the γ B-spline is physically consistent.
        #   dγ/dτ = [dθ/dτ·(lb + lf·cos γ) + fwd·sin γ] / lb
        # where fwd = V·T = ds_x·cos(θ) + ds_y·sin(θ) (τ-derivative form).
        s0_th = (basis @ cp0)[:, 2]         # θ at each sample
        cos0  = np.cos(s0_th)
        sin0  = np.sin(s0_th)
        fwd0  = cos0 * ds0[:, 0] + sin0 * ds0[:, 1]   # V·T (initial)
        tp0   = ds0[:, 2]                              # dθ/dτ (initial)
        lb, lf = self._lb, self._lf

        gamma_init = np.zeros(self._nt)
        gamma_init[0] = self._gamma_entry
        for i in range(self._nt - 1):
            dg = (tp0[i] * (lb + lf * np.cos(gamma_init[i]))
                  + fwd0[i] * np.sin(gamma_init[i])) / lb
            gamma_init[i + 1] = gamma_init[i] + dg / self._nt
            gamma_init[i + 1] = np.clip(gamma_init[i + 1],
                                        -self._gamma_max, self._gamma_max)

        # Fit the integrated γ profile to the γ B-spline via least squares.
        pg_init, _, _, _ = np.linalg.lstsq(basis, gamma_init, rcond=None)

        # Pin first/last control points to the boundary values (overrides fit).
        pg_init[0]  = self._gamma_entry
        pg_init[-1] = self._gamma_exit

        for j in range(self._n_Q):
            opti.set_initial(pg[j], float(pg_init[j]))

    def _add_dynamic_constraints(self, opti, s, ds, dds, ddds, T):
        """! Tractor-trailer kinodynamic constraints.

        Replaces the differential-drive nonholonomic constraint with:
          1. Trailer lateral velocity ≈ 0  (same algebraic form as diff-drive)
          2. Hitch angle coupling equality  lb·dγ = dθ·(lb + lf·cos(γ)) + V·T·sin(γ)
          3. Trailer feedrate and forward-only bounds
          4. Optional tractor velocity bounds
          5. Acceleration / jerk bounds on dims 0,1,2 (trailer x,y,θ)
          6. T > 0

        The hitch angle (jackknife) bound |γ| ≤ γ_max is NOT enforced here;
        see _add_jackknife_constraints, which bounds the γ control points
        directly so the bound is guaranteed over the entire spline arc via
        the B-spline convex-hull property, not just at these sample nodes.
        """
        lb       = self._lb
        lf       = self._lf
        gamma_max = self._gamma_max
        vmax     = self._vel_max
        v_fwd    = vmax[0]
        vlin_min = self._vel_min_lin
        amax     = self._acc_max
        amin     = self._acc_min
        jmax     = self._jerk_max
        jmin     = self._jerk_min
        eps      = self._eps_nonh

        for i in range(self._nt):
            Ti  = T[i]
            fwd = (ds[i, 0] * cs.cos(s[i, 2])
                   + ds[i, 1] * cs.sin(s[i, 2]))    # V · T
            lat = (-ds[i, 0] * cs.sin(s[i, 2])
                   + ds[i, 1] * cs.cos(s[i, 2]))    # lateral · T

            # 1. Trailer nonholonomic: lateral velocity ≈ 0
            opti.subject_to(lat <= eps * Ti)
            opti.subject_to(-eps * Ti <= lat)

            # 2. Hitch coupling ODE: lb·dγ/dτ = dθ/dτ·(lb + lf·cos γ) + V·T·sin γ
            # Enforced as an inequality band (±eps_hitch·T) instead of a strict
            # equality so that the γ B-spline can approximate the ODE solution
            # without over-constraining the NLP (analogous to eps_nonh).
            coupling = (lb * ds[i, 3]
                        - ds[i, 2] * (lb + lf * cs.cos(s[i, 3]))
                        - fwd * cs.sin(s[i, 3]))
            opti.subject_to(coupling <= self._eps_hitch * Ti)
            opti.subject_to(-self._eps_hitch * Ti <= coupling)

            # 3. Trailer feedrate bounds
            opti.subject_to(fwd <= v_fwd * Ti)
            opti.subject_to(vlin_min * Ti <= fwd)

            # 4. Trailer angular velocity bounds (ω_trailer)
            opti.subject_to(self._vel_min[2] * Ti <= ds[i, 2])
            opti.subject_to(ds[i, 2] <= self._vel_max[2] * Ti)

            # 5. Acceleration bounds (trailer x, y, θ only — dims 0,1,2)
            for dim in range(3):
                opti.subject_to(amin[dim] * Ti**2 <= dds[i, dim])
                opti.subject_to(dds[i, dim] <= amax[dim] * Ti**2)

            # 6. Jerk bounds (trailer x, y, θ only — dims 0,1,2)
            for dim in range(3):
                opti.subject_to(jmin[dim] * Ti**3 <= ddds[i, dim])
                opti.subject_to(ddds[i, dim] <= jmax[dim] * Ti**3)

            # 7. Optional tractor velocity bounds
            if self._vel_tractor_max is not None:
                v_tmax = self._vel_tractor_max[0]
                w_tmax = self._vel_tractor_max[1]
                v_sc = (fwd * cs.cos(s[i, 3])
                        - lf * ds[i, 2] * cs.sin(s[i, 3]))   # v_tractor · T
                # w_tractor · T · lb  (avoids a division by lb inside CasADi)
                w_sc = -(lf * ds[i, 2] * cs.cos(s[i, 3])
                         + fwd * cs.sin(s[i, 3]))             # w · lb · T
                opti.subject_to(v_sc <= v_tmax * Ti)
                opti.subject_to(-v_tmax * Ti <= v_sc)
                opti.subject_to(w_sc <= w_tmax * lb * Ti)
                opti.subject_to(-w_tmax * lb * Ti <= w_sc)

            # 8. T must be positive
            opti.subject_to(Ti >= 1e-4)

        # Forward-only at entry/exit using known waypoint headings
        theta_0 = float(self._waypoints[0, 2])
        c0, s0_ = float(np.cos(theta_0)), float(np.sin(theta_0))
        fwd_0 = ds[0, 0] * c0 + ds[0, 1] * s0_
        opti.subject_to(fwd_0 >= self._vel_min_lin * T[0])

        theta_n = float(self._waypoints[-1, 2])
        cn, sn = float(np.cos(theta_n)), float(np.sin(theta_n))
        fwd_n = ds[-1, 0] * cn + ds[-1, 1] * sn
        opti.subject_to(fwd_n >= self._vel_min_lin * T[-1])

        # Optional entry/exit speed pins (same ±1 % band as parent)
        if self._v_entry is not None:
            opti.subject_to(fwd_0 >= self._v_entry * T[0])
            opti.subject_to(fwd_0 <= self._v_entry * 1.01 * T[0])
        if self._v_exit is not None:
            opti.subject_to(fwd_n >= self._v_exit * T[-1])
            opti.subject_to(fwd_n <= self._v_exit * 1.01 * T[-1])

        # Optional entry/exit acceleration pins
        if self._a_entry is not None:
            fwd_acc_0 = dds[0, 0] * c0 + dds[0, 1] * s0_
            opti.subject_to(fwd_acc_0 == self._a_entry * T[0]**2)
        if self._a_exit is not None:
            fwd_acc_n = dds[-1, 0] * cn + dds[-1, 1] * sn
            opti.subject_to(fwd_acc_n == self._a_exit * T[-1]**2)

        # Optional angular velocity / acceleration pins at boundaries
        if self._omega_entry is not None:
            opti.subject_to(ds[0, 2] == self._omega_entry * T[0])
        if self._omega_exit is not None:
            opti.subject_to(ds[-1, 2] == self._omega_exit * T[-1])
        if self._alpha_entry is not None:
            opti.subject_to(dds[0, 2] == self._alpha_entry * T[0]**2)
        if self._alpha_exit is not None:
            opti.subject_to(dds[-1, 2] == self._alpha_exit * T[-1]**2)

    def _add_jackknife_constraints(self, opti, pg):
        """! Convex-hull-guaranteed jackknife bound |γ(τ)| ≤ γ_max for all τ.

        Bounding every γ control point directly, rather than sampling γ(τ)
        at the nt sample nodes, exploits the same B-spline convex-hull
        property used for the corridor constraint (_add_corridor_constraints):
        every point on a B-spline segment is a convex combination of its
        local control points, so if all control points satisfy the bound,
        the entire spline arc does too — including between sample nodes,
        which the previous pointwise formulation could not guarantee.
        """
        gamma_max = self._gamma_max
        for j in range(self._n_Q):
            opti.subject_to(pg[j] <= gamma_max)
            opti.subject_to(-gamma_max <= pg[j])

    def _add_boundary_constraints(self, opti, px, py, pt, pg):
        """! Pin start/end trailer pose and hitch angle to waypoint values."""
        wps = self._waypoints
        opti.subject_to(px[0]  == wps[0, 0])
        opti.subject_to(py[0]  == wps[0, 1])
        opti.subject_to(pt[0]  == wps[0, 2])
        opti.subject_to(pg[0]  == self._gamma_entry)
        opti.subject_to(px[-1] == wps[-1, 0])
        opti.subject_to(py[-1] == wps[-1, 1])
        opti.subject_to(pt[-1] == wps[-1, 2])
        opti.subject_to(pg[-1] == self._gamma_exit)
