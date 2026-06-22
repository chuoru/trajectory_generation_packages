#!/usr/bin/env python3
##
# @file bspline_coverage.py
#
# @brief B-Spline parameterized optimal control for coverage trajectory
# generation on a differential drive robot.
#
# Based on the work of Fabian Friz, Toyohashi University of Technology (08/2024).
# The state (x, y, theta) is represented as a single B-spline in a normalized
# parameter tau in [0, 1]. A per-sample time variable T(i) links the geometric
# parameter to real time. IPOPT minimizes sum(T) subject to velocity,
# acceleration, jerk, nonholonomic, and corridor constraints.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/04/15

# Standard library
import numpy as np

# External library
import casadi as cs
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline as _CubicSpline


class BSplineCoverage:
    """! B-Spline parameterized OCP for differential drive coverage.

    Represents the full robot state (x, y, theta) as a B-spline over a
    normalized parameter tau in [0, 1], with per-sample time variables T(i)
    providing the mapping to real time. Minimizes total traversal time.
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
                 acc_max=None, jerk_max=None):
        """! Constructor.
        @param waypoints<list>: Via-points [[x, y, theta], ...]. At least 2.
        @param bound<float>: Half-width of the corridor around each segment [m].
        @param n_ctrl_pts<int>: Number of B-spline control points per segment.
        @param spline_order<int>: B-spline degree (3 = cubic).
        @param n_sampling<int>: Discrete samples per segment.
        @param vel_max<list|None>: [vx_max, vy_max, omega_max]. Defaults to
            [0.2, 0.2, 0.196] m/s and rad/s.
        @param vel_min_lin<float>: Minimum feedrate (||vx,vy||) [m/s].
        @param eps_nonh<float>: Nonholonomic constraint tolerance.
        @param v_entry<float|None>: Exact linear speed [m/s] at trajectory start.
            None leaves the entry speed free (determined by the OCP).
        @param v_exit<float|None>: Exact linear speed [m/s] at trajectory end.
            None leaves the exit speed free (determined by the OCP).
        @param acc_max<list|None>: [ax_max, ay_max, alpha_max] physical acceleration
            limits [m/s², m/s², rad/s²]. None keeps default [20, 20, 10].
        @param jerk_max<list|None>: [jx_max, jy_max, jalpha_max] physical jerk
            limits [m/s³, m/s³, rad/s³]. None keeps default [1e3, 1e3, 100].
        """
        self._waypoints = np.array(waypoints, dtype=float)
        self._bound = bound
        self._n_ctrl_pts = n_ctrl_pts
        self._degree = spline_order
        self._n_sampling = n_sampling
        self._eps_nonh = eps_nonh
        self._vel_min_lin = vel_min_lin

        if vel_max is None:
            v = 0.2
            self._vel_max = np.array([v, v, (v - 0.001) / (2 * 0.51)])
        else:
            self._vel_max = np.array(vel_max, dtype=float)

        self._vel_min = -self._vel_max
        self._acc_max = np.array([20.0, 20.0, 10.0])
        self._acc_min = -self._acc_max
        self._jerk_max = np.array([1e3, 1e3, 100.0])
        self._jerk_min = -self._jerk_max

        if acc_max is not None:
            self._acc_max = np.array(acc_max, dtype=float)
            self._acc_min = -self._acc_max
        if jerk_max is not None:
            self._jerk_max = np.array(jerk_max, dtype=float)
            self._jerk_min = -self._jerk_max

        self._n_pieces = len(waypoints) - 1
        self._n_Q = self._n_pieces * n_ctrl_pts
        self._nt = n_sampling * (self._n_Q - 1) + 1

        self._v_entry = float(v_entry) if v_entry is not None else None
        self._v_exit  = float(v_exit)  if v_exit  is not None else None
        self._a_entry = float(a_entry) if a_entry is not None else None
        self._a_exit  = float(a_exit)  if a_exit  is not None else None
        self._omega_entry = float(omega_entry) if omega_entry is not None else None
        self._omega_exit  = float(omega_exit)  if omega_exit  is not None else None
        self._alpha_entry = float(alpha_entry) if alpha_entry is not None else None
        self._alpha_exit  = float(alpha_exit)  if alpha_exit  is not None else None

        self._optimizer = cs.Opti()
        self._optimizer.solver(
            'ipopt',
            {'print_time': False},
            {
                'max_iter': 10000,
                'print_level': 3,
                'tol': 1e-5,
                'acceptable_tol': 5e-3,
                'acceptable_iter': 15,
                'constr_viol_tol': 1e-4,
                'hessian_approximation': 'limited-memory',
            }
        )

    def generate_trajectory(self):
        """! Build and solve the OCP.
        @return dict with keys:
            - 'states'  : (nt, 3) array [x, y, theta] along the spline
            - 'time'    : (nt,)   real time at each sample [s]
            - 'v'       : (M,)    linear velocity from inverse kinematics [m/s]
            - 'omega'   : (M,)    angular velocity [rad/s]
            - 'time_ik' : (M,)    time vector for v/omega
            - 'ctrl_pts': (n_Q, 3) optimized control points
        """
        knot = self._build_knot_vector()
        tau = np.linspace(0.0, 1.0, self._nt)

        basis, dot_basis, ddot_basis, dddot_basis = \
            self._build_basis_matrices(tau, knot)

        opti = self._optimizer

        # Decision variables: control points and per-sample time
        px = opti.variable(self._n_Q)
        py = opti.variable(self._n_Q)
        pt = opti.variable(self._n_Q)      # theta control points
        T = opti.variable(self._nt)        # time scaling at each sample

        # Spline values and derivatives (linear in control points)
        P = cs.horzcat(px, py, pt)         # (n_Q, 3)

        s = basis @ P                      # (nt, 3)
        ds = dot_basis @ P                 # (nt, 3)
        dds = ddot_basis @ P               # (nt, 3)
        ddds = dddot_basis @ P             # (nt, 3)

        # Initial conditions for solver
        self._set_initial_conditions(opti, px, py, pt, T, basis, dot_basis)

        # Objective: minimize total time
        opti.minimize(cs.sum1(T))

        # Constraints
        self._add_dynamic_constraints(opti, s, ds, dds, ddds, T)
        self._add_boundary_constraints(opti, px, py, pt)
        self._add_corridor_constraints(opti, px, py)

        # Solve
        try:
            sol = opti.solve()
            dbg = sol
        except Exception:
            dbg = opti.debug

        # Extract solution
        T_val = dbg.value(T)
        s_val = dbg.value(s)
        ds_val = dbg.value(ds)
        ctrl_pts = dbg.value(P)

        # Build real time axis: t_real[i] = sum(T[0..i]) / nt
        t_real = np.cumsum(T_val) / self._nt

        # Velocity at OCP nodes from B-spline derivatives (smooth, no staircase)
        cos_th = np.cos(s_val[:, 2])
        sin_th = np.sin(s_val[:, 2])
        v_ocp  = (cos_th * ds_val[:, 0] + sin_th * ds_val[:, 1]) / T_val
        om_ocp = ds_val[:, 2] / T_val

        ts_des   = 0.01
        t_interp = np.arange(ts_des, t_real[-1], ts_des)
        v_arr    = _CubicSpline(t_real, v_ocp)(t_interp[:-1])
        omega_arr = _CubicSpline(t_real, om_ocp)(t_interp[:-1])

        return {
            'states': s_val,
            'time': t_real,
            'ctrl_pts': ctrl_pts,
            'v': v_arr,
            'omega': omega_arr,
            'time_ik': t_interp[:-1],
        }

    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    def _build_knot_vector(self):
        """! Build clamped uniform knot vector (port of knot_v2.m).
        @return numpy array of length n_Q + degree + 1.
        """
        n_Q = self._n_Q
        k = self._degree
        m = n_Q + k + 1           # total number of knots

        U = np.zeros(m)
        denom = m - 2 * k - 1
        for i in range(1, m + 1):   # 1-indexed like MATLAB
            if i <= k:
                U[i - 1] = 0.0
            elif i > m - k:
                U[i - 1] = 1.0
            else:
                U[i - 1] = (i - k - 1) / denom
        return U

    def _find_span(self, u_val, U):
        """! Find the knot span index (0-indexed) containing u_val."""
        k = self._degree
        if u_val == 1.0:
            return len(U) - k - 2
        idx = int(np.searchsorted(U, u_val, side='right')) - 1
        return min(idx, len(U) - k - 2)

    def _ders_basis_funs(self, span, u_val, U):
        """! Compute B-spline basis functions and derivatives up to order 3.

        Algorithm A2.3 from Piegl & Tiller, "The NURBS Book".

        @param span: 0-indexed knot span.
        @param u_val: parameter value in [0, 1].
        @param U: knot vector (numpy array, 0-indexed).
        @return (4, degree+1) array. Row k = k-th derivative values.
        """
        p = self._degree
        n_derivs = 3

        ndu = np.zeros((p + 1, p + 1))
        left = np.zeros(p + 1)
        right = np.zeros(p + 1)
        a = np.zeros((2, p + 1))
        ders = np.zeros((n_derivs + 1, p + 1))

        ndu[0, 0] = 1.0
        for j in range(1, p + 1):
            left[j] = u_val - U[span + 1 - j]
            right[j] = U[span + j] - u_val
            saved = 0.0
            for r in range(j):
                ndu[j, r] = right[r + 1] + left[j - r]
                temp = ndu[r, j - 1] / ndu[j, r]
                ndu[r, j] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            ndu[j, j] = saved

        for j in range(p + 1):
            ders[0, j] = ndu[j, p]

        for r in range(p + 1):
            s1, s2 = 0, 1
            a[0, 0] = 1.0
            for k in range(1, n_derivs + 1):
                d = 0.0
                rk = r - k
                pk = p - k
                if r >= k:
                    a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                    d = a[s2, 0] * ndu[rk, pk]
                j1 = 1 if rk >= -1 else -rk
                j2 = k - 1 if r - 1 <= pk else p - r
                for j in range(j1, j2 + 1):
                    a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                    d += a[s2, j] * ndu[rk + j, pk]
                if r <= pk:
                    a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                    d += a[s2, k] * ndu[r, pk]
                ders[k, r] = d
                s1, s2 = s2, s1

        r = p
        for k in range(1, n_derivs + 1):
            ders[k, :] *= r
            r *= (p - k)

        return ders

    def _build_basis_matrices(self, tau, U):
        """! Precompute numeric basis matrices for all sample points.

        @param tau: (nt,) array of parameter values in [0, 1].
        @param U: knot vector.
        @return Tuple (B, dB, ddB, dddB) each of shape (nt, n_Q).
                Multiplying by control-point matrix P gives spline values.
        """
        nt = len(tau)
        n_Q = self._n_Q
        k = self._degree

        B = np.zeros((nt, n_Q))
        dB = np.zeros((nt, n_Q))
        ddB = np.zeros((nt, n_Q))
        dddB = np.zeros((nt, n_Q))

        # Compute and cache at t=0 to mirror endpoint at t=1
        span0 = self._find_span(tau[0], U)
        ders0 = self._ders_basis_funs(span0, tau[0], U)

        for ti, u_val in enumerate(tau):
            if u_val == 1.0:
                # Mirror of t=0 endpoint (clamped spline symmetry)
                lo0 = span0 - k
                lo1 = n_Q - 1 - span0
                B[ti, lo1:lo1 + k + 1] = ders0[0, ::-1]
                dB[ti, lo1:lo1 + k + 1] = -ders0[1, ::-1]
                ddB[ti, lo1:lo1 + k + 1] = ders0[2, ::-1]
                dddB[ti, lo1:lo1 + k + 1] = -ders0[3, ::-1]
            else:
                span = self._find_span(u_val, U)
                ders = self._ders_basis_funs(span, u_val, U)
                lo = span - k
                B[ti, lo:lo + k + 1] = ders[0, :]
                dB[ti, lo:lo + k + 1] = ders[1, :]
                ddB[ti, lo:lo + k + 1] = ders[2, :]
                dddB[ti, lo:lo + k + 1] = ders[3, :]

        return B, dB, ddB, dddB

    def _set_initial_conditions(self, opti, px, py, pt, T, basis,
                               dot_basis=None):
        """! Warm-start: lay control points along the piecewise-linear path
        and initialize T so that the initial velocity equals vel_max.
        """
        wps = self._waypoints
        n_ctrl = self._n_ctrl_pts
        n_pieces = self._n_pieces

        # Control-point initial values (uniform interpolation along each segment)
        cp0 = np.zeros((self._n_Q, 3))
        idx = 0
        for seg in range(n_pieces):
            p0 = wps[seg]
            p1 = wps[seg + 1]
            for j in range(n_ctrl):
                alpha = j / (n_ctrl - 1)
                cp0[idx] = (1 - alpha) * p0 + alpha * p1
                opti.set_initial(px[idx], cp0[idx, 0])
                opti.set_initial(py[idx], cp0[idx, 1])
                opti.set_initial(pt[idx], cp0[idx, 2])
                idx += 1

        # Use spline DERIVATIVE magnitudes to estimate T so ||vel|| ~= vel_max.
        # dot_basis @ cp0 gives ds/dtau at each sample; its xy norm is the
        # feedrate scaling.  Fall back to position norms if dot_basis is absent.
        if dot_basis is not None:
            ds0_xy = np.linalg.norm((dot_basis @ cp0)[:, :2], axis=1)
        else:
            ds0_xy = np.linalg.norm((basis @ cp0)[:, :2], axis=1)

        v_max = self._vel_max[0]
        T_init = np.where(ds0_xy > 1e-8,
                          ds0_xy / v_max,
                          1.0 / v_max)

        # Patch T[0] / T[-1] so the initial point approximately satisfies
        # any velocity-BC constraints and gives IPOPT a warm feasible region.
        if self._v_entry is not None and self._v_entry > 1e-6:
            v0 = min(float(self._v_entry), v_max)
            T_init[0] = ds0_xy[0] / v0 if ds0_xy[0] > 1e-8 else 1.0 / v0
        if self._v_exit is not None and self._v_exit > 1e-6:
            vn = min(float(self._v_exit), v_max)
            T_init[-1] = ds0_xy[-1] / vn if ds0_xy[-1] > 1e-8 else 1.0 / vn

        opti.set_initial(T, T_init)

    def _add_dynamic_constraints(self, opti, s, ds, dds, ddds, T):
        """! Add velocity, acceleration, jerk, and nonholonomic constraints."""
        vmax = self._vel_max
        vmin = self._vel_min
        amax = self._acc_max
        amin = self._acc_min
        jmax = self._jerk_max
        jmin = self._jerk_min
        v = vmax[0]          # feedrate limit
        vlin_min = self._vel_min_lin
        eps = self._eps_nonh

        for i in range(self._nt):
            Ti = T[i]

            # Angular velocity
            opti.subject_to(vmin[2] * Ti <= ds[i, 2])
            opti.subject_to(ds[i, 2] <= vmax[2] * Ti)

            # Acceleration (all 3 dimensions)
            for dim in range(3):
                opti.subject_to(amin[dim] * Ti**2 <= dds[i, dim])
                opti.subject_to(dds[i, dim] <= amax[dim] * Ti**2)

            # Jerk (all 3 dimensions)
            for dim in range(3):
                opti.subject_to(jmin[dim] * Ti**3 <= ddds[i, dim])
                opti.subject_to(ddds[i, dim] <= jmax[dim] * Ti**3)

            # Feedrate bounds: vlin_min <= ||vx,vy|| <= v
            feedrate_sq = ds[i, 0]**2 + ds[i, 1]**2
            opti.subject_to(feedrate_sq <= v**2 * Ti**2)
            opti.subject_to(vlin_min**2 * Ti**2 <= feedrate_sq)

            # Nonholonomic: lateral velocity ~ 0
            lateral = -ds[i, 0] * cs.sin(s[i, 2]) + ds[i, 1] * cs.cos(s[i, 2])
            opti.subject_to(lateral <= eps * Ti)
            opti.subject_to(-eps * Ti <= lateral)

            # Forward-only: projection of velocity onto heading must be >= vel_min_lin
            fwd_i = ds[i, 0] * cs.cos(s[i, 2]) + ds[i, 1] * cs.sin(s[i, 2])
            opti.subject_to(fwd_i >= vlin_min * Ti)

            # T must be positive
            opti.subject_to(Ti >= 1e-4)

        # Forward-only constraints at the boundary samples: the heading is
        # effectively known (pinned by the clamped knot + boundary constraints),
        # so these are linear in ds and T.  They prevent backward solutions
        # at entry/exit even without explicit velocity BC.
        theta_0 = float(self._waypoints[0, 2])
        c0, s0 = float(np.cos(theta_0)), float(np.sin(theta_0))
        fwd_0 = ds[0, 0] * c0 + ds[0, 1] * s0
        opti.subject_to(fwd_0 >= self._vel_min_lin * T[0])

        theta_n = float(self._waypoints[-1, 2])
        cn, sn = float(np.cos(theta_n)), float(np.sin(theta_n))
        fwd_n = ds[-1, 0] * cn + ds[-1, 1] * sn
        opti.subject_to(fwd_n >= self._vel_min_lin * T[-1])

        # Optional: pin entry/exit speed using a tight range (±5 %) rather
        # than an equality so that IPOPT always has a feasible interior point.
        if self._v_entry is not None:
            opti.subject_to(fwd_0 >= self._v_entry * T[0])
            opti.subject_to(fwd_0 <= self._v_entry * 1.01 * T[0])

        if self._v_exit is not None:
            opti.subject_to(fwd_n >= self._v_exit * T[-1])
            opti.subject_to(fwd_n <= self._v_exit * 1.01 * T[-1])

        # Pin boundary forward accelerations for smooth stitching with adjacent segments.
        # When a_entry/a_exit = 0, this is a pure linear constraint (homogeneous in ctrl pts).
        if self._a_entry is not None:
            c0_ = float(np.cos(theta_0))
            s0_ = float(np.sin(theta_0))
            fwd_acc_0 = dds[0, 0] * c0_ + dds[0, 1] * s0_
            opti.subject_to(fwd_acc_0 == self._a_entry * T[0]**2)

        if self._a_exit is not None:
            cn_ = float(np.cos(theta_n))
            sn_ = float(np.sin(theta_n))
            fwd_acc_n = dds[-1, 0] * cn_ + dds[-1, 1] * sn_
            opti.subject_to(fwd_acc_n == self._a_exit * T[-1]**2)

        # Pin angular velocity at entry/exit so that omega = 0 at the junction
        # with straight JLAP segments (which always have zero yaw rate).
        # ds[i, 2] = dtheta/dtau; dividing by T[i] gives physical omega [rad/s].
        if self._omega_entry is not None:
            opti.subject_to(ds[0, 2] == self._omega_entry * T[0])
        if self._omega_exit is not None:
            opti.subject_to(ds[-1, 2] == self._omega_exit * T[-1])

        # Pin angular acceleration at entry/exit to eliminate the alpha spike that
        # occurs when omega jumps away from zero immediately after the boundary.
        # dds[i, 2] = d²theta/dtau²; dividing by T[i]² gives alpha [rad/s²].
        if self._alpha_entry is not None:
            opti.subject_to(dds[0, 2] == self._alpha_entry * T[0]**2)
        if self._alpha_exit is not None:
            opti.subject_to(dds[-1, 2] == self._alpha_exit * T[-1]**2)

    def _add_boundary_constraints(self, opti, px, py, pt):
        """! Pin start and end poses to the first and last waypoints."""
        wps = self._waypoints
        opti.subject_to(px[0] == wps[0, 0])
        opti.subject_to(py[0] == wps[0, 1])
        opti.subject_to(pt[0] == wps[0, 2])
        opti.subject_to(px[-1] == wps[-1, 0])
        opti.subject_to(py[-1] == wps[-1, 1])
        opti.subject_to(pt[-1] == wps[-1, 2])

    def _add_corridor_constraints(self, opti, px, py):
        """! Constrain each control point to lie within its segment corridor."""
        wps = self._waypoints
        n_ctrl = self._n_ctrl_pts

        for seg in range(self._n_pieces):
            A_seg, b_seg = self._segment_corridor(
                wps[seg, :2], wps[seg + 1, :2])

            for j in range(n_ctrl):
                cp_idx = seg * n_ctrl + j
                pt_vec = cs.vertcat(px[cp_idx], py[cp_idx])
                opti.subject_to(A_seg @ pt_vec <= b_seg)

            # Continuity at segment junction
            if seg < self._n_pieces - 1:
                junc = (seg + 1) * n_ctrl - 1
                opti.subject_to(px[junc] == px[junc + 1])
                opti.subject_to(py[junc] == py[junc + 1])

    def _segment_corridor(self, A, B):
        """! Build a rectangular corridor polyhedron for segment A→B.

        Returns (A_mat, b_vec) such that A_mat @ [x, y] <= b_vec defines
        a box of half-width `bound` around the segment.
        """
        d = B - A
        d = d / np.linalg.norm(d)
        n = np.array([-d[1], d[0]])        # 90-degree rotation
        bd = self._bound

        # Four half-space inequalities (rectangle corners)
        vertices = np.array([
            A - d * bd + n * bd,
            A - d * bd - n * bd,
            B + d * bd + n * bd,
            B + d * bd - n * bd,
        ])

        hull = ConvexHull(vertices)
        A_mat = hull.equations[:, :2]
        b_vec = -hull.equations[:, 2]
        return A_mat, b_vec

    def _inverse_kinematics(self, states, dt):
        """! Convert a state trajectory to (v, omega) via unicycle IK.

        For each step k→k+1 the angular rate is approximated as
        omega = delta_theta / dt, and the linear velocity is recovered
        by inverting the integral of the rotation matrix.

        @param states: (M, 3) array [x, y, theta].
        @param dt: time step between consecutive states [s].
        @return Tuple of (v, omega) each of length M-1.
        """
        M = len(states)
        v_arr = np.zeros(M - 1)
        omega_arr = np.zeros(M - 1)

        for k in range(M - 1):
            xk = states[k]
            xk1 = states[k + 1]
            theta = xk[2]
            wk = (xk1[2] - xk[2]) / dt

            # Rotation matrix transpose (body to world inverse)
            S_inv = np.array([
                [np.cos(theta),  np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0,              0,             1],
            ])

            delta = xk1 - xk

            if abs(wk) < 1e-5:
                # Taylor expansion: B ≈ diag(dt, 0, dt)
                B_inv = np.diag([1.0 / dt, 0.0, 1.0 / dt])
            else:
                sw = np.sin(wk * dt) / wk
                cw = (1.0 - np.cos(wk * dt)) / wk
                B = np.diag([sw, cw, dt])
                try:
                    B_inv = np.linalg.inv(B)
                except np.linalg.LinAlgError:
                    B_inv = np.diag([1.0 / dt, 0.0, 1.0 / dt])

            # Select rows for [v, omega]
            select = np.array([[1, 0, 0], [0, 0, 1]])
            u_body = select @ B_inv @ S_inv @ delta

            v_arr[k] = u_body[0]
            omega_arr[k] = wk

        return v_arr, omega_arr
