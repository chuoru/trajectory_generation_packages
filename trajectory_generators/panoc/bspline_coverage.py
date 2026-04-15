#!/usr/bin/env python3
##
# @file bspline_coverage.py
#
# @brief PANOC-based B-Spline parameterized OCP for coverage trajectory
# generation on a differential drive robot.
#
# Reformulates the B-Spline OCP for the PANOC solver (via OpEn / opengen):
#   - Decision variables : B-spline control points [px, py, pt] (3 * n_Q)
#   - Parameters         : start pose, end pose, total time, waypoints
#   - Cost               : nonholonomic penalty + jerk smoothness +
#                          soft corridor penalty
#   - AL constraints     : boundary conditions + segment continuity
#   - Box constraints    : position / angle bounds on control points
#
# Total time is a fixed parameter (not optimized), making the problem
# purely a shape-optimisation suited to the first-order PANOC method.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/04/15

# Standard library
import os

# External library
import numpy as np
import casadi as cs
import opengen as og


class BSplineCoverage:
    """! PANOC B-Spline coverage trajectory generator.

    Builds a Rust/PANOC solver once (via OpEn), then calls it at runtime
    with varying start/end poses, total time, and waypoints as parameters.
    The optimizer finds the B-spline control points [px, py, pt] that
    minimise nonholonomic constraint violation, jerk, and corridor deviation.
    """

    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    def __init__(self, n_waypoints=3, n_ctrl_pts=6, spline_order=3,
                 n_sampling=30, bound=0.17, vel_max=0.2,
                 w_nonh=10.0, w_jerk=1e-3, w_corr=50.0):
        """! Constructor.
        @param n_waypoints<int>: Number of via-points (fixed at build time).
        @param n_ctrl_pts<int>: B-spline control points per segment.
        @param spline_order<int>: B-spline degree (3 = cubic).
        @param n_sampling<int>: Discrete samples per spline segment.
        @param bound<float>: Corridor half-width [m].
        @param vel_max<float>: Maximum forward speed for soft penalty [m/s].
        @param w_nonh<float>: Weight on nonholonomic penalty.
        @param w_jerk<float>: Weight on jerk smoothness.
        @param w_corr<float>: Weight on corridor soft penalty.
        """
        self._n_waypoints = n_waypoints
        self._n_ctrl_pts = n_ctrl_pts
        self._degree = spline_order
        self._n_sampling = n_sampling
        self._n_pieces = n_waypoints - 1
        self._n_Q = self._n_pieces * n_ctrl_pts
        self._nt = n_sampling * (self._n_Q - 1) + 1
        self._bound = bound
        self._vel_max = vel_max
        self._w_nonh = w_nonh
        self._w_jerk = w_jerk
        self._w_corr = w_corr

        self._name = 'panoc_bspline_coverage'
        self._build_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', self._name))

        # Precompute numeric basis matrices (fixed for this problem size)
        knot = self._build_knot_vector()
        tau = np.linspace(0.0, 1.0, self._nt)
        self._B, self._dB, self._ddB, self._dddB = \
            self._build_basis_matrices(tau, knot)

        self._is_built = False

    def generate_trajectory(self, waypoints, total_time=10.0):
        """! Generate the B-Spline trajectory using the PANOC solver.
        @param waypoints<list>: Via-points [[x, y, theta], ...].
            Must have exactly n_waypoints entries (fixed at construction).
        @param total_time<float>: Fixed total traversal time [s].
        @return dict with keys:
            - 'states'   : (nt, 3) array [x, y, theta] along the spline
            - 'time'     : (nt,)   real time at each sample [s]
            - 'ctrl_pts' : (n_Q, 3) optimized control points
        """
        assert len(waypoints) == self._n_waypoints, (
            f"Expected {self._n_waypoints} waypoints, got {len(waypoints)}")

        if not self._is_built:
            self._define_problem()
            self._is_built = True

        params = self._build_params(waypoints, total_time)
        u0 = self._build_initial_guess(waypoints)

        manager = og.tcp.OptimizerTcpManager(
            os.path.join(self._build_dir, 'optimizer'))
        manager.start()
        manager.ping()

        solution = manager.call(params, initial_guess=u0)

        manager.kill()

        return self._extract_solution(solution['solution'], total_time)

    # ==========================================================================
    # PRIVATE METHODS — Problem definition
    # ==========================================================================
    def _define_problem(self):
        """! Build CasADi cost + constraints and compile the PANOC solver."""
        n_Q = self._n_Q
        n_ctrl = self._n_ctrl_pts
        n_pieces = self._n_pieces
        n_wps = self._n_waypoints
        bound = self._bound

        # ── Decision variable ────────────────────────────────────────────────
        # u = [px_0..px_{n_Q-1},  py_0..py_{n_Q-1},  pt_0..pt_{n_Q-1}]
        n_dv = 3 * n_Q
        u = cs.MX.sym('u', n_dv)

        px = u[:n_Q]
        py = u[n_Q:2 * n_Q]
        pt = u[2 * n_Q:]

        # ── Parameter vector ─────────────────────────────────────────────────
        # p = [x0(3), xf(3), tf(1), wp0(3), wp1(3), ..., wp_{n_wps-1}(3)]
        n_p = 7 + 3 * n_wps
        p = cs.MX.sym('p', n_p)

        tf = p[6]

        # ── Spline evaluation via precomputed basis matrices ─────────────────
        B_dm = cs.DM(self._B.tolist())
        dB_dm = cs.DM(self._dB.tolist())
        dddB_dm = cs.DM(self._dddB.tolist())

        # Control-point matrix (n_Q × 3)
        P = cs.horzcat(px, py, pt)

        s = cs.mtimes(B_dm, P)       # (nt, 3)  spline position + heading
        ds = cs.mtimes(dB_dm, P)     # (nt, 3)  spline first derivative
        ddds = cs.mtimes(dddB_dm, P) # (nt, 3)  spline third derivative

        s_th = s[:, 2]              # (nt, 1)  heading
        ds_x = ds[:, 0]             # (nt, 1)
        ds_y = ds[:, 1]             # (nt, 1)

        # ── Cost terms ───────────────────────────────────────────────────────

        # (1) Nonholonomic penalty: lateral velocity must vanish
        lateral = -ds_x * cs.sin(s_th) + ds_y * cs.cos(s_th)
        nonh_cost = cs.sumsqr(lateral)

        # (2) Jerk smoothness: minimise third derivative energy
        # Scale by total_time^3 to keep units consistent
        jerk_cost = cs.sumsqr(ddds)

        # (3) Soft corridor penalty: control points stay within ±bound of segment
        corr_cost = 0
        for seg in range(n_pieces):
            wp_a = p[7 + 3 * seg:7 + 3 * seg + 2]        # (2, 1)
            wp_b = p[7 + 3 * (seg + 1):7 + 3 * (seg + 1) + 2]
            diff = wp_b - wp_a
            seg_len = cs.sqrt(diff[0] ** 2 + diff[1] ** 2 + 1e-8)
            d = diff / seg_len                             # unit direction
            n_vec = cs.vertcat(-d[1], d[0])               # unit normal

            cp_start = seg * n_ctrl
            px_seg = px[cp_start:cp_start + n_ctrl]        # (n_ctrl, 1)
            py_seg = py[cp_start:cp_start + n_ctrl]

            rel_x = px_seg - wp_a[0]
            rel_y = py_seg - wp_a[1]
            lat_cp = n_vec[0] * rel_x + n_vec[1] * rel_y  # (n_ctrl, 1)

            corr_cost += cs.sumsqr(cs.fmax(0, lat_cp - bound))
            corr_cost += cs.sumsqr(cs.fmax(0, -lat_cp - bound))

        cost = (self._w_nonh * nonh_cost
                + self._w_jerk * jerk_cost
                + self._w_corr * corr_cost)

        # ── AL constraints: boundary conditions + segment continuity ─────────
        c_start = cs.vertcat(
            px[0] - p[0], py[0] - p[1], pt[0] - p[2])    # start = x0
        c_end = cs.vertcat(
            px[-1] - p[3], py[-1] - p[4], pt[-1] - p[5]) # end = xf

        c_al = cs.vertcat(c_start, c_end)

        for seg in range(n_pieces - 1):
            junc = (seg + 1) * n_ctrl - 1
            c_al = cs.vertcat(c_al,
                              px[junc] - px[junc + 1],
                              py[junc] - py[junc + 1])

        n_al = 6 + 2 * (n_pieces - 1)
        al_bounds = og.constraints.Rectangle(
            xmin=[0.0] * n_al,
            xmax=[0.0] * n_al)

        # ── Box constraints on decision variable ─────────────────────────────
        pos_hi = 20.0
        th_hi = 4 * np.pi
        u_min = [-pos_hi] * (2 * n_Q) + [-th_hi] * n_Q
        u_max = [pos_hi] * (2 * n_Q) + [th_hi] * n_Q
        bounds = og.constraints.Rectangle(u_min, u_max)

        # ── Build the Rust solver ─────────────────────────────────────────────
        problem = (og.builder.Problem(u, p, cost)
                   .with_constraints(bounds)
                   .with_aug_lagrangian_constraints(c_al, al_bounds))

        build_config = (og.config.BuildConfiguration()
                        .with_build_directory(self._build_dir)
                        .with_build_mode('release')
                        .with_tcp_interface_config())

        meta = og.config.OptimizerMeta().with_optimizer_name('optimizer')

        solver_config = (og.config.SolverConfiguration()
                         .with_tolerance(1e-5)
                         .with_initial_tolerance(1e-3)
                         .with_max_outer_iterations(50)
                         .with_max_inner_iterations(10000)
                         .with_penalty_weight_update_factor(5.0)
                         .with_delta_tolerance(1e-4))

        builder = og.builder.OpEnOptimizerBuilder(
            problem, meta, build_config, solver_config)
        builder.build()

    # ==========================================================================
    # PRIVATE METHODS — Helpers
    # ==========================================================================
    def _build_params(self, waypoints, total_time):
        """! Assemble the flat parameter vector passed to the solver at runtime.
        p = [x0(3), xf(3), tf(1), wp0(3), wp1(3), ..., wp_{n-1}(3)]
        """
        wps = np.array(waypoints, dtype=float)
        p = list(wps[0]) + list(wps[-1]) + [total_time]
        for wp in wps:
            p += list(wp)
        return p

    def _build_initial_guess(self, waypoints):
        """! Warm-start: lay control points linearly along the path."""
        wps = np.array(waypoints, dtype=float)
        n_ctrl = self._n_ctrl_pts
        n_Q = self._n_Q

        cp0 = np.zeros((n_Q, 3))
        idx = 0
        for seg in range(self._n_pieces):
            p0, p1 = wps[seg], wps[seg + 1]
            for j in range(n_ctrl):
                alpha = j / (n_ctrl - 1)
                cp0[idx] = (1 - alpha) * p0 + alpha * p1
                idx += 1

        return list(cp0[:, 0]) + list(cp0[:, 1]) + list(cp0[:, 2])

    def _extract_solution(self, u_sol, total_time):
        """! Evaluate the B-spline at the optimized control points.
        @return dict with 'states', 'time', 'ctrl_pts'.
        """
        n_Q = self._n_Q
        u = np.array(u_sol)
        px = u[:n_Q]
        py = u[n_Q:2 * n_Q]
        pt = u[2 * n_Q:]

        cp = np.column_stack([px, py, pt])   # (n_Q, 3)

        states = self._B @ cp                # (nt, 3)
        time = np.linspace(0.0, total_time, self._nt)

        return {
            'states': states,
            'time': time,
            'ctrl_pts': cp,
        }

    # ==========================================================================
    # PRIVATE METHODS — B-Spline utilities (adapted from bspline_coverage.py)
    # ==========================================================================
    def _build_knot_vector(self):
        """! Clamped uniform knot vector of length n_Q + degree + 1."""
        n_Q = self._n_Q
        k = self._degree
        m = n_Q + k + 1
        U = np.zeros(m)
        denom = m - 2 * k - 1
        for i in range(1, m + 1):
            if i <= k:
                U[i - 1] = 0.0
            elif i > m - k:
                U[i - 1] = 1.0
            else:
                U[i - 1] = (i - k - 1) / denom
        return U

    def _find_span(self, u_val, U):
        """! 0-indexed knot span containing u_val."""
        k = self._degree
        if u_val == 1.0:
            return len(U) - k - 2
        idx = int(np.searchsorted(U, u_val, side='right')) - 1
        return min(idx, len(U) - k - 2)

    def _ders_basis_funs(self, span, u_val, U):
        """! B-Spline basis functions and derivatives up to order 3.
        Algorithm A2.3 from Piegl & Tiller, "The NURBS Book".
        """
        p = self._degree
        ndu = np.zeros((p + 1, p + 1))
        left = np.zeros(p + 1)
        right = np.zeros(p + 1)
        a = np.zeros((2, p + 1))
        ders = np.zeros((4, p + 1))

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
            for k in range(1, 4):
                d = 0.0
                rk, pk = r - k, p - k
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
        for k in range(1, 4):
            ders[k, :] *= r
            r *= (p - k)
        return ders

    def _build_basis_matrices(self, tau, U):
        """! Precompute numeric (nt × n_Q) basis matrices."""
        nt = len(tau)
        n_Q = self._n_Q
        k = self._degree

        B = np.zeros((nt, n_Q))
        dB = np.zeros((nt, n_Q))
        ddB = np.zeros((nt, n_Q))
        dddB = np.zeros((nt, n_Q))

        span0 = self._find_span(tau[0], U)
        ders0 = self._ders_basis_funs(span0, tau[0], U)

        for ti, u_val in enumerate(tau):
            if u_val == 1.0:
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
