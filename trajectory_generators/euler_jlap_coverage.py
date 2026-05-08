#!/usr/bin/env python3
##
# @file euler_jlap_coverage.py
#
# @brief Euler-spiral corner smoothing + JLAP velocity profiling for
# differential-drive coverage trajectory generation.
#
# The algorithm (ported from the C++ reference in
# reference/trajectory_generation/) works in three passes:
#
#   1. Corner smoothing  — for every interior waypoint, fits an Euler spiral
#      (clothoid) whose curvature increases linearly with arc length.  The
#      spiral is parameterised by a scaling factor `a_euler` chosen so that
#      the lateral deviation from the straight-line path equals a computed
#      cornering tolerance `epsilon`.
#
#   2. JLAP solver  — for every straight-line segment between corner
#      entry/exit points, solves for the 7 time intervals of a jerk-limited
#      acceleration profile (positive-jerk / constant-acc / negative-jerk /
#      cruise / mirror decel).
#
#   3. Interpolation  — samples the composite path at `sampling_time`
#      intervals to produce position, velocity, acceleration, and heading.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/04/16

# Standard library
import math
import numpy as np


class EulerJLAPCoverage:
    """! Euler-spiral + JLAP trajectory generator for a differential drive.

    Takes a sequence of 2-D waypoints (x, y) and produces a time-sampled
    trajectory with smooth corners and jerk-limited straight-segment
    velocity profiles.

    Robot and motion-limit parameters are fully configurable; defaults
    match the NewMiniAGV used in the reference C++ code.
    """

    # ------------------------------------------------------------------
    # DEFAULT ROBOT / MOTION PARAMETERS  (NewMiniAGV reference values)
    # ------------------------------------------------------------------
    _DEFAULT_ROBOT_PARAMS = {
        'robot_mass':         50.4,    # [kg]
        'robot_width':        0.510,    # [m]  full track width
        'wheel_radius':       0.3,    # [m]
        'gear_ratio':         40.0,     # [ ]  motor→wheel
        'rated_motor_torque': 1.3,      # [Nm]
        'rated_motor_speed':  3500.0,   # [RPM]
        'motor_inertia':      0.66e-4,  # [kgm²]
        'path_vel_lim':       0.5,      # [m/s]  maximum path velocity
    }

    # Number of Taylor-series terms for unit Euler spiral (N_EULER=50 in C++)
    _N_EULER = 50

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def __init__(self, waypoints, sampling_time=0.05, robot_params=None,
                 path_vel_step=0.01, epsilon_offset=0.1, lc_scale=0.4,
                 initial_vel=0.0, final_vel=0.0):
        """! Constructor.

        @param waypoints<list>: Via-points as [[x, y], ...] or [[x, y, theta], ...].
            Only (x, y) are used for path geometry; theta, if present, is ignored.
        @param sampling_time<float>: Output sample period [s].
        @param robot_params<dict|None>: Override any of _DEFAULT_ROBOT_PARAMS.
        @param path_vel_step<float>: Velocity search step in JLAP solver [m/s].
        @param epsilon_offset<float>: Additive offset on minimum cornering
            tolerance [m].  Must be > 0.  C++ value: 0.1.
        @param lc_scale<float>: Rule-of-thumb fraction of segment length
            allowed for corner Euclidean reach (0 < lc_scale < 0.5).
        @param initial_vel<float>: Robot's path velocity at the first waypoint.
        @param final_vel<float>: Robot's path velocity at the last waypoint.
            Defaults to 0.0 (full stop). Set to a positive value to hand off
            velocity continuously to the next segment.
        """
        if len(waypoints) < 2:
            raise ValueError("At least 2 waypoints required.")

        # Store only (x, y)
        self._waypoints = np.array([[w[0], w[1]] for w in waypoints],
                                   dtype=float)

        self._dt = float(sampling_time)

        rp = {**self._DEFAULT_ROBOT_PARAMS, **(robot_params or {})}
        self._r     = rp['wheel_radius']
        self._l     = 0.5 * rp['robot_width']   # half-wheelbase (WHEEL_AXIS_DIST)
        self._v_lim = rp['path_vel_lim']

        # Derived motion limits
        rated_torque = rp['gear_ratio'] * rp['rated_motor_torque']
        inertia      = rp['gear_ratio']**2 * rp['motor_inertia']
        m, r         = rp['robot_mass'], rp['wheel_radius']

        self._a_lim = (0.5 * rated_torque * r
                       / (0.25 * m * r**2 + inertia))          # [m/s²]
        self._j_lim = self._a_lim / (40.0 * self._dt)          # [m/s³]
        self._ang_vel_lim = ((2.0 * math.pi / 60.0)
                             * rp['rated_motor_speed']
                             / rp['gear_ratio'])                 # [rad/s]
        self._ang_acc_lim = self._a_lim / self._r               # [rad/s²]

        self._vel_step      = float(path_vel_step)
        self._eps_offset    = float(epsilon_offset)
        self._lc_scale      = float(lc_scale)
        self._initial_vel   = float(initial_vel)
        self._final_vel     = float(final_vel)

        # Build Euler coefficient table once
        self._euler_table = self._build_euler_table()

    # ------------------------------------------------------------------

    def generate_trajectory(self):
        """! Compute and return the trajectory.

        @return dict with keys:
            - 'states'     : (N, 3) array  [x, y, phi]
            - 'time'       : (N,)   array  [s]
            - 'v'          : (N,)   array  path velocity [m/s]
            - 'acc_path'   : (N,)   array  path acceleration [m/s²]
            - 'omega'      : (N,)   array  yaw rate dphi [rad/s]
            - 'alpha'       : (N,)  array  yaw acceleration ddphi [rad/s²]
            - 'corner_info' : list of dicts, one per interior waypoint
        """
        wps = self._waypoints
        n_pt     = len(wps)
        n_line   = n_pt - 1
        n_corner = n_pt - 2

        # ---- Phase 1: corner smoothing --------------------------------
        corners = self._solve_corners(wps, n_corner)

        # ---- Phase 2: JLAP per line segment ---------------------------
        # Line segments run from:
        #   wps[0]         → corners[0].start   (first line)
        #   corners[i].end → corners[i+1].start (middle lines)
        #   corners[-1].end→ wps[-1]            (last line)
        line_starts = [wps[0]] + [c['pos_end'] for c in corners]
        line_ends   = [c['pos_start'] for c in corners] + [wps[-1]]

        vel_starts = [self._initial_vel] + [c['vel'] for c in corners]
        vel_ends   = [c['vel'] for c in corners] + [self._final_vel]

        jlap_data = []
        for i in range(n_line):
            seg = self._jlap_solver(
                line_starts[i], vel_starts[i], 0.0,
                line_ends[i],   vel_ends[i],   0.0)
            jlap_data.append(seg)

        # ---- Phase 3: interpolation -----------------------------------
        states, times, v_arr, acc_arr, omega_arr, alpha_arr = \
            self._interpolate(wps, corners, jlap_data,
                              line_starts, line_ends)

        omega_r = (v_arr + self._l * omega_arr) / self._r
        omega_l = (v_arr - self._l * omega_arr) / self._r

        return {
            'states':      states,
            'time':        times,
            'v':           v_arr,
            'acc_path':    acc_arr,
            'omega':       omega_arr,
            'alpha':       alpha_arr,
            'omega_r':     omega_r,
            'omega_l':     omega_l,
            'corner_info': corners,
        }

    # ==================================================================
    # PRIVATE — EULER TABLE
    # ==================================================================
    def _build_euler_table(self):
        """Precompute the Taylor-series coefficient table for the unit Euler spiral."""
        table = np.zeros((self._N_EULER, 2))
        for i in range(self._N_EULER):
            # factx = (2i)!
            factx = 1
            for j in range(1, 2 * i + 1):
                factx *= j
            facty = factx * (2 * i + 1)        # (2i+1)!
            table[i, 0] = ((-1)**i) * 2.0 / (factx * (4 * i + 1))
            table[i, 1] = ((-1)**i) * 2.0 / (facty * (4 * i + 3))
        return table

    def _unit_euler_coord(self, phi_abs):
        """Compute (x, y) on the unit Euler spiral at orientation angle phi_abs >= 0."""
        x, y = 0.0, 0.0
        for i in range(self._N_EULER):
            e1 = 0.5 * (4 * i + 1)
            e2 = 0.5 * (4 * i + 3)
            x += self._euler_table[i, 0] * (phi_abs ** e1)
            y += self._euler_table[i, 1] * (phi_abs ** e2)
        sq = math.sqrt(2.0 * math.pi)
        return x / sq, y / sq

    # ==================================================================
    # PRIVATE — CORNER SMOOTHING
    # ==================================================================
    @staticmethod
    def _angle_from_vector(v):
        """Return atan2 angle in (-pi, pi] for a 2-D vector."""
        return math.atan2(v[1], v[0])

    @staticmethod
    def _wrap_to_pi(angle):
        return angle - 2.0 * math.pi * math.floor(
            (angle + math.pi) / (2.0 * math.pi))

    def _solve_corners(self, wps, n_corner):
        """Compute Euler spiral parameters for every interior waypoint."""
        corners = []
        for i in range(n_corner):
            p_start  = wps[i]
            p_corner = wps[i + 1]
            p_end    = wps[i + 2]

            # Unit vectors approaching / leaving the corner
            d_cs           = p_corner - p_start
            uv_start       = d_cs / np.linalg.norm(d_cs)
            theta_start    = self._angle_from_vector(uv_start)

            d_ec           = p_end - p_corner
            uv_end         = d_ec / np.linalg.norm(d_ec)
            theta_end      = self._angle_from_vector(uv_end)

            # Local frame: approach direction is +x
            cos_s, sin_s   = math.cos(theta_start), math.sin(theta_start)
            # R_localw rows: [cos, -sin; sin, cos]
            # transform d_cs into local: R^T * d_cs
            p_corner_local = np.array([
                cos_s * d_cs[0] + sin_s * d_cs[1],
               -sin_s * d_cs[0] + cos_s * d_cs[1],
            ])

            theta_end_local = self._wrap_to_pi(theta_end - theta_start)

            turn_dir  = 1 if theta_end_local >= 0 else -1
            delta_phi = abs(theta_end_local)            # total turning angle

            beta               = math.pi - delta_phi
            phi_mid            = 0.5 * delta_phi

            # Euler spiral at mid-point
            cx_mid, cy_mid = self._unit_euler_coord(phi_mid)

            # Minimum viable Euler scaling factor (wheel clearance)
            sin_b2     = math.sin(beta / 2.0)
            a_euler_min = self._l * math.sqrt(2.0 * math.pi * phi_mid)
            eps_min     = (a_euler_min * cy_mid / sin_b2
                           if sin_b2 > 1e-12 else 0.0)

            # C++ sets: epsilon = epsilon_min + 0.1
            epsilon = eps_min + self._eps_offset

            # Euclidean reach along approach/leave directions
            denom  = (sin_b2 / cy_mid) * cx_mid + math.cos(beta / 2.0) \
                     if cy_mid > 1e-12 else 1.0
            Lc      = epsilon * denom
            a_euler = epsilon * sin_b2 / cy_mid if cy_mid > 1e-12 else a_euler_min

            # Clip: corner must not exceed lc_scale fraction of shortest segment
            Lc_max = self._lc_scale * min(np.linalg.norm(d_cs),
                                          np.linalg.norm(d_ec))
            if Lc > Lc_max:
                Lc      = Lc_max
                epsilon = Lc / denom
                a_euler = epsilon * sin_b2 / cy_mid if cy_mid > 1e-12 else a_euler_min

            # Enforce minimum Euler scaling
            if a_euler_min > a_euler:
                a_euler = a_euler_min
                epsilon = (a_euler_min * cy_mid / sin_b2
                           if sin_b2 > 1e-12 else eps_min)
                Lc      = epsilon * denom

            # Corner geometry in local frame
            theta_eps_local = (delta_phi + 0.5 * beta) * turn_dir

            p_sc_local = p_corner_local + Lc * np.array([-1.0, 0.0])
            p_ep_local = p_corner_local + Lc * np.array([
                math.cos(theta_end_local),
                math.sin(theta_end_local),
            ])
            p_mid_local = p_corner_local + epsilon * np.array([
                math.cos(theta_eps_local),
                math.sin(theta_eps_local),
            ])

            # Rotate back to world frame: R_localw * v_local + p_start
            def to_world(v_loc):
                return np.array([
                    cos_s * v_loc[0] - sin_s * v_loc[1],
                    sin_s * v_loc[0] + cos_s * v_loc[1],
                ]) + p_start

            pos_start_c = to_world(p_sc_local)
            pos_end_c   = to_world(p_ep_local)
            pos_mid_c   = to_world(p_mid_local)

            # Corner path velocity (limited by angular velocity and acceleration)
            if phi_mid > 1e-12 and a_euler > 1e-12:
                v_vel = (self._ang_vel_lim * self._r
                         / (1.0 + (self._l / a_euler)
                            * math.sqrt(2.0 * math.pi * phi_mid)))
                v_acc = math.sqrt(self._ang_acc_lim * self._r
                                  * a_euler**2
                                  / (math.pi * self._l))
            else:
                v_vel = self._v_lim
                v_acc = self._v_lim
            v_corner = min(v_vel, v_acc, self._v_lim)

            # Corner arc length and duration
            s_mid    = a_euler * math.sqrt(2.0 * phi_mid / math.pi) \
                       if phi_mid > 0 else 0.0
            T_corner = 2.0 * s_mid / v_corner if v_corner > 1e-12 else 0.0

            corners.append({
                'pos_start':    pos_start_c,
                'pos_end':      pos_end_c,
                'pos_mid':      pos_mid_c,
                'a_euler':      a_euler,
                'phi_mid':      phi_mid,
                'turn_dir':     turn_dir,
                'vel':          v_corner,
                'T_corner':     T_corner,
                'theta_start':  theta_start,   # approach direction angle
                'epsilon':      epsilon,
                'Lc':           Lc,
            })
        return corners

    # ==================================================================
    # PRIVATE — JLAP SOLVER
    # ==================================================================
    def _jlap_accel_phase(self, v_s, a_s, v_e, a_e):
        """Compute time intervals + displacement for one JLAP accel/decel phase.

        Returns (T1, T2, T3, |S|).
        """
        dv = v_e - v_s
        if abs(dv) < 1e-12:
            return 0.0, 0.0, 0.0, 0.0

        sign     = 1 if dv > 0 else -1
        a_max    = self._a_lim * sign
        j        = self._j_lim * sign

        T1 = (a_max - a_s) / j
        T3 = (a_e - a_max) / (-j)
        T2 = (dv
              - 0.5 * T1 * (a_max + a_s)
              - 0.5 * T3 * (a_max + a_e)) / a_max

        if T2 < 0.0:
            a_max = math.sqrt(0.5 * (a_s**2 + a_e**2 + 2.0 * j * dv))
            a_max *= sign
            T1 = (a_max - a_s) / j
            T3 = (a_e - a_max) / (-j)
            T2 = 0.0

        # Accumulate displacement
        # Phase 1
        S1 = v_s * T1 + (1.0 / 6.0) * j * T1**3
        v1 = v_s + 0.5 * j * T1**2
        # Phase 2
        S2 = S1 + v1 * T2 + 0.5 * a_max * T2**2
        v2 = v1 + a_max * T2
        # Phase 3
        S3 = S2 + v2 * T3 + 0.5 * a_max * T3**2 - (1.0 / 6.0) * j * T3**3

        return T1, T2, T3, abs(S3)

    def _jlap_solver(self, p_start, v_start, a_start, p_end, v_end, a_end):
        """Return 7 JLAP time intervals for one straight-line segment."""
        line_len = np.linalg.norm(p_end - p_start)

        v_max = min(1.1 * max(v_start, v_end), self._v_lim)

        # Iterative refinement
        T1a, T2a, T3a, Sa = self._jlap_accel_phase(v_start, a_start, v_max, 0.0)
        T1d, T2d, T3d, Sd = self._jlap_accel_phase(v_max,   0.0,     v_end,  a_end)
        disp = Sa + Sd

        if disp > line_len:
            while disp > line_len and v_max > v_end + 1e-9:
                v_max = max(v_max - self._vel_step, v_end)
                T1a, T2a, T3a, Sa = self._jlap_accel_phase(
                    v_start, a_start, v_max, 0.0)
                T1d, T2d, T3d, Sd = self._jlap_accel_phase(
                    v_max, 0.0, v_end, a_end)
                disp = Sa + Sd
        else:
            while disp < line_len and v_max < self._v_lim - 1e-9:
                v_max = min(v_max + self._vel_step, self._v_lim)
                T1a, T2a, T3a, Sa = self._jlap_accel_phase(
                    v_start, a_start, v_max, 0.0)
                T1d, T2d, T3d, Sd = self._jlap_accel_phase(
                    v_max, 0.0, v_end, a_end)
                disp = Sa + Sd

        cruise_dist = max(line_len - disp, 0.0)
        Tc = cruise_dist / v_max if v_max > 1e-12 else 0.0

        return {
            'T1a': T1a, 'T2a': T2a, 'T3a': T3a,
            'Tc':  Tc,
            'T1d': T1d, 'T2d': T2d, 'T3d': T3d,
            'v_max': v_max,
            'v_start': v_start, 'v_end': v_end,
        }

    # ==================================================================
    # PRIVATE — INTERPOLATION
    # ==================================================================
    def _interpolate(self, wps, corners, jlap_data,
                     line_starts, line_ends):
        """Sample the composite path at self._dt intervals."""
        pos_list   = []
        phi_list   = []
        v_list     = []
        acc_list   = []
        dphi_list  = []
        ddphi_list = []
        t_list     = []

        n_line   = len(jlap_data)
        t_clock  = 0.0

        def _append(pos, phi, v, acc_val, dphi, ddphi, t_val):
            pos_list.append(pos.copy())
            phi_list.append(phi)
            v_list.append(v)
            acc_list.append(acc_val)
            dphi_list.append(dphi)
            ddphi_list.append(ddphi)
            t_list.append(t_val)

        # Initial sample
        _append(wps[0], self._angle_from_vector(
            line_ends[0] - line_starts[0]), self._initial_vel, 0.0, 0.0, 0.0, 0.0)

        for i in range(n_line):
            seg    = jlap_data[i]
            p0     = line_starts[i]
            p1     = line_ends[i]
            dv_seg = p1 - p0
            norm   = np.linalg.norm(dv_seg)
            if norm < 1e-12:
                continue
            uv = dv_seg / norm
            phi_line = self._angle_from_vector(uv)

            T1a = seg['T1a']; T2a = seg['T2a']; T3a = seg['T3a']
            Tc  = seg['Tc']
            T1d = seg['T1d']; T2d = seg['T2d']; T3d = seg['T3d']
            v_max = seg['v_max']
            v_s   = seg['v_start']

            T_a    = T1a + T2a + T3a
            T_d    = T1d + T2d + T3d
            T_line = T_a + Tc + T_d
            Ns_line = max(1, round(T_line / self._dt))

            # Pre-compute boundary vectors (2-D)
            a_pk    = self._j_lim * T1a          # peak accel in phase 1

            vel_not = v_s * uv
            pos_not = p0.copy()

            acc_1a = a_pk * uv
            vel_1a = vel_not + 0.5 * self._j_lim * T1a**2 * uv
            pos_1a = pos_not + vel_not * T1a + (1.0/6) * self._j_lim * T1a**3 * uv

            vel_2a = vel_1a + acc_1a * T2a
            pos_2a = pos_1a + vel_1a * T2a + 0.5 * acc_1a * T2a**2

            vel_3a = vel_2a + acc_1a * T3a - 0.5 * self._j_lim * T3a**2 * uv
            pos_3a = pos_2a + vel_2a * T3a + 0.5 * acc_1a * T3a**2 \
                     - (1.0/6) * self._j_lim * T3a**3 * uv

            pos_c  = pos_3a + vel_3a * Tc

            acc_1d = -self._j_lim * T1d * uv
            vel_1d = vel_3a - 0.5 * self._j_lim * T1d**2 * uv
            pos_1d = pos_c + vel_3a * T1d - (1.0/6) * self._j_lim * T1d**3 * uv

            vel_2d = vel_1d + acc_1d * T2d
            pos_2d = pos_1d + vel_1d * T2d + 0.5 * acc_1d * T2d**2

            for j in range(1, Ns_line + 1):
                t_j = t_clock + j * self._dt
                z   = t_j - t_clock   # time within this line segment

                if z <= T1a:
                    pos_j = pos_not + z * vel_not + (1.0/6) * self._j_lim * z**3 * uv
                    v_j   = np.linalg.norm(vel_not + 0.5 * self._j_lim * z**2 * uv)
                    a_j   = self._j_lim * z
                elif z <= T1a + T2a:
                    zz    = z - T1a
                    pos_j = pos_1a + zz * vel_1a + 0.5 * acc_1a * zz**2
                    v_j   = np.linalg.norm(vel_1a + acc_1a * zz)
                    a_j   = a_pk
                elif z <= T_a:
                    zz    = z - (T1a + T2a)
                    pos_j = pos_2a + zz * vel_2a + 0.5 * acc_1a * zz**2 \
                            - (1.0/6) * self._j_lim * zz**3 * uv
                    v_j   = np.linalg.norm(vel_2a + acc_1a * zz
                                           - 0.5 * self._j_lim * zz**2 * uv)
                    a_j   = a_pk - self._j_lim * zz
                elif z <= T_a + Tc:
                    zz    = z - T_a
                    pos_j = pos_3a + zz * vel_3a
                    v_j   = v_max
                    a_j   = 0.0
                elif z <= T_a + Tc + T1d:
                    zz    = z - (T_a + Tc)
                    pos_j = pos_c + zz * vel_3a \
                            - (1.0/6) * self._j_lim * zz**3 * uv
                    v_j   = np.linalg.norm(vel_3a
                                           - 0.5 * self._j_lim * zz**2 * uv)
                    a_j   = -self._j_lim * zz
                elif z <= T_a + Tc + T1d + T2d:
                    zz    = z - (T_a + Tc + T1d)
                    pos_j = pos_1d + zz * vel_1d + 0.5 * acc_1d * zz**2
                    v_j   = np.linalg.norm(vel_1d + acc_1d * zz)
                    a_j   = -self._j_lim * T1d
                else:
                    zz    = z - (T_a + Tc + T1d + T2d)
                    pos_j = pos_2d + zz * vel_2d + 0.5 * acc_1d * zz**2 \
                            + (1.0/6) * self._j_lim * zz**3 * uv
                    v_j   = np.linalg.norm(vel_2d + acc_1d * zz
                                           + 0.5 * self._j_lim * zz**2 * uv)
                    a_j   = self._j_lim * (zz - T1d)

                _append(pos_j, phi_line, abs(v_j), abs(a_j), 0.0, 0.0, t_j)

            j_c      = len(t_list) - 1
            t_clock += Ns_line * self._dt

            # Corner after this line (if not last segment)
            if i < n_line - 1:
                c          = corners[i]
                a_euler    = c['a_euler']
                phi_mid    = c['phi_mid']
                turn_dir   = c['turn_dir']
                v_c        = c['vel']
                T_c        = c['T_corner']
                theta_s    = c['theta_start']
                p_sc       = c['pos_start']

                cos_s = math.cos(theta_s)
                sin_s = math.sin(theta_s)

                arc_mid = v_c * 0.5 * T_c       # arc length to midpoint
                phi_mid_local = (math.pi / 2.0) * (arc_mid / a_euler)**2 \
                                if a_euler > 1e-12 else 0.0

                Ns_corner = max(1, round(T_c / self._dt))

                pos_local_prev = np.zeros(2)   # used in the leaving-midpoint half
                vel_local_prev = np.zeros(2)
                acc_local_prev = np.zeros(2)

                for j in range(1, Ns_corner + 1):
                    t_j   = t_clock + j * self._dt
                    arc   = v_c * (t_j - t_clock)

                    if t_j <= t_clock + 0.5 * T_c:
                        # Approaching midpoint
                        phi_abs    = (math.pi / 2.0) * (arc / a_euler)**2 \
                                     if a_euler > 1e-12 else 0.0
                        phi_local  = turn_dir * phi_abs
                        dphi_local = (turn_dir * math.pi * v_c * arc
                                      / a_euler**2
                                      if a_euler > 1e-12 else 0.0)
                        ddphi_loc  = (turn_dir * math.pi * (v_c / a_euler)**2
                                      if a_euler > 1e-12 else 0.0)

                        cx, cy        = self._unit_euler_coord(phi_abs)
                        pos_local     = a_euler * np.array([cx, cy * turn_dir])
                        vel_loc_dir   = np.array([math.cos(phi_local),
                                                  math.sin(phi_local)])
                        acc_loc_dir   = np.array([-math.sin(phi_local),
                                                   math.cos(phi_local)])
                        vel_local     = v_c * vel_loc_dir
                        acc_local_vec = v_c * dphi_local * acc_loc_dir

                        pos_local_prev = pos_local.copy()
                        vel_local_prev = vel_local.copy()
                        acc_local_prev = acc_local_vec.copy()
                    else:
                        # Leaving midpoint
                        phi_abs   = (phi_mid_local
                                     + (math.pi / a_euler**2)
                                     * (-0.5 * arc**2
                                        + 2.0 * arc_mid * arc
                                        - 1.5 * arc_mid**2)
                                     if a_euler > 1e-12 else 0.0)
                        phi_local  = turn_dir * phi_abs
                        dphi_local = (turn_dir * math.pi * v_c
                                      * (2.0 * arc_mid - arc)
                                      / a_euler**2
                                      if a_euler > 1e-12 else 0.0)
                        ddphi_loc  = (-turn_dir * math.pi
                                      * (v_c / a_euler)**2
                                      if a_euler > 1e-12 else 0.0)

                        vel_loc_dir   = np.array([math.cos(phi_local),
                                                  math.sin(phi_local)])
                        acc_loc_dir   = np.array([-math.sin(phi_local),
                                                   math.cos(phi_local)])
                        vel_local     = v_c * vel_loc_dir
                        acc_local_vec = v_c * dphi_local * acc_loc_dir

                        # Euler-integrate position from last known
                        pos_local = (pos_local_prev
                                     + vel_local_prev * self._dt
                                     + 0.5 * acc_local_prev * self._dt**2)
                        pos_local_prev = pos_local.copy()
                        vel_local_prev = vel_local.copy()
                        acc_local_prev = acc_local_vec.copy()

                    # Rotate to world frame
                    pos_world = np.array([
                        cos_s * pos_local[0] - sin_s * pos_local[1],
                        sin_s * pos_local[0] + cos_s * pos_local[1],
                    ]) + p_sc

                    phi_world  = phi_local + theta_s
                    _append(pos_world, phi_world, v_c, 0.0,
                            dphi_local, ddphi_loc, t_j)

                t_clock += Ns_corner * self._dt

        # Pack output
        pos_arr = np.array(pos_list)
        phi_arr = np.array(phi_list)
        states  = np.column_stack([pos_arr, phi_arr])   # (N, 3)

        return (states,
                np.array(t_list),
                np.array(v_list),
                np.array(acc_list),
                np.array(dphi_list),
                np.array(ddphi_list))
