#!/usr/bin/env python3
##
# @file path_segment.py
#
# @brief Kinematically feasible circular arc segment for corner smoothing on
#        a differential-drive robot.
#
# Replaces a sharp corner (characterised by turning angle beta) with a
# circular arc whose radius is derived from centripetal-acceleration and
# robot-body clearance constraints.  The output is expressed in a local
# 2-D frame (arc entry at origin, incoming direction along +x) so that the
# caller can apply an arbitrary rigid-body transform to place it in the world
# frame.
#
# Local frame convention
# ----------------------
#   Arc entry : (0, 0),  heading 0  (incoming along +x)
#   Arc centre : (0, R)             (left of incoming, CCW / left turn)
#   Arc point at angle phi in [0, beta]:
#       x(phi) = R * sin(phi)
#       y(phi) = R * (1 - cos(phi))
#       heading(phi) = phi          (exact tangent direction)
#
# For a right-turn (CW) the caller negates the y and heading columns of the
# returned path_segment array.
#
# Geometry note
# -------------
# L_seg is the *standoff* distance (distance from the corner vertex to the
# arc tangent point along each straight segment), not the arc length.
# The exact inscribed-circle radius is:
#       R = L_seg / tan(beta / 2)
# The spec's "optional" approximation R = L_seg / beta is also returned for
# reference (it corresponds to tan(beta/2) ≈ beta/2, valid only for very
# small beta).
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/04/22

# Standard library
import numpy as np


class PathSegment:
    """! Circular arc segment for kinematically feasible corner smoothing.

    Computes a minimum-radius feasibility window from robot motion limits,
    clamps the caller-supplied parameters into that window, and generates a
    sampled circular arc in a local 2-D frame.

    The output path_segment (n_samples, 3) array contains [x, y, heading]
    columns in the local frame and can be transformed to the world frame by
    the caller with a 2-D rotation and translation.
    """

    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    def __init__(self, beta, L_input, b_input, v_max, a_max,
                 L_wheelbase, n_samples=50):
        """! Constructor.

        @param beta<float>: Corner turning angle [rad]. Must be in (0, pi).
        @param L_input<float>: Available standoff from corner vertex to arc
            tangent point [m]. Must be > 0.
        @param b_input<float>: Desired corner deviation tolerance [m].
            Must be > 0.
        @param v_max<float>: Maximum robot path velocity [m/s]. Must be > 0.
        @param a_max<float>: Maximum lateral (centripetal) acceleration
            [m/s^2]. Must be > 0.
        @param L_wheelbase<float>: Robot wheelbase (full axle-to-axle) [m].
            Must be > 0.
        @param n_samples<int>: Number of sample points along the arc
            (default 50). Must be >= 2.
        """
        if not (0.0 < beta < np.pi):
            raise ValueError("beta must be in (0, pi) [rad].")
        if L_input <= 0.0:
            raise ValueError("L_input must be positive [m].")
        if b_input <= 0.0:
            raise ValueError("b_input must be positive [m].")
        if v_max <= 0.0:
            raise ValueError("v_max must be positive [m/s].")
        if a_max <= 0.0:
            raise ValueError("a_max must be positive [m/s^2].")
        if L_wheelbase <= 0.0:
            raise ValueError("L_wheelbase must be positive [m].")
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2.")

        self._beta = float(beta)
        self._L_input = float(L_input)
        self._b_input = float(b_input)
        self._v_max = float(v_max)
        self._a_max = float(a_max)
        self._L_wheelbase = float(L_wheelbase)
        self._n_samples = int(n_samples)

    def generate_segment(self):
        """! Build the feasible circular arc segment.

        Executes the four algorithmic steps: bounds computation, feasibility
        clamping, arc generation, and result assembly.

        @return dict with keys:
            - 'L_seg'       : float  — clamped standoff distance [m]
            - 'b'           : float  — clamped deviation tolerance [m]
            - 'feasible'    : bool   — True iff both inputs were within bounds
            - 'path_segment': (n_samples, 3) ndarray float64 — [x, y, heading]
                              in local frame [m, m, rad]
            - 'R'           : float  — exact arc radius [m]
            - 'R_approx'    : float  — approximate radius L_seg/beta [m]
            - 'L_min'       : float  — minimum feasible standoff [m]
            - 'b_max'       : float  — maximum feasible deviation [m]
        """
        L_min, b_max = self._compute_bounds()
        L_seg, b, feasible = self._clamp(L_min, b_max)
        R = L_seg / np.tan(self._beta / 2.0)
        path_segment = self._build_arc(R)

        return {
            'L_seg':        L_seg,
            'b':            b,
            'feasible':     feasible,
            'path_segment': path_segment,
            'R':            R,
            'R_approx':     L_seg / self._beta,
            'L_min':        L_min,
            'b_max':        b_max,
        }

    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    def _compute_bounds(self):
        """! Compute kinematic feasibility bounds (Step 1).

        Minimum standoff derives from the centripetal-acceleration limit:
            R_min = v_max^2 / a_max
            L_min = R_min * tan(beta / 2)

        Maximum deviation derives from the robot-body inscribed-circle
        constraint for the wheelbase square (exact for a 90-degree corner):
            b_max = L_wheelbase / (2 * sqrt(2))

        @return Tuple (L_min, b_max), both float [m].
        """
        R_min = self._v_max ** 2 / self._a_max
        L_min = float(R_min * np.tan(self._beta / 2.0))
        b_max = float(self._L_wheelbase / (2.0 * np.sqrt(2.0)))
        return L_min, b_max

    def _clamp(self, L_min, b_max):
        """! Clamp inputs into the feasible window (Step 2).

        L_seg = max(L_input, L_min)   — never shrink below minimum standoff
        b     = min(b_input, b_max)   — never exceed body clearance limit
        feasible = True iff no clamping was applied.

        @param L_min<float>: Minimum feasible standoff [m].
        @param b_max<float>: Maximum feasible deviation [m].
        @return Tuple (L_seg, b, feasible) — float, float, bool.
        """
        L_seg = max(self._L_input, L_min)
        b = min(self._b_input, b_max)
        feasible = (self._L_input >= L_min) and (self._b_input <= b_max)
        return L_seg, b, feasible

    def _build_arc(self, R):
        """! Sample the circular arc in local frame (Step 3).

        At arc parameter phi in [0, beta]:
            x(phi)       = R * sin(phi)
            y(phi)       = R * (1 - cos(phi))
            heading(phi) = phi

        The arc centre is at (0, R), left of the incoming direction.  Entry
        and exit tangent vectors align with the straight segments, guaranteeing
        G1 (heading) continuity.  Curvature is constant at 1/R throughout.

        @param R<float>: Arc radius [m].
        @return ndarray shape (n_samples, 3), dtype float64. Columns: x, y,
                heading [m, m, rad].
        """
        phi = np.linspace(0.0, self._beta, self._n_samples)
        x = R * np.sin(phi)
        y = R * (1.0 - np.cos(phi))
        return np.column_stack([x, y, phi])
