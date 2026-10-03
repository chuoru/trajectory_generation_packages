#!/usr/bin/env python3
##
# @file fig_bspline_waypoints.py
#
# @brief Regenerates a standalone schematic figure for the paper explaining
#        how the corner B-spline OCP's three waypoints (extended entry
#        point, corner vertex, extended exit point) are constructed from
#        the PathSegment arc geometry (Section III-A / III-D of the paper).
#
# Pure geometry -- reuses PathSegment and the same world-frame construction
# as differential_drive_comparison.py / differential_drive_path_segment_combined.py
# (beta=90 deg corner at (5,0), L_input=1.0, b_input=0.08, L_TRANSITION=0.5),
# but does NOT solve the OCP: no casadi dependency, runs in well under a
# second.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/08/19

import sys
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trajectory_generators.path_segment import PathSegment

FIG_OUT_DIR = (pathlib.Path(__file__).resolve().parent.parent.parent
               / 'Writting' / 'energy_aware')

# Same scenario as differential_drive_comparison.py (paper's evaluation corner).
WP_START, WP_CORNER, WP_END = [0.0, 0.0], [5.0, 0.0], [5.0, 5.0]
HEADING_IN, HEADING_OUT = 0.0, np.pi / 2
BETA = np.pi / 2
L_TRANSITION = 0.5
L_INPUT, B_INPUT, V_MAX_SEG, A_MAX_SEG, L_WHEELBASE = 1.0, 0.08, 0.5, 1.0, 0.53

COL_ARC, COL_LEG, COL_CORRIDOR = 'purple', 'gray', 'khaki'


def _segment_corner():
    ps = PathSegment(beta=BETA, L_input=L_INPUT, b_input=B_INPUT,
                      v_max=V_MAX_SEG, a_max=A_MAX_SEG,
                      L_wheelbase=L_WHEELBASE, n_samples=200)
    res = ps.generate_segment()

    L_seg = res['L_seg']
    corner = np.array(WP_CORNER, dtype=float)
    in_dir = np.array([np.cos(HEADING_IN), np.sin(HEADING_IN)])
    out_dir = np.array([np.cos(HEADING_OUT), np.sin(HEADING_OUT)])
    arc_entry = corner - L_seg * in_dir

    c, s = np.cos(HEADING_IN), np.sin(HEADING_IN)
    R_mat = np.array([[c, -s], [s, c]])
    arc_local = res['path_segment']
    xy_world = (R_mat @ arc_local[:, :2].T).T + arc_entry
    arc_world = xy_world
    arc_exit = arc_world[-1].copy()

    arc_entry_ext = arc_entry - L_TRANSITION * in_dir
    arc_exit_ext = arc_exit + L_TRANSITION * out_dir

    res.update(arc_world=arc_world, arc_entry=arc_entry, arc_exit=arc_exit,
               arc_entry_ext=arc_entry_ext, arc_exit_ext=arc_exit_ext,
               corner=corner, in_dir=in_dir, out_dir=out_dir)
    return res


def _dim_line(ax, p0, p1, text, offset, color='black'):
    """Double-headed dimension arrow from p0 to p1, offset perpendicular to
    the segment, annotated with `text` at its midpoint."""
    p0, p1 = np.array(p0), np.array(p1)
    d = p1 - p0
    L = np.hypot(*d)
    if L < 1e-9:
        return
    n = np.array([-d[1], d[0]]) / L
    q0, q1 = p0 + offset * n, p1 + offset * n
    ax.annotate('', xy=q1, xytext=q0,
                arrowprops=dict(arrowstyle='<->', color=color, lw=1.3))
    ax.plot([p0[0], q0[0]], [p0[1], q0[1]], color=color, lw=0.6, ls=':')
    ax.plot([p1[0], q1[0]], [p1[1], q1[1]], color=color, lw=0.6, ls=':')
    mid = (q0 + q1) / 2
    ax.annotate(text, xy=mid, xytext=mid + 0.10 * n / max(np.hypot(*n), 1e-9),
                ha='center', va='center', fontsize=10, color=color)


def main():
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12,
                          'legend.fontsize': 9.5, 'axes.grid': False})
    seg = _segment_corner()
    corner = seg['corner']
    ae, axx = seg['arc_entry'], seg['arc_exit']
    ae_ext, ax_ext = seg['arc_entry_ext'], seg['arc_exit_ext']
    arc = seg['arc_world']

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.set_aspect('equal')

    # Reference straight legs (only the portion near the corner).
    leg_in_far = corner - seg['in_dir'] * (L_INPUT + L_TRANSITION + 0.9)
    leg_out_far = corner + seg['out_dir'] * (L_INPUT + L_TRANSITION + 0.9)
    ax.plot([leg_in_far[0], corner[0]], [leg_in_far[1], corner[1]], '--',
            color=COL_LEG, lw=1.3, zorder=1, label='Reference path')
    ax.plot([corner[0], leg_out_far[0]], [corner[1], leg_out_far[1]], '--',
            color=COL_LEG, lw=1.3, zorder=1)

    # Coverage corridor band around the two corridor segments
    # (entry-ext -> vertex, vertex -> exit-ext), matching eq:corridor.
    d_max = 0.25
    for p0, p1 in [(ae_ext, corner), (corner, ax_ext)]:
        p0a, p1a = np.array(p0), np.array(p1)
        d = p1a - p0a
        Lp = np.hypot(*d)
        n = np.array([-d[1], d[0]]) / Lp
        quad = np.array([p0a + d_max * n, p1a + d_max * n,
                         p1a - d_max * n, p0a - d_max * n])
        ax.fill(quad[:, 0], quad[:, 1], color=COL_CORRIDOR, alpha=0.45,
                zorder=0,
                label=(r'Coverage corridor ($d_{max}$)' if p0 is ae_ext
                       else None))

    # Inscribed arc.
    ax.plot(arc[:, 0], arc[:, 1], '-', color=COL_ARC, lw=2.2, zorder=3,
            label=f'Inscribed arc ($R$={seg["R"]:.2f} m)')

    # Corner vertex.
    ax.plot(*corner, 'o', color='black', ms=9, zorder=6,
            label='Corner vertex')

    # Arc tangent points.
    ax.plot(*ae, 's', color=COL_ARC, ms=8, zorder=6,
            label='Arc tangent points')
    ax.plot(*axx, 's', color=COL_ARC, ms=8, zorder=6)

    # Extended entry / exit points -- the spline's other two waypoints.
    ax.plot(*ae_ext, 'D', color='crimson', ms=8, zorder=6,
            label=r'Extended entry/exit points ($\mathbf{p}^{ext}$)')
    ax.plot(*ax_ext, 'D', color='crimson', ms=8, zorder=6)

    # Text labels for the three B-spline waypoints.
    ax.annotate(r'$\mathbf{p}_{entry}^{ext}$', xy=ae_ext,
                xytext=ae_ext + np.array([-0.35, -0.35]), fontsize=12)
    ax.annotate(r'corner vertex', xy=corner,
                xytext=corner + np.array([0.12, -0.30]), fontsize=12)
    ax.annotate(r'$\mathbf{p}_{exit}^{ext}$', xy=ax_ext,
                xytext=ax_ext + np.array([0.10, 0.05]), fontsize=12)

    # Dimension annotations: L_seg (vertex -> tangent point) and
    # L_T (tangent point -> extended point), on each leg. Entry-leg
    # dimensions sit above the path (clear of the beta arc below the
    # vertex); exit-leg dimensions sit to the right of the path.
    _dim_line(ax, corner, ae, f'$L_{{seg}}$={seg["L_seg"]:.2f} m',
              offset=-0.35, color='black')
    _dim_line(ax, ae, ae_ext, f'$L_T$={L_TRANSITION:.2f} m',
              offset=-0.70, color='dimgray')
    _dim_line(ax, corner, axx, f'$L_{{seg}}$={seg["L_seg"]:.2f} m',
              offset=-0.35, color='black')
    _dim_line(ax, axx, ax_ext, f'$L_T$={L_TRANSITION:.2f} m',
              offset=-0.70, color='dimgray')

    # Turning-angle beta arc at the vertex (bottom-left quadrant, clear of
    # both dimension lines).
    ang0, ang1 = np.degrees(HEADING_IN + np.pi), np.degrees(HEADING_OUT + np.pi)
    theta = np.linspace(np.radians(ang0), np.radians(ang1), 40)
    beta_r = 0.3
    ax.plot(corner[0] + beta_r * np.cos(theta),
            corner[1] + beta_r * np.sin(theta), color='black', lw=1.0)
    ax.annotate(r'$\beta$', xy=corner + beta_r * 1.5 *
                np.array([np.cos(np.mean(theta)), np.sin(np.mean(theta))]),
                fontsize=12, ha='center', va='center')

    ax.set_xlim(corner[0] - L_INPUT - L_TRANSITION - 0.9, corner[0] + 1.0)
    ax.set_ylim(corner[1] - 1.0, corner[1] + L_INPUT + L_TRANSITION + 0.9)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    fig.tight_layout()

    out = FIG_OUT_DIR / 'fig_bspline_waypoints.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'[paper] Saved fig_bspline_waypoints.png -> {out}')
    print(f'  L_seg={seg["L_seg"]:.4f} m  R={seg["R"]:.4f} m  '
          f'b={seg["b"]:.4f} m  feasible={seg["feasible"]}')
    print(f'  arc_entry={ae}  arc_entry_ext={ae_ext}')
    print(f'  arc_exit={axx}  arc_exit_ext={ax_ext}')


if __name__ == '__main__':
    main()
