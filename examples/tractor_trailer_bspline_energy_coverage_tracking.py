#!/usr/bin/env python3
##
# @file tractor_trailer_bspline_energy_coverage_tracking.py
#
# @brief Example: closed-loop Pure Pursuit tracking of a tractor-trailer
# B-Spline coverage trajectory, with explicit hitch-angle feedback.
#
# Companion to tractor_trailer_bspline_energy_coverage.py. That example only
# generates and plots the OPEN-LOOP trajectory. This one closes the loop:
# only the tractor is actuated, so a tractor-pose-only controller (plain
# PurePursuit) leaves the hitch angle gamma fully PASSIVE during tracking --
# gammadot = thetadot_trailer - w_tractor is never observed or targeted, so
# tracking-time disturbance can let gamma drift toward the mechanical
# jackknife limit even while tractor position/heading track well. This was
# found and fixed once already in export_section5_tractor_trailer.py (gamma
# drifted to 82-86 deg against a 44.98 deg limit without a fix); this
# example demonstrates the two complementary, now-reusable fixes:
#
#   (A) Trajectory generation: BSplineEnergyTractorTrailerCoverage's new
#       gamma_margin / gamma_rate_max parameters plan a gamma(t) that stays
#       clear of gamma_max and doesn't demand large hitch rates.
#   (B) Tracking: controllers.tractor_trailer_adapters.
#       HitchStabilizedController feeds the OCP's own planned gamma(t) back
#       into the tractor's commanded turn rate via measured hitch angle.
#
# Three scenarios are simulated against the SAME corridor/waypoints:
#   (a) baseline    -- unmodified trajectory, tractor-pose-only tracking.
#   (b) margin-only -- gamma_margin/gamma_rate_max trajectory, same
#                       tractor-pose-only tracking (no hitch feedback).
#   (c) full fix    -- margin/rate-bounded trajectory + hitch feedback.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/08/12

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trajectory_generators.bspline_energy_tractor_trailer_coverage import (
    BSplineEnergyTractorTrailerCoverage,
)
from controllers.trajectory import Trajectory
from controllers.purepursuit import PurePursuit
from controllers.tractor_trailer_adapters import (
    TrailerToTractorPoseAdapter, HitchStabilizedController,
)
from models.tractor_trailer_articulated import TractorTrailerArticulated
from simulators.time_stepping import TimeStepping

# ---------------------------------------------------------------------------
# Waypoints for the TRAILER rear axle  [x, y, theta_trailer]  -- same
# scenario as tractor_trailer_bspline_energy_coverage.py.
# ---------------------------------------------------------------------------
WAYPOINTS = [
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [2.0, 2.0, np.pi / 2],
]

LB = 0.2   # tractor rear axle -> hitch [m]
LF = 0.8   # hitch -> trailer rear axle [m]
GAMMA_MAX = 0.785

COMMON = dict(
    waypoints=WAYPOINTS,
    bound=0.17,
    n_ctrl_pts=5,
    spline_order=3,
    n_sampling=20,
    vel_max=[0.2, 0.2, 0.196],
    vel_min_lin=0.01,
    eps_nonh=0.001,
    eps_hitch=0.05,
    length_back=LB,
    length_front=LF,
    gamma_max=GAMMA_MAX,
    gamma_entry=0.0,
    gamma_exit=0.0,
    acc_max=[5.0, 5.0, 4.0],
    jerk_max=[50.0, 50.0, 20.0],
    p_electronics=2.0,
    w_time=1.0, w_energy=1.0,
)

# Pure Pursuit gains -- same validated operating point used across the
# repo's other tractor-trailer tracking scripts (purepursuit_fine_sweep.py,
# export_section5_tractor_trailer.py).
PurePursuit.lookahead_distance = 0.25
PurePursuit.lookahead_gain = 0.4
PurePursuit.k = 6.0
PurePursuit.k_i = 0.1
PurePursuit.k_ff = 1.0

# Hitch feedback gain. K_GAMMA=0.3 is what export_section5_tractor_trailer.py
# validated, but that script always tracks a corner WITH gentle JLAP
# straight-line lead-in/lead-out around it, keeping tractor heading error
# small throughout. This example tracks the raw, unstitched corner alone
# (same scenario as tractor_trailer_bspline_energy_coverage.py) -- a harder
# tracking problem where PurePursuit's own tractor-pose error is already
# larger, so pushing K_GAMMA as high as 0.3 buys a bigger max|gamma|
# reduction at a steeper tractor-position-tracking cost. K_GAMMA=0.2 is a
# more moderate operating point for this harsher scenario, still giving a
# clear, monotonic max|gamma| improvement (the metric that actually governs
# jackknife safety) without dominating the tractor-pose tracking loop.
K_GAMMA = 0.2
MAX_GAMMA_CORRECTION = 1.0

DT = 0.05

# ---------------------------------------------------------------------------
# Solve: baseline (no jackknife margin, no hitch-rate bound) ...
# ---------------------------------------------------------------------------
print("Solving baseline trajectory (no gamma margin / rate bound) ...")
gen_base = BSplineEnergyTractorTrailerCoverage(**COMMON)
res_base = gen_base.generate_trajectory()

# ... and improved (Part A: jackknife margin + hitch-rate bound), warm
# started from the baseline solution.
print("Solving margin-aware trajectory (gamma_margin=0.1, "
      "gamma_rate_max=0.5 rad/s) ...")
gen_imp = BSplineEnergyTractorTrailerCoverage(
    **COMMON, gamma_margin=0.1, gamma_rate_max=0.5)
res_imp = gen_imp.generate_trajectory(warm_start=res_base)


# ---------------------------------------------------------------------------
# Build a tractor-frame Trajectory + gamma(t) reference from an OCP result.
# ---------------------------------------------------------------------------
def build_tractor_reference(res, dt=DT):
    """! Resample an OCP result onto a uniform-dt grid and derive the
    tractor-frame reference (what PurePursuit actually tracks) via forward
    kinematics, same geometry as the open-loop example's tractor_xy().
    @param res<dict>: generate_trajectory() output.
    @param dt<float>: Uniform sampling interval [s].
    @return<dict>: 'traj_tractor' (Trajectory), 'traj_trailer' (Trajectory),
        'gamma_ref' (array, same time grid/index as both trajectories).
    """
    t_state = res['time']
    s = res['states']
    t_ik = res['time_ik']

    t_end = min(t_state[-1], t_ik[-1])
    t_uni = np.arange(0.0, t_end, dt)

    x_u = np.interp(t_uni, t_state, s[:, 0])
    y_u = np.interp(t_uni, t_state, s[:, 1])
    th_u = np.interp(t_uni, t_state, s[:, 2])
    gamma_u = np.interp(t_uni, t_state, s[:, 3])
    v_u = np.interp(t_uni, t_ik, res['v'])
    w_u = np.interp(t_uni, t_ik, res['omega'])

    xt_u = x_u + LF * np.cos(th_u) + LB * np.cos(th_u - gamma_u)
    yt_u = y_u + LF * np.sin(th_u) + LB * np.sin(th_u - gamma_u)
    tht_u = th_u - gamma_u

    traj_tractor = Trajectory(
        x=np.column_stack([xt_u, yt_u, tht_u]),
        u=np.vstack([v_u, w_u]), t=t_uni, sampling_time=dt,
    )
    traj_trailer = Trajectory(
        x=np.column_stack([x_u, y_u, th_u]),
        u=np.vstack([v_u, w_u]), t=t_uni, sampling_time=dt,
    )
    return {
        'traj_tractor': traj_tractor,
        'traj_trailer': traj_trailer,
        'gamma_ref': gamma_u,
        'initial_state': [x_u[0], y_u[0], th_u[0], gamma_u[0]],
    }


ref_base = build_tractor_reference(res_base)
ref_imp = build_tractor_reference(res_imp)


# ---------------------------------------------------------------------------
# Run a closed-loop tracking scenario.
# ---------------------------------------------------------------------------
def run_scenario(ref, k_gamma):
    """! Simulate the articulated plant tracking ref['traj_tractor'] with
    PurePursuit, optionally wrapped in hitch-angle feedback.
    @param ref<dict>: build_tractor_reference() output.
    @param k_gamma<float>: Hitch feedback gain. 0.0 disables feedback (gamma
        fully passive).
    @return<TimeStepping>: The completed simulation.
    """
    traj_tractor = ref['traj_tractor']
    model = TractorTrailerArticulated(length_back=LB, length_front=LF)

    inner = PurePursuit(model, traj_tractor)
    pose_ctrl = TrailerToTractorPoseAdapter(
        inner, length_back=LB, length_front=LF)

    if k_gamma > 0.0:
        controller = HitchStabilizedController(
            pose_ctrl, gamma_ref=ref['gamma_ref'], k_gamma=k_gamma,
            max_correction=MAX_GAMMA_CORRECTION,
        )
    else:
        controller = pose_ctrl

    sim = TimeStepping(model, float(traj_tractor.t[-1]), traj_tractor.sampling_time)
    sim.run_with_controller(ref['initial_state'], traj_tractor, controller)
    return sim


print("\nSimulating (a) baseline: unmodified trajectory, no hitch feedback ...")
sim_a = run_scenario(ref_base, k_gamma=0.0)

print("Simulating (b) margin-aware trajectory, no hitch feedback ...")
sim_b = run_scenario(ref_imp, k_gamma=0.0)

print("Simulating (c) margin-aware trajectory + hitch feedback ...")
sim_c = run_scenario(ref_imp, k_gamma=K_GAMMA)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def summarize(sim, ref, label):
    nt = min(sim.x_out.shape[1], len(ref['gamma_ref']))
    gamma_sim = sim.x_out[3, :nt]
    gamma_ref = ref['gamma_ref'][:nt]
    gamma_rmse = np.sqrt(np.mean((gamma_sim - gamma_ref) ** 2))
    pos_err = np.hypot(
        sim.x_out[0, :nt] - ref['traj_trailer'].x[:nt, 0],
        sim.x_out[1, :nt] - ref['traj_trailer'].x[:nt, 1],
    )
    print(f"{label:28s} "
          f"max|gamma|={np.rad2deg(np.abs(gamma_sim).max()):6.2f} deg  "
          f"gamma_rmse={np.rad2deg(gamma_rmse):6.2f} deg  "
          f"mean_pos_err={pos_err.mean() * 100:6.2f} cm  "
          f"max_pos_err={pos_err.max() * 100:6.2f} cm")
    return gamma_sim, gamma_ref


print(f"\n{'':28s} {'max|gamma|':>12s}  {'gamma RMSE':>12s}  "
      f"{'mean pos err':>14s}  {'max pos err':>13s}")
gamma_a, gref_a = summarize(sim_a, ref_base, "(a) baseline")
gamma_b, gref_b = summarize(sim_b, ref_imp, "(b) margin-only")
gamma_c, gref_c = summarize(sim_c, ref_imp, "(c) margin + hitch feedback")

print(
    "\nNote: max|gamma| is the safety-relevant metric here -- it is what "
    "determines jackknife margin against gamma_max. It should decrease "
    "monotonically from (a) to (b) to (c). gamma_rmse and position error "
    "can rise slightly under hitch feedback: this scenario tracks a raw, "
    "unstitched corner (no gentle straight lead-in/lead-out around it, "
    "unlike export_section5_tractor_trailer.py's validated pipeline), so "
    "tractor-pose tracking error is already large, and pulling gamma toward "
    "its reference trades off against pose tracking -- an expected "
    "consequence of only having 2 actuated DOF for 4 coupled states, not a "
    "malfunction. K_GAMMA can be tuned higher for a bigger max|gamma| cut "
    "at a steeper tracking-error cost, or lower for the opposite trade.")

# ---------------------------------------------------------------------------
# Figure -- hitch angle tracking across the three scenarios.
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

ax1.plot(sim_a.t_out[:len(gamma_a)], np.rad2deg(gamma_a), 'r-', lw=1.5,
         label='(a) baseline')
ax1.plot(sim_b.t_out[:len(gamma_b)], np.rad2deg(gamma_b), 'b--', lw=1.5,
         label='(b) margin-only')
ax1.plot(sim_c.t_out[:len(gamma_c)], np.rad2deg(gamma_c), 'g-', lw=1.5,
         label='(c) margin + hitch feedback')
ax1.plot(ref_base['traj_tractor'].t, np.rad2deg(ref_base['gamma_ref']),
         'k:', lw=1.0, alpha=0.6, label='(a) planned gamma_ref')
ax1.plot(ref_imp['traj_tractor'].t, np.rad2deg(ref_imp['gamma_ref']),
         'k-.', lw=1.0, alpha=0.6, label='(b,c) planned gamma_ref')
ax1.axhline(np.rad2deg(GAMMA_MAX), color='k', ls=':', lw=1)
ax1.axhline(-np.rad2deg(GAMMA_MAX), color='k', ls=':', lw=1, label='±gamma_max (mechanical)')
ax1.set_xlabel('time [s]'); ax1.set_ylabel('gamma [deg]')
ax1.set_title('Closed-Loop Hitch Angle Tracking')
ax1.legend(fontsize=8, ncol=2); ax1.grid(True, ls=':', alpha=0.5)

for sim, ref, lbl, ls in [
        (sim_a, ref_base, '(a) baseline', 'r-'),
        (sim_b, ref_imp, '(b) margin-only', 'b--'),
        (sim_c, ref_imp, '(c) margin + hitch feedback', 'g-')]:
    nt = min(sim.x_out.shape[1], ref['traj_trailer'].x.shape[0])
    pos_err = np.hypot(
        sim.x_out[0, :nt] - ref['traj_trailer'].x[:nt, 0],
        sim.x_out[1, :nt] - ref['traj_trailer'].x[:nt, 1],
    )
    ax2.plot(sim.t_out[:nt], pos_err * 100, ls, lw=1.5, label=lbl)
ax2.set_xlabel('time [s]'); ax2.set_ylabel('trailer position error [cm]')
ax2.set_title('Trailer Tracking Error (sanity check)')
ax2.legend(fontsize=8); ax2.grid(True, ls=':', alpha=0.5)

fig.tight_layout()
plt.show()
