"""! Export closed-loop tracking data for the tractor-trailer robot, tracked
by a genuinely ARTICULATED plant (hitch angle propagated through closed-loop
tracking), instead of the DifferentialDrive stand-in used by
tractor_trailer_path_segment_combined.py's own Pure Pursuit demo.

A second simulated sensor -- a rotary hitch-angle IMU/encoder, the exact
instrumentation the paper's own future-work section proposes for hardware
validation (main.tex, sec:conclusion: "a passive trailer, connected through
an instrumented hitch (a rotary hitch-angle sensor)") -- closes an explicit
feedback loop on gamma. Without it, gamma is a purely PASSIVE, unobserved
internal state: tracking only the tractor's pose (necessary, since only the
tractor is actuated) leaves nothing in the loop noticing gamma drift away
from its planned value, which was found to reach 82-86 deg against a
44.98 deg mechanical limit. The hitch IMU lets a proportional correction
nudge the tractor's commanded turn rate to pull gamma back toward its
planned reference, layered additively on top of (not replacing) the
existing tractor-pose-tracking command.

Mirrors export_section5_baseline.py's export format (raw per-seed metrics,
seed-0 timeseries, summary stats, metadata), but for two tractor-trailer
corner methods:
  - 'T': time-optimal corner (w_e=0.0)
  - 'E': the tractor-trailer paper's own VALIDATED energy-aware operating
         point (w_e=0.1, Writting/energy_aware_trailer_tractor/main.tex,
         ssec:res_pareto / Table tab:pareto) -- not a freshly computed
         Pareto-knee, which the paper found fragile for this problem.

The corner OCP is solved with the paper-validated settings (Table
tab:params / tractor_trailer_paper_sweeps.py's common_kwargs), NOT
tractor_trailer_path_segment_combined.py's own _make_bspline_tt defaults,
which the paper found failed to converge for the energy-aware corner.

Both methods reuse the S1(JLAP)->corner(B-spline OCP)->S2(JLAP) pipeline
structure and corridor scenario from tractor_trailer_path_segment_combined.py
(the same "three-stage pipeline" the paper's own tracking figures come from).

Output directory: csv_output_fine_sweep/section5_tractor_trailer_export/
"""
import os
import json
import datetime

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import casadi.casadi as cs

import tractor_trailer_path_segment_combined as ttp
import differential_drive_comparison as dc
import purepursuit_fine_sweep as pps
import section5_purepursuit as s5

from trajectory_generators.bspline_energy_tractor_trailer_coverage import (
    BSplineEnergyTractorTrailerCoverage,
)
from controllers.trajectory import Trajectory
from controllers.purepursuit import PurePursuit
from simulators.time_stepping import TimeStepping

OUT_DIR = os.path.join(os.path.dirname(__file__), 'csv_output_fine_sweep',
                       'section5_tractor_trailer_export')
os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS = s5.N_SEEDS

# Paper-validated corner-OCP settings -- Writting/energy_aware_trailer_tractor/
# main.tex, Table tab:params / tractor_trailer_paper_sweeps.py's common_kwargs.
# The tractor's l/r match the companion diff-drive paper's identified platform,
# so the power model below is reused unmodified rather than re-derived.
ROBOT_PARAMS_TT = {'l': 0.265, 'r': 0.15}
N_SAMPLING = 10
N_CTRL_PTS = 5
MAX_ITER = 1500
METHOD_WE = {'T': 0.0, 'E': 0.1}

# Hitch-angle IMU/rotary encoder + gamma-stabilization feedback gain.
# HITCH_IMU_STD_GAMMA: comparable precision to a rotary encoder (tighter
# than the heading-AHRS noise pps.IMU_STD_THETA=0.01 rad, since a hitch
# encoder is a direct mechanical angle reading, not a fused heading
# estimate). K_GAMMA: proportional gain [1/s] converting a gamma tracking
# error [rad] into an ADDITIVE correction on the tractor's commanded turn
# rate [rad/s] -- see _HitchStabilizedController for the sign derivation
# and MAX_GAMMA_CORRECTION for the clamp.
HITCH_IMU_STD_GAMMA = 0.01
K_GAMMA = 0.3
MAX_GAMMA_CORRECTION = 1.0


# =============================================================================
# CORNER OCP -- paper-validated settings (NOT ttp._solve_corner_tt's defaults)
# =============================================================================
def _solve_corner_paper(waypoints, w_energy, warm_start=None,
                        v_entry=None, v_exit=None, a_entry=0.0, alpha_entry=0.0):
    v_e = float(v_entry) if v_entry is not None else ttp.V_HANDOFF
    v_x = float(v_exit) if v_exit is not None else ttp.V_HANDOFF
    gen = BSplineEnergyTractorTrailerCoverage(
        waypoints=waypoints, bound=0.25, n_ctrl_pts=N_CTRL_PTS, spline_order=3,
        n_sampling=N_SAMPLING, vel_max=[v_e, v_e, 0.196], vel_min_lin=0.01,
        eps_nonh=0.001, v_entry=v_e, v_exit=v_x,
        a_entry=a_entry, alpha_entry=alpha_entry,
        acc_max=[5.0, 5.0, 4.0], jerk_max=[50.0, 50.0, 20.0], eps_hitch=0.05,
        length_back=ttp.LB, length_front=ttp.LF, gamma_max=ttp.GAMMA_MAX,
        gamma_entry=0.0, gamma_exit=0.0, p_electronics=2.0,
        robot_params=ROBOT_PARAMS_TT, w_time=1.0, w_energy=float(w_energy),
    )
    gen._optimizer.solver(
        'ipopt', {'print_time': False},
        {'max_iter': MAX_ITER, 'print_level': 0, 'tol': 1e-5,
         'acceptable_tol': 5e-3, 'acceptable_iter': 15,
         'constr_viol_tol': 1e-4, 'hessian_approximation': 'limited-memory'})
    return gen.generate_trajectory(warm_start=warm_start)


def _jittered_ctrl_pts(res, scale=0.12, rng=None):
    """Warm-start dict for generate_trajectory: needs both 'ctrl_pts' (here
    perturbed) and 'time' (used to seed the per-node time-scaling variable
    T, kept unperturbed -- see BSplineEnergyTractorTrailerCoverage.
    generate_trajectory, lines ~198-207)."""
    rng = rng if rng is not None else np.random.default_rng()
    cp = np.asarray(res['ctrl_pts'])
    sigma = scale * (np.abs(cp).mean() + 1e-6)
    return {'ctrl_pts': cp + rng.normal(0.0, sigma, size=cp.shape),
            'time': res['time']}


def _corner_objective(res, w_energy):
    """Proxy for the OCP's own w_time*(sum(T)/nt) + w_energy*E_motor
    objective, using total mission time as a stand-in for sum(T)/nt (not
    returned directly) -- good enough to RANK two candidates at the same
    w_e against each other."""
    return float(res['time'][-1]) + float(w_energy) * float(res['energy'])


def _solve_corner_multistart(waypoints, w_energy, warm_start=None,
                             extra_warm_starts=(), seed=0, n_jitter=1, **kwargs):
    """Cold start + n_jitter jittered restarts (main.tex "Multi-start
    solving": n_jitter=1 is the paper's own default budget, n_jitter=3 its
    "broader" budget used for harder-to-converge configurations), plus any
    extra already-solved trajectories offered as further warm-start
    candidates (main.tex's own "cross-pollination" fix, ssec:res_pareto /
    line 361, applied when it found its own w_e=0 point could land in a
    worse local optimum than nearby weights). Keeps whichever candidate
    converges to the lower proxy objective. A single cold start is not
    reliable for this problem: the paper documents up to 338% best-to-worst
    spread for time-optimal corner legs from local-optimum sensitivity
    alone."""
    candidates = [_solve_corner_paper(waypoints, w_energy, warm_start=warm_start, **kwargs)]
    labels = ['cold']
    for j in range(n_jitter):
        try:
            jittered = _jittered_ctrl_pts(candidates[0], rng=np.random.default_rng(seed + j))
            candidates.append(_solve_corner_paper(waypoints, w_energy, warm_start=jittered, **kwargs))
            labels.append(f'jittered-{j}')
        except Exception as exc:
            print(f"    jittered restart {j} failed ({exc})")
    for i, ws in enumerate(extra_warm_starts):
        try:
            candidates.append(_solve_corner_paper(waypoints, w_energy, warm_start=ws, **kwargs))
            labels.append(f'cross-warm-start-{i}')
        except Exception as exc:
            print(f"    extra warm-started candidate failed ({exc})")
    best_idx = int(np.argmin([_corner_objective(r, w_energy) for r in candidates]))
    best = candidates[best_idx]
    print(f"    multi-start candidates ({labels}): "
          f"{[round(_corner_objective(r, w_energy), 3) for r in candidates]}  "
          f"-> kept {labels[best_idx]}")
    return best


def _build_s2(res_corner):
    """Post-corner tractor straight run (gamma=0 at exit -> JLAP plans the
    TRACTOR path, offset from the trailer by LF+LB in HEADING_OUT)."""
    st_exit = res_corner['states'][-1:]
    xt_exit, yt_exit = ttp.tractor_xy(st_exit)
    wp_start = [float(xt_exit[0]), float(yt_exit[0])]
    wp_end = [
        ttp.WP_END[0] + (ttp.LF + ttp.LB) * np.cos(ttp.HEADING_OUT),
        ttp.WP_END[1] + (ttp.LF + ttp.LB) * np.sin(ttp.HEADING_OUT),
    ]
    v_exit = float(max(0.0, res_corner['v'][-1]))
    return ttp._run_jlap_seg(wp_start, wp_end, initial_vel=v_exit, final_vel=0.0)


# =============================================================================
# TRAILER-FRAME REFERENCE STITCHING
# =============================================================================
def _stitch_reference(res_s1, res_corner, res_s2_tractor, dt=0.05):
    """Stitch S1 + corner + S2 into a uniform-dt reference carrying BOTH the
    trailer pose (x,y,theta,gamma -- used for tracking-quality METRICS,
    since the OCP plans the trailer's own path) and the tractor pose
    (x_tractor,y_tractor,theta_tractor, derived via forward kinematics --
    used as what PurePursuit actually TRACKS). Only the tractor is
    actuated: a differentially-driven tractor's own body is a unicycle by
    construction (theta_dot_tractor = w_tractor exactly), so PurePursuit's
    raw [v,w] output can drive the real articulated plant's tractor input
    unmodified when tracking the TRACTOR reference -- unlike an earlier
    version of this script that tracked the TRAILER reference directly and
    needed an inverse-kinematics correction to convert a desired trailer
    heading rate into a tractor input; that approach treated the hitch
    angle as if it were actively commandable, when a real trailer is a
    PASSIVE follower, and was found to let gamma wander to an unstable
    ~180 deg equilibrium under closed-loop tracking. Tracking the tractor
    directly (as tractor_trailer_path_segment_combined.py's own, proven
    Pure Pursuit demo already does) while still simulating the full 4-state
    plant underneath lets gamma evolve as a genuine passive consequence of
    the tractor's motion -- exactly what the paper's ssec:res_tracking
    lists as its own future work."""
    T1 = float(res_s1['time'][-1])
    t_ocp = res_corner['time']
    t_ik = res_corner['time_ik']
    sc = res_corner['states']
    Tc = float(t_ocp[-1])

    s1_x = res_s1['states'][:, 0]
    s1_y = res_s1['states'][:, 1]
    s1_th = res_s1['states'][:, 2]

    x_c = np.interp(t_ik, t_ocp, sc[:, 0])
    y_c = np.interp(t_ik, t_ocp, sc[:, 1])
    th_c = np.interp(t_ik, t_ocp, sc[:, 2])
    gamma_c = np.interp(t_ik, t_ocp, sc[:, 3])

    # S2 was JLAP-planned for the tractor -- invert the (LF+LB) offset to
    # recover the trailer's own path (valid since gamma=0 post-corner).
    off_x = (ttp.LF + ttp.LB) * np.cos(ttp.HEADING_OUT)
    off_y = (ttp.LF + ttp.LB) * np.sin(ttp.HEADING_OUT)
    s2_x = res_s2_tractor['states'][:, 0] - off_x
    s2_y = res_s2_tractor['states'][:, 1] - off_y
    s2_th = res_s2_tractor['states'][:, 2]

    t_raw = np.concatenate([res_s1['time'], t_ik + T1,
                            res_s2_tractor['time'][1:] + T1 + Tc])
    x_raw = np.concatenate([s1_x, x_c, s2_x[1:]])
    y_raw = np.concatenate([s1_y, y_c, s2_y[1:]])
    th_raw = np.concatenate([s1_th, th_c, s2_th[1:]])
    gamma_raw = np.concatenate([np.zeros_like(s1_x), gamma_c,
                                np.zeros_like(s2_x[1:])])
    v_raw = np.concatenate([res_s1['v'], res_corner['v'], res_s2_tractor['v'][1:]])
    w_raw = np.concatenate([res_s1['omega'], res_corner['omega'],
                            res_s2_tractor['omega'][1:]])
    acc_raw = np.concatenate([res_s1['acc_path'], res_corner['acc_path'],
                              res_s2_tractor['acc_path'][1:]])
    alpha_raw = np.concatenate([res_s1['alpha'], res_corner['alpha'],
                                res_s2_tractor['alpha'][1:]])

    t_uni = np.arange(t_raw[0], t_raw[-1], dt)
    x_u = np.interp(t_uni, t_raw, x_raw)
    y_u = np.interp(t_uni, t_raw, y_raw)
    th_u = np.interp(t_uni, t_raw, th_raw)
    gamma_u = np.interp(t_uni, t_raw, gamma_raw)

    # Tractor pose via forward kinematics (same formula as ttp.tractor_xy).
    xt_u = x_u + ttp.LF * np.cos(th_u) + ttp.LB * np.cos(th_u - gamma_u)
    yt_u = y_u + ttp.LF * np.sin(th_u) + ttp.LB * np.sin(th_u - gamma_u)
    tht_u = th_u - gamma_u

    return {
        'time':     t_uni,
        'x':        x_u,
        'y':        y_u,
        'theta':    th_u,
        'gamma':    gamma_u,
        'x_tractor':     xt_u,
        'y_tractor':     yt_u,
        'theta_tractor': tht_u,
        'v':        np.interp(t_uni, t_raw, v_raw),
        'omega':    np.interp(t_uni, t_raw, w_raw),
        'acc_path': np.interp(t_uni, t_raw, acc_raw),
        'alpha':    np.interp(t_uni, t_raw, alpha_raw),
    }


# =============================================================================
# CORRECTED ARTICULATED PLANT (matches the OCP's own trailer-centric ODE --
# NOT models/trailer_tractor.py, whose ODE doesn't match this formulation)
# =============================================================================
class _TractorTrailerModel:
    nx = 4   # [x_trailer, y_trailer, theta_trailer, gamma]
    nu = 2   # [v_tractor, w_tractor]

    def __init__(self, length_back, length_front):
        self.length_back = length_back
        self.length_front = length_front

    def function(self, state, input, dt):
        theta = state[2]
        gamma = state[3]
        v = input[0]
        w = input[1]
        lb = self.length_back
        lf = self.length_front
        common = v * cs.cos(gamma) - w * lb * cs.sin(gamma)
        dxdt = cs.cos(theta) * common
        dydt = cs.sin(theta) * common
        dthetadt = -v * cs.sin(gamma) / lf - w * lb * cs.cos(gamma) / lf
        dgammadt = dthetadt - w
        dfdt = cs.vertcat(dxdt, dydt, dthetadt, dgammadt)
        return state + dfdt * dt


class _TrailerToTractorPoseAdapter:
    """Forward-kinematics wrapper: derives the TRACTOR's pose from the real
    articulated plant's true trailer-frame state [x,y,theta,gamma] at each
    control step, then hands that pose to an inner controller that tracks a
    TRACTOR-frame reference (the same frame tractor_trailer_path_segment_
    combined.py's own Pure Pursuit demo already tracks, with the same
    controller gains, so this is known-stable tracking behavior). The inner
    controller's raw [v,w] output can then drive the plant's tractor input
    UNMODIFIED: a differentially-driven tractor is a unicycle in its own
    right (theta_dot_tractor = w_tractor exactly, by construction), unlike
    the trailer, whose heading rate is a derived, passive consequence of
    the hitch coupling, not something directly commandable. An earlier
    version of this script instead tracked the trailer's own pose and
    inverse-kinematics-converted a desired TRAILER heading rate into a
    tractor input; that treated the hitch angle as if it were actively
    regulated, and was found to let gamma wander to an unstable ~180 deg
    equilibrium under closed-loop tracking, even while the trailer's own
    position/heading tracked well (small CTE/heading error) -- an
    underactuated-internal-dynamics failure, not a tracking failure."""

    def __init__(self, controller, length_back, length_front):
        self._ctrl = controller
        self.lb = length_back
        self.lf = length_front

    def initialize(self):
        self._ctrl.initialize()

    def execute(self, state, input, index):
        x, y, theta, gamma = state[0], state[1], state[2], state[3]
        xt = x + self.lf * np.cos(theta) + self.lb * np.cos(theta - gamma)
        yt = y + self.lf * np.sin(theta) + self.lb * np.sin(theta - gamma)
        thetat = theta - gamma
        return self._ctrl.execute(np.array([xt, yt, thetat]), input, index)


class _HitchStabilizedController:
    """Adds explicit hitch-angle feedback, from a simulated second IMU (a
    rotary encoder at the hitch, per main.tex's own hardware-validation
    future work), on top of the existing tractor-pose-tracking controller.

    Without this, gamma is purely PASSIVE: the outer loop (tractor pose)
    never observes or targets it, so small per-step mismatches between the
    commanded and "ideal" w_tractor accumulate over a ~40s corner into tens
    of degrees of drift -- confirmed to reach 82-86 deg against the 44.98
    deg mechanical limit even while tractor heading error stayed under a
    few degrees the whole time. This wrapper measures gamma directly
    (gamma_meas = true gamma + sensor noise) and ADDS a proportional
    correction to whatever w_tractor the inner (tractor-pose-tracking)
    controller already computed, rather than replacing it -- preserving
    the existing, known-good tractor position/heading tracking.

    Sign derivation: the plant ODE gives gamma_dot = theta_dot_trailer -
    w_tractor, so INCREASING w_tractor DECREASES gamma_dot. If
    gamma_meas > gamma_ref (hitch angle too large), gamma_dot should be
    driven negative, so w_tractor must INCREASE:
        correction = +k_gamma * (gamma_meas - gamma_ref),  k_gamma > 0.
    """

    def __init__(self, controller, gamma_ref, k_gamma, gamma_std,
                 max_correction, rng=None):
        self._ctrl = controller
        self._gamma_ref = np.asarray(gamma_ref)
        self._k_gamma = float(k_gamma)
        self._gamma_std = float(gamma_std)
        self._max_correction = float(max_correction)
        self._rng = rng if rng is not None else np.random.default_rng()

    def initialize(self):
        self._ctrl.initialize()

    def execute(self, state, input, index):
        status, uv = self._ctrl.execute(state, input, index)
        if not status:
            return status, uv
        gamma_true = float(state[3])
        gamma_meas = gamma_true + self._rng.normal(0.0, self._gamma_std)
        gamma_ref_t = float(self._gamma_ref[min(index, len(self._gamma_ref) - 1)])
        correction = self._k_gamma * (gamma_meas - gamma_ref_t)
        correction = float(np.clip(correction, -self._max_correction, self._max_correction))
        v_trac, w_trac = float(uv[0]), float(uv[1])
        return status, [v_trac, w_trac + correction]


def _run_purepursuit_tt(traj_tractor, initial_state_trailer, gamma_ref, seed,
                        k_gamma=K_GAMMA):
    """traj_tractor: TRACTOR-frame reference Trajectory, what PurePursuit
    actually tracks. initial_state_trailer: the real plant's initial state
    [x_trailer, y_trailer, theta_trailer, gamma]. gamma_ref: the OCP-planned
    trailer hitch-angle array, same time grid as traj_tractor.t, used by
    the hitch-IMU feedback layer (_HitchStabilizedController)."""
    model = _TractorTrailerModel(length_back=ttp.LB, length_front=ttp.LF)
    PurePursuit.lookahead_distance = pps.LOOKAHEAD_DISTANCE
    PurePursuit.lookahead_gain     = pps.LOOKAHEAD_GAIN
    PurePursuit.k                  = pps.K_SPEED
    PurePursuit.k_i                = pps.K_I_SPEED
    PurePursuit.k_ff               = 1.0
    sim = TimeStepping(model, float(traj_tractor.t[-1]), traj_tractor.sampling_time)
    pose_controller = _TrailerToTractorPoseAdapter(
        pps.GpsImuObserver(
            PurePursuit(model, traj_tractor), sim_dt=traj_tractor.sampling_time,
            gps_rate_hz=pps.GPS_RATE_HZ, gps_std_xy=pps.GPS_STD_XY,
            imu_std_theta=pps.IMU_STD_THETA, imu_std_v=pps.IMU_STD_V,
            rng=np.random.default_rng(seed),
        ),
        length_back=ttp.LB, length_front=ttp.LF,
    )
    controller = _HitchStabilizedController(
        pose_controller, gamma_ref=gamma_ref, k_gamma=k_gamma,
        gamma_std=HITCH_IMU_STD_GAMMA, max_correction=MAX_GAMMA_CORRECTION,
        rng=np.random.default_rng(seed + 1_000_000),
    )
    sim.run_with_controller(initial_state_trailer, traj_tractor, controller)
    return sim


# =============================================================================
# METRICS -- same set as section5_purepursuit._track_one_seed + hitch metrics.
# Power model reused UNMODIFIED: ROBOT_PARAMS_TT['l']=0.265 matches
# purepursuit_fine_sweep.HALF_WHEEL_BASE / dc.ROBOT_PARAMS_BSPLINE['l']
# exactly, and the TT energy coefficients equal ENERGY_COEFFS_RIGHT/LEFT.
# =============================================================================
METRIC_KEYS = [
    'mission_time', 'energy_sim', 'peak_power_sim', 'energy_meas',
    'peak_power_meas', 'model_rmse', 'max_cte_cm', 'mean_cte_cm',
    'rms_cte_cm', 'final_err_cm', 'max_he_deg', 'mean_he_deg',
    'max_verr_ms', 'track_dur_s', 'max_gamma_deg', 'mean_abs_gamma_deg',
    'gamma_rmse_deg',
]


def _track_one_seed_tt(d, mission_time, seed):
    dt = float(d['time'][1] - d['time'][0])
    # Trailer-frame trajectory: what tracking-quality METRICS are measured
    # against (the OCP plans the trailer's own path).
    traj = Trajectory(
        x=np.column_stack([d['x'], d['y'], d['theta']]),
        u=np.vstack([d['v'], d['omega']]),
        t=d['time'], sampling_time=dt,
    )
    # Tractor-frame trajectory: what PurePursuit actually TRACKS (see
    # _stitch_reference / _TrailerToTractorPoseAdapter docstrings).
    traj_tractor = Trajectory(
        x=np.column_stack([d['x_tractor'], d['y_tractor'], d['theta_tractor']]),
        u=np.vstack([d['v'], d['omega']]),
        t=d['time'], sampling_time=dt,
    )
    initial_state_trailer = [d['x'][0], d['y'][0], d['theta'][0], d['gamma'][0]]
    sim = _run_purepursuit_tt(traj_tractor, initial_state_trailer,
                              gamma_ref=d['gamma'], seed=seed)

    nt = min(traj.x.shape[0], sim.x_out.shape[1])
    trk_xy = sim.x_out[:2, :].T
    ref_xy = traj.x[:, :2]
    cte = pps.cross_track_error(ref_xy, trk_xy)
    he = pps.heading_error(traj.x[:nt, 2], sim.x_out[2, :nt])
    v_err = np.abs(traj.u[0, :nt] - sim.u_out[0, :nt])
    final_err = np.hypot(sim.x_out[0, -1] - traj.x[-1, 0],
                         sim.x_out[1, -1] - traj.x[-1, 1])

    power_ref = dc._compute_power_from_accel(d['v'], d['omega'],
                                              d['acc_path'], d['alpha'])
    total_energy_ref = float(np.trapezoid(power_ref, d['time']))
    peak_power_ref = float(power_ref.max())

    v_trk = sim.u_out[0, :nt]
    om_trk = sim.u_out[1, :nt]
    t_trk = sim.t_out[:nt]
    power_trk = pps.compute_motor_power(v_trk, om_trk, t_trk)
    total_energy_trk = float(np.trapezoid(power_trk, t_trk))
    peak_power_trk = float(power_trk.max())

    gamma_trk = sim.x_out[3, :nt]
    gamma_rmse = float(np.sqrt(np.mean((gamma_trk - d['gamma'][:nt]) ** 2)))

    metrics = {
        'mission_time':     mission_time,
        'energy_sim':       total_energy_ref,
        'peak_power_sim':   peak_power_ref,
        'energy_meas':      total_energy_trk,
        'peak_power_meas':  peak_power_trk,
        'model_rmse':       float(np.sqrt(np.mean(
            (np.interp(t_trk, d['time'], power_ref) - power_trk) ** 2))),
        'max_cte_cm':       float(cte.max()  * 1e2),
        'mean_cte_cm':      float(cte.mean() * 1e2),
        'rms_cte_cm':       float(np.sqrt((cte ** 2).mean()) * 1e2),
        'final_err_cm':     float(final_err  * 1e2),
        'max_he_deg':       float(np.rad2deg(np.abs(he).max())),
        'mean_he_deg':      float(np.rad2deg(np.abs(he).mean())),
        'max_verr_ms':      float(v_err.max()),
        'track_dur_s':      float(sim.t_out[-1]),
        'max_gamma_deg':       float(np.rad2deg(np.abs(gamma_trk).max())),
        'mean_abs_gamma_deg':  float(np.rad2deg(np.abs(gamma_trk).mean())),
        'gamma_rmse_deg':      float(np.rad2deg(gamma_rmse)),
    }
    return metrics, traj, sim


# =============================================================================
# MAIN
# =============================================================================
print("=" * 60)
print("Step 1: PathSegment - corner arc geometry")
print("=" * 60)
seg_info = ttp._segment_corner()
arc_entry_ext = seg_info['arc_entry_ext_world']
arc_exit_ext = seg_info['arc_exit_ext_world']
corner_wps = ttp._build_corner_waypoints(arc_entry_ext, arc_exit_ext)
print(f"  feasible = {seg_info['feasible']}   L_seg = {seg_info['L_seg']:.3f} m")

print()
print("=" * 60)
print("Step 2: EulerJLAPCoverage - segment 1 (shared by both methods)")
print("=" * 60)
res_s1 = ttp._run_jlap_seg(ttp.WP_START, arc_entry_ext.tolist(),
                           initial_vel=0.0, final_vel=ttp.V_HANDOFF)
a_s1_exit = float(res_s1['acc_path'][-1])
alpha_s1_exit = float(res_s1['alpha'][-1])
print(f"  T = {res_s1['time'][-1]:.3f} s   v_exit = {res_s1['v'][-1]:.3f} m/s")

print()
print("=" * 60)
print("Step 3: BSplineEnergyTractorTrailerCoverage corner solves "
      "(paper-validated Table tab:params settings)")
print("=" * 60)
# Solved w_e=0.1 FIRST: an earlier single-cold-start run of this script hit
# exactly the failure mode main.tex's own Section V documents -- a pure
# time-minimization (w_e=0) cold start landing in a local optimum worse than
# a weighted-objective solve at the SAME geometry (T=61s vs T=15s). The
# paper's own fix (ssec:res_pareto, "companion check") was to give w_e=0 the
# other already-solved trajectories as extra warm-start candidates; applied
# here by solving w_e=0.1 first and cross-warm-starting w_e=0.0 from it.
print("  w_e=0.1 (paper-validated energy-aware operating point), "
      "cold start + jittered restart ...")
res_corner_energy = _solve_corner_multistart(
    corner_wps, w_energy=0.1, seed=1,
    v_entry=ttp.V_HANDOFF, v_exit=ttp.V_HANDOFF,
    a_entry=a_s1_exit, alpha_entry=alpha_s1_exit,
)
print(f"    kept: T={res_corner_energy['time'][-1]:.3f}s  "
      f"E={res_corner_energy['energy']:.3f}J")

print("  w_e=0.0 (time-optimal), cold + 3 jittered restarts (main.tex's "
      "broader multi-start budget), plus cross-warm-started from the kept "
      "w_e=0.1 result ...")
res_corner_time = _solve_corner_multistart(
    corner_wps, w_energy=0.0, seed=0, n_jitter=3,
    extra_warm_starts=[res_corner_energy],
    v_entry=ttp.V_HANDOFF, v_exit=ttp.V_HANDOFF,
    a_entry=a_s1_exit, alpha_entry=alpha_s1_exit,
)
print(f"    kept: T={res_corner_time['time'][-1]:.3f}s  "
      f"E={res_corner_time['energy']:.3f}J")

RES_CORNER_BY_LABEL = {'T': res_corner_time, 'E': res_corner_energy}

METHODS = {}
for label, we in METHOD_WE.items():
    res_corner = RES_CORNER_BY_LABEL[label]
    res_s2 = _build_s2(res_corner)
    d = _stitch_reference(res_s1, res_corner, res_s2)
    METHODS[label] = (d, float(d['time'][-1]))

summary = {}
for label, (d, mission_time) in METHODS.items():
    print(f"\nTracking Method {label} (w_e={METHOD_WE[label]}): {N_SEEDS} seeds ...")
    raw_rows = []
    seed0_traj = seed0_sim = None
    for seed in range(N_SEEDS):
        metrics, traj, sim = _track_one_seed_tt(d, mission_time, seed)
        row = {'seed': seed}
        row.update(metrics)
        raw_rows.append(row)
        if seed == 0:
            seed0_traj, seed0_sim = traj, sim

    # --- raw per-seed CSV -------------------------------------------------
    raw_path = os.path.join(OUT_DIR, f'method_{label}_raw_per_seed.csv')
    header = ['seed'] + METRIC_KEYS
    with open(raw_path, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in raw_rows:
            f.write(','.join(str(row[k]) for k in header) + '\n')
    print(f"  Saved {raw_path}")

    # --- summary stats ------------------------------------------------------
    stats = {}
    for k in METRIC_KEYS:
        vals = np.array([row[k] for row in raw_rows])
        stats[k] = {'mean': float(vals.mean()), 'std': float(vals.std()),
                    'min': float(vals.min()), 'max': float(vals.max())}
    summary[label] = stats

    # --- seed-0 representative time series -----------------------------------
    nt = min(seed0_traj.x.shape[0], seed0_sim.x_out.shape[1])
    ts_path = os.path.join(OUT_DIR, f'method_{label}_seed0_timeseries.csv')
    with open(ts_path, 'w') as f:
        f.write('t,x_ref,y_ref,theta_ref,gamma_ref,v_ref,omega_ref,'
                'x_trk,y_trk,theta_trk,gamma_trk,v_trk,omega_trk\n')
        for i in range(nt):
            f.write(f"{seed0_sim.t_out[i]},"
                    f"{seed0_traj.x[i,0]},{seed0_traj.x[i,1]},{seed0_traj.x[i,2]},{d['gamma'][i]},"
                    f"{seed0_traj.u[0,i]},{seed0_traj.u[1,i]},"
                    f"{seed0_sim.x_out[0,i]},{seed0_sim.x_out[1,i]},{seed0_sim.x_out[2,i]},{seed0_sim.x_out[3,i]},"
                    f"{seed0_sim.u_out[0,i]},{seed0_sim.u_out[1,i]}\n")
    print(f"  Saved {ts_path}")

# --- summary stats JSON (both methods) -------------------------------------
summary_path = os.path.join(OUT_DIR, 'summary_stats.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved {summary_path}")

# --- metadata ---------------------------------------------------------------
metadata = {
    'export_timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
    'n_seeds': N_SEEDS,
    'reference_scenario': {
        'waypoints_m': [ttp.WP_START, ttp.WP_CORNER, ttp.WP_END],
        'corridor_half_width_m': 0.25,
        'corner_angle_deg': 90,
        'note': "S1(JLAP)->corner(B-spline OCP)->S2(JLAP) three-stage pipeline "
            "structure reused from tractor_trailer_path_segment_combined.py, "
            "matching the scenario described in the tractor-trailer paper's "
            "ssec:res_tracking (Fig. 7/8 -- tt_fig_stitched.png / "
            "tt_fig_purepursuit.png). The corridor bound (0.25 m) and arc/"
            "PathSegment feasibility geometry are the combined script's own, "
            "independent of the paper's Table tab:params corner-OCP numerics "
            "below.",
    },
    'tractor_trailer_geometry': {
        'length_back_lb_m': ttp.LB,
        'length_front_lf_m': ttp.LF,
        'gamma_max_rad': ttp.GAMMA_MAX,
        'gamma_max_deg': float(np.rad2deg(ttp.GAMMA_MAX)),
    },
    'corner_ocp_settings': {
        'source': "Writting/energy_aware_trailer_tractor/main.tex, Table "
            "tab:params / examples/tractor_trailer_paper_sweeps.py "
            "(common_kwargs, _make_gen) -- NOT "
            "tractor_trailer_path_segment_combined.py's own _make_bspline_tt "
            "defaults, which the paper found failed to converge for the "
            "energy-aware corner (1709s, hit IPOPT's iteration cap).",
        'robot_params': ROBOT_PARAMS_TT,
        'n_sampling': N_SAMPLING,
        'n_ctrl_pts': N_CTRL_PTS,
        'acc_max_mps2_radps2': [5.0, 5.0, 4.0],
        'jerk_max_mps3_radps3': [50.0, 50.0, 20.0],
        'ipopt_max_iter': MAX_ITER,
        'multistart_budget': 'cold start + one 12%-jittered restart per '
            'corner solve, keeping the lower-proxy-objective candidate '
            '(main.tex "Multi-start solving" default budget). '
            'return_status is not exposed by this OCP class\'s result dict; '
            'the multi-start comparison in the run log is the convergence-'
            'quality signal recorded here.',
        'corner_solve_objective_by_label': {
            'T': _corner_objective(res_corner_time, 0.0),
            'E': _corner_objective(res_corner_energy, 0.1),
        },
        'known_limitation': "For THIS export's corridor scenario (the "
            "stitched S1/corner/S2 waypoints from tractor_trailer_path_"
            "segment_combined.py, NOT the paper's own isolated 2m-leg "
            "corner), the w_e=0.0 (time-optimal) corner repeatedly "
            "converges to a mission time LONGER than the w_e=0.1 "
            "(energy-aware) corner's, even across a 5-candidate multi-start "
            "budget (cold + 3 jittered restarts + cross-warm-start from the "
            "w_e=0.1 result) where 2 of the 3 extra jittered restarts "
            "converged to still-worse local optima. This mirrors, but goes "
            "beyond, the exact failure mode main.tex's ssec:res_pareto "
            "documents for its own w_e=0 point (there resolved only through "
            "full densification/dominance-check validation, out of scope "
            "here). The 'T' method below should be read as 'the best "
            "w_e=0.0 corner found within this budget', not a certified "
            "global time-optimum -- it remains a valid, trackable reference "
            "for the closed-loop tracking comparison, just not necessarily "
            "the fastest possible one for this corridor.",
    },
    'method_energy_weights': METHOD_WE,
    'method_energy_weight_note': "w_e=0.1 is the tractor-trailer paper's own "
        "VALIDATED energy-aware operating point (ssec:res_pareto, Table "
        "tab:pareto), NOT a freshly computed Pareto-knee: the paper found a "
        "naive geometric knee-point construction fragile for this problem "
        "(an 11-point w_e sweep initially looked like a smooth trade-off "
        "curve but failed dominance/weighted-sum-consistency checks) and "
        "instead validated w_e=0.1 as the smallest weight reaching the "
        "achievable energy floor, through densification and multi-start "
        "solving out of scope for this export script. Reference values from "
        "Table tab:pareto at w_e=0.1: T=21.03s, E=113.17J, P_peak=8.37W, "
        "|gamma|max=44.98deg, margin~=0%.",
    'plant_model': {
        'class': '_TractorTrailerModel (local to this script)',
        'state': '[x_trailer, y_trailer, theta_trailer, gamma]',
        'input': '[v_tractor, w_tractor]',
        'ode': "xdot = cos(theta)*[v*cos(gamma) - w*lb*sin(gamma)]; "
            "ydot = sin(theta)*[v*cos(gamma) - w*lb*sin(gamma)]; "
            "thetadot = -v*sin(gamma)/lf - w*lb*cos(gamma)/lf; "
            "gammadot = thetadot - w",
        'note': "Matches trajectory_generators/bspline_tractor_trailer_"
            "coverage.py's documented ODE exactly (the model that generated "
            "the reference trajectory), NOT models/trailer_tractor.py's "
            "TrailerTractor class, whose ODE does not match that "
            "formulation and would make the simulated plant diverge from "
            "the path it is meant to track.",
    },
    'tracking_architecture': {
        'class': '_TrailerToTractorPoseAdapter (local to this script)',
        'note': "PurePursuit tracks the TRACTOR's own reference pose (same "
            "frame tractor_trailer_path_segment_combined.py's own, proven "
            "Pure Pursuit demo tracks), not the trailer's: a differentially"
            "-driven tractor is a unicycle in its own right "
            "(theta_dot_tractor = w_tractor exactly), so PurePursuit's raw "
            "[v, w] output can drive the real 4-state plant's tractor "
            "input unmodified. A wrapper derives the tractor's pose from "
            "the plant's TRUE trailer-frame state [x,y,theta,gamma] at "
            "each control step via forward kinematics (same formula as "
            "ttp.tractor_xy) before handing it to the tracking controller. "
            "An earlier version of this script instead tracked the "
            "trailer's own pose directly and inverse-kinematics-converted "
            "a desired TRAILER heading rate into a tractor input; that "
            "treated the passive, underactuated hitch angle as if it were "
            "directly commandable and was found to let gamma wander to an "
            "unstable ~180 deg equilibrium under closed-loop tracking, "
            "even while the trailer's own position/heading tracked well "
            "(small CTE/heading error) -- confirming it was an "
            "underactuated-internal-dynamics failure, not a tracking-loop "
            "failure. Tracking the tractor directly instead lets gamma "
            "evolve as a genuinely PASSIVE consequence of tractor motion, "
            "matching both real trailer physics (only the tractor is "
            "actuated) and the paper's own framing of this as its stated "
            "future work (ssec:res_tracking). Tracking the tractor alone "
            "still lets gamma drift (see hitch_angle_feedback below) since "
            "it remains unobserved by this layer.",
    },
    'hitch_angle_feedback': {
        'class': '_HitchStabilizedController (local to this script)',
        'sensor': 'simulated rotary hitch-angle IMU/encoder, gamma_meas = '
            'true gamma + N(0, hitch_imu_std_gamma_rad) at the 20 Hz '
            'control rate -- the exact instrumentation main.tex\'s own '
            'future-work section proposes for hardware validation '
            '(sec:conclusion: "a passive trailer, connected through an '
            'instrumented hitch (a rotary hitch-angle sensor)").',
        'law': 'correction = clip(k_gamma * (gamma_meas - gamma_ref(t)), '
            '-max_gamma_correction, +max_gamma_correction); '
            'w_tractor_final = w_tractor_from_pose_tracking + correction',
        'hitch_imu_std_gamma_rad': HITCH_IMU_STD_GAMMA,
        'k_gamma_per_s': K_GAMMA,
        'max_gamma_correction_radps': MAX_GAMMA_CORRECTION,
        'note': "Added on top of (not replacing) the tractor-pose-tracking "
            "command from tracking_architecture above, so tractor "
            "position/heading tracking is preserved while gamma is "
            "additionally pulled toward its OCP-planned reference. "
            "Compare gamma_rmse_deg / max_gamma_deg in the raw-per-seed "
            "CSVs against the same metrics from before this feedback was "
            "added (max_gamma_deg was 82-86 deg for Method T, 60-64 deg "
            "for Method E, against a 44.98 deg mechanical limit) to see "
            "how much of that drift this closes.",
    },
    'sensor_noise_model': {
        'gps_rate_hz': pps.GPS_RATE_HZ,
        'gps_std_xy_m': pps.GPS_STD_XY,
        'imu_control_rate_hz': 1.0 / pps.SIM_DT,
        'imu_heading_std_rad': pps.IMU_STD_THETA,
        'imu_velocity_std_mps': pps.IMU_STD_V,
        'hitch_imu_std_gamma_rad': HITCH_IMU_STD_GAMMA,
    },
    'pure_pursuit_controller': {
        'lookahead_distance_m': pps.LOOKAHEAD_DISTANCE,
        'lookahead_gain_kv_s': pps.LOOKAHEAD_GAIN,
        'speed_kp_per_s': pps.K_SPEED,
        'speed_ki_per_s2': pps.K_I_SPEED,
        'feedforward_gain_kff': 1.0,
        'control_rate_hz': 1.0 / pps.SIM_DT,
        'note': "PurePursuit tracks the TRACTOR's own reference pose and "
            "its raw [v, w] output drives the tractor input directly -- "
            "see tracking_architecture above for why, and for the "
            "underactuated-hitch failure mode that ruled out tracking the "
            "trailer's pose instead. lookahead_gain is a documented no-op "
            "in this codebase (same as export_section5_baseline.py): "
            "TimeStepping.run_with_controller always passes input=[0,0] "
            "to the controller, so the speed-adaptive lookahead term "
            "never contributes and lookahead_distance is effectively "
            "constant.",
    },
    'motor_power_model': "Reused UNMODIFIED from differential_drive_"
        "comparison.py / purepursuit_fine_sweep.py "
        "(_compute_power_from_accel / compute_motor_power): the tractor's "
        "half-wheelbase/wheel radius (l=0.265, r=0.15) and TJ108 energy "
        "coefficients are numerically identical to the diff-drive companion "
        "platform's, per Table tab:params.",
    'robot_params': {'mass_kg': 50.4, 'half_wheelbase_l_m': ROBOT_PARAMS_TT['l'],
                     'wheel_radius_r_m': ROBOT_PARAMS_TT['r']},
    'metric_definitions': {
        'mission_time': 'planned trajectory duration [s]',
        'energy_sim': 'open-loop (planned) total energy [J], from acc_path/alpha fields',
        'peak_power_sim': 'open-loop (planned) peak total power [W]',
        'energy_meas': 'closed-loop (tracked) total energy [J], integrated from tracked v/omega',
        'peak_power_meas': 'closed-loop (tracked) peak total power [W]',
        'model_rmse': 'RMSE [W] between open-loop reference power and tracked power',
        'max_cte_cm/mean_cte_cm/rms_cte_cm': 'cross-track error [cm], point-to-polyline vs reference path (trailer frame)',
        'final_err_cm': 'final trailer position error [cm]',
        'max_he_deg/mean_he_deg': 'trailer heading error [deg], wrap-corrected tracked-minus-reference',
        'max_verr_ms': 'max absolute tractor forward-velocity tracking error [m/s]',
        'track_dur_s': 'actual simulated tracking duration [s]',
        'max_gamma_deg/mean_abs_gamma_deg': 'simulated closed-loop hitch angle [deg] -- '
            'sanity check that the articulated plant stays within gamma_max',
        'gamma_rmse_deg': 'RMSE [deg] between the simulated closed-loop hitch angle and '
            'its OCP-planned reference -- direct measure of how well hitch_angle_feedback '
            'is regulating gamma',
    },
    'source_paper': 'Energy-Aware Function-Parameterized Optimal Control for '
        'Tractor-Trailer Trajectory Generation, Section V',
    'note': "ALL data in this export is SIMULATED (GPS/IMU noise model + a "
        "corrected, trailer-centric articulated-plant simulation), not from "
        "physical hardware. Unlike tractor_trailer_path_segment_combined."
        "py's own closed-loop tracking figure, which tracks only the "
        "derived tractor path with a DifferentialDrive plant, this export "
        "genuinely propagates the trailer's hitch angle through closed-loop "
        "tracking -- the future-work item noted in the paper's "
        "ssec:res_tracking ('propagating the trailer's own closed-loop "
        "response through its passive kinematics during tracking').",
}
meta_path = os.path.join(OUT_DIR, 'metadata.json')
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Saved {meta_path}")

print(f"\nAll tractor-trailer Section V closed-loop tracking data exported to: {OUT_DIR}")
