# Jerk Improvement Summary

## Problem

The corner B-spline segment had sudden jerk spikes of ±200–400 rad/s³ at the entry and exit,
caused by double numerical differentiation (`np.gradient` twice). Additionally, JLAP segment
jerk values were expressed in rad/s³ (up to ±28) rather than the physical limit of
J_LIM = 3.547 m/s³.

---

## Changes

### 1. `trajectory_generators/euler_jlap_coverage.py` — Decel-phase acceleration bug

**Bug:** `_interpolate()` used `a_pk = j_lim * T1a` (accel-phase ramp peak) for the T2d and T3d
deceleration phases, giving `acc_path[-1] ≈ 0.77 m/s²` instead of ≈ 0 at segment exit.

**Fix:** Replace with decel-phase ramp time `T1d`:
- T2d phase: `a_j = -j_lim * T1d`
- T3d phase: `a_j = j_lim * (zz - T1d)`

**Result:** Exit acceleration dropped from 0.77 m/s² to ~0.03 m/s².

---

### 2. `trajectory_generators/bspline_energy_coverage.py` — OCP wheel-jerk constraints + analytical jerk output

**a) Added wheel-level jerk constraints to the OCP:**

Constrain the wheel linear jerk directly in the OCP (per node, both wheels):
```
(cos·ddds_x + sin·ddds_y ± l·ddds_θ) / T³  ≤  J_LIM
```
This guarantees `|jerk_r|, |jerk_l| ≤ J_LIM = 3.547 m/s³` at every OCP node.

**b) Extract `ddds_val` from OCP solution:**

Added extraction of the 3rd B-spline derivative after the solve:
```python
ddds_val = np.array(dbg.value(ddds))
```

**c) Compute `acc_path` and `alpha` analytically from OCP 2nd derivatives:**

Instead of differentiating the interpolated v/omega (which caused spikes), compute directly:
```python
a_ocp     = (cos_th * dds_val[:, 0] + sin_th * dds_val[:, 1]) / T_val**2
alpha_ocp = dds_val[:, 2] / T_val**2
```
Interpolate with boundary-clamped CubicSpline (acc_path) and Pchip (alpha, monotone-preserving).

**d) Return jerk in m/s³ from OCP 3rd derivatives (no numerical differentiation):**

```python
body_jerk_ocp = (cos_th * ddds_val[:, 0] + sin_th * ddds_val[:, 1]) / T_val**3
ang_jerk_ocp  = ddds_val[:, 2] / T_val**3
jerk_r_ocp    = body_jerk_ocp + l * ang_jerk_ocp   # m/s³
jerk_l_ocp    = body_jerk_ocp - l * ang_jerk_ocp   # m/s³
```
Pchip-interpolate to fine grid. Added `jerk_r` and `jerk_l` keys to the return dict.

---

### 3. `examples/differential_drive_path_segment_combined.py` — Units and figures

**a) CSV export — all jerk columns now in m/s³:**

- **Corner:** use `res_corner_opt['jerk_r']` directly (from OCP, already m/s³)
- **JLAP segments:** multiply `np.gradient(alpha_r, dt)` by `r_ref` to convert rad/s³ → m/s³:
  ```python
  jrkr_s1 = np.gradient(alr_s1, JLAP_DT) * r_ref   # m/s³
  ```

**b) `_compute_wheel_kinematics()` — use OCP jerk when available:**

```python
if 'jerk_r' in res and 'jerk_l' in res:
    jerk_r = res['jerk_r']          # OCP-level m/s³ (corner)
    jerk_l = res['jerk_l']
else:
    jerk_r = np.gradient(alpha_r, dt) * r   # m/s³ (JLAP)
    jerk_l = np.gradient(alpha_l, dt) * r
```

**c) Figure 5 — corrected label and limit lines:**

- Y-axis label changed: `jerk [rad/s³]` → `jerk [m/s³]`
- Red dotted lines added at ±J_LIM = ±3.547 m/s³

**d) V_HANDOFF check — limit updated to m/s³:**

Changed `jerk_wheel_lim = J_LIM / r_ref` (rad/s³) → `jerk_wheel_lim = J_LIM` (m/s³).

---

## Results

| Segment | `jerk_r` [m/s³] | `jerk_l` [m/s³] | Violations |
|---|---|---|---|
| segment1_jlap.csv | [−3.547, +3.547] | [−3.547, +3.547] | 0 |
| corner_bspline.csv | [−0.237, +2.687] | [−3.547, +1.026] | 0 |
| segment2_jlap.csv | [−3.547, +3.547] | [−3.547, +3.547] | 0 |

V_HANDOFF probe output: `edge jerk = 3.13 m/s³ (limit 3.55 m/s³)`

All jerk values across CSVs and Figure 5 are within ±J_LIM = ±3.547 m/s³.
