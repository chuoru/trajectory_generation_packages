import numpy as np

data = np.loadtxt('examples/csv_output/trajectory_stitched.csv', delimiter=',', skiprows=1)
time, x, y, theta, v, omega, omega_r, omega_l, seg = data.T

s1_end   = np.where(seg == 1)[0][-1]
c_start  = np.where(seg == 2)[0][0]
c_end    = np.where(seg == 2)[0][-1]
s2_start = np.where(seg == 3)[0][0]

print('=== S1 -> Corner transition ===')
print(f'  S1 last   : v={v[s1_end]:.4f}  omega={omega[s1_end]:.5f} rad/s')
print(f'  Corner 1st: v={v[c_start]:.4f}  omega={omega[c_start]:.5f} rad/s')
print(f'  Delta v   : {abs(v[c_start]-v[s1_end]):.5f} m/s')
print(f'  Delta omega: {abs(omega[c_start]-omega[s1_end]):.5f} rad/s')

print()
print('=== Corner -> S2 transition ===')
print(f'  Corner last: v={v[c_end]:.4f}  omega={omega[c_end]:.5f} rad/s')
print(f'  S2 1st     : v={v[s2_start]:.4f}  omega={omega[s2_start]:.5f} rad/s')
print(f'  Delta v    : {abs(v[s2_start]-v[c_end]):.5f} m/s')
print(f'  Delta omega: {abs(omega[s2_start]-omega[c_end]):.5f} rad/s')

# Per-wheel jerk near transitions
alpha_r = np.gradient(omega_r, time)
alpha_l = np.gradient(omega_l, time)
jerk_r  = np.gradient(alpha_r, time)
jerk_l  = np.gradient(alpha_l, time)

print()
print('=== Peak |jerk| in ±5 samples around each junction ===')
for label, idx in [('S1->Corner', s1_end), ('Corner->S2', c_end)]:
    lo = max(0, idx - 5)
    hi = min(len(time) - 1, idx + 5)
    print(f'  {label}: |jerk_r|={np.max(np.abs(jerk_r[lo:hi])):.2f}  '
          f'|jerk_l|={np.max(np.abs(jerk_l[lo:hi])):.2f}  rad/s^3')

print()
print('=== Peak |jerk| over full trajectory ===')
print(f'  max |jerk_r| = {np.max(np.abs(jerk_r)):.2f} rad/s^3')
print(f'  max |jerk_l| = {np.max(np.abs(jerk_l)):.2f} rad/s^3')
