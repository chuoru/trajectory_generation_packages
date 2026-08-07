"""! Build the corner-angle generalisation figure (peak power vs. w_e, and
peak power vs. total energy) for the 45/90/135-degree reduced-resolution
sweeps reported in Table 2 of the paper (Section III-E). Data below is
transcribed directly from the sweep logs (angle_sweep_generalization.py /
angle_sweep_135_only.py output); no new solves are performed by this script.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import matplotlib.pyplot as plt

import differential_drive_path_segment_fine_sweep as m

m._set_paper_style()

# (w_e, T [s], E [J], Peak P [W]) per angle, high-to-low w_e as swept.
DATA_45 = np.array([
    [1.000000, 19.692, 76.121, 17.871],
    [0.928571, 19.379, 75.813, 17.871],
    [0.857143, 18.805, 75.273, 17.871],
    [0.785714, 17.974, 74.576, 17.871],
    [0.714286, 17.300, 74.114, 17.871],
    [0.642857, 16.697, 74.349, 17.871],
    [0.571429, 15.920, 73.456, 17.871],
    [0.500000, 15.313, 73.323, 17.871],
    [0.428571, 14.049, 73.402, 17.871],
    [0.357143, 13.063, 73.892, 17.871],
    [0.285714, 11.721, 75.334, 17.871],
    [0.214286, 10.307, 78.204, 17.871],
    [0.142857,  8.594, 84.548, 17.871],
    [0.071429,  6.347, 102.103, 17.871],
    [0.000000,  5.935, 113.762, 29.645],
])

DATA_90 = np.array([
    [1.000000, 18.497, 70.845, 17.871],
    [0.928571, 18.030, 70.395, 17.871],
    [0.857143, 17.500, 69.928, 17.871],
    [0.785714, 16.939, 69.489, 17.871],
    [0.714286, 16.316, 69.072, 17.871],
    [0.642857, 15.685, 68.740, 17.871],
    [0.571429, 15.054, 68.517, 17.871],
    [0.500000, 14.335, 68.425, 17.871],
    [0.428571, 13.521, 68.556, 17.871],
    [0.357143, 12.824, 69.609, 17.871],
    [0.285714, 12.092, 69.749, 17.871],
    [0.214286, 11.271, 71.413, 17.871],
    [0.142857, 10.313, 74.965, 17.871],
    [0.071429,  9.182, 83.888, 17.871],
    [0.000000,  8.612, 100.731, 21.608],
])

DATA_135 = np.array([
    [1.000000, 16.729, 59.226, 17.871],
    [0.857143, 16.157, 59.399, 17.872],
    [0.714286, 15.620, 58.306, 17.871],
    [0.571429, 15.075, 58.072, 17.871],
    [0.428571, 14.402, 58.095, 17.871],
    [0.285714, 13.669, 59.824, 17.871],
    [0.142857, 12.850, 60.904, 17.871],
    [0.000000, 12.234, 72.895, 20.470],
])

KNEE = {45: 0.071429, 90: 0.071429, 135: 0.142857}

ANGLES = [(45, DATA_45, '#1f77b4', 'o'),
          (90, DATA_90, '#d62728', 's'),
          (135, DATA_135, '#2ca02c', '^')]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                num='Corner-angle generalisation')

for angle, data, color, marker in ANGLES:
    we, T, E, Pk = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    order = np.argsort(we)
    ax1.plot(we[order], Pk[order], color=color, marker=marker, ms=5,
              lw=1.5, label=f'{angle}$^\\circ$')
    # Mark the knee point.
    k = KNEE[angle]
    ki = np.argmin(np.abs(we - k))
    ax1.scatter([we[ki]], [Pk[ki]], color=color, marker=marker, s=110,
                edgecolor='black', linewidth=1.0, zorder=5)

ax1.set_xlabel('$w_e$')
ax1.set_ylabel('Peak Motor Power [W]')
ax1.legend(title='Corner angle', fontsize=9)
ax1.set_title('(a) Peak power saturates to the\nsame floor at every angle')

for angle, data, color, marker in ANGLES:
    we, T, E, Pk = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    order = np.argsort(E)
    ax2.plot(E[order], Pk[order], color=color, marker=marker, ms=5,
              lw=1.5, label=f'{angle}$^\\circ$')
    k = KNEE[angle]
    ki = np.argmin(np.abs(we - k))
    ax2.scatter([E[ki]], [Pk[ki]], color=color, marker=marker, s=110,
                edgecolor='black', linewidth=1.0, zorder=5,
                label=f'{angle}$^\\circ$ knee' if False else None)

ax2.set_xlabel('Total Energy [J]')
ax2.set_ylabel('Peak Motor Power [W]')
ax2.legend(title='Corner angle', fontsize=9)
ax2.set_title('(b) Pareto front shape is consistent\nacross angles (outlined markers: knee)')

fig.tight_layout()
m._savefig(fig, 'fig_angle_generalization.png')
print('Saved fig_angle_generalization.png')
