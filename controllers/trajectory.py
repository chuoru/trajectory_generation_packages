from dataclasses import dataclass
import numpy as np


@dataclass
class Trajectory:
    """Container bridging trajectory generator output to the controller interface.

    Attributes:
        x: Reference states, shape (N, nx) — each row is a state [x, y, theta].
        u: Reference controls, shape (nu, N) — each column is a control [v, w].
        t: Time array, shape (N,).
        sampling_time: Time step dt in seconds.

    Note:
        x is row-major (N, nx) while u is column-major (nu, N) to match
        the convention used by the controllers in this package.
        To convert from TimeStepping output: x = simulator.x_out.T
    """
    x: np.ndarray
    u: np.ndarray
    t: np.ndarray
    sampling_time: float

    @classmethod
    def from_stitched_csv(cls, path, dt=0.05):
        """Load a stitched trajectory CSV and resample to uniform dt.

        Expected CSV columns: time, x, y, theta, v, omega, ...
        Compatible with trajectory_stitched.csv written by
        differential_drive_path_segment_combined.py.

        @param path<str>: Path to the CSV file.
        @param dt<float>: Target uniform sampling interval [s].
        @return Trajectory instance resampled to dt.
        """
        data = np.loadtxt(path, delimiter=',', skiprows=1)
        t_raw  = data[:, 0]
        x_raw  = data[:, 1]
        y_raw  = data[:, 2]
        th_raw = data[:, 3]
        v_raw  = data[:, 4]
        w_raw  = data[:, 5]

        t_uni  = np.arange(t_raw[0], t_raw[-1], dt)
        x_uni  = np.interp(t_uni, t_raw, x_raw)
        y_uni  = np.interp(t_uni, t_raw, y_raw)
        th_uni = np.interp(t_uni, t_raw, th_raw)
        v_uni  = np.interp(t_uni, t_raw, v_raw)
        w_uni  = np.interp(t_uni, t_raw, w_raw)

        states   = np.column_stack([x_uni, y_uni, th_uni])  # (N, 3)
        controls = np.vstack([v_uni, w_uni])                  # (2, N)

        return cls(x=states, u=controls, t=t_uni, sampling_time=dt)
