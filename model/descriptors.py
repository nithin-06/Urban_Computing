import numpy as np


def sensor_descriptors(data_path, weekend_dows=(5, 6), n_days=7,
                       steps_per_day=288, bins=24):
    """Average flow per hour for weekdays and weekends over the first n_days.

    Returns an array of shape (N, 2 * bins). Only the first n_days of the
    training period are used, as promised in the proposal.
    """
    data = np.load(f"{data_path}/data.npz")["data"]  # (T, N, 3): flow, tod, dow
    n_steps = n_days * steps_per_day
    flow = data[:n_steps, :, 0].astype(np.float64)
    tod = data[:n_steps, 0, 1]
    dow = data[:n_steps, 0, 2].astype(int)
    hour = np.minimum((tod * bins).astype(int), bins - 1)
    weekend = np.isin(dow, weekend_dows)

    z = np.zeros((flow.shape[1], 2 * bins))
    for part, mask in enumerate((~weekend, weekend)):
        for h in range(bins):
            sel = mask & (hour == h)
            if not sel.any():
                raise ValueError(f"no samples for part={part} hour={h}")
            z[:, part * bins + h] = flow[sel].mean(axis=0)
    return z


def normalise_descriptors(z, fit_idx):
    """Scalar z-score using only the sensors in fit_idx (the seen sensors)."""
    mu = z[fit_idx].mean()
    sd = z[fit_idx].std()
    return (z - mu) / sd


def check_weekend_encoding(data_path):
    """Print mean flow per day-of-week value. The two lowest are the weekend."""
    data = np.load(f"{data_path}/data.npz")["data"]
    dow = data[:, 0, 2].astype(int)
    for d in range(7):
        print(d, round(float(data[dow == d, :, 0].mean()), 2))
