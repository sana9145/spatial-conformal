"""landscape.py -- controlled environmental-monitoring landscape simulator.

Ground truth is known at every one of GRID*GRID cells, so interval coverage is
measurable everywhere (including unmonitored regions). Monitoring sites are
drawn with probability proportional to accessibility**bias; the exact selection
probabilities are returned so that ORACLE weighted conformal can use them.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

GRID = 56
N_REG = 3          # -> 9 regions


def _n(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-12)


def _grf(shape, scale, rng):
    f = gaussian_filter(rng.standard_normal(shape), sigma=scale, mode="reflect")
    return (f - f.mean()) / f.std()


def make_landscape(rng, corr=6.0):
    g = GRID
    ys, xs = np.mgrid[0:g, 0:g] / (g - 1)
    elevation = _grf((g, g), corr, rng) + 1.5 * ys
    river_x = 0.5 + 0.18 * np.sin(2 * np.pi * ys)
    dist_river = np.abs(xs - river_x)
    climate = 1.2 * xs + 0.5 * _grf((g, g), corr, rng)
    wetness = _grf((g, g), corr * 0.8, rng)
    target = (2.5 * np.tanh(2 * (0.4 - dist_river)) + 1.8 * elevation
              + 1.3 * climate * (dist_river < 0.25) + 1.0 * np.sin(3 * wetness)
              + 0.6 * _grf((g, g), corr * 0.5, rng))
    X = np.stack([elevation, dist_river, climate, wetness], -1).reshape(-1, 4)
    y = target.reshape(-1)
    coords = np.stack([xs.reshape(-1), ys.reshape(-1)], -1)
    access = ((1 - _n(elevation)) * (1 - _n(dist_river))).reshape(-1)
    return X, y, coords, access


def region_ids(coords, k=N_REG):
    rx = np.clip((coords[:, 0] * k).astype(int), 0, k - 1)
    ry = np.clip((coords[:, 1] * k).astype(int), 0, k - 1)
    return ry * k + rx


def selection_prob(access, bias):
    """Normalised monitoring-selection probability over all cells."""
    p = np.power(access, bias)          # bias=0 -> uniform
    return p / p.sum()


def sample_monitoring(access, rng, bias, n=500):
    p = selection_prob(access, bias)
    idx = rng.choice(len(access), size=n, replace=False, p=p)
    return idx, p
