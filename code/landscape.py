"""landscape.py -- controlled environmental-monitoring landscape simulator.

Ground truth is known at every one of GRID*GRID cells, so interval coverage is
measurable everywhere (including unmonitored regions). Monitoring sites are
drawn without replacement with per-draw probability proportional to
accessibility**bias.

Two accessibility mechanisms are provided:
  * "covariate": accessibility is a function of two model covariates
    (low elevation, near the river). Selection therefore acts directly on the
    feature space in which the Dissimilarity Index is computed.
  * "hidden": accessibility is a smooth road-access field that is NOT a model
    covariate and is independent of the covariates and of the target. Any
    covariate shift arises only indirectly, through the spatial clustering of
    the monitored sites.

`inclusion_prob` returns the without-replacement inclusion probabilities pi(x)
used for the oracle weighted-conformal baseline.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import brentq

GRID = 56
N_REG = 3          # -> 9 regions


def _n(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-12)


def _grf(shape, scale, rng):
    f = gaussian_filter(rng.standard_normal(shape), sigma=scale, mode="reflect")
    return (f - f.mean()) / f.std()


def make_landscape(rng, corr=6.0, access_mode="covariate"):
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
    if access_mode == "covariate":
        access = ((1 - _n(elevation)) * (1 - _n(dist_river))).reshape(-1)
    elif access_mode == "hidden":
        # drawn AFTER all landscape fields, so the covariates and target of a
        # given seed are identical across the two access modes
        road = _grf((g, g), corr * 1.5, rng)
        access = (1 - _n(road)).reshape(-1)
    else:
        raise ValueError(access_mode)
    return X, y, coords, access


def region_ids(coords, k=N_REG):
    rx = np.clip((coords[:, 0] * k).astype(int), 0, k - 1)
    ry = np.clip((coords[:, 1] * k).astype(int), 0, k - 1)
    return ry * k + rx


def selection_prob(access, bias):
    """Normalised per-draw selection intensity over all cells."""
    p = np.power(access, bias)          # bias=0 -> uniform
    return p / p.sum()


def sample_monitoring(access, rng, bias, n=500):
    p = selection_prob(access, bias)
    idx = rng.choice(len(access), size=n, replace=False, p=p)
    return idx, p


def inclusion_prob(p, n, reps=20000, seed=12345, batch=2000):
    """Inclusion probabilities pi_i for successive sampling of n units without
    replacement with per-draw probabilities p (the design of sample_monitoring).

    Successive sampling is equivalent to exponential order sampling: unit i
    enters the sample iff E_i / p_i is among the n smallest values, with E_i
    i.i.d. Exp(1). We estimate pi by Monte Carlo over `reps` replicate designs.
    Units never selected in the replicates (pi below ~1/reps) are assigned the
    standard large-population approximation pi_i = 1 - exp(-lam * p_i), with lam
    solving sum_i pi_i = n; units with p_i = 0 have pi_i = 0 exactly.
    """
    p = np.asarray(p, float)
    N = len(p)
    if np.all(p == p[0]):                  # equal-probability design: pi = n/N exactly
        return np.full(N, n / N)
    pos = p > 0
    rng = np.random.default_rng(seed)
    hits = np.zeros(N)
    with np.errstate(divide="ignore"):
        inv = np.where(pos, 1.0 / np.where(pos, p, 1.0), np.inf)
    done = 0
    while done < reps:
        b = min(batch, reps - done)
        key = rng.exponential(size=(b, N)) * inv
        sel = np.argpartition(key, n - 1, axis=1)[:, :n]
        np.add.at(hits, sel.ravel(), 1.0)
        done += b
    pi_mc = hits / reps
    f = lambda lam: np.sum(-np.expm1(-lam * p)) - n
    lam = brentq(f, 1e-9, 1e12) if np.count_nonzero(pos) > n else np.inf
    pi_ap = np.where(pos, -np.expm1(-lam * p), 0.0)
    pi = np.where(pi_mc > 0, pi_mc, pi_ap)
    pi[~pos] = 0.0
    return pi
