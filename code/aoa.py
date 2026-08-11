"""aoa.py -- Area of Applicability Dissimilarity Index (Meyer & Pebesma 2021).

Leakage control: the DI of ANY point (calibration or test) is computed only
against the FITTING set X_fit. Test labels are never used. Feature weights come
from a model fitted on the fitting set alone.
"""
import numpy as np
from scipy.spatial.distance import cdist


def dissimilarity_index(X_fit, X_query, weights=None):
    """Return (di_query, di_fit, aoa_threshold).

    di = min distance (importance-weighted, standardised feature space) from a
    query point to the FITTING set, normalised by the mean pairwise distance
    among fitting points. Standardisation uses fitting-set mean/sd only.
    AOA threshold = Q75 + 1.5*IQR of the fitting-set DI (outlier-adjusted max).
    """
    X_fit = np.asarray(X_fit, float)
    X_query = np.asarray(X_query, float)
    mu = X_fit.mean(0)
    sd = X_fit.std(0)
    sd[sd == 0] = 1e-12
    Xf = (X_fit - mu) / sd
    Xq = (X_query - mu) / sd
    if weights is None:
        weights = np.ones(Xf.shape[1])
    weights = np.asarray(weights, float)
    weights = weights / weights.sum() * len(weights)
    w = np.sqrt(weights)
    Xf = Xf * w
    Xq = Xq * w
    dff = cdist(Xf, Xf)
    n = dff.shape[0]
    mean_pairwise = dff.sum() / (n * (n - 1))
    np.fill_diagonal(dff, np.inf)
    di_fit = dff.min(1) / mean_pairwise
    di_query = cdist(Xq, Xf).min(1) / mean_pairwise
    q25, q75 = np.percentile(di_fit, [25, 75])
    thr = q75 + 1.5 * (q75 - q25)
    return di_query, di_fit, thr


def transform(X_fit, X_query, weights=None):
    """Importance-weighted, fitting-standardised feature embedding used by the DI.
    Localized conformal operates in this same space (fitting statistics only)."""
    X_fit = np.asarray(X_fit, float); X_query = np.asarray(X_query, float)
    mu = X_fit.mean(0); sd = X_fit.std(0); sd[sd == 0] = 1e-12
    if weights is None:
        weights = np.ones(X_fit.shape[1])
    weights = np.asarray(weights, float)
    weights = weights / weights.sum() * len(weights)
    return ((X_query - mu) / sd) * np.sqrt(weights)
