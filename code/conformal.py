"""conformal.py -- conformal prediction methods for spatial regression.

Every method takes calibration quantities and query quantities and returns
(lo, hi, half) prediction intervals at level 1-alpha.

Leakage prevention (applies throughout):
  * The mean model mu() and quantile models are fit ONLY on the fitting set.
  * Calibration and test/query DI values are computed ONLY against the fitting
    set (see aoa.dissimilarity_index); test labels are never used.
  * DI bin edges are built from CALIBRATION DI values only (no test labels).
  * Finite-sample conformal quantile within a set of m scores uses the level
    ceil((m+1)(1-alpha))/m, implemented as np.quantile(., level, 'higher').
  * Monotonicity: per-bin quantiles are made non-decreasing in the DI bin index
    via a running maximum, so a more-dissimilar query never gets a narrower
    interval than a less-dissimilar one.
  * Empty / small bins (fewer than min_n calibration points) fall back to the
    GLOBAL conformal quantile computed on all calibration scores.
  * Query DI beyond the calibration DI range is clipped into the top bin (the
    widest), never extrapolated to a narrower one.
  * region-Mondrian: a region with fewer than min_n calibration points uses the
    GLOBAL conformal quantile as its fallback.
"""
import numpy as np


def conformal_quantile(scores, alpha):
    """Finite-sample split-conformal quantile of 1-D scores."""
    m = len(scores)
    if m == 0:
        return np.inf
    level = min(1.0, np.ceil((m + 1) * (1 - alpha)) / m)
    return float(np.quantile(scores, level, method="higher"))


# ---------------------------------------------------------------- basic
def split(mu_cal, y_cal, mu_q, alpha):
    q = conformal_quantile(np.abs(y_cal - mu_cal), alpha)
    half = np.full_like(mu_q, q)
    return mu_q - half, mu_q + half, half


def normalized(mu_cal, y_cal, sig_cal, mu_q, sig_q, alpha, eps=1e-6):
    s = np.abs(y_cal - mu_cal) / (sig_cal + eps)
    q = conformal_quantile(s, alpha)
    half = q * (sig_q + eps)
    return mu_q - half, mu_q + half, half


def di_normalized(mu_cal, y_cal, di_cal, mu_q, di_q, alpha, floor=0.25):
    return normalized(mu_cal, y_cal, floor + di_cal, mu_q, floor + di_q, alpha)


# ---------------------------------------------------------------- Mondrian
def region_mondrian(mu_cal, y_cal, reg_cal, mu_q, reg_q, alpha, min_n=10):
    s = np.abs(y_cal - mu_cal)
    qg = conformal_quantile(s, alpha)
    half = np.empty_like(mu_q)
    for r in np.unique(reg_q):
        m = reg_cal == r
        qr = conformal_quantile(s[m], alpha) if m.sum() >= min_n else qg
        half[reg_q == r] = qr
    return mu_q - half, mu_q + half, half


def _di_bin_quantiles(scores_cal, di_cal, alpha, K, monotone, min_n):
    """Per-DI-bin conformal quantiles (+ bin edges, per-bin counts)."""
    edges = np.quantile(di_cal, np.linspace(0, 1, K + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    qg = conformal_quantile(scores_cal, alpha)
    b = np.clip(np.digitize(di_cal, edges) - 1, 0, K - 1)
    qbins, counts = [], []
    for k in range(K):
        m = b == k
        counts.append(int(m.sum()))
        qbins.append(conformal_quantile(scores_cal[m], alpha) if m.sum() >= min_n else qg)
    qbins = np.array(qbins, float)
    if monotone:
        qbins = np.maximum.accumulate(qbins)
    return edges, qbins, counts


def di_mondrian(mu_cal, y_cal, di_cal, mu_q, di_q, alpha,
                K=5, monotone=True, min_n=15, return_info=False):
    s = np.abs(y_cal - mu_cal)
    edges, qbins, counts = _di_bin_quantiles(s, di_cal, alpha, K, monotone, min_n)
    bq = np.clip(np.digitize(di_q, edges) - 1, 0, K - 1)
    half = qbins[bq]
    out = (mu_q - half, mu_q + half, half)
    return (out + (counts,)) if return_info else out


# ---------------------------------------------------------------- CQR
def cqr(qlo_cal, qhi_cal, y_cal, qlo_q, qhi_q, alpha):
    E = np.maximum(qlo_cal - y_cal, y_cal - qhi_cal)
    Q = conformal_quantile(E, alpha)
    lo, hi = qlo_q - Q, qhi_q + Q
    return lo, hi, (hi - lo) / 2.0


def di_cqr(qlo_cal, qhi_cal, y_cal, di_cal, qlo_q, qhi_q, di_q, alpha,
           K=5, monotone=True, min_n=15, return_info=False):
    E = np.maximum(qlo_cal - y_cal, y_cal - qhi_cal)
    edges, Qbins, counts = _di_bin_quantiles(E, di_cal, alpha, K, monotone, min_n)
    bq = np.clip(np.digitize(di_q, edges) - 1, 0, K - 1)
    Qv = Qbins[bq]
    lo, hi = qlo_q - Qv, qhi_q + Qv
    out = (lo, hi, (hi - lo) / 2.0)
    return (out + (counts,)) if return_info else out


# ---------------------------------------------------------------- weighted
def weighted_split(mu_cal, y_cal, w_cal, mu_q, w_q, alpha, cap=None):
    """Weighted split conformal (Tibshirani et al. 2019) for covariate shift.

    For each query with weight w0, the interval half-width is the weighted
    (1-alpha) quantile of calibration scores under normalised weights
    w_i/(sum_j w_j + w0), with a point mass w0/(...) at +inf. If the total
    calibration weight fraction is below 1-alpha the quantile is +inf (interval
    unbounded); such cases are capped at `cap` (if given) and counted.
    """
    s = np.abs(y_cal - mu_cal)
    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = np.asarray(w_cal, float)[order]
    cumw = np.cumsum(w_sorted)
    S = cumw[-1]
    rhs = (1 - alpha) * (S + np.asarray(w_q, float))     # per-query threshold
    idx = np.searchsorted(cumw, rhs, side="left")
    half = np.empty_like(mu_q, dtype=float)
    finite = idx < len(s_sorted)
    half[finite] = s_sorted[np.clip(idx[finite], 0, len(s_sorted) - 1)]
    half[~finite] = np.inf
    if cap is not None:
        half = np.where(np.isfinite(half), half, cap)
    return mu_q - half, mu_q + half, half


def localized_conformal(mu_cal, y_cal, feat_cal, mu_q, feat_q, alpha,
                        bandwidth=None, batch=512):
    """Localized conformal prediction (Guan 2023): the interval half-width at a
    query is a *kernel-weighted* conformal quantile of calibration
    non-conformity scores, with weights = Gaussian kernel of feature-space
    distance between the query and each calibration point. Continuous cousin of
    DI-binning. `feat_*` are the (fitting-standardised, importance-weighted)
    embeddings; use coordinates for a geographic-localized variant.

    Leakage: scores/feat from fitting+calibration only; bandwidth from the
    median calibration pairwise distance (no test labels).
    """
    from scipy.spatial.distance import cdist
    s = np.abs(y_cal - mu_cal)
    order = np.argsort(s)
    s_sorted = s[order]
    fc = np.asarray(feat_cal, float)[order]
    if bandwidth is None:
        dc = cdist(fc, fc)
        bandwidth = np.median(dc[dc > 0]) + 1e-12
    n = len(s_sorted)
    level = min(1.0, (1 - alpha) * (1 + 1.0 / n))
    fq = np.asarray(feat_q, float)
    half = np.empty(len(fq))
    for i in range(0, len(fq), batch):                 # batch to bound memory
        D = cdist(fq[i:i+batch], fc)
        W = np.exp(-(D ** 2) / (2 * bandwidth ** 2)) + 1e-12
        frac = np.cumsum(W, axis=1) / W.sum(1, keepdims=True)
        idx = np.clip((frac >= level).argmax(axis=1), 0, n - 1)
        half[i:i+batch] = s_sorted[idx]
    return mu_q - half, mu_q + half, half


def localized_tuned(mu_cal, y_cal, feat_cal, mu_q, feat_q, alpha,
                    mults=(0.5, 1.0, 2.0)):
    """Localized conformal with the kernel bandwidth chosen fairly on the
    calibration set only: for each candidate bandwidth we form leave-one-out
    localized intervals for the calibration points and pick the bandwidth with the
    lowest calibration interval (Winkler) score. No test labels are used."""
    from scipy.spatial.distance import cdist
    s = np.abs(y_cal - mu_cal)
    fc = np.asarray(feat_cal, float)
    dcc = cdist(fc, fc)
    base_bw = np.median(dcc[dcc > 0]) + 1e-12
    n = len(s)
    level = min(1.0, (1 - alpha) * (1 + 1.0 / n))
    order = np.argsort(s)
    s_sorted = s[order]

    best_bw, best_score = base_bw, np.inf
    for mlt in mults:
        bw = base_bw * mlt
        W = np.exp(-(dcc ** 2) / (2 * bw ** 2))
        np.fill_diagonal(W, 0.0)                       # leave-one-out
        Wc = W[:, order]
        frac = np.cumsum(Wc, axis=1) / (Wc.sum(1, keepdims=True) + 1e-12)
        idx = np.clip((frac >= level).argmax(axis=1), 0, n - 1)
        half_cal = s_sorted[idx]
        sc = np.mean(_interval_score(y_cal, mu_cal - half_cal, mu_cal + half_cal, alpha))
        if sc < best_score:
            best_score, best_bw = sc, bw
    return localized_conformal(mu_cal, y_cal, feat_cal, mu_q, feat_q, alpha, bandwidth=best_bw)


def _interval_score(y, lo, hi, alpha):
    width = hi - lo
    pen = (2.0 / alpha) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    return width + pen


def width_matched_global(mu_q, target_mean_width):
    """Constant-width interval around mu whose mean width equals target."""
    half = np.full_like(mu_q, target_mean_width / 2.0)
    return mu_q - half, mu_q + half, half


def localized_cqr(qlo_cal, qhi_cal, y_cal, feat_cal, qlo_q, qhi_q, feat_q, alpha,
                  mults=(0.5, 1.0, 2.0), batch=512):
    """Kernel-localized CQR: the offset applied to a query's quantile-regression
    interval is a Gaussian-kernel-weighted quantile (in feature space) of the CQR
    non-conformity scores E_i = max(q_lo(x_i)-y_i, y_i-q_hi(x_i)). This is the
    apples-to-apples localized counterpart of DI-CQR (same kernel and
    calibration-only bandwidth tuning as the kernel-localized residual comparator,
    but on CQR scores and applied to the quantile interval), isolating whether
    DI-CQR's edge is the DI grouping rather than merely the CQR score.
    Bandwidth is chosen on the calibration set only by leave-one-out interval score;
    no test labels are used."""
    from scipy.spatial.distance import cdist
    E = np.maximum(qlo_cal - y_cal, y_cal - qhi_cal)
    fc = np.asarray(feat_cal, float)
    dcc = cdist(fc, fc)
    base_bw = np.median(dcc[dcc > 0]) + 1e-12
    n = len(E)
    level = min(1.0, (1 - alpha) * (1 + 1.0 / n))
    order = np.argsort(E)
    E_sorted = E[order]
    # tune bandwidth on calibration LOO CQR interval score
    best_bw, best_score = base_bw, np.inf
    for mlt in mults:
        bw = base_bw * mlt
        W = np.exp(-(dcc ** 2) / (2 * bw ** 2))
        np.fill_diagonal(W, 0.0)
        Wc = W[:, order]
        frac = np.cumsum(Wc, axis=1) / (Wc.sum(1, keepdims=True) + 1e-12)
        idx = np.clip((frac >= level).argmax(axis=1), 0, n - 1)
        Qcal = E_sorted[idx]
        sc = np.mean(_interval_score(y_cal, qlo_cal - Qcal, qhi_cal + Qcal, alpha))
        if sc < best_score:
            best_score, best_bw = sc, bw
    # apply at queries with best bandwidth
    fc_ord = fc[order]
    fq = np.asarray(feat_q, float)
    Q = np.empty(len(fq))
    for i in range(0, len(fq), batch):
        D = cdist(fq[i:i+batch], fc_ord)
        Wq = np.exp(-(D ** 2) / (2 * best_bw ** 2)) + 1e-12
        frac = np.cumsum(Wq, axis=1) / Wq.sum(1, keepdims=True)
        idx = np.clip((frac >= level).argmax(axis=1), 0, n - 1)
        Q[i:i+batch] = E_sorted[idx]
    lo, hi = qlo_q - Q, qhi_q + Q
    return lo, hi, (hi - lo) / 2.0


# ---------------------------------------------------------------- hyperparameter
# selection (calibration-valid: chosen by cross-conformal interval score on the
# CALIBRATION set only; no test labels, no fitting labels beyond the already-fit
# models). This makes DI-CQR / DI-normalized deployable without a fixed a-priori K
# or kappa and without test-set tuning.
def _xval_folds(n, n_folds, seed):
    perm = np.random.default_rng(seed).permutation(n)
    return [perm[i::n_folds] for i in range(n_folds)]


def select_K_di_cqr(qlo_cal, qhi_cal, y_cal, di_cal, alpha,
                    K_grid=(3, 4, 5, 6, 8, 10), n_folds=5, min_n=15, seed=0):
    """Pick K minimizing the cross-conformal interval score on the calibration set.
    Each fold is scored by a DI-CQR model calibrated on the other folds only."""
    n = len(y_cal)
    if n < 2 * n_folds:
        return 5
    folds = _xval_folds(n, n_folds, seed)
    best_K, best = K_grid[0], np.inf
    for K in K_grid:
        sc = []
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            if len(tr) < min_n:
                continue
            lo, hi, _ = di_cqr(qlo_cal[tr], qhi_cal[tr], y_cal[tr], di_cal[tr],
                               qlo_cal[te], qhi_cal[te], di_cal[te], alpha,
                               K=K, min_n=min_n)
            sc.append(_interval_score(y_cal[te], lo, hi, alpha))
        if not sc:
            continue
        m = float(np.mean(np.concatenate(sc)))
        if m < best:
            best, best_K = m, K
    return best_K


def select_kappa_di_normalized(mu_cal, y_cal, di_cal, alpha,
                               kappa_grid=(0.1, 0.25, 0.5, 1.0), n_folds=5, seed=0):
    """Pick the DI-normalized floor kappa minimizing the cross-conformal interval
    score on the calibration set (calibration-valid, no test labels)."""
    n = len(y_cal)
    if n < 2 * n_folds:
        return 0.25
    folds = _xval_folds(n, n_folds, seed)
    best_k, best = kappa_grid[0], np.inf
    for kap in kappa_grid:
        sc = []
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            lo, hi, _ = di_normalized(mu_cal[tr], y_cal[tr], di_cal[tr],
                                      mu_cal[te], di_cal[te], alpha, floor=kap)
            sc.append(_interval_score(y_cal[te], lo, hi, alpha))
        m = float(np.mean(np.concatenate(sc)))
        if m < best:
            best, best_k = m, kap
    return best_k
