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


def conformal_rank(m, alpha):
    """k = ceil((m+1)(1-alpha)), guarded against floating-point round-up
    (e.g. 20 * 0.9 evaluates to 18.000000000000004)."""
    return int(np.ceil((m + 1) * (1 - alpha) - 1e-9))


def conformal_quantile(scores, alpha):
    """Finite-sample split-conformal quantile: the k-th smallest of m scores with
    k = ceil((m+1)(1-alpha)). If k > m (fewer than about 1/alpha scores) the
    largest score is used; this fallback does not keep the usual guarantee."""
    m = len(scores)
    if m == 0:
        return np.inf
    k = conformal_rank(m, alpha)
    return float(np.sort(np.asarray(scores, float))[min(k, m) - 1])


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


def di_normalized_clipped(mu_cal, y_cal, di_cal, mu_q, di_q, alpha, floor=0.25):
    """DI-normalized with the query DI truncated at the largest calibration DI.
    Diagnostic variant: it bounds the normalizer exactly where DI-CQR clips its
    bins, so it isolates whether DI-normalized's over-widening comes from the
    unbounded growth of g(d) beyond the calibration DI range."""
    d_q = np.minimum(di_q, np.max(di_cal))
    return normalized(mu_cal, y_cal, floor + di_cal, mu_q, floor + d_q, alpha)


# ---------------------------------------------------------------- localized
def median_bandwidth(feat_fit, max_n=2000, seed=0):
    """Median pairwise Euclidean distance among FITTING-set feature vectors.
    Label-free and independent of the calibration and test data, so the kernel
    is fixed before conformalization (as Guan's guarantee requires)."""
    from scipy.spatial.distance import pdist
    F = np.asarray(feat_fit, float)
    if len(F) > max_n:
        F = F[np.random.default_rng(seed).choice(len(F), max_n, replace=False)]
    return float(np.median(pdist(F))) + 1e-12


def lcp_threshold(scores_cal, feat_cal, feat_q, alpha, bandwidth, batch=256):
    """Localized conformal prediction (Guan 2023, Biometrika 110:33-50).

    Gaussian localizer H(x, x') = exp(-|x - x'|^2 / (2 h^2)) with a FIXED
    bandwidth. For the augmented sample (calibration points plus the query with
    a candidate score v), each point i gets the localized CDF
        F_i = sum_j H_ij / (sum_k H_ik) * delta_{V_j}
    and the statistic c_i = F_i(V_i^-), the localized mass strictly below its own
    score. Guan's level adjustment (choose the smallest level alpha~ such that a
    fraction >= 1 - alpha of the n+1 points satisfy V_i <= Q(alpha~; F_i)) makes
    the candidate v admissible iff c_{n+1}(v) is no larger than the k-th
    smallest of {c_1(v), ..., c_{n+1}(v)}, with k = ceil((1-alpha)(n+1)).
    Because the c_i are a permutation-equivariant function of the n+1
    exchangeable points, this has finite-sample marginal coverage >= 1 - alpha
    under exchangeability. The admissible set is {v <= v*}; v* is found by a
    vectorised bisection over the sorted calibration scores. v* = +inf when the
    query carries too little localized calibration mass (the query's own atom
    then dominates its localized distribution).

    Returns (v_star, local_mass) where local_mass = sum_j H(x_q, x_j) is the
    kernel-weighted number of calibration points near each query.
    """
    from scipy.spatial.distance import cdist
    V = np.asarray(scores_cal, float)
    order = np.argsort(V, kind="mergesort")
    Vs = V[order]
    Fc = np.asarray(feat_cal, float)[order]
    n = len(Vs)
    k = conformal_rank(n, alpha)
    g = 1.0 / (2.0 * bandwidth ** 2)
    H = np.exp(-g * cdist(Fc, Fc) ** 2)                  # includes H_ii = 1
    S = H.sum(1)
    n_less = np.searchsorted(Vs, Vs, side="left")        # #{j: V_j < V_i}
    Hc = np.cumsum(H, axis=1)
    B = np.where(n_less > 0, Hc[np.arange(n), np.maximum(n_less - 1, 0)], 0.0)

    Fq = np.asarray(feat_q, float)
    v_star = np.empty(len(Fq))
    mass = np.empty(len(Fq))
    for s0 in range(0, len(Fq), batch):
        h = np.exp(-g * cdist(Fq[s0:s0 + batch], Fc) ** 2)      # (Q, n)
        Q = h.shape[0]
        sh = h.sum(1)
        ch = np.cumsum(h, axis=1)
        mass[s0:s0 + batch] = sh

        def admissible(m):
            # candidate v in the open cell (Vs[m-1], Vs[m]); m = n means v > all
            c_test = np.where(m > 0, ch[np.arange(Q), np.maximum(m - 1, 0)], 0.0) / (sh + 1.0)
            thr = Vs[np.minimum(m, n - 1)]
            above = Vs[None, :] >= thr[:, None]
            above[m == n] = False
            c = (B[None, :] + h * above) / (S[None, :] + h)
            return (c < c_test[:, None]).sum(1) <= k - 1

        lo = np.zeros(Q, int)              # admissible(0) is always True
        hi = np.full(Q, n + 1, int)        # sentinel: first inadmissible cell
        while np.any(hi - lo > 1):
            mid = (lo + hi) // 2
            ok = admissible(mid)
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        v_star[s0:s0 + batch] = np.where(lo < n, Vs[np.minimum(lo, n - 1)], np.inf)
    return v_star, mass


def lcp(mu_cal, y_cal, feat_cal, mu_q, feat_q, alpha, bandwidth, cap=None,
        return_mass=False):
    """Localized conformal on absolute residuals: mu(x) +/- v*(x)."""
    v, mass = lcp_threshold(np.abs(y_cal - mu_cal), feat_cal, feat_q, alpha, bandwidth)
    if cap is not None:
        v = np.where(np.isfinite(v), v, cap)
    out = (mu_q - v, mu_q + v, v)
    return (out + (mass,)) if return_mass else out


def lcp_cqr(qlo_cal, qhi_cal, y_cal, feat_cal, qlo_q, qhi_q, feat_q, alpha,
            bandwidth, cap=None):
    """Localized conformal on CQR scores: [q_lo(x) - v*, q_hi(x) + v*]. The
    apples-to-apples localized counterpart of DI-CQR (same score, same quantile
    models; only the way calibration is localized differs)."""
    E = np.maximum(qlo_cal - y_cal, y_cal - qhi_cal)
    v, _ = lcp_threshold(E, feat_cal, feat_q, alpha, bandwidth)
    if cap is not None:
        v = np.where(np.isfinite(v), v, cap)
    lo, hi = qlo_q - v, qhi_q + v
    return lo, hi, (hi - lo) / 2.0


def _interval_score(y, lo, hi, alpha):
    width = hi - lo
    pen = (2.0 / alpha) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    return width + pen


def width_matched_global(mu_q, target_mean_width):
    """Constant-width interval around mu whose mean width equals target."""
    half = np.full_like(mu_q, target_mean_width / 2.0)
    return mu_q - half, mu_q + half, half


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
