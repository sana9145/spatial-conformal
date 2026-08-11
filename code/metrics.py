"""metrics.py -- coverage, width, interval (Winkler) score, and per-region /
per-DI-bin conditional coverage diagnostics. Nominal = 1 - alpha."""
import numpy as np


def interval_score(y, lo, hi, alpha):
    """Winkler / interval score (lower is better)."""
    width = hi - lo
    penalty = (2.0 / alpha) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    return width + penalty


def marginal_coverage(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


def per_region(region_ids, y, lo, hi, alpha):
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    isc = interval_score(y, lo, hi, alpha)
    out = {}
    for r in np.unique(region_ids):
        m = region_ids == r
        out[int(r)] = dict(coverage=float(covered[m].mean()),
                           width=float(width[m].mean()),
                           interval_score=float(isc[m].mean()),
                           n=int(m.sum()))
    return out


def summarize(region_table, y, lo, hi, alpha, nominal):
    covs = np.array([v["coverage"] for v in region_table.values()])
    return dict(
        true_marginal=marginal_coverage(y, lo, hi),
        mean_region_coverage=float(covs.mean()),
        worst_region_coverage=float(covs.min()),
        best_region_coverage=float(covs.max()),
        coverage_gap=float(covs.max() - covs.min()),
        coverage_rmse=float(np.sqrt(np.mean((covs - nominal) ** 2))),
        mean_width=float(np.mean(hi - lo)),
        median_width=float(np.median(hi - lo)),
        mean_interval_score=float(np.mean(interval_score(y, lo, hi, alpha))),
        frac_infinite=float(np.mean(~np.isfinite(hi - lo))),
    )


def conditional_coverage_by_di(y, lo, hi, di, nbins=10, nominal=0.9):
    """Coverage within fixed DI quantile bins of the TEST set (diagnostic only).
    Returns per-bin coverage/width and the mean & max |coverage-nominal|."""
    edges = np.quantile(di, np.linspace(0, 1, nbins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    covered = (y >= lo) & (y <= hi)
    b = np.clip(np.digitize(di, edges) - 1, 0, nbins - 1)
    covs, mids, wids = [], [], []
    for k in range(nbins):
        m = b == k
        if m.sum() == 0:
            continue
        covs.append(float(covered[m].mean()))
        mids.append(float(np.median(di[m])))
        wids.append(float(np.mean((hi - lo)[m])))
    covs = np.array(covs)
    return dict(bin_mid_di=mids, bin_coverage=covs.tolist(), bin_width=wids,
                cond_cov_error_mean=float(np.mean(np.abs(covs - nominal))),
                cond_cov_error_max=float(np.max(np.abs(covs - nominal))))
