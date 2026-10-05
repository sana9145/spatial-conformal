"""run_sim.py -- main simulation benchmark (resumable).

Regimes: covariate-driven accessibility at four bias strengths -- none (b=0),
mild (b=2), moderate (b=6), severe (b=12) -- plus two regimes in which
accessibility is a hidden road-access field that is not a model covariate
(hidden_moderate, b=6; hidden_severe, b=12).
Per (regime, seed): fit models on the FITTING split only, calibrate on the
CALIBRATION split, and score intervals on the HELD-OUT unmonitored cells only
(true y known everywhere). Also computes the DI-bin ablation, kappa and region
sensitivity, clipping and localized-mass diagnostics, and a coverage-vs-width
multiplier sweep. Appends one JSON line per run (resumable).

Usage:
  python run_sim.py             # run all remaining (regime, seed) pairs
  python run_sim.py agg         # aggregate -> ../results/*.csv/json + fields
Set the environment variable BUDGET (seconds) to stop early and resume later.
"""
import json, os, sys, time, numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import aoa, metrics as MET, conformal as CF


def _estimated_weights(Xf, Xc, Xall, seed, clip=(1e-2, 1e2)):
    """Cross-fitted, balanced logistic density-ratio p_test/p_calib.
    Query (grid) weights: classifier trained on all calibration (class 0) vs a
    size-matched random grid sample (class 1). Calibration weights: 5-fold
    cross-fit so a calibration point is never in its own training fold."""
    rng = np.random.default_rng(1000 + seed)
    sc = StandardScaler().fit(Xf)
    Zc, Zall = sc.transform(Xc), sc.transform(Xall)
    n = len(Xc)

    def ratio(clf, Z):
        p = np.clip(clf.predict_proba(Z)[:, 1], 1e-4, 1 - 1e-4)
        return np.clip(p / (1 - p), *clip)

    # query weights (grid), full calibration as class0
    g_idx = rng.choice(len(Xall), size=n, replace=False)
    clf_all = LogisticRegression(max_iter=500).fit(
        np.vstack([Zc, Zall[g_idx]]), np.r_[np.zeros(n), np.ones(n)])
    w_all = ratio(clf_all, Zall)

    # cross-fitted calibration weights
    w_c = np.empty(n)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(Zc):
        gi = rng.choice(len(Xall), size=len(tr), replace=False)
        clf = LogisticRegression(max_iter=500).fit(
            np.vstack([Zc[tr], Zall[gi]]), np.r_[np.zeros(len(tr)), np.ones(len(tr))])
        w_c[te] = ratio(clf, Zc[te])
    return w_c, w_all
from landscape import (make_landscape, region_ids, sample_monitoring, selection_prob,
                       inclusion_prob, GRID)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
RAW = os.path.join(RESULTS, "sim_raw.jsonl")
ALPHA = 0.10
NOMINAL = 1 - ALPHA
N_MON = 500
CAL_FRAC = 0.4
K_MAIN = 5
SEEDS = 30
REGIMES = {"none": (0.0, "covariate"), "mild": (2.0, "covariate"),
           "moderate": (6.0, "covariate"), "severe": (12.0, "covariate"),
           "hidden_moderate": (6.0, "hidden"), "hidden_severe": (12.0, "hidden")}
RF = dict(n_estimators=150, n_jobs=-1, min_samples_leaf=3, random_state=0)
BASE_METHODS = ["split", "normalized", "di_normalized", "di_normalized_auto",
                "di_normalized_clip", "region_mondrian", "di_mondrian", "cqr",
                "di_cqr", "di_cqr_auto", "lcp", "lcp_cqr", "geo_lcp",
                "weighted_oracle", "weighted_estimated", "weighted_proxy",
                "width_matched_global"]
UNBOUNDED_METHODS = ["lcp", "lcp_cqr", "geo_lcp", "weighted_oracle", "weighted_estimated",
                     "weighted_proxy"]
KAPPAS = [0.1, 0.25, 0.5, 1.0]
REGION_KS = [2, 3, 4]
ABLATION_K = [3, 4, 5, 6, 8, 10]
MULTS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
BUDGET = float(os.environ.get("BUDGET", "inf"))


def _summ(reg, y, lo, hi, di):
    rt = MET.per_region(reg, y, lo, hi, ALPHA)
    s = MET.summarize(rt, y, lo, hi, ALPHA, NOMINAL)
    cc = MET.conditional_coverage_by_di(y, lo, hi, di, nbins=10, nominal=NOMINAL)
    s["cond_cov_error_mean"] = cc["cond_cov_error_mean"]
    s["cond_cov_error_max"] = cc["cond_cov_error_max"]
    return s, cc


def one_run(regime, seed, save_fields=False):
    bias, access_mode = REGIMES[regime]
    rng = np.random.default_rng(seed)
    X, y, coords, access = make_landscape(rng, access_mode=access_mode)
    reg = region_ids(coords)
    p_sel = selection_prob(access, bias)
    pi = inclusion_prob(p_sel, N_MON)          # exact design inclusion probabilities
    mon, _ = sample_monitoring(access, rng, bias, n=N_MON)
    rng.shuffle(mon)
    ncal = int(len(mon) * CAL_FRAC)
    cal, fit = mon[:ncal], mon[ncal:]
    Xf, yf = X[fit], y[fit]
    Xc, yc = X[cal], y[cal]

    mu_model = RandomForestRegressor(**RF).fit(Xf, yf)
    mu_all, mu_c = mu_model.predict(X), mu_model.predict(Xc)
    w_imp = mu_model.feature_importances_

    sig_model = RandomForestRegressor(**{**RF, "random_state": 1}).fit(
        Xf, np.abs(yf - mu_model.predict(Xf)))
    sig_all, sig_c = sig_model.predict(X), sig_model.predict(Xc)

    qlo_m = HistGradientBoostingRegressor(loss="quantile", quantile=ALPHA/2,
                                          max_iter=200, random_state=seed).fit(Xf, yf)
    qhi_m = HistGradientBoostingRegressor(loss="quantile", quantile=1-ALPHA/2,
                                          max_iter=200, random_state=seed).fit(Xf, yf)
    qlo_all, qhi_all = qlo_m.predict(X), qhi_m.predict(X)
    qlo_c, qhi_c = qlo_m.predict(Xc), qhi_m.predict(Xc)

    di_all, _, _ = aoa.dissimilarity_index(Xf, X, weights=w_imp)
    di_c, _, _ = aoa.dissimilarity_index(Xf, Xc, weights=w_imp)
    feat_all = aoa.transform(Xf, X, w_imp)
    feat_c = aoa.transform(Xf, Xc, w_imp)

    # oracle weighted conformal: the evaluation target is the unmonitored
    # complement, so the test/calibration density ratio at a cell is
    # (1 - pi)/pi with pi the without-replacement inclusion probability
    with np.errstate(divide="ignore"):
        w_oracle_all = (1.0 - pi) / pi
    w_oracle_c = w_oracle_all[cal]
    # estimated density ratio p_test/p_calib via balanced, cross-fitted logistic
    # classifier (calibration=class0 vs a size-matched random grid sample=class1).
    # Balanced classes remove the base-rate; cross-fitting avoids memorising the
    # calibration points; logistic keeps weights smooth. Weights clipped to [1e-2,1e2].
    w_est_c, w_est_all = _estimated_weights(Xf, Xc, X, seed)
    cap = 3.0 * (yf.max() - yf.min())

    di_cqr_intervals = CF.di_cqr(qlo_c, qhi_c, yc, di_c, qlo_all, qhi_all, di_all,
                                 ALPHA, K=K_MAIN, monotone=True, return_info=True)
    # HELD-OUT evaluation set: score only on UNMONITORED cells (the fitting and
    # calibration cells are not held out). Defined here so the width-matched global
    # baseline is matched to DI-CQR's mean width ON THE EVALUATION SET (otherwise the
    # two mean widths differ and the "matched" control is not actually matched).
    ev = np.setdiff1d(np.arange(len(y)), mon)
    di_cqr_half_mean = float(np.mean(di_cqr_intervals[2][ev]))

    # calibration-valid hyperparameter selection (no test labels; Sec. auto)
    K_sel = CF.select_K_di_cqr(qlo_c, qhi_c, yc, di_c, ALPHA, min_n=15, seed=seed)
    kap_sel = CF.select_kappa_di_normalized(mu_c, yc, di_c, ALPHA, seed=seed)

    # localized conformal (Guan 2023) with a fixed kernel: bandwidth = median
    # pairwise distance among FITTING points (label-free, chosen before
    # calibration), in the DI feature space and in coordinate space
    bw_feat = CF.median_bandwidth(aoa.transform(Xf, Xf, w_imp))
    bw_geo = CF.median_bandwidth(coords[fit])
    lcp_out = CF.lcp(mu_c, yc, feat_c, mu_all, feat_all, ALPHA, bw_feat, return_mass=True)

    raw = {
        "split": CF.split(mu_c, yc, mu_all, ALPHA),
        "normalized": CF.normalized(mu_c, yc, sig_c, mu_all, sig_all, ALPHA),
        "di_normalized": CF.di_normalized(mu_c, yc, di_c, mu_all, di_all, ALPHA),
        "di_normalized_auto": CF.di_normalized(mu_c, yc, di_c, mu_all, di_all, ALPHA, floor=kap_sel),
        "di_cqr_auto": CF.di_cqr(qlo_c, qhi_c, yc, di_c, qlo_all, qhi_all, di_all, ALPHA, K=K_sel),
        "region_mondrian": CF.region_mondrian(mu_c, yc, reg[cal], mu_all, reg, ALPHA),
        "di_mondrian": CF.di_mondrian(mu_c, yc, di_c, mu_all, di_all, ALPHA, K=K_MAIN),
        "cqr": CF.cqr(qlo_c, qhi_c, yc, qlo_all, qhi_all, ALPHA),
        "di_cqr": di_cqr_intervals[:3],
        "di_normalized_clip": CF.di_normalized_clipped(mu_c, yc, di_c, mu_all, di_all, ALPHA),
        "lcp": lcp_out[:3],
        "lcp_cqr": CF.lcp_cqr(qlo_c, qhi_c, yc, feat_c, qlo_all, qhi_all, feat_all, ALPHA, bw_feat),
        "geo_lcp": CF.lcp(mu_c, yc, coords[cal], mu_all, coords, ALPHA, bw_geo),
        "weighted_oracle": CF.weighted_split(mu_c, yc, w_oracle_c, mu_all, w_oracle_all, ALPHA),
        "weighted_estimated": CF.weighted_split(mu_c, yc, w_est_c, mu_all, w_est_all, ALPHA),
        "weighted_proxy": CF.weighted_split(mu_c, yc, 1.0 / p_sel[cal], mu_all, 1.0 / p_sel, ALPHA),
        "width_matched_global": CF.width_matched_global(mu_all, di_cqr_half_mean * 2),
    }
    # methods that can return unbounded intervals: record the unbounded fraction
    # on held-out cells, then cap the offset at `cap` so that width and interval
    # score stay finite (the cap and the unbounded fraction are both reported)
    unbounded = {m: float(np.mean(~np.isfinite(raw[m][2][ev]))) for m in UNBOUNDED_METHODS}
    intervals = {}
    for name, (lo, hi, half) in raw.items():
        if name == "lcp_cqr":
            lo = np.where(np.isfinite(lo), lo, qlo_all - cap)
            hi = np.where(np.isfinite(hi), hi, qhi_all + cap)
            half = (hi - lo) / 2.0
        elif name in UNBOUNDED_METHODS:
            half = np.where(np.isfinite(half), half, cap)
            lo, hi = mu_all - half, mu_all + half
        intervals[name] = (lo, hi, half)
    res = {"regime": regime, "bias": bias, "access": access_mode, "seed": seed,
           "n_eval": int(len(ev)), "n_mon": int(len(mon)), "methods": {}}
    for name, (lo, hi, half) in intervals.items():
        s, _ = _summ(reg[ev], y[ev], lo[ev], hi[ev], di_all[ev])
        res["methods"][name] = s
    res["di_cqr_bin_counts"] = di_cqr_intervals[3]
    res["selected"] = {"K": int(K_sel), "kappa": float(kap_sel)}

    # fraction of held-out cells with an unbounded interval (before capping)
    res["unbounded"] = unbounded
    res["cap_half_width"] = float(cap)
    # Kish effective sample size of the calibration weights (oracle / estimated)
    res["weight_ess"] = {nm: float(w.sum() ** 2 / np.sum(w ** 2)) for nm, w in
                         [("weighted_oracle", w_oracle_c), ("weighted_estimated", w_est_c),
                          ("weighted_proxy", 1.0 / p_sel[cal])]}

    # ---- region-definition sensitivity: worst-region coverage (held-out) under
    # 2x2 / 3x3 / 4x4 partitions, to show the "per-region" headline is not an
    # artifact of one grid (k=3 reproduces the main worst-region coverage).
    yv = y[ev]
    region_sens = {}
    for name in ["split", "di_normalized", "di_cqr"]:
        lo, hi, _ = intervals[name]
        cov = (yv >= lo[ev]) & (yv <= hi[ev])
        d = {}
        for k in REGION_KS:
            rk = region_ids(coords, k)[ev]
            d[str(k)] = float(min(cov[rk == r].mean() for r in np.unique(rk)))
        region_sens[name] = d
    res["region_sens"] = region_sens

    # ---- kappa sensitivity for DI-normalized (worst-region cov + interval score)
    kappa_sens = {}
    for kap in KAPPAS:
        lo, hi, _ = CF.di_normalized(mu_c, yc, di_c, mu_all, di_all, ALPHA, floor=kap)
        s, _ = _summ(reg[ev], y[ev], lo[ev], hi[ev], di_all[ev])
        kappa_sens[str(kap)] = dict(worst_region_coverage=s["worst_region_coverage"],
                                    mean_interval_score=s["mean_interval_score"],
                                    mean_width=s["mean_width"])
    res["kappa_sens"] = kappa_sens

    # ---- clipping diagnostics (DI-CQR): queries whose DI exceeds the maximum
    # calibration DI are assigned the top (widest) bin. Report how many held-out
    # points are clipped and their coverage / width / interval score, so the
    # extrapolative tail is not hidden behind the marginal average.
    lo_d, hi_d, _ = intervals["di_cqr"]
    di_cal_max = float(di_c.max())
    di_ev = di_all[ev]
    clipped = di_ev > di_cal_max
    ycov = (y[ev] >= lo_d[ev]) & (y[ev] <= hi_d[ev])
    isc = CF._interval_score(y[ev], lo_d[ev], hi_d[ev], ALPHA)
    wid = hi_d[ev] - lo_d[ev]

    def _cd(mask):
        if int(mask.sum()) == 0:
            return dict(n=0, frac=0.0, coverage=None,
                        mean_width=None, mean_interval_score=None)
        return dict(n=int(mask.sum()), frac=float(mask.mean()),
                    coverage=float(ycov[mask].mean()),
                    mean_width=float(wid[mask].mean()),
                    mean_interval_score=float(isc[mask].mean()))
    res["clip"] = {"clipped": _cd(clipped), "unclipped": _cd(~clipped),
                   "di_cal_max": di_cal_max}
    # localized-conformal mechanism: kernel-weighted calibration mass near each
    # query (the query's own atom has weight 1) and the unbounded rate, split by
    # whether the query lies beyond the calibration DI range
    mass_ev = lcp_out[3][ev]
    inf_ev = ~np.isfinite(raw["lcp"][2][ev])
    res["lcp_mass"] = {
        grp: (dict(median_mass=float(np.median(mass_ev[msk])),
                   frac_unbounded=float(inf_ev[msk].mean()))
              if msk.sum() else dict(median_mass=None, frac_unbounded=None))
        for grp, msk in [("clipped", clipped), ("unclipped", ~clipped)]}

    # DI vs the oracle covariate-shift weight (only meaningful when biased),
    # on held-out cells. (1-pi)/pi is a decreasing function of p_sel, so its
    # rank correlation with DI equals that of 1/p_sel.
    if bias > 0:
        from scipy.stats import spearmanr
        w_ev = w_oracle_all[ev]
        res["di_selweight_spearman"] = float(spearmanr(di_all[ev], w_ev).correlation)
        res["di_logselweight_pearson"] = float(np.corrcoef(di_all[ev], np.log(w_ev))[0, 1])
    else:
        res["di_selweight_spearman"] = None
        res["di_logselweight_pearson"] = None

    # ---- DI-bin ablation (reuse E, di) : di_cqr and di_mondrian, K x monotone
    E = np.maximum(qlo_c - yc, yc - qhi_c)
    s_res = np.abs(yc - mu_c)
    abl = {}
    for K in ABLATION_K:
        for mono in [True, False]:
            edges, Qb, counts = CF._di_bin_quantiles(E, di_c, ALPHA, K, mono, min_n=15)
            bq = np.clip(np.digitize(di_all, edges) - 1, 0, K - 1)
            Qv = Qb[bq]
            lo, hi = qlo_all - Qv, qhi_all + Qv
            s, _ = _summ(reg[ev], y[ev], lo[ev], hi[ev], di_all[ev])
            s["min_bin_count"] = int(min(counts)); s["mean_bin_count"] = float(np.mean(counts))
            abl[f"di_cqr_K{K}_mono{int(mono)}"] = s
    res["ablation"] = abl

    # ---- coverage vs width multiplier sweep
    sweep = {}
    for name in ["split", "cqr", "di_cqr", "weighted_oracle"]:
        lo, hi, half = intervals[name]
        mu_center = 0.5 * (lo + hi)
        rows = []
        for mult in MULTS:
            lo2 = mu_center - half * mult; hi2 = mu_center + half * mult
            rt = MET.per_region(reg[ev], y[ev], lo2[ev], hi2[ev], ALPHA)
            covs = np.array([v["coverage"] for v in rt.values()])
            rows.append(dict(mult=mult, marginal=MET.marginal_coverage(y[ev], lo2[ev], hi2[ev]),
                             worst=float(covs.min()),
                             mean_width=float(np.mean((hi2 - lo2)[ev]))))
        sweep[name] = rows
    res["sweep"] = sweep

    if save_fields:
        lo_s, hi_s, _ = intervals["split"]; lo_d, hi_d, hf_d = intervals["di_cqr"]
        res["_fields"] = dict(GRID=GRID, di=di_all.tolist(), reg=reg.tolist(),
                              in_split=((y >= lo_s) & (y <= hi_s)).astype(int).tolist(),
                              in_dicqr=((y >= lo_d) & (y <= hi_d)).astype(int).tolist(),
                              half_split=(0.5*(hi_s-lo_s)).tolist(), half_dicqr=hf_d.tolist(),
                              cal=cal.tolist(), fit=fit.tolist(), mon=mon.tolist(),
                              coords=coords.tolist(), access=access.tolist(),
                              p_sel=p_sel.tolist(), inv_psel=(1.0/p_sel).tolist(),
                              w_oracle=w_oracle_all.tolist(), ev=ev.tolist())
    return res


def done():
    s = set()
    if os.path.exists(RAW):
        for l in open(RAW):
            try:
                r = json.loads(l); s.add((r["regime"], r["seed"]))
            except Exception: pass
    return s


def work():
    t0 = time.time(); d = done()
    pairs = [(rg, sd) for rg in REGIMES for sd in range(SEEDS)]
    todo = [p for p in pairs if p not in d]
    n = 0
    with open(RAW, "a") as f:
        for rg, sd in todo:
            if time.time() - t0 > BUDGET: break
            r = one_run(rg, sd)
            f.write(json.dumps(r) + "\n"); f.flush(); n += 1
    print(f"processed {n}; total {len(d)+n}/{len(pairs)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "agg":
        import aggregate; aggregate.main()
    else:
        work()
