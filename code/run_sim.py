"""run_sim.py -- main simulation benchmark (resumable).

Regimes: no-bias (b=0), mild (b=2), moderate (b=6), severe (b=12).
Per (regime, seed): fit models on the FITTING split only, calibrate on the
CALIBRATION split, evaluate interval coverage over the whole landscape (true y
known). Also computes the DI-bin ablation (cheap re-binning) and a
coverage-vs-width multiplier sweep. Append one JSON line per run.

Usage:
  python3 run_sim.py            # process pairs within time budget
  python3 run_sim.py agg        # aggregate -> ../results/*.csv/json + fields
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
from landscape import make_landscape, region_ids, sample_monitoring, selection_prob, GRID

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
RAW = os.path.join(RESULTS, "sim_raw.jsonl")
ALPHA = 0.10
NOMINAL = 1 - ALPHA
N_MON = 500
CAL_FRAC = 0.4
K_MAIN = 5
SEEDS = 30
REGIMES = {"none": 0.0, "mild": 2.0, "moderate": 6.0, "severe": 12.0}
RF = dict(n_estimators=150, n_jobs=-1, min_samples_leaf=3, random_state=0)
BASE_METHODS = ["split", "normalized", "di_normalized", "region_mondrian",
                "di_mondrian", "cqr", "di_cqr", "localized", "geo_localized",
                "weighted_oracle", "weighted_estimated", "width_matched_global"]
ABLATION_K = [3, 4, 5, 6, 8, 10]
MULTS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
BUDGET = 37.0


def _summ(reg, y, lo, hi, di):
    rt = MET.per_region(reg, y, lo, hi, ALPHA)
    s = MET.summarize(rt, y, lo, hi, ALPHA, NOMINAL)
    cc = MET.conditional_coverage_by_di(y, lo, hi, di, nbins=10, nominal=NOMINAL)
    s["cond_cov_error_mean"] = cc["cond_cov_error_mean"]
    s["cond_cov_error_max"] = cc["cond_cov_error_max"]
    return s, cc


def one_run(regime, seed, save_fields=False):
    bias = REGIMES[regime]
    rng = np.random.default_rng(seed)
    X, y, coords, access = make_landscape(rng)
    reg = region_ids(coords)
    p_sel = selection_prob(access, bias)
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

    # weighted conformal weights
    w_oracle_c = 1.0 / p_sel[cal]
    w_oracle_all = 1.0 / p_sel
    # estimated density ratio p_test/p_calib via balanced, cross-fitted logistic
    # classifier (calibration=class0 vs a size-matched random grid sample=class1).
    # Balanced classes remove the base-rate; cross-fitting avoids memorising the
    # calibration points; logistic keeps weights smooth. Weights clipped to [1e-2,1e2].
    w_est_c, w_est_all = _estimated_weights(Xf, Xc, X, seed)
    cap = 3.0 * (yf.max() - yf.min())

    di_cqr_intervals = CF.di_cqr(qlo_c, qhi_c, yc, di_c, qlo_all, qhi_all, di_all,
                                 ALPHA, K=K_MAIN, monotone=True, return_info=True)
    di_cqr_half_mean = float(np.mean(di_cqr_intervals[2]))

    intervals = {
        "split": CF.split(mu_c, yc, mu_all, ALPHA),
        "normalized": CF.normalized(mu_c, yc, sig_c, mu_all, sig_all, ALPHA),
        "di_normalized": CF.di_normalized(mu_c, yc, di_c, mu_all, di_all, ALPHA),
        "region_mondrian": CF.region_mondrian(mu_c, yc, reg[cal], mu_all, reg, ALPHA),
        "di_mondrian": CF.di_mondrian(mu_c, yc, di_c, mu_all, di_all, ALPHA, K=K_MAIN),
        "cqr": CF.cqr(qlo_c, qhi_c, yc, qlo_all, qhi_all, ALPHA),
        "di_cqr": di_cqr_intervals[:3],
        "localized": CF.localized_tuned(mu_c, yc, feat_c, mu_all, feat_all, ALPHA),
        "geo_localized": CF.localized_tuned(mu_c, yc, coords[cal], mu_all, coords, ALPHA),
        "weighted_oracle": CF.weighted_split(mu_c, yc, w_oracle_c, mu_all, w_oracle_all, ALPHA, cap=cap),
        "weighted_estimated": CF.weighted_split(mu_c, yc, w_est_c, mu_all, w_est_all, ALPHA, cap=cap),
        "width_matched_global": CF.width_matched_global(mu_all, di_cqr_half_mean * 2),
    }
    res = {"regime": regime, "bias": bias, "seed": seed, "methods": {}}
    for name, (lo, hi, half) in intervals.items():
        s, _ = _summ(reg, y, lo, hi, di_all)
        res["methods"][name] = s
    res["di_cqr_bin_counts"] = di_cqr_intervals[3]

    # DI vs inverse selection weight bridge (only meaningful when biased)
    if bias > 0:
        from scipy.stats import spearmanr
        inv = 1.0 / p_sel
        res["di_selweight_spearman"] = float(spearmanr(di_all, inv).correlation)
        res["di_logselweight_pearson"] = float(np.corrcoef(di_all, np.log(inv))[0, 1])
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
            s, _ = _summ(reg, y, lo, hi, di_all)
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
            rt = MET.per_region(reg, y, lo2, hi2, ALPHA)
            covs = np.array([v["coverage"] for v in rt.values()])
            rows.append(dict(mult=mult, marginal=MET.marginal_coverage(y, lo2, hi2),
                             worst=float(covs.min()),
                             mean_width=float(np.mean(hi2 - lo2))))
        sweep[name] = rows
    res["sweep"] = sweep

    if save_fields:
        lo_s, hi_s, _ = intervals["split"]; lo_d, hi_d, hf_d = intervals["di_cqr"]
        res["_fields"] = dict(GRID=GRID, di=di_all.tolist(), reg=reg.tolist(),
                              in_split=((y >= lo_s) & (y <= hi_s)).astype(int).tolist(),
                              in_dicqr=((y >= lo_d) & (y <= hi_d)).astype(int).tolist(),
                              half_split=(0.5*(hi_s-lo_s)).tolist(), half_dicqr=hf_d.tolist(),
                              cal=cal.tolist(), coords=coords.tolist(),
                              inv_psel=(1.0/p_sel).tolist())
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
