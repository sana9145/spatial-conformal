"""run_meuse.py -- semi-synthetic monitoring-bias experiment on REAL data.

Meuse dataset (155 floodplain soil samples; gstat/sp ecosystem, bundled offline
via scikit-gstat). Target: log zinc. Predictors: dist (normalised distance to
river), elev, om (organic matter), ffreq (flood-frequency class), soil, dist.m.

This is a SEMI-SYNTHETIC monitoring-bias experiment on real environmental
observations, NOT a naturally biased monitoring network: we impose the bias by
repeatedly drawing a geographically CLUSTERED set of monitored sites (nearest to
a random seed location), splitting it into fitting and calibration subsets, and
holding out the geographically SEPARATED remaining measured sites as the test
set. Ground truth at test sites is the real measured zinc.

K=4 DI bins chosen a priori (small calibration set ~40 -> ~10 points/bin), not
tuned on test outcomes.
"""
import json, os, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import aoa, metrics as MET, conformal as CF

RES = os.path.join(os.path.dirname(__file__), "..", "results")
PRED = ["dist", "elev", "om", "ffreq", "soil", "dist.m"]
ALPHA = 0.10
NOMINAL = 0.9
N_MON = 100          # monitored (clustered) of 155; rest are far test sites
CAL_FRAC = 0.4
K = 4
REPS = 30
RF = dict(n_estimators=300, n_jobs=-1, min_samples_leaf=2, random_state=0)
METHODS = ["split", "normalized", "di_normalized", "region_mondrian",
           "di_mondrian", "cqr", "di_cqr", "localized", "geo_localized",
           "weighted_estimated"]


def load():
    df = pd.read_csv(os.path.join(RES, "meuse_raw.csv"))
    df = df.copy()
    df["logzinc"] = np.log(df["zinc"].to_numpy())
    return df


def est_weights(Xf, Xc, Xt, seed, clip=(1e-2, 1e2)):
    sc = StandardScaler().fit(Xf)
    Zc, Zt = sc.transform(Xc), sc.transform(Xt)
    rng = np.random.default_rng(1000 + seed)
    n = len(Zc)
    def ratio(clf, Z):
        p = np.clip(clf.predict_proba(Z)[:, 1], 1e-4, 1-1e-4)
        return np.clip(p/(1-p), *clip)
    m = min(n, len(Zt))
    ti = rng.choice(len(Zt), size=m, replace=False)
    clf = LogisticRegression(max_iter=500).fit(
        np.vstack([Zc[:m] if m < n else Zc, Zt[ti]]),
        np.r_[np.zeros(m if m < n else n), np.ones(m)])
    w_t = ratio(clf, Zt)
    # cross-fit calibration weights
    from sklearn.model_selection import KFold
    w_c = np.empty(n)
    for tr, te in KFold(min(5, n), shuffle=True, random_state=seed).split(Zc):
        mm = min(len(tr), len(Zt))
        gi = rng.choice(len(Zt), size=mm, replace=False)
        clf2 = LogisticRegression(max_iter=500).fit(
            np.vstack([Zc[tr][:mm], Zt[gi]]), np.r_[np.zeros(mm), np.ones(mm)])
        w_c[te] = ratio(clf2, Zc[te])
    return w_c, w_t


def one_rep(df, seed):
    rng = np.random.default_rng(seed)
    coords = df[["x", "y"]].to_numpy(float)
    center = coords[rng.integers(len(df))]
    d = np.linalg.norm(coords - center, axis=1)
    mon = np.argsort(d)[:N_MON]                       # clustered monitored
    test = np.setdiff1d(np.arange(len(df)), mon)      # far held-out sites
    rng.shuffle(mon)
    ncal = int(len(mon) * CAL_FRAC)
    cal, fit = mon[:ncal], mon[ncal:]

    # median-impute om using FIT statistics only (leakage control)
    Xall = df[PRED].to_numpy(float)
    fit_med = np.nanmedian(Xall[fit], axis=0)
    Xall = np.where(np.isnan(Xall), fit_med, Xall)
    y = df["logzinc"].to_numpy()

    Xf, yf = Xall[fit], y[fit]
    Xc, yc = Xall[cal], y[cal]
    Xt, yt = Xall[test], y[test]

    # regions: 4 spatial quadrants from KMeans on ALL coords (fixed geography)
    reg_all = KMeans(4, n_init=10, random_state=0).fit_predict(coords)
    reg_t = reg_all[test]

    mu = RandomForestRegressor(**RF).fit(Xf, yf)
    w_imp = mu.feature_importances_
    mu_t, mu_c = mu.predict(Xt), mu.predict(Xc)
    sig = RandomForestRegressor(**{**RF, "random_state": 1}).fit(Xf, np.abs(yf-mu.predict(Xf)))
    sig_t, sig_c = sig.predict(Xt), sig.predict(Xc)
    qlo = HistGradientBoostingRegressor(loss="quantile", quantile=ALPHA/2, max_iter=150, random_state=seed).fit(Xf, yf)
    qhi = HistGradientBoostingRegressor(loss="quantile", quantile=1-ALPHA/2, max_iter=150, random_state=seed).fit(Xf, yf)
    qlo_t, qhi_t = qlo.predict(Xt), qhi.predict(Xt)
    qlo_c, qhi_c = qlo.predict(Xc), qhi.predict(Xc)
    di_t, _, _ = aoa.dissimilarity_index(Xf, Xt, weights=w_imp)
    di_c, _, _ = aoa.dissimilarity_index(Xf, Xc, weights=w_imp)
    feat_t = aoa.transform(Xf, Xt, w_imp); feat_c = aoa.transform(Xf, Xc, w_imp)
    coords_t, coords_c = coords[test], coords[cal]
    w_c, w_t = est_weights(Xf, Xc, Xt, seed)
    cap = 3.0 * (yf.max() - yf.min())

    iv = {
        "split": CF.split(mu_c, yc, mu_t, ALPHA),
        "normalized": CF.normalized(mu_c, yc, sig_c, mu_t, sig_t, ALPHA),
        "di_normalized": CF.di_normalized(mu_c, yc, di_c, mu_t, di_t, ALPHA),
        "region_mondrian": CF.region_mondrian(mu_c, yc, reg_all[cal], mu_t, reg_t, ALPHA, min_n=6),
        "di_mondrian": CF.di_mondrian(mu_c, yc, di_c, mu_t, di_t, ALPHA, K=K, min_n=6),
        "cqr": CF.cqr(qlo_c, qhi_c, yc, qlo_t, qhi_t, ALPHA),
        "di_cqr": CF.di_cqr(qlo_c, qhi_c, yc, di_c, qlo_t, qhi_t, di_t, ALPHA, K=K, min_n=6),
        "localized": CF.localized_tuned(mu_c, yc, feat_c, mu_t, feat_t, ALPHA),
        "geo_localized": CF.localized_tuned(mu_c, yc, coords_c, mu_t, coords_t, ALPHA),
        "weighted_estimated": CF.weighted_split(mu_c, yc, w_c, mu_t, w_t, ALPHA, cap=cap),
    }
    # DI-CQR bin diagnostics: exact fit/cal sizes, per-bin calibration counts,
    # how many of the K bins fell back to the global quantile (count < min_n=6),
    # and the fraction of test queries clipped into the top DI bin.
    E_c = np.maximum(qlo_c - yc, yc - qhi_c)
    edges_m, _, counts_m = CF._di_bin_quantiles(E_c, di_c, ALPHA, K, True, min_n=6)
    n_fallback = int(sum(c < 6 for c in counts_m))
    frac_clip = float(np.mean(di_t > di_c.max()))
    out = {"seed": seed, "n_test": int(len(test)),
           "n_fit": int(len(fit)), "n_cal": int(len(cal)), "K": K, "min_n": 6,
           "bin_counts": [int(c) for c in counts_m], "n_fallback_bins": n_fallback,
           "frac_test_clipped": frac_clip, "methods": {}}
    for name, (lo, hi, half) in iv.items():
        rt = MET.per_region(reg_t, yt, lo, hi, ALPHA)
        covs = np.array([v["coverage"] for v in rt.values()])
        out["methods"][name] = dict(
            marginal=MET.marginal_coverage(yt, lo, hi),
            worst_region=float(covs.min()), coverage_gap=float(covs.max()-covs.min()),
            mean_width=float(np.mean(hi-lo)),
            mean_interval_score=float(np.mean(MET.interval_score(yt, lo, hi, ALPHA))))
    return out


RAW = os.path.join(RES, "meuse_raw_results.jsonl")


def work():
    import time
    t0 = time.time()
    df = load()
    done = set()
    if os.path.exists(RAW):
        done = {json.loads(l)["seed"] for l in open(RAW)}
    n = 0
    with open(RAW, "a") as f:
        for s in range(REPS):
            if s in done:
                continue
            if time.time() - t0 > 35:
                break
            f.write(json.dumps(one_rep(df, s)) + "\n"); f.flush(); n += 1
    print(f"processed {n}; total {len(done)+n}/{REPS}")


def main():
    runs = [json.loads(l) for l in open(RAW)]
    from scipy import stats
    rows = []
    for m in METHODS:
        for k in ["marginal", "worst_region", "coverage_gap", "mean_width", "mean_interval_score"]:
            v = np.array([r["methods"][m][k] for r in runs])
            se = v.std(ddof=1)/np.sqrt(len(v)); t = stats.t.ppf(0.975, len(v)-1)
            rows.append(dict(method=m, metric=k, mean=v.mean(), sd=v.std(ddof=1),
                             ci_lo=v.mean()-t*se, ci_hi=v.mean()+t*se))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "meuse_summary.csv"), index=False)

    # DI-CQR bin diagnostics (exact sizes + fallback frequency), averaged over reps
    diag = dict(
        n_fit=int(np.median([r.get("n_fit", np.nan) for r in runs])),
        n_cal=int(np.median([r.get("n_cal", np.nan) for r in runs])),
        K=int(runs[0].get("K", 4)), min_n=int(runs[0].get("min_n", 6)),
        mean_bin_count=float(np.mean([np.mean(r["bin_counts"]) for r in runs if "bin_counts" in r])),
        min_bin_count=float(np.mean([min(r["bin_counts"]) for r in runs if "bin_counts" in r])),
        mean_fallback_bins=float(np.mean([r.get("n_fallback_bins", 0) for r in runs])),
        frac_reps_any_fallback=float(np.mean([r.get("n_fallback_bins", 0) > 0 for r in runs])),
        mean_frac_test_clipped=float(np.mean([r.get("frac_test_clipped", np.nan) for r in runs])))
    json.dump(diag, open(os.path.join(RES, "meuse_diag.json"), "w"), indent=2)

    # print
    piv = pd.DataFrame(rows).pivot(index="method", columns="metric", values="mean")
    print(piv.loc[METHODS, ["marginal","worst_region","coverage_gap","mean_width","mean_interval_score"]].round(3).to_string())
    print("DIAG:", diag)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "agg":
        main()
    else:
        work()
