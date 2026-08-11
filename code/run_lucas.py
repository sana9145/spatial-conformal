"""run_lucas.py -- semi-synthetic monitoring-bias experiment on LUCAS 2015 topsoil.

Real environmental data (21,687 EU points; predict log soil organic carbon from
bioclim + terrain covariates). As with Meuse, we impose a SEMI-SYNTHETIC
monitoring bias: repeatedly draw a geographically clustered monitored set (nearest
N_MON points to a random seed), split it into fitting/calibration, and hold out
the geographically separated remaining sites as test (subsampled for tractability).
Worst-region coverage uses the fixed 10 k-means regions. Same methods, metrics and
leakage controls as run_meuse. Resumable.
"""
import json, os, sys, time, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import aoa, metrics as MET, conformal as CF
from run_meuse import est_weights

RES = os.path.join(os.path.dirname(__file__), "..", "results")
RAW = os.path.join(RES, "lucas_raw.jsonl")
ALPHA, NOMINAL = 0.10, 0.90
N_MON, CAL_FRAC, N_TEST, K = 1800, 0.4, 2500, 5
REPS = 20
BUDGET = 33.0
RF = dict(n_estimators=150, n_jobs=-1, min_samples_leaf=3, random_state=0)
METHODS = ["split", "normalized", "di_normalized", "region_mondrian",
           "di_mondrian", "cqr", "di_cqr", "localized", "geo_localized",
           "weighted_estimated"]
_DF = None
_PRED = None


def load():
    global _DF, _PRED
    if _DF is None:
        _DF = pd.read_csv(os.path.join(RES, "lucas_prepared.csv"))
        _PRED = [c for c in _DF.columns if c not in ("Point_ID", "x", "y", "region", "logOC")]
    return _DF, _PRED


def one_rep(seed):
    df, PRED = load()
    rng = np.random.default_rng(seed)
    coords = df[["x", "y"]].to_numpy(float)
    X = df[PRED].to_numpy(float)
    y = df["logOC"].to_numpy(float)
    region = df["region"].to_numpy(int)

    center = coords[rng.integers(len(df))]
    d = np.linalg.norm(coords - center, axis=1)
    mon = np.argsort(d)[:N_MON]
    far = np.setdiff1d(np.arange(len(df)), mon)
    test = far if len(far) <= N_TEST else rng.choice(far, size=N_TEST, replace=False)
    rng.shuffle(mon)
    ncal = int(len(mon) * CAL_FRAC)
    cal, fit = mon[:ncal], mon[ncal:]

    Xf, yf = X[fit], y[fit]
    Xc, yc = X[cal], y[cal]
    Xt, yt = X[test], y[test]
    reg_t = region[test]

    mu = RandomForestRegressor(**RF).fit(Xf, yf)
    w_imp = mu.feature_importances_
    mu_t, mu_c = mu.predict(Xt), mu.predict(Xc)
    sig = RandomForestRegressor(**{**RF, "random_state": 1}).fit(Xf, np.abs(yf - mu.predict(Xf)))
    sig_t, sig_c = sig.predict(Xt), sig.predict(Xc)
    qlo = HistGradientBoostingRegressor(loss="quantile", quantile=ALPHA/2, max_iter=200, random_state=seed).fit(Xf, yf)
    qhi = HistGradientBoostingRegressor(loss="quantile", quantile=1-ALPHA/2, max_iter=200, random_state=seed).fit(Xf, yf)
    qlo_t, qhi_t, qlo_c, qhi_c = qlo.predict(Xt), qhi.predict(Xt), qlo.predict(Xc), qhi.predict(Xc)
    di_t, _, _ = aoa.dissimilarity_index(Xf, Xt, weights=w_imp)
    di_c, _, _ = aoa.dissimilarity_index(Xf, Xc, weights=w_imp)
    feat_t, feat_c = aoa.transform(Xf, Xt, w_imp), aoa.transform(Xf, Xc, w_imp)
    ct, cc = coords[test], coords[cal]
    w_c, w_t = est_weights(Xf, Xc, Xt, seed)
    cap = 3.0 * (yf.max() - yf.min())

    iv = {
        "split": CF.split(mu_c, yc, mu_t, ALPHA),
        "normalized": CF.normalized(mu_c, yc, sig_c, mu_t, sig_t, ALPHA),
        "di_normalized": CF.di_normalized(mu_c, yc, di_c, mu_t, di_t, ALPHA),
        "region_mondrian": CF.region_mondrian(mu_c, yc, region[cal], mu_t, reg_t, ALPHA, min_n=20),
        "di_mondrian": CF.di_mondrian(mu_c, yc, di_c, mu_t, di_t, ALPHA, K=K),
        "cqr": CF.cqr(qlo_c, qhi_c, yc, qlo_t, qhi_t, ALPHA),
        "di_cqr": CF.di_cqr(qlo_c, qhi_c, yc, di_c, qlo_t, qhi_t, di_t, ALPHA, K=K),
        "localized": CF.localized_tuned(mu_c, yc, feat_c, mu_t, feat_t, ALPHA),
        "geo_localized": CF.localized_tuned(mu_c, yc, cc, mu_t, ct, ALPHA),
        "weighted_estimated": CF.weighted_split(mu_c, yc, w_c, mu_t, w_t, ALPHA, cap=cap),
    }
    out = {"seed": seed, "n_test": int(len(test)), "methods": {}}
    for name, (lo, hi, half) in iv.items():
        rt = MET.per_region(reg_t, yt, lo, hi, ALPHA)
        covs = np.array([v["coverage"] for v in rt.values()])
        out["methods"][name] = dict(marginal=MET.marginal_coverage(yt, lo, hi),
            worst_region=float(covs.min()), coverage_gap=float(covs.max()-covs.min()),
            mean_width=float(np.mean(hi-lo)),
            mean_interval_score=float(np.mean(MET.interval_score(yt, lo, hi, ALPHA))))
    return out


def work():
    t0 = time.time()
    done = {json.loads(l)["seed"] for l in open(RAW)} if os.path.exists(RAW) else set()
    n = 0
    with open(RAW, "a") as f:
        for s in range(REPS):
            if s in done: continue
            if time.time() - t0 > BUDGET: break
            f.write(json.dumps(one_rep(s)) + "\n"); f.flush(); n += 1
    print(f"processed {n}; total {len(done)+n}/{REPS}")


def agg():
    from scipy import stats
    runs = [json.loads(l) for l in open(RAW)]
    rows = []
    for r in runs:
        for m in METHODS:
            s = r["methods"][m]
            rows.append(dict(seed=r["seed"], method=m, **s))
    df = pd.DataFrame(rows)
    g = df.groupby("method").agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    g = g.reset_index()
    g.to_csv(os.path.join(RES, "lucas_summary.csv"), index=False)
    sig = {}
    for tgt in ["di_normalized", "di_cqr"]:
        for base in ["split", "localized"]:
            a = df[df.method == tgt].sort_values("seed")["mean_interval_score"].to_numpy()
            b = df[df.method == base].sort_values("seed")["mean_interval_score"].to_numpy()
            sig[f"{tgt}-{base}"] = dict(diff=float((a-b).mean()),
                t_p=float(stats.ttest_rel(a, b).pvalue), w_p=float(stats.wilcoxon(a, b).pvalue))
    json.dump(sig, open(os.path.join(RES, "lucas_sig.json"), "w"), indent=2)
    print(g[["method", "marginal_mean", "worst_region_mean", "mean_width_mean", "mean_interval_score_mean"]].round(3).to_string(index=False))
    print("sig:", {k: (round(v["diff"], 2), f"{v['t_p']:.1e}", f"{v['w_p']:.1e}") for k, v in sig.items()})


if __name__ == "__main__":
    (agg if len(sys.argv) > 1 and sys.argv[1] == "agg" else work)()
