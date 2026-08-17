"""run_robust.py -- second base-model family robustness check.

Repeats the benchmark with a HISTOGRAM GRADIENT BOOSTING mean model in place of
the random forest (quantile models are histogram-boosting in both settings), to
test that the DI-conditioning finding is not specific to random forests. DI
feature weights are taken from permutation importance of the HGB mean model
(random forests expose feature_importances_; HGB does not). Moderate and severe
regimes, 30 seeds; same methods, metrics, and leakage controls as run_sim.

  python3 run_robust.py         # resumable -> ../results/robust_raw.jsonl
  python3 run_robust.py agg     # -> ../results/robust_summary.csv, table_robust.tex
"""
import json, os, sys, time, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import aoa, metrics as MET, conformal as CF
from landscape import make_landscape, region_ids, sample_monitoring, selection_prob

RES = os.path.join(os.path.dirname(__file__), "..", "results")
PAP = os.path.join(os.path.dirname(__file__), "..", "paper")
RAW = os.path.join(RES, "robust_raw.jsonl")
ALPHA, NOMINAL, N_MON, CAL_FRAC, K = 0.10, 0.90, 500, 0.4, 5
REGIMES = {"moderate": 6.0, "severe": 12.0}
SEEDS = 30
BUDGET = 34.0
HGB = dict(max_iter=200)
METHODS = ["split", "normalized", "di_normalized", "cqr", "di_cqr",
           "localized", "weighted_oracle", "width_matched_global"]


def one_run(regime, seed):
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

    # HGB mean model (the changed base-model family)
    mu_model = HistGradientBoostingRegressor(random_state=seed, **HGB).fit(Xf, yf)
    mu_all, mu_c = mu_model.predict(X), mu_model.predict(Xc)
    # DI weights via permutation importance of the HGB mean model (fitting split)
    pi = permutation_importance(mu_model, Xf, yf, n_repeats=5, random_state=seed)
    w_imp = np.clip(pi.importances_mean, 0, None)
    if w_imp.sum() <= 1e-12:
        w_imp = np.ones(Xf.shape[1])

    sig_model = HistGradientBoostingRegressor(random_state=seed + 1, **HGB).fit(
        Xf, np.abs(yf - mu_model.predict(Xf)))
    sig_all, sig_c = sig_model.predict(X), sig_model.predict(Xc)

    qlo = HistGradientBoostingRegressor(loss="quantile", quantile=ALPHA/2, random_state=seed, **HGB).fit(Xf, yf)
    qhi = HistGradientBoostingRegressor(loss="quantile", quantile=1-ALPHA/2, random_state=seed, **HGB).fit(Xf, yf)
    qlo_all, qhi_all, qlo_c, qhi_c = qlo.predict(X), qhi.predict(X), qlo.predict(Xc), qhi.predict(Xc)

    di_all, _, _ = aoa.dissimilarity_index(Xf, X, weights=w_imp)
    di_c, _, _ = aoa.dissimilarity_index(Xf, Xc, weights=w_imp)
    feat_all, feat_c = aoa.transform(Xf, X, w_imp), aoa.transform(Xf, Xc, w_imp)
    cap = 3.0 * (yf.max() - yf.min())

    dicqr = CF.di_cqr(qlo_c, qhi_c, yc, di_c, qlo_all, qhi_all, di_all, ALPHA, K=K)
    iv = {
        "split": CF.split(mu_c, yc, mu_all, ALPHA),
        "normalized": CF.normalized(mu_c, yc, sig_c, mu_all, sig_all, ALPHA),
        "di_normalized": CF.di_normalized(mu_c, yc, di_c, mu_all, di_all, ALPHA),
        "cqr": CF.cqr(qlo_c, qhi_c, yc, qlo_all, qhi_all, ALPHA),
        "di_cqr": dicqr,
        "localized": CF.localized_tuned(mu_c, yc, feat_c, mu_all, feat_all, ALPHA),
        "weighted_oracle": CF.weighted_split(mu_c, yc, 1.0/p_sel[cal], mu_all, 1.0/p_sel, ALPHA, cap=cap),
        "width_matched_global": CF.width_matched_global(mu_all, float(np.mean(dicqr[2]))*2),
    }
    # HELD-OUT evaluation: unmonitored cells only (exclude fitting+calibration).
    ev = np.setdiff1d(np.arange(len(y)), mon)
    res = {"regime": regime, "seed": seed, "n_eval": int(len(ev)), "methods": {}}
    for name, (lo, hi, half) in iv.items():
        rt = MET.per_region(reg[ev], y[ev], lo[ev], hi[ev], ALPHA)
        res["methods"][name] = MET.summarize(rt, y[ev], lo[ev], hi[ev], ALPHA, NOMINAL)
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
    todo = [(rg, sd) for rg in REGIMES for sd in range(SEEDS) if (rg, sd) not in d]
    n = 0
    with open(RAW, "a") as f:
        for rg, sd in todo:
            if time.time() - t0 > BUDGET: break
            f.write(json.dumps(one_run(rg, sd)) + "\n"); f.flush(); n += 1
    print(f"processed {n}; total {len(d)+n}/{len(REGIMES)*SEEDS}")


def agg():
    import pandas as pd
    from scipy import stats
    runs = [json.loads(l) for l in open(RAW)]
    rows = []
    for r in runs:
        for m in METHODS:
            s = r["methods"][m]
            rows.append(dict(regime=r["regime"], seed=r["seed"], method=m,
                             marginal=s["true_marginal"], worst=s["worst_region_coverage"],
                             interval_score=s["mean_interval_score"], width=s["mean_width"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "robust_per_run.csv"), index=False)
    g = df.groupby(["regime", "method"]).agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    g.reset_index().to_csv(os.path.join(RES, "robust_summary.csv"), index=False)

    # significance: DI-normalized & DI-CQR vs split and localized (interval score)
    sig = {}
    for rg in REGIMES:
        for tgt in ["di_normalized", "di_cqr"]:
            for base in ["split", "localized"]:
                a = df[(df.regime == rg) & (df.method == tgt)].sort_values("seed")["interval_score"].to_numpy()
                b = df[(df.regime == rg) & (df.method == base)].sort_values("seed")["interval_score"].to_numpy()
                sig[f"{rg}:{tgt}-{base}"] = dict(diff=float((a-b).mean()),
                    t_p=float(stats.ttest_rel(a, b).pvalue),
                    w_p=float(stats.wilcoxon(a, b).pvalue))
    json.dump(sig, open(os.path.join(RES, "robust_sig.json"), "w"), indent=2)

    # LaTeX table (mean +- sd), moderate & severe
    pretty = {"split": "Split", "normalized": "Normalized", "di_normalized": r"\textbf{DI-normalized}",
              "cqr": "CQR", "di_cqr": "DI-CQR", "localized": "Localized (Guan)",
              "weighted_oracle": "Weighted (oracle)", "width_matched_global": "Width-matched global"}
    def cell(rg, m, col, dec=2):
        r = g.reset_index(); r = r[(r.regime == rg) & (r.method == m)].iloc[0]
        return f"{r[col+'_mean']:.{dec}f}\\,$\\pm$\\,{r[col+'_std']:.{dec}f}"
    L = [r"\begin{tabular}{l cc cc cc}", r"\toprule",
         r" & \multicolumn{2}{c}{Marginal cov.} & \multicolumn{2}{c}{Worst-region cov.} & \multicolumn{2}{c}{Interval score} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Method (HGB mean model) & mod. & sev. & mod. & sev. & mod. & sev. \\", r"\midrule"]
    for m in METHODS:
        L.append(f"{pretty[m]} & {cell('moderate',m,'marginal')} & {cell('severe',m,'marginal')} & "
                 f"{cell('moderate',m,'worst')} & {cell('severe',m,'worst')} & "
                 f"{cell('moderate',m,'interval_score',1)} & {cell('severe',m,'interval_score',1)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAP, "table_robust.tex"), "w").write("\n".join(L))
    print("wrote robust_summary.csv, robust_sig.json, table_robust.tex")
    gr = g.reset_index()
    def isc(rg, m):
        return float(gr[(gr.regime == rg) & (gr.method == m)]["interval_score_mean"].iloc[0])
    for rg in REGIMES:
        print(f"[{rg}]  IS  di_normalized={isc(rg,'di_normalized'):.2f}  di_cqr={isc(rg,'di_cqr'):.2f}  "
              f"split={isc(rg,'split'):.2f}  localized={isc(rg,'localized'):.2f}")


if __name__ == "__main__":
    (agg if len(sys.argv) > 1 and sys.argv[1] == "agg" else work)()
