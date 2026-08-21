"""aggregate.py -- turn sim_raw.jsonl into tidy CSV/JSON result tables with
mean, sd, 95% CI, and paired seed-level differences (DI-CQR vs each baseline)."""
import json, os, numpy as np, pandas as pd
from scipy import stats

RES = os.path.join(os.path.dirname(__file__), "..", "results")
RAW = os.path.join(RES, "sim_raw.jsonl")
METRICS = ["true_marginal", "worst_region_coverage", "coverage_gap",
           "mean_width", "mean_interval_score", "coverage_rmse",
           "cond_cov_error_mean", "cond_cov_error_max", "frac_infinite"]
REGIME_ORDER = ["none", "mild", "moderate", "severe"]


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    se = x.std(ddof=1) / np.sqrt(len(x))
    t = stats.t.ppf(0.975, len(x) - 1)
    return (x.mean() - t * se, x.mean() + t * se)


def load():
    return [json.loads(l) for l in open(RAW)]


def main():
    runs = load()
    methods = list(runs[0]["methods"].keys())

    # ---- long per-run table
    rows = []
    for r in runs:
        for m in methods:
            row = dict(regime=r["regime"], bias=r["bias"], seed=r["seed"], method=m)
            row.update({k: r["methods"][m][k] for k in METRICS})
            rows.append(row)
    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(RES, "sim_per_run.csv"), index=False)

    # ---- summary mean/sd/CI
    srows = []
    for rg in REGIME_ORDER:
        for m in methods:
            sub = long[(long.regime == rg) & (long.method == m)]
            d = dict(regime=rg, method=m, n=len(sub))
            for k in METRICS:
                v = sub[k].to_numpy()
                lo, hi = ci95(v)
                d[f"{k}_mean"] = v.mean(); d[f"{k}_sd"] = v.std(ddof=1)
                d[f"{k}_ci_lo"] = lo; d[f"{k}_ci_hi"] = hi
            srows.append(d)
    summary = pd.DataFrame(srows)
    summary.to_csv(os.path.join(RES, "sim_summary.csv"), index=False)

    # ---- paired seed-level differences: DI-CQR minus each baseline
    prows = []
    for rg in REGIME_ORDER:
        base = long[(long.regime == rg) & (long.method == "di_cqr")].set_index("seed")
        for m in methods:
            if m == "di_cqr":
                continue
            other = long[(long.regime == rg) & (long.method == m)].set_index("seed")
            seeds = base.index.intersection(other.index)
            for k in ["true_marginal", "worst_region_coverage", "mean_interval_score", "mean_width"]:
                a = base.loc[seeds, k].to_numpy(); b = other.loc[seeds, k].to_numpy()
                diff = a - b
                lo, hi = ci95(diff)
                tstat, p = stats.ttest_rel(a, b)
                try:
                    wp = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
                except Exception:
                    wp = float("nan")
                prows.append(dict(regime=rg, metric=k, baseline=m,
                                  mean_diff=float(diff.mean()), sd=float(diff.std(ddof=1)),
                                  ci_lo=float(lo), ci_hi=float(hi),
                                  p_value=float(p), wilcoxon_p=wp))
    paired = pd.DataFrame(prows)
    paired.to_csv(os.path.join(RES, "sim_paired_diffs.csv"), index=False)

    # ---- ablation table
    arows = []
    for r in runs:
        for key, s in r["ablation"].items():
            _, kpart, mpart = key.split("_")[0], key.split("_")[2], key.split("_")[3]
            K = int(kpart[1:]); mono = int(mpart[4:])
            arows.append(dict(regime=r["regime"], seed=r["seed"], K=K, monotone=mono,
                              min_bin_count=s["min_bin_count"], mean_bin_count=s["mean_bin_count"],
                              true_marginal=s["true_marginal"],
                              worst_region_coverage=s["worst_region_coverage"],
                              coverage_gap=s["coverage_gap"], mean_width=s["mean_width"],
                              mean_interval_score=s["mean_interval_score"]))
    abl = pd.DataFrame(arows)
    abl.to_csv(os.path.join(RES, "sim_ablation_per_run.csv"), index=False)
    abl_summary = (abl.groupby(["regime", "K", "monotone"])
                   .agg(["mean", "std"]).reset_index())
    abl_summary.columns = ["_".join([c for c in col if c]).strip("_") for col in abl_summary.columns]
    abl_summary.to_csv(os.path.join(RES, "sim_ablation_summary.csv"), index=False)

    # ---- coverage vs width sweep (averaged over seeds)
    swrows = []
    for r in runs:
        for m, arr in r["sweep"].items():
            for pt in arr:
                swrows.append(dict(regime=r["regime"], seed=r["seed"], method=m, **pt))
    sw = pd.DataFrame(swrows)
    sw_summary = (sw.groupby(["regime", "method", "mult"])
                  .agg(marginal=("marginal", "mean"), worst=("worst", "mean"),
                       mean_width=("mean_width", "mean")).reset_index())
    sw_summary.to_csv(os.path.join(RES, "sim_sweep_summary.csv"), index=False)

    # ---- DI vs inverse-selection-weight bridge (per biased regime)
    brows = []
    for rg in ["mild", "moderate", "severe"]:
        sp = [r["di_selweight_spearman"] for r in runs
              if r["regime"] == rg and r.get("di_selweight_spearman") is not None]
        pe = [r["di_logselweight_pearson"] for r in runs
              if r["regime"] == rg and r.get("di_logselweight_pearson") is not None]
        lo, hi = ci95(sp)
        brows.append(dict(regime=rg, n=len(sp),
                          spearman_DI_invpsel_mean=float(np.mean(sp)),
                          spearman_sd=float(np.std(sp, ddof=1)),
                          spearman_ci_lo=float(lo), spearman_ci_hi=float(hi),
                          pearson_DI_logInvPsel_mean=float(np.mean(pe))))
    pd.DataFrame(brows).to_csv(os.path.join(RES, "sim_bridge.csv"), index=False)

    # ---- region-definition sensitivity: worst-region coverage under 2x2/3x3/4x4
    rs_rows = []
    for r in runs:
        for m, dk in (r.get("region_sens") or {}).items():
            for k, worst in dk.items():
                rs_rows.append(dict(regime=r["regime"], seed=r["seed"], method=m,
                                    k=int(k), worst=float(worst)))
    if rs_rows:
        rsdf = pd.DataFrame(rs_rows)
        (rsdf.groupby(["regime", "method", "k"])["worst"].agg(["mean", "std"])
         .reset_index().to_csv(os.path.join(RES, "sim_region_sens.csv"), index=False))

    # ---- kappa sensitivity for DI-normalized
    ks_rows = []
    for r in runs:
        for kap, d in (r.get("kappa_sens") or {}).items():
            ks_rows.append(dict(regime=r["regime"], seed=r["seed"], kappa=float(kap), **d))
    if ks_rows:
        ksdf = pd.DataFrame(ks_rows)
        (ksdf.groupby(["regime", "kappa"]).agg(
            worst=("worst_region_coverage", "mean"),
            interval_score=("mean_interval_score", "mean"),
            width=("mean_width", "mean")).reset_index()
         .to_csv(os.path.join(RES, "sim_kappa_sens.csv"), index=False))

    # ---- mechanism: does the per-seed DI-1/p_sel Spearman predict the per-seed
    # DI-CQR gain? (turns the bridge correlation into predictive evidence)
    from scipy.stats import pearsonr
    recs = []
    for r in runs:
        sp = r.get("di_selweight_spearman")
        if sp is None:
            continue
        M = r["methods"]
        recs.append(dict(regime=r["regime"], seed=r["seed"], spear=float(sp),
                         gain_IS_vs_cqr=M["cqr"]["mean_interval_score"] - M["di_cqr"]["mean_interval_score"],
                         gain_IS_vs_split=M["split"]["mean_interval_score"] - M["di_cqr"]["mean_interval_score"],
                         gain_worst_vs_split=M["di_cqr"]["worst_region_coverage"] - M["split"]["worst_region_coverage"]))
    mdf = pd.DataFrame(recs)
    groups = [(rg, mdf[mdf.regime == rg]) for rg in ["mild", "moderate", "severe"]]
    groups.append(("pooled_biased", mdf))
    mech = []
    for grp, sub in groups:
        for gcol in ["gain_IS_vs_cqr", "gain_IS_vs_split", "gain_worst_vs_split"]:
            if len(sub) >= 3 and sub["spear"].nunique() > 2 and sub[gcol].nunique() > 2:
                pr, pp = pearsonr(sub["spear"].to_numpy(), sub[gcol].to_numpy())
                mech.append(dict(group=grp, gain=gcol, n=len(sub),
                                 pearson_r=float(pr), p_value=float(pp)))
    pd.DataFrame(mech).to_csv(os.path.join(RES, "sim_mechanism.csv"), index=False)

    # partial correlation controlling for regime (within-regime centering) with a
    # seed-clustered bootstrap CI, since (a) the pooled r conflates between-regime
    # variation and (b) seeds are shared across regimes so runs are not independent.
    biased = mdf[mdf.regime.isin(["mild", "moderate", "severe"])].copy()

    def _partial_r(d, gcol):
        sc = d["spear"] - d.groupby("regime")["spear"].transform("mean")
        gc = d[gcol] - d.groupby("regime")[gcol].transform("mean")
        if sc.std(ddof=0) == 0 or gc.std(ddof=0) == 0:
            return np.nan
        return float(np.corrcoef(sc, gc)[0, 1])

    rng = np.random.default_rng(0)
    seeds = sorted(biased.seed.unique())
    by_seed = {s: biased[biased.seed == s] for s in seeds}
    mechp = []
    for gcol in ["gain_IS_vs_cqr", "gain_IS_vs_split", "gain_worst_vs_split"]:
        r = _partial_r(biased, gcol)
        boots = []
        for _ in range(1000):
            samp = rng.choice(seeds, size=len(seeds), replace=True)
            dd = pd.concat([by_seed[s] for s in samp], ignore_index=True)
            rb = _partial_r(dd, gcol)
            if not np.isnan(rb):
                boots.append(rb)
        boots = np.array(boots)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
        mechp.append(dict(gain=gcol, partial_r=r, ci_lo=float(lo), ci_hi=float(hi),
                          boot_p=float(p), n_seeds=len(seeds)))
    pd.DataFrame(mechp).to_csv(os.path.join(RES, "sim_mechanism_partial.csv"), index=False)

    # ---- DI-CQR clipping diagnostics (queries above max calibration DI -> top bin)
    crows = []
    for r in runs:
        c = r.get("clip")
        if not c:
            continue
        for grp in ["clipped", "unclipped"]:
            d = c[grp]
            crows.append(dict(regime=r["regime"], seed=r["seed"], group=grp,
                              frac=d["frac"], n=d["n"], coverage=d["coverage"],
                              mean_width=d["mean_width"],
                              mean_interval_score=d["mean_interval_score"]))
    if crows:
        clip = pd.DataFrame(crows)
        clip.to_csv(os.path.join(RES, "sim_clip_per_run.csv"), index=False)
        csum = []
        for rg in REGIME_ORDER:
            for grp in ["clipped", "unclipped"]:
                sub = clip[(clip.regime == rg) & (clip.group == grp)]
                cov = sub["coverage"].dropna().to_numpy()
                csum.append(dict(regime=rg, group=grp,
                                 frac_mean=float(sub["frac"].mean()),
                                 frac_sd=float(sub["frac"].std(ddof=1)),
                                 coverage_mean=float(cov.mean()) if len(cov) else float("nan"),
                                 mean_width=float(sub["mean_width"].dropna().mean()),
                                 mean_interval_score=float(sub["mean_interval_score"].dropna().mean())))
        pd.DataFrame(csum).to_csv(os.path.join(RES, "sim_clip_summary.csv"), index=False)

    # ---- calibration-valid selected K and kappa (distribution by regime)
    sel_rows = []
    for r in runs:
        s = r.get("selected")
        if s:
            sel_rows.append(dict(regime=r["regime"], seed=r["seed"],
                                 K=s["K"], kappa=s["kappa"]))
    if sel_rows:
        sd = pd.DataFrame(sel_rows)
        srows = []
        for rg in REGIME_ORDER:
            sub = sd[sd.regime == rg]
            srows.append(dict(regime=rg, K_mean=float(sub.K.mean()), K_sd=float(sub.K.std(ddof=1)),
                              K_mode=int(sub.K.mode().iloc[0]),
                              kappa_mean=float(sub.kappa.mean()), kappa_sd=float(sub.kappa.std(ddof=1)),
                              kappa_mode=float(sub.kappa.mode().iloc[0])))
        pd.DataFrame(srows).to_csv(os.path.join(RES, "sim_selection.csv"), index=False)

    # ---- weighted-conformal capped-interval fraction by regime
    wrows = []
    for r in runs:
        wc = r.get("weighted_capped")
        if not wc:
            continue
        for m, frac in wc.items():
            wrows.append(dict(regime=r["regime"], seed=r["seed"], method=m, frac_capped=frac))
    if wrows:
        wdf = pd.DataFrame(wrows)
        wsum = (wdf.groupby(["regime", "method"])["frac_capped"]
                .agg(["mean", "std"]).reset_index())
        wsum.to_csv(os.path.join(RES, "sim_weighted_capped.csv"), index=False)

    # ---- representative fields for maps (severe seed 0)
    import run_sim as R
    rep = R.one_run("severe", 0, save_fields=True)
    json.dump(rep["_fields"], open(os.path.join(RES, "sim_fields_severe.json"), "w"))

    print("wrote sim_per_run.csv, sim_summary.csv, sim_paired_diffs.csv,")
    print("      sim_ablation_per_run.csv, sim_ablation_summary.csv, sim_sweep_summary.csv,")
    print("      sim_fields_severe.json")
    # quick console check
    for rg in REGIME_ORDER:
        s = summary[(summary.regime == rg) & (summary.method == "split")].iloc[0]
        d = summary[(summary.regime == rg) & (summary.method == "di_cqr")].iloc[0]
        print(f"[{rg}] split marg={s['true_marginal_mean']:.2f} | di_cqr marg={d['true_marginal_mean']:.2f} "
              f"worst={d['worst_region_coverage_mean']:.2f} IS={d['mean_interval_score_mean']:.2f}")


if __name__ == "__main__":
    main()
