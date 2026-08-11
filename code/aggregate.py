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
