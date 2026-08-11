"""make_tables.py -- emit LaTeX table fragments straight from result CSVs so no
number is transcribed by hand. Also emits headline_numbers.json for prose checks."""
import os, json, numpy as np, pandas as pd

RES = os.path.join(os.path.dirname(__file__), "..", "results")
PAP = os.path.join(os.path.dirname(__file__), "..", "paper")
S = pd.read_csv(os.path.join(RES, "sim_summary.csv"))
P = pd.read_csv(os.path.join(RES, "sim_paired_diffs.csv"))
A = pd.read_csv(os.path.join(RES, "sim_ablation_summary.csv"))
M = pd.read_csv(os.path.join(RES, "meuse_summary.csv"))
PRETTY = {"split": "Split", "normalized": "Normalized", "di_normalized": "\\textbf{DI-normalized}",
          "region_mondrian": "Spatial-Mondrian", "di_mondrian": "DI-Mondrian", "cqr": "CQR",
          "di_cqr": "DI-CQR", "localized": "Localized (Guan)", "geo_localized": "Geo-localized",
          "weighted_oracle": "Weighted (oracle)",
          "weighted_estimated": "Weighted (est.)", "width_matched_global": "Width-matched global"}
ORDER = ["split", "region_mondrian", "normalized", "di_normalized", "cqr", "di_mondrian",
         "di_cqr", "localized", "geo_localized", "weighted_oracle", "weighted_estimated",
         "width_matched_global"]


def ms(rg, m, col):
    r = S[(S.regime == rg) & (S.method == m)].iloc[0]
    return r[col + "_mean"], r[col + "_sd"]


def cell(rg, m, col, dec=2):
    mean, sd = ms(rg, m, col)
    return f"{mean:.{dec}f}\\,$\\pm$\\,{sd:.{dec}f}"


def main_table():
    lines = [r"\begin{tabular}{l cc cc cc}", r"\toprule",
             r" & \multicolumn{2}{c}{Marginal cov.} & \multicolumn{2}{c}{Worst-region cov.} & \multicolumn{2}{c}{Interval score} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Method & moderate & severe & moderate & severe & moderate & severe \\", r"\midrule"]
    for m in ORDER:
        lines.append(f"{PRETTY[m]} & {cell('moderate',m,'true_marginal')} & {cell('severe',m,'true_marginal')} & "
                     f"{cell('moderate',m,'worst_region_coverage')} & {cell('severe',m,'worst_region_coverage')} & "
                     f"{cell('moderate',m,'mean_interval_score',1)} & {cell('severe',m,'mean_interval_score',1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAP, "table_main.tex"), "w").write("\n".join(lines))


def controls_table():
    lines = [r"\begin{tabular}{l cccc}", r"\toprule",
             r"Method & none & mild & moderate & severe \\", r"\midrule"]
    for m in ["split", "cqr", "di_cqr", "weighted_oracle"]:
        vals = " & ".join(cell(rg, m, "true_marginal") for rg in ["none", "mild", "moderate", "severe"])
        lines.append(f"{PRETTY[m]} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAP, "table_controls.tex"), "w").write("\n".join(lines))


def width_table():
    lines = [r"\begin{tabular}{l cc cc}", r"\toprule",
             r" & \multicolumn{2}{c}{Mean width} & \multicolumn{2}{c}{Coverage gap} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Method & moderate & severe & moderate & severe \\", r"\midrule"]
    for m in ORDER:
        lines.append(f"{PRETTY[m]} & {cell('moderate',m,'mean_width',1)} & {cell('severe',m,'mean_width',1)} & "
                     f"{cell('moderate',m,'coverage_gap')} & {cell('severe',m,'coverage_gap')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAP, "table_width.tex"), "w").write("\n".join(lines))


def paired_table():
    lines = [r"\begin{tabular}{l cc cc}", r"\toprule",
             r" & \multicolumn{2}{c}{$\Delta$ Interval score} & \multicolumn{2}{c}{$\Delta$ Marginal cov.} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Baseline (DI-CQR $-$ baseline) & mean [95\% CI] & $p$ & mean [95\% CI] & $p$ \\", r"\midrule"]
    for m in [x for x in ORDER if x != "di_cqr"]:
        ri = P[(P.regime == "severe") & (P.metric == "mean_interval_score") & (P.baseline == m)].iloc[0]
        rm = P[(P.regime == "severe") & (P.metric == "true_marginal") & (P.baseline == m)].iloc[0]
        lines.append(f"{PRETTY[m].replace(chr(92)+'textbf','').strip('{}')} & "
                     f"{ri['mean_diff']:+.2f} [{ri['ci_lo']:+.2f}, {ri['ci_hi']:+.2f}] & {ri['p_value']:.1g} & "
                     f"{rm['mean_diff']:+.3f} [{rm['ci_lo']:+.3f}, {rm['ci_hi']:+.3f}] & {rm['p_value']:.1g} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAP, "table_paired.tex"), "w").write("\n".join(lines))


def ablation_table():
    lines = [r"\begin{tabular}{c c ccc ccc}", r"\toprule",
             r"$K$ & min pts/bin & \multicolumn{3}{c}{moderate} & \multicolumn{3}{c}{severe} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r" & & marg & worst & IntScore & marg & worst & IntScore \\", r"\midrule"]
    for K in [3, 4, 5, 6, 8, 10]:
        rmo = A[(A.regime == "moderate") & (A.K == K) & (A.monotone == 1)].iloc[0]
        rse = A[(A.regime == "severe") & (A.K == K) & (A.monotone == 1)].iloc[0]
        star = r"$^\star$" if K == 5 else ""
        lines.append(f"{K}{star} & {rmo['min_bin_count_mean']:.0f} & "
                     f"{rmo['true_marginal_mean']:.2f} & {rmo['worst_region_coverage_mean']:.2f} & {rmo['mean_interval_score_mean']:.1f} & "
                     f"{rse['true_marginal_mean']:.2f} & {rse['worst_region_coverage_mean']:.2f} & {rse['mean_interval_score_mean']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAP, "table_ablation.tex"), "w").write("\n".join(lines))


def meuse_table():
    mm = ["split", "region_mondrian", "normalized", "di_normalized", "cqr", "di_mondrian",
          "di_cqr", "localized", "geo_localized", "weighted_estimated"]
    def mc(m, met):
        r = M[(M.method == m) & (M.metric == met)].iloc[0]
        return f"{r['mean']:.2f}\\,$\\pm$\\,{r['sd']:.2f}"
    lines = [r"\begin{tabular}{l cccc}", r"\toprule",
             r"Method & Marginal & Worst-region & Coverage gap & Interval score \\", r"\midrule"]
    for m in mm:
        lines.append(f"{PRETTY[m]} & {mc(m,'marginal')} & {mc(m,'worst_region')} & {mc(m,'coverage_gap')} & {mc(m,'mean_interval_score')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAP, "table_meuse.tex"), "w").write("\n".join(lines))


def bridge_table():
    B = pd.read_csv(os.path.join(RES, "sim_bridge.csv")).set_index("regime")
    lines = [r"\begin{tabular}{l ccc}", r"\toprule",
             r"Regime & Spearman$(\mathrm{DI},1/p_{\text{sel}})$ & 95\% CI & Pearson$(\mathrm{DI},\log 1/p_{\text{sel}})$ \\", r"\midrule",
             r"none & \multicolumn{3}{c}{n/a (uniform selection, constant weight)} \\"]
    for rg in ["mild", "moderate", "severe"]:
        r = B.loc[rg]
        lines.append(f"{rg} & {r['spearman_DI_invpsel_mean']:.2f} & "
                     f"[{r['spearman_ci_lo']:.2f}, {r['spearman_ci_hi']:.2f}] & "
                     f"{r['pearson_DI_logInvPsel_mean']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAP, "table_bridge.tex"), "w").write("\n".join(lines))


def lucas_table():
    L = pd.read_csv(os.path.join(RES, "lucas_summary.csv")).set_index("method")
    mm = ["split", "region_mondrian", "normalized", "di_normalized", "cqr", "di_mondrian",
          "di_cqr", "localized", "geo_localized", "weighted_estimated"]
    pretty = dict(PRETTY); pretty["di_cqr"] = r"\textbf{DI-CQR}"; pretty["di_normalized"] = "DI-normalized"
    def c(m, col, dec=2):
        r = L.loc[m]
        return f"{r[col+'_mean']:.{dec}f}\\,$\\pm$\\,{r[col+'_std']:.{dec}f}"
    lines = [r"\begin{tabular}{l cccc}", r"\toprule",
             r"Method & Marginal & Worst-region & Mean width & Interval score \\", r"\midrule"]
    for m in mm:
        lines.append(f"{pretty[m]} & {c(m,'marginal')} & {c(m,'worst_region')} & "
                     f"{c(m,'mean_width',1)} & {c(m,'mean_interval_score',1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAP, "table_lucas.tex"), "w").write("\n".join(lines))


def headline():
    h = {}
    for rg in ["none", "mild", "moderate", "severe"]:
        h[rg] = {m: {c: round(float(ms(rg, m, c)[0]), 3) for c in
                     ["true_marginal", "worst_region_coverage", "coverage_gap", "mean_width", "mean_interval_score"]}
                 for m in ORDER}
    json.dump(h, open(os.path.join(RES, "headline_numbers.json"), "w"), indent=2)


if __name__ == "__main__":
    main_table(); controls_table(); width_table(); paired_table(); ablation_table()
    meuse_table(); bridge_table(); lucas_table(); headline()
    print("wrote LaTeX tables + headline_numbers.json")
