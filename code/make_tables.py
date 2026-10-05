"""make_tables.py -- write every LaTeX table and every number quoted in the
manuscript straight from the result files, so nothing is transcribed by hand.

Outputs (in ../paper):
  table_*.tex   table bodies, \\input by paper.tex
  numbers.tex   one macro per quoted number; the text uses \\res{key}
and ../results/headline_numbers.json (a flat copy of numbers.tex).
"""
import os, json, numpy as np, pandas as pd

RES = os.path.join(os.path.dirname(__file__), "..", "results")
PAP = os.path.join(os.path.dirname(__file__), "..", "paper")
os.makedirs(PAP, exist_ok=True)   # the manuscript itself is not part of the public repository


def rd(name):
    return pd.read_csv(os.path.join(RES, name))


def rj(name):
    return json.load(open(os.path.join(RES, name)))


S = rd("sim_summary.csv")
P = rd("sim_paired_diffs.csv")
A = rd("sim_ablation_summary.csv")
M = rd("meuse_summary.csv")
L = rd("lucas_summary.csv").set_index("method")
RB = rd("robust_summary.csv")

REG = {"none": "none", "mild": "mild", "moderate": "mod", "severe": "sev",
       "hidden_moderate": "hmod", "hidden_severe": "hsev"}
PRETTY = {"split": "Split", "normalized": "Normalized", "di_normalized": "DI-normalized",
          "di_normalized_clip": "DI-normalized (DI capped)",
          "di_normalized_auto": r"DI-normalized (auto-$\kappa$)",
          "region_mondrian": "Spatial-Mondrian", "di_mondrian": "DI-Mondrian", "cqr": "CQR",
          "di_cqr": r"\textbf{DI-CQR}", "di_cqr_auto": r"DI-CQR (auto-$K$)", "di_cqr_K2": r"DI-CQR ($K{=}2$)",
          "lcp": "LCP", "lcp_cqr": "LCP-CQR", "geo_lcp": "Geo-LCP",
          "weighted_oracle": "Weighted (oracle)", "weighted_estimated": "Weighted (estimated)",
          "weighted_proxy": r"Weighted ($1/p_{\mathrm{sel}}$ proxy)",
          "width_matched_global": "Width-matched"}
SIM_ORDER = ["split", "region_mondrian", "normalized", "cqr", "di_normalized", "di_mondrian",
             "di_cqr", "lcp", "lcp_cqr", "geo_lcp", "weighted_oracle", "weighted_estimated",
             "width_matched_global"]
NUM = {}


def put(key, val):
    NUM[key] = val


def f2(x):
    return f"{x:.2f}"


def f1(x):
    return f"{x:.1f}"


def pct(x):
    return f"{100 * x:.0f}"


def prel(p):
    """p-value with its relation, for use as $p\\res{...}$."""
    if not np.isfinite(p):
        return "{=}\\,\\text{n/a}"
    if p < 1e-3:
        return "{<}\\,0.001"
    return "{=}\\," + (f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def ptab(p):
    if p < 1e-3:
        return "$<$0.001"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def sv(rg, m, col):
    r = S[(S.regime == rg) & (S.method == m)]
    return float(r.iloc[0][col + "_mean"]), float(r.iloc[0][col + "_sd"])


def cell(rg, m, col, dec=2):
    mean, sd = sv(rg, m, col)
    return f"{mean:.{dec}f}\\,$\\pm$\\,{sd:.{dec}f}"


def write(name, lines):
    open(os.path.join(PAP, name), "w", encoding="utf8", newline="\n").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ numbers
def sim_numbers():
    for rg, k in REG.items():
        for m in S[S.regime == rg].method.unique():
            put(f"sim:{k}:{m}:marg", f2(sv(rg, m, "true_marginal")[0]))
            put(f"sim:{k}:{m}:worst", f2(sv(rg, m, "worst_region_coverage")[0]))
            put(f"sim:{k}:{m}:gap", f2(sv(rg, m, "coverage_gap")[0]))
            put(f"sim:{k}:{m}:width", f1(sv(rg, m, "mean_width")[0]))
            put(f"sim:{k}:{m}:is", f1(sv(rg, m, "mean_interval_score")[0]))
    for _, r in P.iterrows():
        k = f"pd:{REG[r.regime]}:{r.baseline}:{ {'true_marginal': 'marg', 'worst_region_coverage': 'worst', 'mean_interval_score': 'is', 'mean_width': 'width'}[r.metric] }"
        dec = 2 if r.metric in ("true_marginal", "worst_region_coverage") else 2
        put(k + ":d", f"{r.mean_diff:+.{dec}f}".replace("+", "+").replace("-", "$-$"))
        put(k + ":dabs", f"{abs(r.mean_diff):.{dec}f}")
        put(k + ":lo", f"{r.ci_lo:.{dec}f}"); put(k + ":hi", f"{r.ci_hi:.{dec}f}")
        put(k + ":p", prel(r.p_value)); put(k + ":wp", prel(r.wilcoxon_p))
        put(k + ":ph", prel(r.p_holm) if pd.notna(r.p_holm) else prel(np.nan))
        put(k + ":wph", prel(r.wilcoxon_p_holm) if pd.notna(r.wilcoxon_p_holm) else prel(np.nan))
        if pd.notna(r.p_holm):
            put(k + ":phv", "<0.001" if r.p_holm < 1e-3 else (f"{r.p_holm:.3f}" if r.p_holm < 0.01 else f"{r.p_holm:.2f}"))
    # unbounded fractions, ESS, LCP mass, clipping
    U = rd("sim_unbounded.csv")
    for _, r in U.iterrows():
        put(f"unb:{REG[r.regime]}:{r.method}", pct(r["mean"]))
    E = rd("sim_weight_ess.csv")
    for _, r in E.iterrows():
        put(f"ess:{REG[r.regime]}:{r.method}", f1(r["median"]))
    LM = rd("sim_lcp_mass.csv")
    for _, r in LM.iterrows():
        put(f"lcpmass:{REG[r.regime]}:{r.group}", f1(r.median_mass))
        put(f"lcpunb:{REG[r.regime]}:{r.group}", pct(r.frac_unbounded))
    C = rd("sim_clip_summary.csv")
    for _, r in C.iterrows():
        k = f"clip:{REG[r.regime]}:{r.group}"
        put(k + ":frac", pct(r.frac_mean))
        if np.isfinite(r.coverage_mean):
            put(k + ":cov", f2(r.coverage_mean)); put(k + ":is", f1(r.mean_interval_score))
    B = rd("sim_bridge.csv")
    for _, r in B.iterrows():
        put(f"bridge:{REG[r.regime]}:sp", f2(r.spearman_DI_invpsel_mean))
        put(f"bridge:{REG[r.regime]}:pe", f2(r.pearson_DI_logInvPsel_mean))
    MM = rd("sim_mechanism.csv")
    for _, r in MM[MM.group == "pooled_biased"].iterrows():
        put(f"mech:pooled:{r.gain}:r", f2(r.pearson_r)); put(f"mech:pooled:{r.gain}:p", prel(r.p_value))
    MP = rd("sim_mechanism_partial.csv")
    for _, r in MP.iterrows():
        put(f"mech:partial:{r.gain}:r", f2(r.partial_r)); put(f"mech:partial:{r.gain}:p", prel(r.boot_p))
    RS = rd("sim_region_sens.csv")
    for _, r in RS.iterrows():
        put(f"region:{REG[r.regime]}:{r.method}:{int(r.k)}", f2(r["mean"]))
    KS = rd("sim_kappa_sens.csv")
    for _, r in KS.iterrows():
        kk = {0.1: "a", 0.25: "b", 0.5: "c", 1.0: "d"}[float(r.kappa)]
        put(f"kappa:{REG[r.regime]}:{kk}:worst", f2(r.worst)); put(f"kappa:{REG[r.regime]}:{kk}:is", f1(r.interval_score))
    SEL = rd("sim_selection.csv")
    for _, r in SEL.iterrows():
        put(f"sel:{REG[r.regime]}:Kmode", f"{int(r.K_mode)}"); put(f"sel:{REG[r.regime]}:kappa", f2(r.kappa_mean))
        put(f"sel:{REG[r.regime]}:Kmean", f1(r.K_mean))
    for _, r in A[A.monotone == 1].iterrows():
        k = f"abl:{REG[r.regime]}:K{int(r.K)}"
        put(k + ":worst", f2(r.worst_region_coverage_mean)); put(k + ":is", f1(r.mean_interval_score_mean))
        put(k + ":marg", f2(r.true_marginal_mean)); put(k + ":minbin", f"{r.min_bin_count_mean:.0f}")
    # reduction of the worst-region coverage shortfall (0.90 - worst) of DI-CQR
    # relative to split, over the four strong-bias regimes
    red = [1 - (0.9 - sv(rg, "di_cqr", "worst_region_coverage")[0]) /
           (0.9 - sv(rg, "split", "worst_region_coverage")[0])
           for rg in ["moderate", "severe", "hidden_moderate", "hidden_severe"]]
    put("defred:min", pct(min(red))); put("defred:max", pct(max(red)))
    # derived ratios quoted in text
    for rg in ["moderate", "severe", "hidden_moderate", "hidden_severe"]:
        put(f"ratio:{REG[rg]}:oracle_width", f"{sv(rg, 'weighted_oracle', 'mean_width')[0] / sv(rg, 'di_cqr', 'mean_width')[0]:.0f}")


def real_numbers():
    for m in M.method.unique():
        for met, key, fmt in [("marginal", "marg", f2), ("worst_region", "worst", f2),
                              ("coverage_gap", "gap", f2), ("mean_width", "width", f2),
                              ("mean_interval_score", "is", f2)]:
            put(f"meuse:{m}:{key}", fmt(float(M[(M.method == m) & (M.metric == met)]["mean"].iloc[0])))
    D = rj("meuse_diag.json")
    put("meusediag:levelone", pct(D["frac_bins_level_one"]))
    put("meusediag:mintest", f"{D['min_test_per_region']}")
    put("meusediag:medtest", f"{D['median_test_per_region']:.0f}")
    put("meusediag:clipped", pct(D["mean_frac_test_clipped"]))
    put("meusediag:fallback", f"{D['mean_fallback_bins']:.0f}")
    for m, f in D["unbounded"].items():
        put(f"meuseunb:{m}", pct(f))
    MS = rj("meuse_sig.json")
    for k, v in MS.items():
        kk = k.replace("mean_interval_score", "is").replace("worst_region", "worst")
        put(f"meusesig:{kk}:d", f"{v['diff']:+.2f}".replace("-", "$-$"))
        put(f"meusesig:{kk}:p", prel(v["t_p"])); put(f"meusesig:{kk}:wp", prel(v["w_p"]))
        put(f"meusesig:{kk}:dabs", f"{abs(v['diff']):.2f}")
    for m in L.index:
        for col, key, fmt in [("marginal", "marg", f2), ("worst_region", "worst", f2),
                              ("coverage_gap", "gap", f2), ("mean_width", "width", f1),
                              ("mean_interval_score", "is", f1)]:
            put(f"lucas:{m}:{key}", (f2 if key == "is" else fmt)(float(L.loc[m][col + "_mean"])))
    LS = rj("lucas_sig.json")
    for k, v in LS.items():
        put(f"lucassig:{k}:d", f"{v['diff']:+.2f}".replace("-", "$-$"))
        put(f"lucassig:{k}:p", prel(v["t_p"])); put(f"lucassig:{k}:wp", prel(v["w_p"]))
    LD = rj("lucas_diag.json")
    for m, f in LD["unbounded"].items():
        put(f"lucasunb:{m}", pct(f))
    for k, v in LD["di"].items():
        put(f"lucasdi:{k}", pct(v) if k.startswith("frac") else f2(v))
    put("lucasdi:ratio", f"{LD['di']['test_median'] / LD['di']['cal_mean']:.0f}")
    put("lucasratio:dinorm_dicqr", f"{L.loc['di_normalized']['mean_width_mean'] / L.loc['di_cqr']['mean_width_mean']:.1f}")
    for _, r in RB.iterrows():
        k = f"rob:{REG[r.regime]}:{r.method}"
        put(k + ":marg", f2(r.marginal_mean)); put(k + ":worst", f2(r.worst_mean))
        put(k + ":is", f1(r.interval_score_mean)); put(k + ":width", f1(r.width_mean))
    RSg = rj("robust_sig.json")
    for k, v in RSg.items():
        rg, pair = k.split(":")
        put(f"robsig:{REG[rg]}:{pair}:d", f"{v['diff']:+.2f}".replace("-", "$-$"))
        put(f"robsig:{REG[rg]}:{pair}:p", prel(v["t_p"])); put(f"robsig:{REG[rg]}:{pair}:wp", prel(v["w_p"]))
    RU = rd("robust_unbounded.csv")
    for _, r in RU.iterrows():
        put(f"robunb:{REG[r.regime]}:{r.method}", pct(r.frac))


# ------------------------------------------------------------------ tables
def sim_table(name, regs, labels, rows=SIM_ORDER):
    L_ = [r"\begin{tabular}{l cc cc cc}", r"\toprule",
          r" & \multicolumn{2}{c}{Marginal coverage} & \multicolumn{2}{c}{Worst-region coverage} & \multicolumn{2}{c}{Interval score} \\",
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
          f"Method & {labels[0]} & {labels[1]} & {labels[0]} & {labels[1]} & {labels[0]} & {labels[1]} \\\\", r"\midrule"]
    for m in rows:
        L_.append(f"{PRETTY[m]} & {cell(regs[0],m,'true_marginal')} & {cell(regs[1],m,'true_marginal')} & "
                  f"{cell(regs[0],m,'worst_region_coverage')} & {cell(regs[1],m,'worst_region_coverage')} & "
                  f"{cell(regs[0],m,'mean_interval_score',1)} & {cell(regs[1],m,'mean_interval_score',1)} \\\\")
        if m == "width_matched_global":
            pass
    L_ += [r"\bottomrule", r"\end{tabular}"]
    write(name, L_)


def controls_table():
    L_ = [r"\begin{tabular}{l cccc cc}", r"\toprule",
          r" & \multicolumn{4}{c}{Covariate-driven access} & \multicolumn{2}{c}{Hidden access} \\",
          r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
          r"Method & none & mild & moderate & severe & moderate & severe \\", r"\midrule"]
    for m in ["split", "cqr", "lcp", "di_normalized", "di_cqr", "weighted_oracle"]:
        vals = " & ".join(cell(rg, m, "true_marginal") for rg in REG)
        L_.append(f"{PRETTY[m]} & {vals} \\\\")
    L_ += [r"\bottomrule", r"\end{tabular}"]
    write("table_controls.tex", L_)


def paired_table():
    base = [m for m in SIM_ORDER if m != "di_cqr"]
    L_ = [r"\begin{tabular}{l cc cc}", r"\toprule",
          r" & \multicolumn{2}{c}{Severe (covariate access)} & \multicolumn{2}{c}{Severe (hidden access)} \\",
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
          r"Baseline & $\Delta$ interval score & $\Delta$ worst-region & $\Delta$ interval score & $\Delta$ worst-region \\", r"\midrule"]

    def c(rg, m, met):
        r = P[(P.regime == rg) & (P.metric == met) & (P.baseline == m)].iloc[0]
        d = f"{r.mean_diff:+.2f}" if met == "mean_interval_score" else f"{r.mean_diff:+.2f}"
        return f"{d.replace('-', '$-$')} ({ptab(r.p_holm)})"
    for m in base:
        L_.append(f"{PRETTY[m].replace(chr(92)+'textbf{','').rstrip('}') if 'textbf' in PRETTY[m] else PRETTY[m]} & "
                  f"{c('severe',m,'mean_interval_score')} & {c('severe',m,'worst_region_coverage')} & "
                  f"{c('hidden_severe',m,'mean_interval_score')} & {c('hidden_severe',m,'worst_region_coverage')} \\\\")
    L_ += [r"\bottomrule", r"\end{tabular}"]
    write("table_paired.tex", L_)


def diag_table():
    U = rd("sim_unbounded.csv"); E = rd("sim_weight_ess.csv"); LM = rd("sim_lcp_mass.csv")
    C = rd("sim_clip_summary.csv")

    def u(rg, m):
        r = U[(U.regime == rg) & (U.method == m)]
        return pct(float(r["mean"].iloc[0]))

    def e(rg, m):
        r = E[(E.regime == rg) & (E.method == m)]
        return f1(float(r["median"].iloc[0]))

    def lm(rg, g):
        r = LM[(LM.regime == rg) & (LM.group == g)]
        return f1(float(r.median_mass.iloc[0])) if len(r) else "--"

    def cl(rg):
        r = C[(C.regime == rg) & (C.group == "clipped")]
        return pct(float(r.frac_mean.iloc[0]))
    L_ = [r"\begin{tabular}{l c cc c cc}", r"\toprule",
          r" & DI-CQR & \multicolumn{2}{c}{Unbounded intervals (\%)} & Oracle-weight & \multicolumn{2}{c}{LCP local mass} \\",
          r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}",
          r"Regime & beyond cal.\ DI (\%) & LCP & Weighted (oracle) & ESS (of 200) & inside & beyond \\", r"\midrule"]
    lab = {"none": "None", "mild": "Mild", "moderate": "Moderate", "severe": "Severe",
           "hidden_moderate": "Hidden, moderate", "hidden_severe": "Hidden, severe"}
    for rg in REG:
        L_.append(f"{lab[rg]} & {cl(rg)} & {u(rg,'lcp')} & {u(rg,'weighted_oracle')} & {e(rg,'weighted_oracle')} & "
                  f"{lm(rg,'unclipped')} & {lm(rg,'clipped')} \\\\")
    L_ += [r"\bottomrule", r"\end{tabular}"]
    write("table_diag.tex", L_)


def width_table():
    U = rd("sim_unbounded.csv")

    def u(rg, m):
        r = U[(U.regime == rg) & (U.method == m)]
        return pct(float(r["mean"].iloc[0])) if len(r) else "0"
    L_ = [r"\begin{tabular}{l cc cc cc}", r"\toprule",
          r" & \multicolumn{2}{c}{Mean width} & \multicolumn{2}{c}{Coverage gap} & \multicolumn{2}{c}{Unbounded (\%)} \\",
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
          r"Method & moderate & severe & moderate & severe & moderate & severe \\", r"\midrule"]
    for m in SIM_ORDER + ["weighted_proxy"]:
        L_.append(f"{PRETTY[m]} & {cell('moderate',m,'mean_width',1)} & {cell('severe',m,'mean_width',1)} & "
                  f"{cell('moderate',m,'coverage_gap')} & {cell('severe',m,'coverage_gap')} & "
                  f"{u('moderate',m)} & {u('severe',m)} \\\\")
    L_ += [r"\bottomrule", r"\end{tabular}"]
    write("table_width.tex", L_)


def ablation_table():
    L_ = [r"\begin{tabular}{c c ccc ccc}", r"\toprule",
          r"$K$ & min.\ cal.\ points per bin & \multicolumn{3}{c}{moderate} & \multicolumn{3}{c}{severe} \\",
          r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
          r" & & marginal & worst & int.\ score & marginal & worst & int.\ score \\", r"\midrule"]
    for K in [3, 4, 5, 6, 8, 10]:
        rmo = A[(A.regime == "moderate") & (A.K == K) & (A.monotone == 1)].iloc[0]
        rse = A[(A.regime == "severe") & (A.K == K) & (A.monotone == 1)].iloc[0]
        star = r"$^\star$" if K == 5 else ""
        L_.append(f"{K}{star} & {rmo['min_bin_count_mean']:.0f} & "
                  f"{rmo['true_marginal_mean']:.2f} & {rmo['worst_region_coverage_mean']:.2f} & {rmo['mean_interval_score_mean']:.1f} & "
                  f"{rse['true_marginal_mean']:.2f} & {rse['worst_region_coverage_mean']:.2f} & {rse['mean_interval_score_mean']:.1f} \\\\")
    L_ += [r"\bottomrule", r"\end{tabular}"]
    write("table_ablation.tex", L_)


def meuse_table():
    mm = ["split", "normalized", "cqr", "di_normalized", "di_mondrian", "di_cqr", "di_cqr_K2",
          "lcp", "lcp_cqr", "geo_lcp", "weighted_estimated"]

    def mc(m, met):
        r = M[(M.method == m) & (M.metric == met)].iloc[0]
        return f"{r['mean']:.2f}\\,$\\pm$\\,{r['sd']:.2f}"
    L_ = [r"\begin{tabular}{l ccccc}", r"\toprule",
          r"Method & Marginal & Worst-region & Coverage gap & Mean width & Interval score \\", r"\midrule"]
    for m in mm:
        L_.append(f"{PRETTY[m]} & {mc(m,'marginal')} & {mc(m,'worst_region')} & {mc(m,'coverage_gap')} & "
                  f"{mc(m,'mean_width')} & {mc(m,'mean_interval_score')} \\\\")
    L_ += [r"\bottomrule", r"\end{tabular}"]
    write("table_meuse.tex", L_)


def lucas_table():
    mm = ["split", "normalized", "cqr", "di_normalized", "di_normalized_clip", "di_normalized_auto",
          "di_mondrian", "di_cqr", "di_cqr_auto", "lcp", "lcp_cqr", "geo_lcp", "weighted_estimated"]
    LD = rj("lucas_diag.json")

    def c(m, col, dec=2):
        r = L.loc[m]
        return f"{r[col+'_mean']:.{dec}f}\\,$\\pm$\\,{r[col+'_std']:.{dec}f}"
    L_ = [r"\begin{tabular}{l ccccc}", r"\toprule",
          r"Method & Marginal & Worst-region & Mean width & Interval score & Unbounded (\%) \\", r"\midrule"]
    for m in mm:
        ub = pct(LD["unbounded"][m]) if m in LD["unbounded"] else "0"
        L_.append(f"{PRETTY[m]} & {c(m,'marginal')} & {c(m,'worst_region')} & "
                  f"{c(m,'mean_width',1)} & {c(m,'mean_interval_score',2)} & {ub} \\\\")
    L_ += [r"\bottomrule", r"\end{tabular}"]
    write("table_lucas.tex", L_)


def bridge_table():
    B = rd("sim_bridge.csv").set_index("regime")
    L_ = [r"\begin{tabular}{l ccc}", r"\toprule",
          r"Regime & Spearman$(\mathrm{DI}, w)$ & 95\% CI & Pearson$(\mathrm{DI}, \log w)$ \\", r"\midrule",
          r"None & \multicolumn{3}{c}{n/a (uniform selection, constant weight)} \\"]
    lab = {"mild": "Mild", "moderate": "Moderate", "severe": "Severe",
           "hidden_moderate": "Hidden, moderate", "hidden_severe": "Hidden, severe"}
    for rg in lab:
        r = B.loc[rg]
        L_.append(f"{lab[rg]} & {r['spearman_DI_invpsel_mean']:.2f} & "
                  f"[{r['spearman_ci_lo']:.2f}, {r['spearman_ci_hi']:.2f}] & {r['pearson_DI_logInvPsel_mean']:.2f} \\\\")
    L_ += [r"\bottomrule", r"\end{tabular}"]
    write("table_bridge.tex", L_)


def clip_table():
    C = rd("sim_clip_summary.csv")

    def v(rg, grp, col, dec=2):
        r = C[(C.regime == rg) & (C.group == grp)].iloc[0]
        x = r[col]
        return "--" if not np.isfinite(x) else f"{x:.{dec}f}"
    lab = {"none": "None", "mild": "Mild", "moderate": "Moderate", "severe": "Severe",
           "hidden_moderate": "Hidden, moderate", "hidden_severe": "Hidden, severe"}
    L_ = [r"\begin{tabular}{l cccc}", r"\toprule",
          r"Regime & Beyond cal.\ DI (\%) & Coverage (beyond) & Coverage (inside) & Int.\ score (beyond / inside) \\",
          r"\midrule"]
    for rg in REG:
        fr = C[(C.regime == rg) & (C.group == "clipped")].iloc[0]["frac_mean"]
        L_.append(f"{lab[rg]} & {pct(fr)} & {v(rg,'clipped','coverage_mean')} & {v(rg,'unclipped','coverage_mean')} & "
                  f"{v(rg,'clipped','mean_interval_score',1)} / {v(rg,'unclipped','mean_interval_score',1)} \\\\")
    L_ += [r"\bottomrule", r"\end{tabular}"]
    write("table_clip.tex", L_)


def region_table():
    R = rd("sim_region_sens.csv")

    def v(rg, m, k):
        r = R[(R.regime == rg) & (R.method == m) & (R.k == k)]
        return f"{float(r['mean'].iloc[0]):.2f}" if len(r) else "--"
    pretty = {"split": "Split", "di_normalized": "DI-normalized", "di_cqr": "DI-CQR"}
    L_ = [r"\begin{tabular}{l ccc ccc}", r"\toprule",
          r" & \multicolumn{3}{c}{moderate} & \multicolumn{3}{c}{severe} \\",
          r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
          r"Method & $2\times2$ & $3\times3$ & $4\times4$ & $2\times2$ & $3\times3$ & $4\times4$ \\", r"\midrule"]
    for m in ["split", "di_normalized", "di_cqr"]:
        L_.append(f"{pretty[m]} & {v('moderate',m,2)} & {v('moderate',m,3)} & {v('moderate',m,4)} & "
                  f"{v('severe',m,2)} & {v('severe',m,3)} & {v('severe',m,4)} \\\\")
    L_ += [r"\bottomrule", r"\end{tabular}"]
    write("table_region.tex", L_)


def kappa_table():
    K = rd("sim_kappa_sens.csv")

    def v(rg, kap, col, dec):
        r = K[(K.regime == rg) & (np.isclose(K.kappa, kap))]
        return f"{float(r[col].iloc[0]):.{dec}f}" if len(r) else "--"
    L_ = [r"\begin{tabular}{c cc cc cc}", r"\toprule",
          r" & \multicolumn{2}{c}{no bias} & \multicolumn{2}{c}{moderate} & \multicolumn{2}{c}{severe} \\",
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
          r"$\kappa$ & worst & int.\ score & worst & int.\ score & worst & int.\ score \\", r"\midrule"]
    for kap in [0.1, 0.25, 0.5, 1.0]:
        star = r"$^\star$" if kap == 0.25 else ""
        L_.append(f"{kap}{star} & {v('none',kap,'worst',2)} & {v('none',kap,'interval_score',1)} & "
                  f"{v('moderate',kap,'worst',2)} & {v('moderate',kap,'interval_score',1)} & "
                  f"{v('severe',kap,'worst',2)} & {v('severe',kap,'interval_score',1)} \\\\")
    L_ += [r"\bottomrule", r"\end{tabular}"]
    write("table_kappa.tex", L_)


def mechanism_table():
    Mm = rd("sim_mechanism.csv"); Mp = rd("sim_mechanism_partial.csv")
    glab = {"gain_IS_vs_cqr": r"Interval-score gain vs.\ CQR",
            "gain_IS_vs_split": r"Interval-score gain vs.\ split",
            "gain_worst_vs_split": r"Worst-region gain vs.\ split"}

    def pooled(gain):
        r = Mm[(Mm.group == "pooled_biased") & (Mm.gain == gain)]
        return f"{float(r['pearson_r'].iloc[0]):.2f} ({ptab(float(r['p_value'].iloc[0]))})" if len(r) else "--"

    def partial(gain):
        r = Mp[Mp.gain == gain]
        if not len(r):
            return "--"
        r = r.iloc[0]
        return f"{r['partial_r']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] ({ptab(r['boot_p'])})"
    L_ = [r"\begin{tabular}{l cc}", r"\toprule",
          r"Per-seed gain of DI-CQR & pooled $r$ ($p$) & within-regime partial $r$ [95\% CI] ($p$) \\", r"\midrule"]
    for g in glab:
        L_.append(f"{glab[g]} & {pooled(g)} & {partial(g)} \\\\")
    L_ += [r"\bottomrule", r"\end{tabular}"]
    write("table_mechanism.tex", L_)


def auto_table():
    rows_m = ["di_cqr", "di_cqr_auto", "di_normalized", "di_normalized_auto", "di_normalized_clip"]
    L_ = [r"\begin{tabular}{l cc cc}", r"\toprule",
          r" & \multicolumn{2}{c}{moderate} & \multicolumn{2}{c}{severe} \\",
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
          r"Method & worst-region & int.\ score & worst-region & int.\ score \\", r"\midrule"]
    for m in rows_m:
        L_.append(f"{PRETTY[m]} & {cell('moderate',m,'worst_region_coverage')} & "
                  f"{cell('moderate',m,'mean_interval_score',1)} & "
                  f"{cell('severe',m,'worst_region_coverage')} & {cell('severe',m,'mean_interval_score',1)} \\\\")
    L_ += [r"\bottomrule", r"\end{tabular}"]
    write("table_auto.tex", L_)


def numbers_file():
    lines = ["% Auto-generated by code/make_tables.py -- do not edit by hand.",
             "% Usage in the text: \\res{key}. An unknown key prints a bold ?? marker.",
             r"\makeatletter",
             r"\newcommand{\res}[1]{\ifcsname res@#1\endcsname\csname res@#1\endcsname\else\textbf{??\detokenize{#1}}\fi}"]
    for k in sorted(NUM):
        lines.append(f"\\expandafter\\def\\csname res@{k}\\endcsname{{{NUM[k]}}}")
    lines.append(r"\makeatother")
    write("numbers.tex", lines)
    json.dump(NUM, open(os.path.join(RES, "headline_numbers.json"), "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    sim_numbers(); real_numbers()
    sim_table("table_main.tex", ["moderate", "severe"], ["moderate", "severe"])
    sim_table("table_hidden.tex", ["hidden_moderate", "hidden_severe"], ["moderate", "severe"])
    controls_table(); paired_table(); diag_table(); width_table(); ablation_table()
    meuse_table(); lucas_table(); bridge_table(); clip_table(); region_table()
    kappa_table(); mechanism_table(); auto_table(); numbers_file()
    print(f"wrote LaTeX tables and {len(NUM)} quoted-number macros")
