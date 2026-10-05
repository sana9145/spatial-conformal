"""verify.py -- check the manuscript against the saved results.

1. Every number quoted in the text comes from paper/numbers.tex; this script
   regenerates the macros from the result files and fails if numbers.tex is out
   of date, or if the text uses a key that does not exist.
2. Every qualitative claim made in the text (orderings, significance calls,
   "does / does not") is asserted below against the raw summaries.
3. Design invariants (held-out scoring, width matching, Holm columns, LCP
   validity under exchangeability) are re-checked.
Exits nonzero on any failure.
"""
import os, re, json, subprocess, sys, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
PAP = os.path.join(HERE, "..", "paper")
fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


S = pd.read_csv(os.path.join(RES, "sim_summary.csv"))
P = pd.read_csv(os.path.join(RES, "sim_paired_diffs.csv"))
M = pd.read_csv(os.path.join(RES, "meuse_summary.csv"))
L = pd.read_csv(os.path.join(RES, "lucas_summary.csv")).set_index("method")
RB = pd.read_csv(os.path.join(RES, "robust_summary.csv"))
U = pd.read_csv(os.path.join(RES, "sim_unbounded.csv"))


def v(rg, m, col):
    return float(S[(S.regime == rg) & (S.method == m)].iloc[0][col + "_mean"])


def pdiff(rg, base, metric):
    return P[(P.regime == rg) & (P.baseline == base) & (P.metric == metric)].iloc[0]


def mv(m, met):
    return float(M[(M.method == m) & (M.metric == met)]["mean"].iloc[0])


def lv(m, col):
    return float(L.loc[m][col + "_mean"])


def rv(rg, m, col):
    return float(RB[(RB.regime == rg) & (RB.method == m)].iloc[0][col + "_mean"])


def unb(rg, m):
    r = U[(U.regime == rg) & (U.method == m)]
    return float(r["mean"].iloc[0])


# ---------------------------------------------------------------- 1. numbers
numbers_path = os.path.join(PAP, "numbers.tex")
old = open(numbers_path, encoding="utf8").read() if os.path.exists(numbers_path) else ""
subprocess.run([sys.executable, os.path.join(HERE, "make_tables.py")], check=True,
               capture_output=True, cwd=HERE)
new = open(numbers_path, encoding="utf8").read()
if old:
    check("paper/numbers.tex is up to date with results/", old == new)
keys = set(re.findall(r"\\csname res@(.+?)\\endcsname\{", new))
tex_path = os.path.join(PAP, "paper.tex")
if os.path.exists(tex_path):        # the manuscript is not in the public repository
    tex = open(tex_path, encoding="utf8").read()
    used = set(re.findall(r"\\res\{([^}]+)\}", tex))
    missing = sorted(used - keys)
    check(f"all {len(used)} \\res keys used in paper.tex exist ({len(keys)} defined)", not missing)
    if missing:
        print("   missing:", missing[:20])
else:
    print(f"(paper/paper.tex not present: skipped the text-key check; {len(keys)} macros generated)")

# ---------------------------------------------------------------- 3. design invariants
r0 = json.loads(open(os.path.join(RES, "sim_raw.jsonl")).readline())
check("simulation scores held-out cells only (n_eval = 3136 - 500)",
      r0["n_eval"] == 3136 - 500 and r0["n_mon"] == 500)
runs = [json.loads(l) for l in open(os.path.join(RES, "sim_raw.jsonl"))]
check("simulation complete: 6 regimes x 30 seeds",
      sorted({(r["regime"], r["seed"]) for r in runs}) ==
      sorted({(rg, s) for rg in ["none", "mild", "moderate", "severe", "hidden_moderate", "hidden_severe"]
              for s in range(30)}))
for rg in ["moderate", "severe", "hidden_moderate", "hidden_severe"]:
    check(f"[{rg}] width-matched baseline has DI-CQR's mean width",
          abs(v(rg, "di_cqr", "mean_width") - v(rg, "width_matched_global", "mean_width")) < 1e-9)
check("Holm-adjusted p-values present for the baseline families",
      P[P.regime.isin(["moderate", "severe"]) & (P.baseline != "di_cqr_auto")]["p_holm"].notna().all())
check("no-bias: oracle weights are constant so weighted conformal equals split",
      abs(v("none", "weighted_oracle", "mean_width") - v("none", "split", "mean_width")) < 1e-9)
check("no-bias: weighted conformal never unbounded; LCP unbounded at < 1% of cells",
      unb("none", "weighted_oracle") == 0 and unb("none", "weighted_estimated") == 0
      and unb("none", "lcp") < 0.01)

# LCP implementation: finite-sample validity under exchangeability (quick check)
sys.path.insert(0, HERE)
import conformal as CF
rng = np.random.default_rng(7)
cov = []
for _ in range(120):
    X = rng.uniform(-2, 2, size=(330, 2))
    y = X[:, 0] + rng.normal(size=330) * (0.3 + np.abs(X[:, 1]))
    lo, hi, _ = CF.lcp(X[:130, 0], y[:130], X[:130], X[130:, 0], X[130:], 0.1, bandwidth=0.7)
    cov.append(np.mean((y[130:] >= lo) & (y[130:] <= hi)))
check(f"LCP marginal coverage under exchangeability >= 0.88 (got {np.mean(cov):.3f})", np.mean(cov) >= 0.88)

# ---------------------------------------------------------------- 2. claims
# Each check below corresponds to a qualitative statement in paper.tex.
def rd(n):
    return pd.read_csv(os.path.join(RES, n))


def rj(n):
    return json.load(open(os.path.join(RES, n)))


ESS = rd("sim_weight_ess.csv"); LM = rd("sim_lcp_mass.csv"); CL = rd("sim_clip_summary.csv")
BR = rd("sim_bridge.csv").set_index("regime"); AB = rd("sim_ablation_summary.csv")
SEL = rd("sim_selection.csv").set_index("regime"); KS = rd("sim_kappa_sens.csv")
RS = rd("sim_region_sens.csv"); MM = rd("sim_mechanism.csv"); MP = rd("sim_mechanism_partial.csv")
MD = rj("meuse_diag.json"); LD = rj("lucas_diag.json"); LS = rj("lucas_sig.json"); RSG = rj("robust_sig.json")
ALL_M = sorted(S.method.unique())
STRONG = ["moderate", "severe", "hidden_moderate", "hidden_severe"]
BIASED = ["mild"] + STRONG
COV = ["moderate", "severe"]; HID = ["hidden_moderate", "hidden_severe"]


def holm_sig(rg, base, met):
    return pdiff(rg, base, met)["p_holm"] < 0.05


def ess(rg, m):
    return float(ESS[(ESS.regime == rg) & (ESS.method == m)]["median"].iloc[0])


def lm(rg, g, col):
    return float(LM[(LM.regime == rg) & (LM.group == g)][col].iloc[0])


def cl(rg, g, col):
    return float(CL[(CL.regime == rg) & (CL.group == g)][col].iloc[0])


# 5.1 coverage falls with bias
check("none: split marginal near nominal (0.88-0.94)", 0.88 <= v("none", "split", "true_marginal") <= 0.94)
check("none: all methods within 0.88-0.96 marginal and interval score within 20% of split",
      all(0.88 <= v("none", m, "true_marginal") <= 0.96 for m in ALL_M) and
      all(abs(v("none", m, "mean_interval_score") / v("none", "split", "mean_interval_score") - 1) < 0.2 for m in ALL_M))
sm = [v(r, "split", "true_marginal") for r in ["none", "mild", "moderate", "severe"]]
check("split marginal decreases none > mild > moderate > severe", all(sm[i] > sm[i + 1] for i in range(3)))
check("hidden access collapse milder than covariate access (split marginal)",
      v("hidden_moderate", "split", "true_marginal") > v("moderate", "split", "true_marginal") and
      v("hidden_severe", "split", "true_marginal") > v("severe", "split", "true_marginal"))

# 5.2 principled methods
for rg in BIASED:
    others = [m for m in ALL_M if m not in ("weighted_oracle", "weighted_proxy")]
    check(f"[{rg}] oracle-weighted has the highest marginal and worst-region coverage (excl. 1/p proxy)",
          all(v(rg, "weighted_oracle", c) >= v(rg, m, c) for m in others
              for c in ["true_marginal", "worst_region_coverage"]))
for rg in COV:
    check(f"[{rg}] oracle weights' median ESS below 20 of 200", ess(rg, "weighted_oracle") < 20)
    check(f"[{rg}] 1/p_sel proxy changes little vs exact oracle",
          abs(v(rg, "weighted_proxy", "true_marginal") - v(rg, "weighted_oracle", "true_marginal")) < 0.02 and
          abs(v(rg, "weighted_proxy", "worst_region_coverage") - v(rg, "weighted_oracle", "worst_region_coverage")) < 0.03 and
          abs(unb(rg, "weighted_proxy") - unb(rg, "weighted_oracle")) < 0.03)
    share = cl(rg, "clipped", "frac_mean") * lm(rg, "clipped", "frac_unbounded")
    total = share + cl(rg, "unclipped", "frac_mean") * lm(rg, "unclipped", "frac_unbounded")
    check(f"[{rg}] most (>= 75%) of LCP's unbounded intervals lie beyond the calibration DI range ({share/total:.2f})",
          share / total >= 0.75)
    check(f"[{rg}] LCP local mass much lower beyond the calibration DI range",
          lm(rg, "clipped", "median_mass") < 0.5 * lm(rg, "unclipped", "median_mass"))
    check(f"[{rg}] LCP-CQR covers more than LCP; similar unbounded share",
          v(rg, "lcp_cqr", "true_marginal") > v(rg, "lcp", "true_marginal") and
          abs(unb(rg, "lcp_cqr") - unb(rg, "lcp")) < 0.02)
for rg in STRONG:
    check(f"[{rg}] LCP marginal coverage well below nominal (<= 0.75)", v(rg, "lcp", "true_marginal") <= 0.75)
check("severe: Geo-LCP worst-region barely above split (0 < diff < 0.05)",
      0 < v("severe", "geo_lcp", "worst_region_coverage") - v("severe", "split", "worst_region_coverage") < 0.05)

# 5.3 DI conditioning
for rg in COV:
    for met in ["worst_region_coverage", "mean_interval_score"]:
        for b in ["split", "cqr"]:
            r = pdiff(rg, b, met)
            good = r["mean_diff"] > 0 if met == "worst_region_coverage" else r["mean_diff"] < 0
            check(f"[{rg}] DI-CQR better than {b} on {met} (Holm p < 0.001)", good and r["p_holm"] < 1e-3)
    check(f"[{rg}] DI-CQR vs DI-normalized interval score not significant", not holm_sig(rg, "di_normalized", "mean_interval_score"))
    check(f"[{rg}] DI-normalized (auto kappa) interval score below DI-CQR's",
          v(rg, "di_normalized_auto", "mean_interval_score") < v(rg, "di_cqr", "mean_interval_score"))
    for b in ["lcp", "lcp_cqr"]:
        r = pdiff(rg, b, "mean_interval_score")
        check(f"[{rg}] DI-CQR beats {b} on interval score (Holm p < 0.001)", r["mean_diff"] < 0 and r["p_holm"] < 1e-3)
    check(f"[{rg}] DI-CQR vs LCP-CQR worst-region not significant", not holm_sig(rg, "lcp_cqr", "worst_region_coverage"))
    r = pdiff(rg, "width_matched_global", "mean_interval_score")
    check(f"[{rg}] width-matched: DI-CQR lower interval score but not Holm-significant",
          r["mean_diff"] < 0 and r["p_holm"] > 0.05)
    check(f"[{rg}] width-matched: worst-region difference not significant",
          not holm_sig(rg, "width_matched_global", "worst_region_coverage"))
    check(f"[{rg}] DI-CQR covers less beyond the calibration DI range",
          cl(rg, "clipped", "coverage_mean") < cl(rg, "unclipped", "coverage_mean") - 0.1)

# 5.4 hidden selection
for c, h in [("moderate", "hidden_moderate"), ("severe", "hidden_severe")]:
    check(f"[{h}] DI-weight Spearman lower than covariate access but > 0.5",
          0.5 < BR.loc[h, "spearman_DI_invpsel_mean"] < BR.loc[c, "spearman_DI_invpsel_mean"])
FIXED = ["split", "region_mondrian", "normalized", "cqr", "di_normalized", "di_normalized_clip",
         "di_mondrian", "di_cqr", "lcp", "lcp_cqr", "geo_lcp", "weighted_oracle", "weighted_estimated",
         "weighted_proxy", "width_matched_global"]
for rg in HID:
    check(f"[{rg}] DI-CQR worst-region above split and CQR",
          v(rg, "di_cqr", "worst_region_coverage") > max(v(rg, "split", "worst_region_coverage"), v(rg, "cqr", "worst_region_coverage")))
    check(f"[{rg}] DI-CQR has the lowest interval score of the fixed-hyperparameter methods",
          min(FIXED, key=lambda m: v(rg, m, "mean_interval_score")) == "di_cqr")
    for b in ["cqr", "lcp", "lcp_cqr"]:
        r = pdiff(rg, b, "mean_interval_score")
        check(f"[{rg}] DI-CQR beats {b} on interval score (Holm p < 0.001)", r["mean_diff"] < 0 and r["p_holm"] < 1e-3)
    r = pdiff(rg, "di_normalized", "worst_region_coverage")
    check(f"[{rg}] DI-CQR worst-region above DI-normalized but not Holm-significant",
          r["mean_diff"] > 0 and r["p_holm"] > 0.05)
    check(f"[{rg}] estimated weights raise worst-region somewhat but less than DI-CQR",
          v(rg, "split", "worst_region_coverage") < v(rg, "weighted_estimated", "worst_region_coverage") < v(rg, "di_cqr", "worst_region_coverage"))
    check(f"[{rg}] LCP rarely unbounded (<2%) and covers about as split (|diff| < 0.02)",
          unb(rg, "lcp") < 0.02 and abs(v(rg, "lcp", "true_marginal") - v(rg, "split", "true_marginal")) < 0.02)

# no finite-interval method reaches nominal worst-region coverage under strong bias (simulation)
finite = [m for m in ALL_M if m not in ("lcp", "lcp_cqr", "geo_lcp", "weighted_oracle", "weighted_estimated", "weighted_proxy")]
check("simulation: no method without unbounded intervals reaches 0.90 worst-region under strong bias",
      all(v(rg, m, "worst_region_coverage") < 0.9 for rg in STRONG for m in finite))

# 5.5 sensitivity
def ab(rg, K, col):
    return float(AB[(AB.regime == rg) & (AB.K == K) & (AB.monotone == 1)].iloc[0][col])


check("ablation: severe worst-region rises from K=3 to K=8", ab("severe", 8, "worst_region_coverage_mean") > ab("severe", 3, "worst_region_coverage_mean"))
check("hidden_moderate: estimated weights barely improve split's interval score (0 < gain < 0.5)",
      0 < v("hidden_moderate", "split", "mean_interval_score") - v("hidden_moderate", "weighted_estimated", "mean_interval_score") < 0.5)
check("hidden_severe: estimated weights give a worse interval score than split",
      v("hidden_severe", "weighted_estimated", "mean_interval_score") > v("hidden_severe", "split", "mean_interval_score"))
check("auto-K picks fewer bins than the fixed K=5 on average (severe)", SEL.loc["severe", "K_mean"] < 5)
for rg in COV:
    r = pdiff(rg, "di_cqr_auto", "worst_region_coverage")
    check(f"[{rg}] auto-K slightly below fixed K=5 in worst-region (0 < diff < 0.06)", 0 < r["mean_diff"] < 0.06)
    kw = [float(KS[(KS.regime == rg) & np.isclose(KS.kappa, k)]["worst"].iloc[0]) for k in [0.1, 0.25, 0.5, 1.0]]
    check(f"[{rg}] smaller kappa gives higher worst-region coverage (monotone)", all(kw[i] > kw[i + 1] for i in range(3)))
check("auto-kappa smaller under severe bias than without bias", SEL.loc["severe", "kappa_mean"] < SEL.loc["none", "kappa_mean"])
check("region partition: DI-CQR > split worst-region at every k in all strong regimes",
      all(float(RS[(RS.regime == rg) & (RS.method == "di_cqr") & (RS.k == k)]["mean"].iloc[0]) >
          float(RS[(RS.regime == rg) & (RS.method == "split") & (RS.k == k)]["mean"].iloc[0])
          for rg in STRONG for k in [2, 3, 4]))
r = MM[(MM.group == "pooled_biased") & (MM.gain == "gain_IS_vs_split")].iloc[0]
check("mechanism: pooled r > 0.4, p < 0.001", r["pearson_r"] > 0.4 and r["p_value"] < 1e-3)
r = MP[MP.gain == "gain_IS_vs_split"].iloc[0]
check("mechanism: within-regime partial r ~ 0 and not significant", abs(r["partial_r"]) < 0.1 and r["boot_p"] > 0.05)
for rg in COV:
    check(f"[robust/{rg}] DI-CQR interval score below split, CQR, LCP-CQR",
          all(rv(rg, "di_cqr", "interval_score") < rv(rg, b, "interval_score") for b in ["split", "cqr", "lcp_cqr"]))
    check(f"[robust/{rg}] DI-CQR vs split/CQR/LCP/LCP-CQR significant (t & Wilcoxon < 0.01)",
          all(RSG[f"{rg}:di_cqr-{b}"]["diff"] < 0 and RSG[f"{rg}:di_cqr-{b}"]["t_p"] < 0.01 and RSG[f"{rg}:di_cqr-{b}"]["w_p"] < 0.01
              for b in ["split", "cqr", "lcp", "lcp_cqr"]))
    check(f"[robust/{rg}] DI-CQR worst-region above split and CQR",
          rv(rg, "di_cqr", "worst") > max(rv(rg, "split", "worst"), rv(rg, "cqr", "worst")))

# 5.6 Meuse
MW = M[M.metric == "worst_region"].set_index("method")["mean"]
MI = M[M.metric == "mean_interval_score"].set_index("method")["mean"]
MG = M[M.metric == "coverage_gap"].set_index("method")["mean"]
check("meuse: mild shift, split marginal rounds to 0.90", round(mv("split", "marginal"), 2) >= 0.90)
check("meuse: LCP and split have the two lowest interval scores", set(MI.nsmallest(2).index) == {"lcp", "split"})
check("meuse: LCP and split have the two lowest worst-region coverages", set(MW.nsmallest(2).index) == {"lcp", "split"})
check("meuse: DI-CQR has highest worst-region and smallest gap (excluding its K=2 variant)",
      MW.drop("di_cqr_K2").idxmax() == "di_cqr" and MG.drop("di_cqr_K2").idxmin() == "di_cqr")
check("meuse: every DI-CQR bin uses level 1 (bin maximum)", MD["frac_bins_level_one"] == 1.0)
MSG = rj("meuse_sig.json")
check("meuse: K=2 worst-region below K=4 but above split and CQR (t & Wilcoxon p < 0.01)",
      MW["di_cqr_K2"] < MW["di_cqr"] - 0.03 and
      all(MSG[f"di_cqr_K2-{b}:worst_region"]["diff"] > 0 and MSG[f"di_cqr_K2-{b}:worst_region"]["t_p"] < 0.01
          and MSG[f"di_cqr_K2-{b}:worst_region"]["w_p"] < 0.01 for b in ["split", "cqr"]))
check("meuse: K=2 interval score close to CQR's (|diff| < 0.1)", abs(MI["di_cqr_K2"] - MI["cqr"]) < 0.1)
check("meuse: some region gets a single held-out site", MD["min_test_per_region"] == 1)

# LUCAS
check("lucas: most held-out sites beyond the calibration DI range", LD["di"]["frac_test_above_cal_max"] > 0.5)
check("lucas: DI-CQR vs CQR interval score not significant", LS["di_cqr-cqr"]["t_p"] > 0.05)
check("lucas: DI-CQR interval score below split (p < 0.01)", LS["di_cqr-split"]["diff"] < 0 and LS["di_cqr-split"]["t_p"] < 0.01)
check("lucas: DI-CQR worst-region above split and CQR", lv("di_cqr", "worst_region") > max(lv("split", "worst_region"), lv("cqr", "worst_region")))
DIM = ["di_normalized", "di_normalized_clip", "di_normalized_auto", "di_mondrian", "di_cqr", "di_cqr_auto"]
check("lucas: DI-normalized has the worst interval score of the DI methods",
      max(DIM, key=lambda m: lv(m, "mean_interval_score")) == "di_normalized")
check("lucas: capping DI roughly halves DI-normalized width (ratio 0.4-0.6)",
      0.4 < lv("di_normalized_clip", "mean_width") / lv("di_normalized", "mean_width") < 0.6)
check("lucas: capped DI-normalized has lower interval score than uncapped (p < 0.01)",
      LS["di_normalized_clip-di_normalized"]["diff"] < 0 and LS["di_normalized_clip-di_normalized"]["t_p"] < 0.01)
check("lucas: capped DI-normalized still wider than DI-CQR and no better than split",
      lv("di_normalized_clip", "mean_width") > lv("di_cqr", "mean_width") and
      lv("di_normalized_clip", "mean_interval_score") >= lv("split", "mean_interval_score") and
      LS["di_normalized_clip-split"]["t_p"] > 0.05)
for rg in COV:
    check(f"[{rg}] simulation: capping DI makes DI-normalized worse",
          v(rg, "di_normalized_clip", "mean_interval_score") > v(rg, "di_normalized", "mean_interval_score"))

# coverage by DI decile (partition-free) and cap sensitivity
DC = rd("sim_di_curve.csv"); LCV = rd("lucas_di_curve.csv")
CS = rd("sim_cap_sens.csv"); LCS = rd("lucas_cap_sens.csv")


def dc(rg, m):
    return DC[(DC.regime == rg) & (DC.method == m)].sort_values("decile")["cov_mean"].to_numpy()


def lcv(m):
    return LCV[LCV.method == m].sort_values("decile")["cov_mean"].to_numpy()


sp, wm, di_, cq = dc("severe", "split"), dc("severe", "width_matched_global"), dc("severe", "di_cqr"), dc("severe", "cqr")
check("deciles/severe: split coverage falls steeply (first - last > 0.4)", sp[0] - sp[-1] > 0.4)
check("deciles/severe: DI-CQR profile flatter than split and width-matched",
      di_[0] - di_[-1] < min(sp[0] - sp[-1], wm[0] - wm[-1]))
check("deciles/severe: width-matched over-covers the first decile (> DI-CQR) and falls below DI-CQR in the last",
      wm[0] > di_[0] and wm[-1] < di_[-1])
check("deciles/severe: DI-CQR above CQR in every decile", np.all(di_ > cq))
check("deciles/severe: LCP-CQR and oracle weighted exceed DI-CQR in the last two deciles",
      np.all(dc("severe", "lcp_cqr")[-2:] > di_[-2:]) and np.all(dc("severe", "weighted_oracle")[-2:] > di_[-2:]))
for rg in STRONG:
    d_, w_ = dc(rg, "di_cqr"), dc(rg, "width_matched_global")
    check(f"[{rg}] deciles: DI-CQR coverage range narrower than width-matched (more even spread)",
          d_.max() - d_.min() < w_.max() - w_.min())
h = {m: dc("hidden_severe", m)[-1] for m in ["di_cqr", "width_matched_global", "cqr", "split"]}
check("deciles/hidden_severe: last decile DI-CQR > width-matched > CQR > split",
      h["di_cqr"] > h["width_matched_global"] > h["cqr"] > h["split"])
check("deciles/LUCAS: DI-CQR above CQR and split in every decile",
      np.all(lcv("di_cqr") > lcv("cqr")) and np.all(lcv("di_cqr") > lcv("split")))
UNB5 = ["lcp", "lcp_cqr", "geo_lcp", "weighted_oracle", "weighted_estimated"]
check("cap sensitivity (simulation): DI-CQR interval score below every unbounded method at caps 2-10x, all strong regimes",
      all(v(rg, "di_cqr", "mean_interval_score") <
          CS[(CS.regime == rg) & (CS.mult == k) & CS.method.isin(UNB5)]["mean_interval_score"].min()
          for rg in STRONG for k in [2, 3, 5, 10]))
check("cap sensitivity (LUCAS): DI-CQR interval score below every unbounded method at caps 2-10x",
      all(lv("di_cqr", "mean_interval_score") < LCS[LCS.mult == k]["mean_interval_score"].min() for k in [2, 3, 5, 10]))
check("cap 3x reproduces the main simulation results",
      all(abs(float(CS[(CS.regime == rg) & (CS.mult == 3) & (CS.method == m)]["mean_interval_score"].iloc[0]) -
              v(rg, m, "mean_interval_score")) < 1e-9 for rg in STRONG for m in UNB5))
for rg in HID:
    r = pdiff(rg, "width_matched_global", "mean_interval_score")
    check(f"[{rg}] width-matched: the one t/Wilcoxon disagreement (W-Holm < 0.05, t-Holm > 0.05, diff < 0)",
          r["mean_diff"] < 0 and r["wilcoxon_p_holm"] < 0.05 and r["p_holm"] > 0.05)

# abstract
check("severe: oracle covers more than DI-CQR; LCP covers less (marginal and worst)",
      v("severe", "weighted_oracle", "true_marginal") > v("severe", "di_cqr", "true_marginal") and
      v("severe", "lcp", "true_marginal") < v("severe", "di_cqr", "true_marginal") and
      v("severe", "lcp", "worst_region_coverage") < v("severe", "di_cqr", "worst_region_coverage"))

# methods: Wilcoxon agrees with the Holm-adjusted t-test for every comparison reported
REPORTED = [(rg, b, m) for rg in COV for b, m in
            [("split", "worst_region_coverage"), ("split", "mean_interval_score"), ("cqr", "worst_region_coverage"),
             ("cqr", "mean_interval_score"), ("di_normalized", "mean_interval_score"), ("lcp", "mean_interval_score"),
             ("lcp_cqr", "mean_interval_score"), ("lcp_cqr", "worst_region_coverage"),
             ("width_matched_global", "mean_interval_score"), ("width_matched_global", "worst_region_coverage")]] + \
           [(rg, b, m) for rg in HID for b, m in
            [("cqr", "mean_interval_score"), ("lcp", "mean_interval_score"), ("lcp_cqr", "mean_interval_score"),
             ("di_normalized", "worst_region_coverage")]]
check("Holm-adjusted Wilcoxon agrees with Holm-adjusted t at 0.05 for every reported comparison",
      all((pdiff(rg, b, m)["p_holm"] < 0.05) == (pdiff(rg, b, m)["wilcoxon_p_holm"] < 0.05) for rg, b, m in REPORTED))

# LCP implementation equals a brute-force evaluation of Guan's definitions (1,020 cases)
from scipy.spatial.distance import cdist


def _brute(V, F, x0, vv, alpha, bw):
    Vall = np.r_[V, vv]; Fall = np.vstack([F, x0]); n1 = len(Vall)
    H = np.exp(-cdist(Fall, Fall) ** 2 / (2 * bw ** 2))
    o = np.argsort(Vall); Vo = Vall[o]
    CW = np.cumsum(H[:, o], axis=1) / H.sum(1, keepdims=True)
    betas = np.unique(np.r_[CW.ravel(), CW.ravel() + 1e-12, 0.0, 1.0]); betas = betas[(betas >= 0) & (betas <= 1)]
    idx = np.array([np.searchsorted(CW[i], betas - 1e-15, side="left") for i in range(n1)])
    Qv = Vo[np.minimum(idx, n1 - 1)]
    cnt = (Vall[:, None] <= Qv).sum(0)
    j = np.argmax(cnt >= CF.conformal_rank(n1 - 1, alpha))
    return vv <= Qv[-1, j]


rng = np.random.default_rng(0); mism = tests = 0
for t in range(20):
    F = rng.normal(size=(25, 2)); V = np.abs(rng.normal(size=25)); bw = [0.5, 1.0, 2.0][t % 3]
    x0 = rng.normal(size=(1, 2)) * (1 + 2 * (t % 4 == 0))
    vstar, _ = CF.lcp_threshold(V, F, x0, 0.1, bw)
    Vs = np.sort(V)
    for vv in np.r_[Vs - 1e-7, Vs + 1e-7, Vs.max() + 5]:
        tests += 1; mism += _brute(V, F, x0, vv, 0.1, bw) != (vv <= vstar[0])
check(f"LCP equals brute-force Guan definitions on all {tests} cases", mism == 0 and tests == 1020)

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
raise SystemExit(1 if fails else 0)
