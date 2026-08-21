"""verify.py -- assert every headline claim and every number quoted in the
manuscript against the saved raw output. Exits nonzero on any failure."""
import os, json, numpy as np, pandas as pd
RES = os.path.join(os.path.dirname(__file__), "..", "results")
S = pd.read_csv(os.path.join(RES, "sim_summary.csv"))
P = pd.read_csv(os.path.join(RES, "sim_paired_diffs.csv"))
A = pd.read_csv(os.path.join(RES, "sim_ablation_summary.csv"))
M = pd.read_csv(os.path.join(RES, "meuse_summary.csv"))
PRACT = ["split", "normalized", "di_normalized", "region_mondrian", "di_mondrian",
         "cqr", "di_cqr", "width_matched_global"]
fails = []


def val(rg, m, col):
    return float(S[(S.regime == rg) & (S.method == m)].iloc[0][col + "_mean"])


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def approx(a, b, tol=0.02):
    return abs(a - b) <= tol


# controls
check("no-bias split marginal near nominal (0.88-0.96)", 0.88 <= val("none", "split", "true_marginal") <= 0.96)
sm = [val(r, "split", "true_marginal") for r in ["none", "mild", "moderate", "severe"]]
check("split marginal monotonically decreasing with bias", all(sm[i] > sm[i+1] for i in range(3)))

# di_normalized lowest interval score among practical; di_cqr best marginal among practical
for rg in ["moderate", "severe"]:
    isc = {m: val(rg, m, "mean_interval_score") for m in PRACT}
    check(f"[{rg}] di_normalized has lowest practical interval score", min(isc, key=isc.get) == "di_normalized")
    mar = {m: val(rg, m, "true_marginal") for m in PRACT}
    check(f"[{rg}] di_cqr has highest practical marginal coverage", max(mar, key=mar.get) == "di_cqr")

# weighted-oracle best coverage but hugely wide
for rg in ["moderate", "severe"]:
    allm = S[S.regime == rg].set_index("method")["true_marginal_mean"]
    check(f"[{rg}] weighted_oracle has highest marginal coverage overall", allm.idxmax() == "weighted_oracle")
    check(f"[{rg}] weighted_oracle width >= 5x di_cqr width",
          val(rg, "weighted_oracle", "mean_width") >= 5 * val(rg, "di_cqr", "mean_width"))

# di_cqr does NOT beat oracle on marginal (paired, severe): negative & significant
row = P[(P.regime == "severe") & (P.metric == "true_marginal") & (P.baseline == "weighted_oracle")].iloc[0]
check("di_cqr marginal significantly LOWER than weighted_oracle (severe)", row["mean_diff"] < 0 and row["p_value"] < 0.05)

# width-matched: MUST be exactly width-matched to di_cqr on the evaluation set
for rg in ["moderate", "severe"]:
    dv = val(rg, "di_cqr", "mean_width"); wv = val(rg, "width_matched_global", "mean_width")
    check(f"[{rg}] width-matched global mean width == di_cqr mean width", approx(dv, wv, 1e-6))
    dsd = float(S[(S.regime == rg) & (S.method == "di_cqr")].iloc[0]["mean_width_sd"])
    wsd = float(S[(S.regime == rg) & (S.method == "width_matched_global")].iloc[0]["mean_width_sd"])
    check(f"[{rg}] width-matched global width SD == di_cqr width SD", approx(dsd, wsd, 1e-6))
# at matched width, di_cqr interval-score gain is modest: raw-significant moderate, NOT severe
rmod = P[(P.regime == "moderate") & (P.metric == "mean_interval_score") & (P.baseline == "width_matched_global")].iloc[0]
rsev = P[(P.regime == "severe") & (P.metric == "mean_interval_score") & (P.baseline == "width_matched_global")].iloc[0]
check("di_cqr vs width-matched IS: raw-significant moderate, not severe",
      rmod["mean_diff"] < 0 and rmod["p_value"] < 0.05 and rsev["p_value"] > 0.05)

# di_cqr vs di_normalized interval score NOT significant (both regimes)
for rg in ["moderate", "severe"]:
    row = P[(P.regime == rg) & (P.metric == "mean_interval_score") & (P.baseline == "di_normalized")].iloc[0]
    check(f"[{rg}] di_cqr vs di_normalized interval score NOT significant (p>0.05)", row["p_value"] > 0.05)

# ablation: worst-region coverage increases from K=3 to K=8 (severe, mono)
w3 = A[(A.regime == "severe") & (A.K == 3) & (A.monotone == 1)].iloc[0]["worst_region_coverage_mean"]
w8 = A[(A.regime == "severe") & (A.K == 8) & (A.monotone == 1)].iloc[0]["worst_region_coverage_mean"]
check("ablation: severe worst-region rises K=3 -> K=8", w8 > w3)

# meuse: di_cqr best worst-region among methods
mw = M[M.metric == "worst_region"].set_index("method")["mean"]
check("meuse: di_cqr has best worst-region coverage", mw.idxmax() == "di_cqr")
# meuse: localized has best (lowest) interval score (boundary reversal)
mi = M[M.metric == "mean_interval_score"].set_index("method")["mean"]
check("meuse: localized has lowest interval score (boundary reversal)", mi.idxmin() == "localized")

# --- localized-conformal baseline present and beaten under strong bias ---
B = pd.read_csv(os.path.join(RES, "sim_bridge.csv")).set_index("regime")
for rg in ["moderate", "severe"]:
    check(f"[{rg}] di_cqr present vs localized in results", "localized" in
          S[S.regime == rg]["method"].values)
    ri = P[(P.regime == rg) & (P.metric == "mean_interval_score") & (P.baseline == "localized")].iloc[0]
    rw = P[(P.regime == rg) & (P.metric == "worst_region_coverage") & (P.baseline == "localized")].iloc[0]
    check(f"[{rg}] di_cqr beats localized on interval score (t & wilcoxon)",
          ri["mean_diff"] < 0 and ri["p_value"] < 0.05 and ri["wilcoxon_p"] < 0.05)
    check(f"[{rg}] di_cqr beats localized on worst-region (t & wilcoxon)",
          rw["mean_diff"] > 0 and rw["p_value"] < 0.05 and rw["wilcoxon_p"] < 0.05)

# --- bridge: DI vs 1/p_sel correlation increases with bias, severe>0.9 ---
sp = {rg: float(B.loc[rg, "spearman_DI_invpsel_mean"]) for rg in ["mild", "moderate", "severe"]}
check("bridge: Spearman increases mild<moderate<severe", sp["mild"] < sp["moderate"] < sp["severe"])
check("bridge: severe Spearman > 0.9", sp["severe"] > 0.9)

# --- exact numbers quoted in the manuscript prose ---
quoted = [
    ("moderate di_normalized IS ~7.1", approx(val("moderate", "di_normalized", "mean_interval_score"), 7.1, 0.15)),
    ("severe di_normalized IS ~9.6", approx(val("severe", "di_normalized", "mean_interval_score"), 9.6, 0.2)),
    ("moderate di_cqr IS ~7.2", approx(val("moderate", "di_cqr", "mean_interval_score"), 7.2, 0.15)),
    ("severe di_cqr IS ~10.1", approx(val("severe", "di_cqr", "mean_interval_score"), 10.1, 0.2)),
    ("moderate di_cqr marginal ~0.83", approx(val("moderate", "di_cqr", "true_marginal"), 0.83, 0.02)),
    ("severe di_cqr marginal ~0.73", approx(val("severe", "di_cqr", "true_marginal"), 0.73, 0.02)),
    ("moderate di_cqr worst ~0.59", approx(val("moderate", "di_cqr", "worst_region_coverage"), 0.59, 0.03)),
    ("moderate split marginal ~0.51", approx(val("moderate", "split", "true_marginal"), 0.51, 0.02)),
    ("severe split marginal ~0.35", approx(val("severe", "split", "true_marginal"), 0.35, 0.02)),
    ("weighted_oracle width moderate ~33.2", approx(val("moderate", "weighted_oracle", "mean_width"), 33.2, 1.5)),
    ("weighted_oracle width severe ~42.7", approx(val("severe", "weighted_oracle", "mean_width"), 42.7, 2.0)),
    ("meuse di_cqr worst ~0.93", approx(float(M[(M.method=="di_cqr")&(M.metric=="worst_region")]["mean"].iloc[0]), 0.93, 0.03)),
    ("meuse split marginal ~0.93", approx(float(M[(M.method=="split")&(M.metric=="marginal")]["mean"].iloc[0]), 0.93, 0.03)),
    ("localized IS moderate ~10.1", approx(val("moderate", "localized", "mean_interval_score"), 10.1, 0.4)),
    ("localized IS severe ~17.0", approx(val("severe", "localized", "mean_interval_score"), 17.0, 0.7)),
    ("bridge spearman mild ~0.68", approx(sp["mild"], 0.68, 0.03)),
    ("bridge spearman moderate ~0.91", approx(sp["moderate"], 0.91, 0.03)),
    ("bridge spearman severe ~0.94", approx(sp["severe"], 0.94, 0.03)),
    ("di_cqr-localized IS diff moderate ~-2.93",
     approx(float(P[(P.regime=="moderate")&(P.metric=="mean_interval_score")&(P.baseline=="localized")]["mean_diff"].iloc[0]), -2.93, 0.3)),
    ("di_cqr-localized IS diff severe ~-6.87",
     approx(float(P[(P.regime=="severe")&(P.metric=="mean_interval_score")&(P.baseline=="localized")]["mean_diff"].iloc[0]), -6.87, 0.5)),
    ("meuse localized IS ~1.66", approx(float(M[(M.method=="localized")&(M.metric=="mean_interval_score")]["mean"].iloc[0]), 1.66, 0.06)),
    ("meuse di_cqr IS ~2.13", approx(float(M[(M.method=="di_cqr")&(M.metric=="mean_interval_score")]["mean"].iloc[0]), 2.13, 0.06)),
]
for name, cond in quoted:
    check("quoted: " + name, cond)

# --- HGB base-model robustness ---
import os as _os
_rp = _os.path.join(RES, "robust_summary.csv")
if _os.path.exists(_rp):
    RB = pd.read_csv(_rp)
    def rval(rg, m, col):
        return float(RB[(RB.regime == rg) & (RB.method == m)].iloc[0][col + "_mean"])
    prac = ["split", "normalized", "di_normalized", "cqr", "di_cqr", "localized", "width_matched_global"]
    for rg in ["moderate", "severe"]:
        isc = {m: rval(rg, m, "interval_score") for m in prac}
        check(f"[robust/{rg}] di_normalized lowest practical interval score", min(isc, key=isc.get) == "di_normalized")
        check(f"[robust/{rg}] di_normalized & di_cqr beat split & localized on IS",
              rval(rg, "di_normalized", "interval_score") < rval(rg, "split", "interval_score") and
              rval(rg, "di_cqr", "interval_score") < rval(rg, "localized", "interval_score"))
    RS = json.load(open(_os.path.join(RES, "robust_sig.json")))
    check("robust: all DI-vs-{split,localized} IS diffs significant (t & wilcoxon <1e-4)",
          all(v["diff"] < 0 and v["t_p"] < 1e-4 and v["w_p"] < 1e-4 for v in RS.values()))
    for nm, cond in [
        ("robust di_normalized IS moderate ~6.9", approx(rval("moderate", "di_normalized", "interval_score"), 6.9, 0.4)),
        ("robust di_normalized IS severe ~10.0", approx(rval("severe", "di_normalized", "interval_score"), 10.0, 0.5)),
        ("robust split IS severe ~19.5", approx(rval("severe", "split", "interval_score"), 19.5, 1.0)),
        ("robust localized IS severe ~16.6", approx(rval("severe", "localized", "interval_score"), 16.6, 0.9)),
    ]:
        check("quoted: " + nm, cond)

# --- LUCAS real-data (EU-scale, high-dim) ---
_lp = _os.path.join(RES, "lucas_summary.csv")
if _os.path.exists(_lp):
    LU = pd.read_csv(_lp).set_index("method")
    def lv(m, col):
        return float(LU.loc[m][col + "_mean"])
    check("[lucas] di_cqr improves worst-region over split & cqr",
          lv("di_cqr", "worst_region") > lv("split", "worst_region") and
          lv("di_cqr", "worst_region") > lv("cqr", "worst_region"))
    check("[lucas] di_cqr interval score below split & localized",
          lv("di_cqr", "mean_interval_score") < lv("split", "mean_interval_score") and
          lv("di_cqr", "mean_interval_score") < lv("localized", "mean_interval_score"))
    check("[lucas] DI-normalized OVER-WIDENS (width > 2x di_cqr; IS worse than split)",
          lv("di_normalized", "mean_width") > 2 * lv("di_cqr", "mean_width") and
          lv("di_normalized", "mean_interval_score") > lv("split", "mean_interval_score"))
    LS = json.load(open(_os.path.join(RES, "lucas_sig.json")))
    check("[lucas] di_cqr sig. better than split & localized on IS (t & wilcoxon <0.01)",
          all(LS[f"di_cqr-{b}"]["diff"] < 0 and LS[f"di_cqr-{b}"]["t_p"] < 0.01 and LS[f"di_cqr-{b}"]["w_p"] < 0.01
              for b in ["split", "localized"]))
    check("[lucas] di_normalized sig. WORSE than split on IS",
          LS["di_normalized-split"]["diff"] > 0 and LS["di_normalized-split"]["t_p"] < 0.01)
    for nm, cond in [
        ("lucas di_cqr worst ~0.79", approx(lv("di_cqr", "worst_region"), 0.79, 0.03)),
        ("lucas split worst ~0.57", approx(lv("split", "worst_region"), 0.57, 0.03)),
        ("lucas di_normalized width ~10.8", approx(lv("di_normalized", "mean_width"), 10.8, 1.5)),
        ("lucas di_cqr IS ~4.65", approx(lv("di_cqr", "mean_interval_score"), 4.65, 0.2)),
    ]:
        check("quoted: " + nm, cond)

# --- held-out evaluation: sim raw runs score only unmonitored cells ---
_sr = _os.path.join(RES, "sim_raw.jsonl")
if _os.path.exists(_sr):
    r0 = json.loads(open(_sr).readline())
    check("sim evaluates held-out cells only (n_eval == 3136 - n_mon)",
          r0.get("n_eval") == 3136 - r0.get("n_mon", 500) and r0.get("n_mon") == 500)

# --- clipping diagnostics: clipped points under-covered vs unclipped; fraction grows ---
_cp = _os.path.join(RES, "sim_clip_summary.csv")
if _os.path.exists(_cp):
    C = pd.read_csv(_cp)
    def cl(rg, grp, col):
        return float(C[(C.regime == rg) & (C.group == grp)].iloc[0][col])
    check("clip: severe clipped fraction > moderate > mild",
          cl("severe", "clipped", "frac_mean") > cl("moderate", "clipped", "frac_mean") > cl("mild", "clipped", "frac_mean"))
    for rg in ["moderate", "severe"]:
        check(f"[{rg}] clipped points under-covered vs unclipped",
              cl(rg, "clipped", "coverage_mean") < cl(rg, "unclipped", "coverage_mean"))
    check("clip: severe clipped coverage ~0.65", approx(cl("severe", "clipped", "coverage_mean"), 0.65, 0.03))
    check("clip: severe clipped fraction ~0.60", approx(cl("severe", "clipped", "frac_mean"), 0.60, 0.05))

# --- Meuse bin diagnostics: 60 fit / 40 cal, no fallback, ~7% clipped ---
_md = _os.path.join(RES, "meuse_diag.json")
if _os.path.exists(_md):
    D = json.load(open(_md))
    check("meuse: 60 fit / 40 cal, K=4, min_n=6", D["n_fit"] == 60 and D["n_cal"] == 40 and D["K"] == 4 and D["min_n"] == 6)
    check("meuse: zero fallback bins (bins stay active, DI-CQR != CQR)", D["mean_fallback_bins"] == 0)
    check("meuse: ~7% of test queries clipped", approx(D["mean_frac_test_clipped"], 0.067, 0.03))

# --- weighted-conformal capped fraction grows with bias (documents the cap dependence) ---
_wc = _os.path.join(RES, "sim_weighted_capped.csv")
if _os.path.exists(_wc):
    WC = pd.read_csv(_wc)
    def wc(rg, m):
        r = WC[(WC.regime == rg) & (WC.method == m)]
        return float(r["mean"].iloc[0]) if len(r) else float("nan")
    check("weighted-oracle capped fraction grows none<moderate<severe",
          wc("none", "weighted_oracle") < wc("moderate", "weighted_oracle") < wc("severe", "weighted_oracle"))
    check("weighted-oracle severe capped fraction ~0.77", approx(wc("severe", "weighted_oracle"), 0.77, 0.05))
    check("weighted-oracle none capped fraction == 0", wc("none", "weighted_oracle") == 0.0)

# --- Localized-CQR control: DI-CQR beats it (so the gain is not just CQR scores) ---
if "localized_cqr" in S["method"].values:
    for rg in ["moderate", "severe"]:
        ri = P[(P.regime == rg) & (P.metric == "mean_interval_score") & (P.baseline == "localized_cqr")].iloc[0]
        rw = P[(P.regime == rg) & (P.metric == "worst_region_coverage") & (P.baseline == "localized_cqr")].iloc[0]
        check(f"[{rg}] di_cqr beats Localized-CQR on interval score (sig.)",
              ri["mean_diff"] < 0 and ri["p_value"] < 0.05)
        check(f"[{rg}] di_cqr beats Localized-CQR on worst-region (sig.)",
              rw["mean_diff"] > 0 and rw["p_value"] < 0.05)
    # Localized-CQR should itself beat residual-localized (CQR score helps it)
    check("Localized-CQR interval score < residual-localized (severe)",
          val("severe", "localized_cqr", "mean_interval_score") < val("severe", "localized", "mean_interval_score"))

# --- mechanism: pooled DI-weight coupling predicts DI-CQR gain ---
_mp = _os.path.join(RES, "sim_mechanism.csv")
if _os.path.exists(_mp):
    MECH = pd.read_csv(_mp)
    def mrow(grp, gain):
        r = MECH[(MECH.group == grp) & (MECH.gain == gain)]
        return r.iloc[0] if len(r) else None
    r = mrow("pooled_biased", "gain_IS_vs_split")
    check("mechanism: pooled DI-weight corr predicts DI-CQR IS gain vs split (r>0.4, p<1e-3)",
          r is not None and r["pearson_r"] > 0.4 and r["p_value"] < 1e-3)
    r2 = mrow("pooled_biased", "gain_IS_vs_cqr")
    check("mechanism: pooled corr predicts DI-CQR IS gain vs CQR (r>0.3, p<1e-3)",
          r2 is not None and r2["pearson_r"] > 0.3 and r2["p_value"] < 1e-3)

# --- mechanism (honest): regime-controlled partial correlation is near zero / n.s. ---
_mpp = _os.path.join(RES, "sim_mechanism_partial.csv")
if _os.path.exists(_mpp):
    MP = pd.read_csv(_mpp)
    rr = MP[MP.gain == "gain_IS_vs_split"].iloc[0]
    check("mechanism: within-regime partial corr (IS vs split) is small and n.s.",
          abs(rr["partial_r"]) < 0.2 and rr["boot_p"] > 0.05)

# --- region sensitivity: di_cqr beats split at every partition granularity ---
_rp2 = _os.path.join(RES, "sim_region_sens.csv")
if _os.path.exists(_rp2):
    RS = pd.read_csv(_rp2)
    def rw2(rg, m, k):
        r = RS[(RS.regime == rg) & (RS.method == m) & (RS.k == k)]
        return float(r["mean"].iloc[0]) if len(r) else float("nan")
    ok = all(rw2(rg, "di_cqr", k) > rw2(rg, "split", k)
             for rg in ["moderate", "severe"] for k in [2, 3, 4])
    check("region sensitivity: di_cqr > split worst-region at every k (moderate+severe)", ok)

# --- kappa sensitivity: monotone (smaller kappa -> higher worst-region), 0.25 present ---
_kp = _os.path.join(RES, "sim_kappa_sens.csv")
if _os.path.exists(_kp):
    KS = pd.read_csv(_kp)
    def kw(rg, kap, col="worst"):
        r = KS[(KS.regime == rg) & (np.isclose(KS.kappa, kap))]
        return float(r[col].iloc[0]) if len(r) else float("nan")
    check("kappa sensitivity: moderate worst decreases 0.1->1.0 (monotone)",
          kw("moderate", 0.1) > kw("moderate", 0.25) > kw("moderate", 0.5) > kw("moderate", 1.0))
    check("kappa sensitivity: a-priori kappa=0.25 present", not np.isnan(kw("moderate", 0.25)))

# --- calibration-valid auto-selection: present, improves DI-normalized, adaptive kappa ---
if "di_normalized_auto" in S["method"].values:
    for rg in ["moderate", "severe"]:
        check(f"[{rg}] auto-kappa improves DI-normalized worst-region coverage",
              val(rg, "di_normalized_auto", "worst_region_coverage") > val(rg, "di_normalized", "worst_region_coverage"))
    check("severe: auto-kappa lowers DI-normalized interval score",
          val("severe", "di_normalized_auto", "mean_interval_score") < val("severe", "di_normalized", "mean_interval_score"))
    check("di_cqr_auto present and within 0.05 worst-region of a-priori di_cqr (severe)",
          "di_cqr_auto" in S["method"].values and
          abs(val("severe", "di_cqr_auto", "worst_region_coverage") - val("severe", "di_cqr", "worst_region_coverage")) < 0.05)
_selp = _os.path.join(RES, "sim_selection.csv")
if _os.path.exists(_selp):
    SEL = pd.read_csv(_selp)
    def kap(rg):
        return float(SEL[SEL.regime == rg]["kappa_mean"].iloc[0])
    check("auto-kappa adapts: mean selected kappa smaller under severe than no bias",
          kap("severe") < kap("none"))

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
raise SystemExit(1 if fails else 0)
