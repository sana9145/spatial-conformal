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

# width-matched: di_cqr interval score significantly lower (severe)
row = P[(P.regime == "severe") & (P.metric == "mean_interval_score") & (P.baseline == "width_matched_global")].iloc[0]
check("di_cqr interval score sig. lower than width-matched global (severe)", row["mean_diff"] < 0 and row["p_value"] < 0.05)

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
    ("moderate di_normalized IS ~6.3", approx(val("moderate", "di_normalized", "mean_interval_score"), 6.3, 0.1)),
    ("severe di_normalized IS ~8.4", approx(val("severe", "di_normalized", "mean_interval_score"), 8.4, 0.15)),
    ("moderate di_cqr IS ~6.4", approx(val("moderate", "di_cqr", "mean_interval_score"), 6.4, 0.1)),
    ("severe di_cqr IS ~8.8", approx(val("severe", "di_cqr", "mean_interval_score"), 8.8, 0.15)),
    ("moderate di_cqr marginal ~0.86", approx(val("moderate", "di_cqr", "true_marginal"), 0.86, 0.02)),
    ("severe di_cqr marginal ~0.77", approx(val("severe", "di_cqr", "true_marginal"), 0.77, 0.02)),
    ("moderate di_cqr worst ~0.61", approx(val("moderate", "di_cqr", "worst_region_coverage"), 0.61, 0.03)),
    ("moderate split marginal ~0.58", approx(val("moderate", "split", "true_marginal"), 0.58, 0.02)),
    ("severe split marginal ~0.44", approx(val("severe", "split", "true_marginal"), 0.44, 0.02)),
    ("weighted_oracle width moderate ~28.5", approx(val("moderate", "weighted_oracle", "mean_width"), 28.5, 1.0)),
    ("weighted_oracle width severe ~36.4", approx(val("severe", "weighted_oracle", "mean_width"), 36.4, 1.5)),
    ("meuse di_cqr worst ~0.93", approx(float(M[(M.method=="di_cqr")&(M.metric=="worst_region")]["mean"].iloc[0]), 0.93, 0.03)),
    ("meuse split marginal ~0.93", approx(float(M[(M.method=="split")&(M.metric=="marginal")]["mean"].iloc[0]), 0.93, 0.03)),
    ("localized IS moderate ~8.8", approx(val("moderate", "localized", "mean_interval_score"), 8.8, 0.3)),
    ("localized IS severe ~14.5", approx(val("severe", "localized", "mean_interval_score"), 14.5, 0.6)),
    ("bridge spearman mild ~0.68", approx(sp["mild"], 0.68, 0.03)),
    ("bridge spearman moderate ~0.91", approx(sp["moderate"], 0.91, 0.03)),
    ("bridge spearman severe ~0.94", approx(sp["severe"], 0.94, 0.03)),
    ("di_cqr-localized IS diff moderate ~-2.45",
     approx(float(P[(P.regime=="moderate")&(P.metric=="mean_interval_score")&(P.baseline=="localized")]["mean_diff"].iloc[0]), -2.45, 0.3)),
    ("di_cqr-localized IS diff severe ~-5.75",
     approx(float(P[(P.regime=="severe")&(P.metric=="mean_interval_score")&(P.baseline=="localized")]["mean_diff"].iloc[0]), -5.75, 0.5)),
    ("meuse localized IS ~1.66", approx(float(M[(M.method=="localized")&(M.metric=="mean_interval_score")]["mean"].iloc[0]), 1.66, 0.05)),
    ("meuse di_cqr IS ~2.13", approx(float(M[(M.method=="di_cqr")&(M.metric=="mean_interval_score")]["mean"].iloc[0]), 2.13, 0.05)),
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
        ("robust di_normalized IS moderate ~6.0", approx(rval("moderate", "di_normalized", "interval_score"), 6.0, 0.3)),
        ("robust di_normalized IS severe ~8.7", approx(rval("severe", "di_normalized", "interval_score"), 8.7, 0.4)),
        ("robust split IS severe ~16.6", approx(rval("severe", "split", "interval_score"), 16.6, 0.8)),
        ("robust localized IS severe ~14.2", approx(rval("severe", "localized", "interval_score"), 14.2, 0.8)),
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

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
raise SystemExit(1 if fails else 0)
