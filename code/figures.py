"""figures.py -- publication figures from the saved result CSV/JSON files."""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
plt.rcParams.update({"font.size": 9, "savefig.dpi": 300, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
NOM = 0.9
REG = ["none", "mild", "moderate", "severe", "hidden_moderate", "hidden_severe"]
REGLAB = ["none", "mild", "moderate", "severe", "hidden\nmoderate", "hidden\nsevere"]
PRETTY = {"split": "Split", "normalized": "Normalized", "di_normalized": "DI-normalized",
          "region_mondrian": "Spatial-Mondrian", "di_mondrian": "DI-Mondrian",
          "cqr": "CQR", "di_cqr": "DI-CQR", "lcp": "LCP", "lcp_cqr": "LCP-CQR",
          "geo_lcp": "Geo-LCP", "weighted_oracle": "Weighted (oracle)",
          "weighted_estimated": "Weighted (estimated)", "width_matched_global": "Width-matched",
          "di_cqr_K2": "DI-CQR (K=2)"}
COL = {"split": "#4c78a8", "cqr": "#f58518", "di_cqr": "#d62728", "di_normalized": "#ff9896",
       "weighted_oracle": "#54a24b", "lcp": "#9467bd", "lcp_cqr": "#c5b0d5"}
S = pd.read_csv(os.path.join(RES, "sim_summary.csv"))


def g(rg, m, col):
    r = S[(S.regime == rg) & (S.method == m)].iloc[0]
    return r[col + "_mean"], r[col + "_sd"]


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)


F = json.load(open(os.path.join(RES, "sim_fields_severe.json")))
FH = json.load(open(os.path.join(RES, "sim_fields_hidden_severe.json")))

# ---- Fig 1: the spatial problem ------------------------------------------------
G = F["GRID"]; ext = [0, 1, 0, 1]


def grid(a):
    return np.array(a, float).reshape(G, G)


def regions(ax, F_, c="w"):
    # boundaries of the 3 x 3 region partition used for worst-region coverage
    for t in (1 / 3, 2 / 3):
        ax.axhline(t, c=c, lw=0.7, alpha=0.8); ax.axvline(t, c=c, lw=0.7, alpha=0.8)


fig, ax = plt.subplots(1, 4, figsize=(15, 3.9))
for j, (F_, lab) in enumerate([(F, "(a) covariate-driven access"), (FH, "(b) hidden road access")]):
    co = np.array(F_["coords"]); fi = np.array(F_["fit"]); ca = np.array(F_["cal"])
    im = ax[j].imshow(np.log10(grid(F_["p_sel"]) + 1e-300).clip(-12, None), origin="lower",
                      extent=ext, cmap="viridis")
    regions(ax[j], F_)
    ax[j].scatter(co[fi, 0], co[fi, 1], s=4, c="#ff7f0e", lw=0, label="fitting")
    ax[j].scatter(co[ca, 0], co[ca, 1], s=4, c="cyan", lw=0, label="calibration")
    ax[j].set_title(lab, fontsize=9, loc="left")
    cb = fig.colorbar(im, ax=ax[j], fraction=.046); cb.set_label(r"$\log_{10} p_{\mathrm{sel}}$")
ax[0].legend(loc="upper right", fontsize=7, framealpha=0.85, markerscale=2)
mon = np.zeros(G * G, bool); mon[np.array(F["mon"])] = True
ax[2].imshow(np.where(mon, 1.0, 0.12).reshape(G, G), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1)
regions(ax[2], F, c="#d62728")
ax[2].set_title("(c) held-out cells (light), as in (a)", fontsize=9, loc="left")
im = ax[3].imshow(grid(F["di"]), origin="lower", extent=ext, cmap="magma")
regions(ax[3], F)
ax[3].set_title("(d) Dissimilarity Index, as in (a)", fontsize=9, loc="left")
fig.colorbar(im, ax=ax[3], fraction=.046)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); save(fig, "fig1_setup.png")

# ---- Fig 2: coverage across regimes -----------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
lines = ["split", "cqr", "lcp", "di_normalized", "di_cqr", "weighted_oracle"]
for j, col in enumerate(["true_marginal", "worst_region_coverage"]):
    for k, m in enumerate(lines):
        x = np.arange(len(REG)) + (k - 2.5) * 0.06
        ys = [g(rg, m, col)[0] for rg in REG]; es = [g(rg, m, col)[1] for rg in REG]
        ax[j].errorbar(x, ys, yerr=es, marker="o", ms=3.5, capsize=2, lw=1.2, label=PRETTY[m], color=COL[m])
    ax[j].axhline(NOM, ls="--", c="k", lw=0.8)
    ax[j].axvline(3.5, c="0.6", lw=0.6)
    ax[j].set_xticks(range(len(REG))); ax[j].set_xticklabels(REGLAB)
    ax[j].set_ylim(0, 1.02)
ax[0].set_ylabel("marginal coverage"); ax[1].set_ylabel("worst-region coverage")
ax[0].set_title("(a)", loc="left"); ax[1].set_title("(b)", loc="left")
ax[1].legend(fontsize=7, loc="lower left", ncol=2)
fig.tight_layout(); save(fig, "fig2_coverage_regimes.png")

# ---- Fig 3: method comparison, severe covariate and hidden regimes -----------------
order = ["split", "region_mondrian", "normalized", "cqr", "di_normalized", "di_mondrian",
         "di_cqr", "lcp", "lcp_cqr", "geo_lcp", "weighted_estimated", "weighted_oracle",
         "width_matched_global"]
fig, ax = plt.subplots(2, 2, figsize=(12, 6.8))
for ci, (rg, lab) in enumerate([("severe", "severe, covariate access"), ("hidden_severe", "severe, hidden access")]):
    for ri, col in enumerate(["mean_interval_score", "worst_region_coverage"]):
        a = ax[ri, ci]
        vals = [g(rg, m, col)[0] for m in order]; es = [g(rg, m, col)[1] for m in order]
        colors = ["#d62728" if m in ("di_cqr", "di_normalized") else
                  ("#9467bd" if "lcp" in m else ("#54a24b" if "weighted" in m else "#9ab0c8")) for m in order]
        a.bar(range(len(order)), vals, yerr=es, capsize=1.5, color=colors, edgecolor="k", lw=.3,
              error_kw=dict(lw=0.6))
        a.set_xticks(range(len(order)))
        a.set_xticklabels([PRETTY[m] for m in order], rotation=45, ha="right", fontsize=7.5)
        if col == "worst_region_coverage":
            a.axhline(NOM, ls="--", c="k", lw=0.8); a.set_ylim(0, 1); a.set_ylabel("worst-region coverage")
        else:
            a.set_ylabel("interval score (log scale; lower is better)"); a.set_yscale("log")
            a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
            a.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
                lambda x, _: f"{x:g}" if f"{x:g}"[0] in "25" else ""))
        a.set_title(f"({'abcd'[2 * ri + ci]}) {lab}", loc="left", fontsize=9)
fig.tight_layout(); save(fig, "fig3_methods.png")

# ---- Fig 4: coverage by decile of the dissimilarity index, averaged ----------------------
DC = pd.read_csv(os.path.join(RES, "sim_di_curve.csv"))
LCV = pd.read_csv(os.path.join(RES, "lucas_di_curve.csv"))
curve_m = ["split", "cqr", "di_normalized", "di_cqr", "lcp_cqr", "weighted_oracle", "width_matched_global"]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9), sharey=True)
panels = [(DC[DC.regime == "severe"], "(a) simulation, severe covariate access", 30),
          (DC[DC.regime == "hidden_severe"], "(b) simulation, severe hidden access", 30),
          (LCV, "(c) LUCAS 2015", 20)]
for j, (d, lab, nrep) in enumerate(panels):
    for m in curve_m:
        dm = d[d.method == m].sort_values("decile")
        if not len(dm):
            continue
        se = dm["cov_sd"] / np.sqrt(nrep)
        c_ = COL.get(m, "0.45")
        ls_ = "--" if m == "width_matched_global" else "-"
        ax[j].plot(dm["decile"], dm["cov_mean"], ls_, marker="o", ms=3.5, lw=1.4, color=c_, label=PRETTY[m])
        ax[j].fill_between(dm["decile"], dm["cov_mean"] - se, dm["cov_mean"] + se, color=c_, alpha=0.15, lw=0)
    ax[j].axhline(NOM, ls="--", c="k", lw=0.8)
    ax[j].set_xticks(range(1, 11)); ax[j].set_ylim(0, 1.02)
    ax[j].set_xlabel("decile of the dissimilarity index (1 = least dissimilar)")
    ax[j].set_title(lab, loc="left", fontsize=9)
ax[0].set_ylabel("coverage")
ax[0].legend(fontsize=7, loc="lower left")
fig.tight_layout(); save(fig, "fig4_coverage_vs_di.png")

# ---- Fig A1: coverage-width frontier ---------------------------------------------------
sw = pd.read_csv(os.path.join(RES, "sim_sweep_summary.csv"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for j, rg in enumerate(["moderate", "severe"]):
    for m in ["split", "cqr", "di_cqr", "weighted_oracle"]:
        d = sw[(sw.regime == rg) & (sw.method == m)].sort_values("mean_width")
        ax[j].plot(d["mean_width"], d["marginal"], marker="o", ms=3.5, label=PRETTY[m], color=COL[m])
    ax[j].axhline(NOM, ls="--", c="k", lw=0.8); ax[j].set_xlabel("mean interval width (log scale)")
    ax[j].set_ylabel("marginal coverage"); ax[j].set_title(f"({'ab'[j]}) {rg}", loc="left")
    ax[j].set_xscale("log")
ax[0].legend(fontsize=7, loc="lower right")
fig.tight_layout(); save(fig, "figA1_frontier.png")

# ---- Fig A2: DI-bin ablation -----------------------------------------------------------
A = pd.read_csv(os.path.join(RES, "sim_ablation_summary.csv"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for j, col in enumerate(["mean_interval_score_mean", "worst_region_coverage_mean"]):
    for rg, c in [("moderate", "#f58518"), ("severe", "#d62728")]:
        for mono, ls in [(1, "-"), (0, "--")]:
            d = A[(A.regime == rg) & (A.monotone == mono)].sort_values("K")
            ax[j].plot(d["K"], d[col], ls, marker="o", ms=3.5, color=c,
                       label=f"{rg}, monotone {'on' if mono else 'off'}")
    ax[j].set_xlabel("number of DI bins $K$"); ax[j].set_title(f"({'ab'[j]})", loc="left")
ax[0].set_ylabel("interval score"); ax[1].set_ylabel("worst-region coverage")
ax[0].legend(fontsize=7)
fig.tight_layout(); save(fig, "figA2_ablation.png")

# ---- Fig A3: DI against the oracle shift weight ------------------------------------------
B = pd.read_csv(os.path.join(RES, "sim_bridge.csv")).set_index("regime")
fig, ax = plt.subplots(1, 3, figsize=(14, 3.7))
for j, (F_, lab) in enumerate([(F, "covariate access, severe"), (FH, "hidden access, severe")]):
    e_ = np.array(F_["ev"]); w = np.array(F_["w_oracle"])[e_]; d_ = np.array(F_["di"])[e_]
    sub = np.random.default_rng(0).choice(len(w), size=min(2000, len(w)), replace=False)
    ax[j].scatter(w[sub], d_[sub], s=5, alpha=.35, c="#4c78a8", lw=0)
    ax[j].set_xscale("log"); ax[j].set_xlabel(r"oracle weight $w=(1-\pi)/\pi$ (log scale)")
    ax[j].set_ylabel("Dissimilarity Index"); ax[j].set_title(f"({'ab'[j]}) {lab}", loc="left")
rg = ["mild", "moderate", "severe", "hidden_moderate", "hidden_severe"]
sp = [B.loc[r, "spearman_DI_invpsel_mean"] for r in rg]
lo = [B.loc[r, "spearman_ci_lo"] for r in rg]; hi = [B.loc[r, "spearman_ci_hi"] for r in rg]
ax[2].errorbar(range(5), sp, yerr=[[s - l for s, l in zip(sp, lo)], [h - s for s, h in zip(sp, hi)]],
               fmt="o", capsize=3, c="#d62728")
ax[2].set_xticks(range(5)); ax[2].set_xticklabels(["mild", "moderate", "severe", "hidden\nmoderate", "hidden\nsevere"])
ax[2].set_ylim(0, 1); ax[2].set_ylabel(r"Spearman(DI, $w$)"); ax[2].set_title("(c)", loc="left")
fig.tight_layout(); save(fig, "figA3_bridge.png")

# ---- Fig A4: Meuse ---------------------------------------------------------------------
M = pd.read_csv(os.path.join(RES, "meuse_summary.csv"))
mm = ["split", "normalized", "cqr", "di_normalized", "di_mondrian", "di_cqr", "di_cqr_K2",
      "lcp", "lcp_cqr", "geo_lcp", "weighted_estimated"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for j, met in enumerate(["worst_region", "mean_interval_score"]):
    vals = [M[(M.method == m) & (M.metric == met)]["mean"].iloc[0] for m in mm]
    sd = [M[(M.method == m) & (M.metric == met)]["sd"].iloc[0] for m in mm]
    colors = ["#d62728" if m.startswith("di_cqr") else ("#9467bd" if "lcp" in m else
              ("#54a24b" if "weighted" in m else "#9ab0c8")) for m in mm]
    ax[j].bar(range(len(mm)), vals, yerr=sd, capsize=1.5, color=colors, edgecolor="k", lw=.3, error_kw=dict(lw=0.6))
    ax[j].set_xticks(range(len(mm))); ax[j].set_xticklabels([PRETTY[m] for m in mm], rotation=45, ha="right", fontsize=7.5)
    ax[j].set_title(f"({'ab'[j]})", loc="left")
    if met == "worst_region":
        ax[j].axhline(NOM, ls="--", c="k", lw=0.8); ax[j].set_ylabel("worst-region coverage"); ax[j].set_ylim(0, 1.05)
    else:
        ax[j].set_ylabel("interval score")
fig.tight_layout(); save(fig, "figA4_meuse.png")

print("figures written:", sorted(os.listdir(FIG)))
