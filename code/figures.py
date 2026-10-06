"""figures.py -- publication figures from the saved result CSV/JSON files.

Colours follow the Okabe-Ito palette, which stays distinguishable under the
common forms of colour vision deficiency; every method also has its own marker,
and all symbols are explained inside the figures. Each figure is written as a
300 dpi PNG and as a vector PDF (fig01-fig04 in the main text, figB1-figB4 in
Appendix B)."""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
plt.rcParams.update({"font.size": 9, "savefig.dpi": 300, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
NOM = 0.9
REG = ["none", "mild", "moderate", "severe", "hidden_moderate", "hidden_severe"]
REGLAB = ["none", "mild", "moderate", "severe", "moderate", "severe"]
PRETTY = {"split": "Split", "normalized": "Normalized", "di_normalized": "DI-normalized",
          "region_mondrian": "Spatial-Mondrian", "di_mondrian": "DI-Mondrian",
          "cqr": "CQR", "di_cqr": "DI-CQR", "lcp": "LCP", "lcp_cqr": "LCP-CQR",
          "geo_lcp": "Geo-LCP", "weighted_oracle": "Weighted (oracle)",
          "weighted_estimated": "Weighted (estimated)", "width_matched_global": "Width-matched",
          "di_cqr_K2": "DI-CQR (K = 2)"}
# Okabe-Ito colours + a distinct marker per method
STY = {"split": ("#0072B2", "o", "-"), "cqr": ("#E69F00", "s", "-"),
       "lcp": ("#56B4E9", "v", "-"), "lcp_cqr": ("#56B4E9", "^", "-"),
       "di_normalized": ("#CC79A7", "D", "-"), "di_cqr": ("#D55E00", "o", "-"),
       "weighted_oracle": ("#009E73", "P", "-"), "width_matched_global": ("#000000", "x", "--")}
FAMILY = {"di": ("#D55E00", "//", "DI-conditioned"), "lcp": ("#56B4E9", "..", "localized conformal"),
          "w": ("#009E73", "xx", "weighted conformal"), "other": ("#BBBBBB", "", "other methods")}
S = pd.read_csv(os.path.join(RES, "sim_summary.csv"))


def fam(m):
    if m.startswith("di_"):
        return "di"
    if "lcp" in m:
        return "lcp"
    if "weighted" in m:
        return "w"
    return "other"


def g(rg, m, col):
    r = S[(S.regime == rg) & (S.method == m)].iloc[0]
    return r[col + "_mean"], r[col + "_sd"]


NOMH = Line2D([], [], ls=":", c="0.35", lw=1.0, label="nominal coverage (0.90)")


def nominal(ax, x=None):
    ax.axhline(NOM, ls=":", c="0.35", lw=1.0)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def family_legend(ax, fams, loc="upper left"):
    ax.legend(handles=[Patch(facecolor=FAMILY[f][0], hatch=FAMILY[f][1], edgecolor="k", lw=0.3,
                             label=FAMILY[f][2]) for f in fams] + [NOMH],
              fontsize=7, loc=loc, framealpha=0.9)


def bars(ax, methods, vals, errs):
    for i, (m, v, e) in enumerate(zip(methods, vals, errs)):
        c, h, _ = FAMILY[fam(m)]
        ax.bar(i, v, yerr=e, capsize=1.5, color=c, hatch=h, edgecolor="k", lw=0.3,
               error_kw=dict(lw=0.6))
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([PRETTY[m] for m in methods], rotation=45, ha="right", fontsize=7.5)


F = json.load(open(os.path.join(RES, "sim_fields_severe.json")))
FH = json.load(open(os.path.join(RES, "sim_fields_hidden_severe.json")))

# ---- Fig 1: the spatial problem ------------------------------------------------
G = F["GRID"]; ext = [0, 1, 0, 1]
STROKE = [pe.Stroke(linewidth=1.6, foreground="k"), pe.Normal()]


def grid(a):
    return np.array(a, float).reshape(G, G)


def regions(ax, c="w"):
    for t in (1 / 3, 2 / 3):
        for f_ in (ax.axhline, ax.axvline):
            f_(t, c=c, lw=0.8, path_effects=STROKE if c == "w" else None)


fig, ax = plt.subplots(1, 4, figsize=(15, 3.9))
for j, (F_, lab) in enumerate([(F, "(a) covariate-driven access"), (FH, "(b) hidden road access")]):
    co = np.array(F_["coords"]); fi = np.array(F_["fit"]); ca = np.array(F_["cal"])
    im = ax[j].imshow(np.log10(grid(F_["p_sel"]) + 1e-300).clip(-12, None), origin="lower",
                      extent=ext, cmap="viridis")
    regions(ax[j])
    ax[j].scatter(co[fi, 0], co[fi, 1], s=7, c="#E69F00", edgecolors="k", lw=0.25, label="fitting site")
    ax[j].scatter(co[ca, 0], co[ca, 1], s=7, c="w", edgecolors="k", lw=0.25, label="calibration site")
    ax[j].set_title(lab, fontsize=9, loc="left")
    cb = fig.colorbar(im, ax=ax[j], fraction=.046); cb.set_label(r"$\log_{10}$ selection intensity")
h, l = ax[0].get_legend_handles_labels()
h.append(Line2D([], [], c="w", lw=1.2, path_effects=STROKE)); l.append("region boundary")
ax[0].legend(h, l, loc="upper right", fontsize=6.5, framealpha=0.9, markerscale=1.6, facecolor="0.85")
mon = np.zeros(G * G, bool); mon[np.array(F["mon"])] = True
ax[2].imshow(np.where(mon, 1.0, 0.12).reshape(G, G), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1)
regions(ax[2], c="#D55E00")
ax[2].legend(handles=[Patch(facecolor="#E8E8E8", edgecolor="k", lw=0.3, label="held-out (scored)"),
                      Patch(facecolor="k", label="monitored (not scored)"),
                      Line2D([], [], c="#D55E00", lw=1.2, label="region boundary")],
             loc="upper right", fontsize=6.5, framealpha=0.95)
ax[2].set_title("(c) evaluation cells for the design in (a)", fontsize=9, loc="left")
im = ax[3].imshow(grid(F["di"]), origin="lower", extent=ext, cmap="magma")
regions(ax[3])
ax[3].set_title("(d) dissimilarity index for the design in (a)", fontsize=9, loc="left")
cb = fig.colorbar(im, ax=ax[3], fraction=.046); cb.set_label("dissimilarity index")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); save(fig, "fig01")

# ---- Fig 2: coverage across regimes -----------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
lines = ["split", "cqr", "lcp", "di_normalized", "di_cqr", "weighted_oracle"]
for j, col in enumerate(["true_marginal", "worst_region_coverage"]):
    for k, m in enumerate(lines):
        c, mk, ls = STY[m]
        x = np.arange(len(REG)) + (k - 2.5) * 0.06
        ys = [g(rg, m, col)[0] for rg in REG]; es = [g(rg, m, col)[1] for rg in REG]
        ax[j].errorbar(x, ys, yerr=es, marker=mk, ms=4, capsize=2, lw=1.3, ls=ls, label=PRETTY[m],
                       color=c, mec="k", mew=0.3)
    nominal(ax[j], x=0.36)
    ax[j].axvline(3.5, c="0.6", lw=0.6)
    for xs, txt in [(1.5, "covariate-driven access"), (4.5, "hidden access")]:
        ax[j].text(xs, 1.06, txt, ha="center", va="bottom", fontsize=8, color="0.25")
    ax[j].set_xticks(range(len(REG))); ax[j].set_xticklabels(REGLAB)
    ax[j].set_xlabel("monitoring bias"); ax[j].set_ylim(0, 1.05)
ax[0].set_ylabel("marginal coverage"); ax[1].set_ylabel("worst-region coverage")
ax[0].set_title("(a)", loc="left", pad=16); ax[1].set_title("(b)", loc="left", pad=16)
h, l = ax[0].get_legend_handles_labels()
ax[0].legend(h + [NOMH], l + [NOMH.get_label()], fontsize=7, loc="lower left", ncol=1)
fig.tight_layout(); save(fig, "fig02")

# ---- Fig 3: method comparison, severe covariate and hidden regimes -----------------
order = ["split", "region_mondrian", "normalized", "cqr", "di_normalized", "di_mondrian",
         "di_cqr", "lcp", "lcp_cqr", "geo_lcp", "weighted_estimated", "weighted_oracle",
         "width_matched_global"]
fig, ax = plt.subplots(2, 2, figsize=(12, 7.0))
for ci, (rg, lab) in enumerate([("severe", "severe, covariate-driven access"), ("hidden_severe", "severe, hidden access")]):
    for ri, col in enumerate(["mean_interval_score", "worst_region_coverage"]):
        a = ax[ri, ci]
        bars(a, order, [g(rg, m, col)[0] for m in order], [g(rg, m, col)[1] for m in order])
        if col == "worst_region_coverage":
            nominal(a, x=0.6); a.set_ylim(0, 1.05); a.set_ylabel("worst-region coverage")
        else:
            a.set_ylabel("interval score (log scale; lower is better)"); a.set_yscale("log")
            a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
            a.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
                lambda x, _: f"{x:g}" if f"{x:g}"[0] in "25" else ""))
        a.set_title(f"({'abcd'[2 * ri + ci]}) {lab}", loc="left", fontsize=9)
family_legend(ax[0, 0], ["di", "lcp", "w", "other"])
fig.tight_layout(); save(fig, "fig03")

# ---- Fig 4: coverage by decile of the dissimilarity index, averaged ----------------------
DC = pd.read_csv(os.path.join(RES, "sim_di_curve.csv"))
LCV = pd.read_csv(os.path.join(RES, "lucas_di_curve.csv"))
curve_m = ["split", "cqr", "di_normalized", "di_cqr", "lcp_cqr", "weighted_oracle", "width_matched_global"]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.1), sharey=True)
panels = [(DC[DC.regime == "severe"], "(a) simulation, severe covariate-driven access", 30),
          (DC[DC.regime == "hidden_severe"], "(b) simulation, severe hidden access", 30),
          (LCV, "(c) LUCAS 2015", 20)]
for j, (d, lab, nrep) in enumerate(panels):
    for m in curve_m:
        dm = d[d.method == m].sort_values("decile")
        if not len(dm):
            continue
        c, mk, ls = STY[m]
        se = dm["cov_sd"] / np.sqrt(nrep)
        ax[j].plot(dm["decile"], dm["cov_mean"], ls, marker=mk, ms=4, lw=1.4, color=c,
                   mec="k", mew=0.3, label=PRETTY[m])
        ax[j].fill_between(dm["decile"], dm["cov_mean"] - se, dm["cov_mean"] + se, color=c, alpha=0.15, lw=0)
    nominal(ax[j], x=0.99)
    ax[j].set_xticks(range(1, 11)); ax[j].set_ylim(0, 1.05)
    ax[j].set_xlabel("decile of the dissimilarity index (1 = least dissimilar)")
    ax[j].set_title(lab, loc="left", fontsize=9)
ax[0].set_ylabel("coverage (mean ± 1 standard error)")
h, l = ax[0].get_legend_handles_labels()
ax[2].legend(h + [NOMH], l + [NOMH.get_label()], fontsize=7, loc="lower left", ncol=2)
fig.tight_layout(); save(fig, "fig04")

# ---- Fig B1: coverage-width frontier ---------------------------------------------------
sw = pd.read_csv(os.path.join(RES, "sim_sweep_summary.csv"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
for j, rg in enumerate(["moderate", "severe"]):
    for m in ["split", "cqr", "di_cqr", "weighted_oracle"]:
        c, mk, ls = STY[m]
        d = sw[(sw.regime == rg) & (sw.method == m)].sort_values("mean_width")
        ax[j].plot(d["mean_width"], d["marginal"], ls, marker=mk, ms=4, label=PRETTY[m], color=c, mec="k", mew=0.3)
    nominal(ax[j], x=0.99)
    ax[j].set_xlabel("mean interval width (log scale)")
    ax[j].set_ylabel("marginal coverage"); ax[j].set_title(f"({'ab'[j]}) {rg} covariate-driven bias", loc="left")
    ax[j].set_xscale("log")
h, l = ax[0].get_legend_handles_labels()
ax[0].legend(h + [NOMH], l + [NOMH.get_label()], fontsize=7, loc="lower right")
fig.tight_layout(); save(fig, "figB1")

# ---- Fig B2: DI-bin ablation -----------------------------------------------------------
A = pd.read_csv(os.path.join(RES, "sim_ablation_summary.csv"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
for j, col in enumerate(["mean_interval_score_mean", "worst_region_coverage_mean"]):
    for rg, c, mk in [("moderate", "#E69F00", "s"), ("severe", "#D55E00", "o")]:
        for mono, ls in [(1, "-"), (0, "--")]:
            d = A[(A.regime == rg) & (A.monotone == mono)].sort_values("K")
            ax[j].plot(d["K"], d[col], ls, marker=mk, ms=4, color=c, mec="k", mew=0.3,
                       label=f"{rg} bias, monotone corrections {'on' if mono else 'off'}")
    ax[j].set_xlabel("number of DI bins, $K$"); ax[j].set_title(f"({'ab'[j]})", loc="left")
ax[0].set_ylabel("interval score"); ax[1].set_ylabel("worst-region coverage")
ax[0].legend(fontsize=7)
fig.tight_layout(); save(fig, "figB2")

# ---- Fig B3: DI against the oracle shift weight ------------------------------------------
B = pd.read_csv(os.path.join(RES, "sim_bridge.csv")).set_index("regime")
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for j, (F_, lab) in enumerate([(F, "severe, covariate-driven access"), (FH, "severe, hidden access")]):
    e_ = np.array(F_["ev"]); w = np.array(F_["w_oracle"])[e_]; d_ = np.array(F_["di"])[e_]
    sub = np.random.default_rng(0).choice(len(w), size=min(2000, len(w)), replace=False)
    ax[j].scatter(w[sub], d_[sub], s=5, alpha=.35, c="#0072B2", lw=0, label="held-out cell")
    ax[j].set_xscale("log"); ax[j].set_xlabel(r"oracle weight $w=(1-\pi)/\pi$ (log scale)")
    ax[j].set_ylabel("dissimilarity index"); ax[j].set_title(f"({'ab'[j]}) {lab}", loc="left")
    ax[j].legend(fontsize=7, loc="upper left", markerscale=2)
rg = ["mild", "moderate", "severe", "hidden_moderate", "hidden_severe"]
sp = [B.loc[r, "spearman_DI_invpsel_mean"] for r in rg]
lo = [B.loc[r, "spearman_ci_lo"] for r in rg]; hi = [B.loc[r, "spearman_ci_hi"] for r in rg]
ax[2].errorbar(range(5), sp, yerr=[[s - l for s, l in zip(sp, lo)], [h - s for s, h in zip(sp, hi)]],
               fmt="o", capsize=3, c="#D55E00", mec="k", mew=0.3, label="mean and 95 % confidence interval")
ax[2].set_xticks(range(5)); ax[2].set_xticklabels(["mild", "moderate", "severe", "hidden,\nmoderate", "hidden,\nsevere"])
ax[2].set_ylim(0, 1); ax[2].set_ylabel(r"Spearman correlation of DI and $w$"); ax[2].set_title("(c)", loc="left")
ax[2].legend(fontsize=7, loc="lower left")
fig.tight_layout(); save(fig, "figB3")

# ---- Fig B4: Meuse ---------------------------------------------------------------------
M = pd.read_csv(os.path.join(RES, "meuse_summary.csv"))
mm = ["split", "normalized", "cqr", "di_normalized", "di_mondrian", "di_cqr", "di_cqr_K2",
      "lcp", "lcp_cqr", "geo_lcp", "weighted_estimated"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
for j, met in enumerate(["worst_region", "mean_interval_score"]):
    vals = [M[(M.method == m) & (M.metric == met)]["mean"].iloc[0] for m in mm]
    sd = [M[(M.method == m) & (M.metric == met)]["sd"].iloc[0] for m in mm]
    bars(ax[j], mm, vals, sd)
    ax[j].set_title(f"({'ab'[j]})", loc="left")
    if met == "worst_region":
        nominal(ax[j], x=0.45); ax[j].set_ylabel("worst-region coverage"); ax[j].set_ylim(0, 1.12)
    else:
        ax[j].set_ylabel("interval score")
family_legend(ax[1], ["di", "lcp", "w", "other"])
fig.tight_layout(); save(fig, "figB4")

print("figures written:", sorted(os.listdir(FIG)))
