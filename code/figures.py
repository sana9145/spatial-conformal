"""figures.py -- publication figures from saved result CSV/JSON."""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
plt.rcParams.update({"font.size": 10, "savefig.dpi": 200, "font.family": "DejaVu Sans"})
NOM = 0.9
REG = ["none", "mild", "moderate", "severe"]
PRETTY = {"split": "Split", "normalized": "Normalized", "di_normalized": "DI-normalized",
          "region_mondrian": "Spatial-Mondrian", "di_mondrian": "DI-Mondrian",
          "cqr": "CQR", "di_cqr": "DI-CQR", "localized": "Localized (Guan)",
          "geo_localized": "Geo-localized", "weighted_oracle": "Weighted (oracle)",
          "weighted_estimated": "Weighted (est.)", "width_matched_global": "Width-matched global"}
S = pd.read_csv(os.path.join(RES, "sim_summary.csv"))


def g(rg, m, col):
    r = S[(S.regime == rg) & (S.method == m)].iloc[0]
    return r[col + "_mean"], r[col + "_sd"]


# ---- Fig 1: controls + collapse (marginal & worst-region vs regime) ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
lines = ["split", "localized", "di_normalized", "di_cqr", "weighted_oracle"]
cols = {"split": "#4c78a8", "cqr": "#f58518", "di_cqr": "#e45756", "weighted_oracle": "#54a24b",
        "localized": "#9467bd", "di_normalized": "#d62728"}
for j, col in enumerate(["true_marginal", "worst_region_coverage"]):
    for m in lines:
        ys = [g(rg, m, col)[0] for rg in REG]
        es = [g(rg, m, col)[1] for rg in REG]
        ax[j].errorbar(range(4), ys, yerr=es, marker="o", capsize=3, label=PRETTY[m], color=cols[m])
    ax[j].axhline(NOM, ls="--", c="k", lw=1); ax[j].set_xticks(range(4)); ax[j].set_xticklabels(REG)
    ax[j].set_ylim(0, 1.02); ax[j].set_xlabel("monitoring-bias regime")
ax[0].set_ylabel("marginal coverage"); ax[0].set_title("(a) Marginal coverage")
ax[1].set_ylabel("worst-region coverage"); ax[1].set_title("(b) Worst-region coverage")
ax[0].legend(fontsize=8, loc="lower left")
ax[0].text(0.05, NOM+.01, "nominal 0.90", fontsize=8)
fig.suptitle("Split conformal is near-nominal only when calibration and test match (no bias);\n"
             "it degrades as monitoring bias grows. Weighted-oracle recovers coverage; DI-CQR partially recovers it.", y=1.03)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_controls.png"), bbox_inches="tight"); plt.close(fig)


# ---- Fig 2: method comparison bars (interval score + worst-region) moderate & severe ----
order = ["split", "region_mondrian", "normalized", "di_normalized", "cqr",
         "di_mondrian", "di_cqr", "localized", "geo_localized",
         "weighted_estimated", "weighted_oracle", "width_matched_global"]
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
for col_i, rg in enumerate(["moderate", "severe"]):
    for row_i, col in enumerate(["mean_interval_score", "worst_region_coverage"]):
        a = ax[row_i, col_i]
        vals = [g(rg, m, col)[0] for m in order]; es = [g(rg, m, col)[1] for m in order]
        colors = ["#e45756" if m in ("di_cqr", "di_normalized")
                  else ("#9467bd" if "localized" in m else ("#54a24b" if "weighted" in m else "#4c78a8"))
                  for m in order]
        a.bar(range(len(order)), vals, yerr=es, capsize=2, color=colors, edgecolor="k", lw=.4)
        a.set_xticks(range(len(order))); a.set_xticklabels([PRETTY[m] for m in order], rotation=40, ha="right", fontsize=8)
        if col == "worst_region_coverage":
            a.axhline(NOM, ls="--", c="k", lw=1); a.set_ylim(0, 1)
            a.set_ylabel("worst-region coverage")
        else:
            a.set_ylabel("interval score (lower=better)")
        a.set_title(f"{rg} bias — {'interval score' if row_i==0 else 'worst-region coverage'}")
fig.suptitle("Weighted-oracle attains the best coverage but the worst interval score (very wide);\n"
             "DI-CQR gives the best interval score among practical methods and improves worst-region coverage", y=1.01)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_methods.png"), bbox_inches="tight"); plt.close(fig)


# ---- Fig 3: coverage-vs-width efficiency frontier (severe) ----
sw = pd.read_csv(os.path.join(RES, "sim_sweep_summary.csv"))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for j, rg in enumerate(["moderate", "severe"]):
    for m in ["split", "cqr", "di_cqr", "weighted_oracle"]:
        d = sw[(sw.regime == rg) & (sw.method == m)].sort_values("mean_width")
        ax[j].plot(d["mean_width"], d["marginal"], marker="o", label=PRETTY[m], color=cols[m])
    ax[j].axhline(NOM, ls="--", c="k", lw=1); ax[j].set_xlabel("mean interval width")
    ax[j].set_ylabel("marginal coverage"); ax[j].set_title(f"({'a' if j==0 else 'b'}) {rg} bias")
    ax[j].set_xscale("log")
ax[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Coverage–width frontier: DI-CQR reaches a given coverage at far smaller width than weighted-oracle,\n"
             "and dominates split/CQR (up-and-left is better)", y=1.03)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_frontier.png"), bbox_inches="tight"); plt.close(fig)


# ---- Fig 4: conditional coverage & width vs DI (severe fields) ----
F = json.load(open(os.path.join(RES, "sim_fields_severe.json")))
di = np.array(F["di"]); ins = np.array(F["in_split"]); inp = np.array(F["in_dicqr"])
hs = np.array(F["half_split"]); hp = np.array(F["half_dicqr"])
edges = np.quantile(di, np.linspace(0, 1, 11)); mid = .5*(edges[:-1]+edges[1:])
def binstat(v): return [v[(di>=edges[i])&(di<edges[i+1])].mean() for i in range(10)]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(mid, binstat(ins), "o-", c=cols["split"], label="Split")
ax[0].plot(mid, binstat(inp), "s-", c=cols["di_cqr"], label="DI-CQR")
ax[0].axhline(NOM, ls="--", c="k", lw=1); ax[0].set_xlabel("Dissimilarity Index (extrapolation →)")
ax[0].set_ylabel("coverage"); ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8)
ax[0].set_title("(a) Coverage vs dissimilarity (severe, one run)")
ax[1].plot(mid, [2*x for x in binstat(hs)], "o-", c=cols["split"], label="Split")
ax[1].plot(mid, [2*x for x in binstat(hp)], "s-", c=cols["di_cqr"], label="DI-CQR")
ax[1].set_xlabel("Dissimilarity Index (extrapolation →)"); ax[1].set_ylabel("interval width"); ax[1].legend(fontsize=8)
ax[1].set_title("(b) DI-CQR widens intervals with dissimilarity")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_vs_di.png"), bbox_inches="tight"); plt.close(fig)


# ---- Fig 5: DI-bin ablation ----
A = pd.read_csv(os.path.join(RES, "sim_ablation_summary.csv"))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for j, col in enumerate(["mean_interval_score_mean", "worst_region_coverage_mean"]):
    for rg, c in [("moderate", "#f58518"), ("severe", "#e45756")]:
        for mono, ls in [(1, "-"), (0, "--")]:
            d = A[(A.regime == rg) & (A.monotone == mono)].sort_values("K")
            ax[j].plot(d["K"], d[col], ls, marker="o", color=c,
                       label=f"{rg}, mono={'on' if mono else 'off'}")
    ax[j].set_xlabel("number of DI bins K")
    if "worst" in col:
        ax[j].axhline(NOM, ls=":", c="k"); ax[j].set_ylabel("worst-region coverage")
        ax[j].set_title("(b) Worst-region coverage vs K")
    else:
        ax[j].set_ylabel("interval score"); ax[j].set_title("(a) Interval score vs K")
ax[0].legend(fontsize=7)
fig.suptitle("DI-bin ablation: larger K improves coverage/interval score (fewer points per bin); "
             "monotonicity has a small positive effect. K=5 fixed a priori for main results.", y=1.03)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_ablation.png"), bbox_inches="tight"); plt.close(fig)


# ---- Fig 6: Meuse real-data ----
M = pd.read_csv(os.path.join(RES, "meuse_summary.csv"))
mm = ["split", "region_mondrian", "normalized", "di_normalized", "cqr", "di_mondrian", "di_cqr", "weighted_estimated"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for j, met in enumerate(["worst_region", "mean_interval_score"]):
    vals = [M[(M.method == m) & (M.metric == met)]["mean"].iloc[0] for m in mm]
    lo = [M[(M.method == m) & (M.metric == met)]["ci_lo"].iloc[0] for m in mm]
    hi = [M[(M.method == m) & (M.metric == met)]["ci_hi"].iloc[0] for m in mm]
    err = [[v-l for v, l in zip(vals, lo)], [h-v for v, h in zip(vals, hi)]]
    colors = ["#e45756" if m == "di_cqr" else ("#54a24b" if "weighted" in m else "#4c78a8") for m in mm]
    ax[j].bar(range(len(mm)), vals, yerr=err, capsize=2, color=colors, edgecolor="k", lw=.4)
    ax[j].set_xticks(range(len(mm))); ax[j].set_xticklabels([PRETTY[m] for m in mm], rotation=40, ha="right", fontsize=8)
    if met == "worst_region":
        ax[j].axhline(NOM, ls="--", c="k"); ax[j].set_ylabel("worst-region coverage"); ax[j].set_ylim(0, 1)
        ax[j].set_title("(a) Worst-region coverage (95% CI)")
    else:
        ax[j].set_ylabel("interval score"); ax[j].set_title("(b) Interval score (95% CI)")
fig.suptitle("Meuse semi-synthetic monitoring bias (real soil data): milder shift, split does not collapse;\n"
             "DI-CQR gives the best worst-region coverage at a modest width cost", y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_meuse.png"), bbox_inches="tight"); plt.close(fig)


# ---- Fig 7: maps (severe) ----
G = F["GRID"]
def grid(a): return np.array(a).reshape(G, G)
fig, ax = plt.subplots(1, 3, figsize=(13, 4.3)); ext = [0, 1, 0, 1]
coords = np.array(F["coords"]); cal = np.array(F["cal"])
im = ax[0].imshow(grid(F["di"]), origin="lower", extent=ext, cmap="magma")
ax[0].scatter(coords[cal, 0], coords[cal, 1], s=4, c="cyan", lw=0)
ax[0].set_title("(a) Dissimilarity Index + calibration sites"); fig.colorbar(im, ax=ax[0], fraction=.046)
for k, (fld, t) in enumerate([("in_split", "(b) Split: miscovered (red)"),
                              ("in_dicqr", "(c) DI-CQR: miscovered (red)")]):
    ax[k+1].imshow(1-grid(F[fld]), origin="lower", extent=ext, cmap="Reds", vmin=0, vmax=1)
    ax[k+1].set_title(t)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle("Spatial miscoverage (severe, one run): DI-CQR recovers much of the unmonitored periphery", y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_maps.png"), bbox_inches="tight"); plt.close(fig)

# ---- Fig 8: DI vs inverse selection weight bridge ----
B = pd.read_csv(os.path.join(RES, "sim_bridge.csv")).set_index("regime")
inv = np.array(F["inv_psel"]); di_f = np.array(F["di"])
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
sub = np.random.default_rng(0).choice(len(inv), size=2000, replace=False)
ax[0].scatter(inv[sub], di_f[sub], s=6, alpha=.3, c="#4c78a8", lw=0)
ax[0].set_xscale("log"); ax[0].set_xlim(np.percentile(inv, 1), np.percentile(inv, 99))
ax[0].set_xlabel("inverse selection weight  $1/p_{sel}(x)$ (1--99 pct)")
ax[0].set_ylabel("Dissimilarity Index");
ax[0].set_title(f"(a) DI tracks 1/$p_{{sel}}$ (severe run)\nSpearman={B.loc['severe','spearman_DI_invpsel_mean']:.2f}")
rg = ["mild", "moderate", "severe"]
sp = [B.loc[r, "spearman_DI_invpsel_mean"] for r in rg]
lo = [B.loc[r, "spearman_ci_lo"] for r in rg]; hi = [B.loc[r, "spearman_ci_hi"] for r in rg]
err = [[s-l for s, l in zip(sp, lo)], [h-s for s, h in zip(sp, hi)]]
ax[1].errorbar(range(3), sp, yerr=err, marker="o", capsize=4, c="#e45756")
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(rg); ax[1].set_ylim(0, 1)
ax[1].set_xlabel("monitoring-bias regime"); ax[1].set_ylabel("Spearman(DI, 1/$p_{sel}$)")
ax[1].set_title("(b) Correlation strengthens with bias\n(explains the boundary)")
fig.suptitle("Why DI-conditioning works: the Dissimilarity Index is a computable proxy for the\n"
             "covariate-shift weight — and only strongly so once bias is large", y=1.03)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig8_bridge.png"), bbox_inches="tight"); plt.close(fig)


# ---- Fig 9: spatial experimental setup (severe run) ----
# Central spatial problem, visualised: where monitoring concentrates, which cells
# are held out for evaluation, the region partition, and the DI surface.
Gm = F["GRID"]
def _grid(a): return np.array(a).reshape(Gm, Gm)
coords = np.array(F["coords"])
fit_i = np.array(F["fit"]); cal_i = np.array(F["cal"])
reg_g = _grid(F["reg"]); ext = [0, 1, 0, 1]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))

# (a) selection intensity + region boundaries + monitored sites
im0 = ax[0].imshow(_grid(F["p_sel"]), origin="lower", extent=ext, cmap="viridis")
ax[0].contour(np.linspace(0, 1, Gm), np.linspace(0, 1, Gm), reg_g,
              levels=np.arange(reg_g.max()+1)+0.5, colors="w", linewidths=0.6, alpha=0.6)
ax[0].scatter(coords[fit_i, 0], coords[fit_i, 1], s=6, c="#ff7f0e", lw=0, label="fitting")
ax[0].scatter(coords[cal_i, 0], coords[cal_i, 1], s=6, c="cyan", lw=0, label="calibration")
ax[0].set_title("(a) Monitoring-selection intensity $p_{sel}$\n+ region boundaries + monitored sites")
ax[0].legend(loc="upper right", fontsize=7, framealpha=0.8); fig.colorbar(im0, ax=ax[0], fraction=.046)

# (b) held-out evaluation cells (unmonitored) vs monitored
mon_mask = np.zeros(Gm*Gm, bool); mon_mask[np.array(F["mon"])] = True
panel = np.where(mon_mask, 1.0, 0.15).reshape(Gm, Gm)  # dark = monitored, light = held-out eval
ax[1].imshow(panel, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.0)
ax[1].contour(np.linspace(0, 1, Gm), np.linspace(0, 1, Gm), reg_g,
              levels=np.arange(reg_g.max()+1)+0.5, colors="#e45756", linewidths=0.7, alpha=0.8)
ax[1].scatter(coords[mon_mask, 0], coords[mon_mask, 1], s=4, c="#1f77b4", lw=0)
ax[1].set_title("(b) Held-out evaluation cells (light)\nmonitored cells excluded (dark, blue points)")

# (c) DI surface + region boundaries
im2 = ax[2].imshow(_grid(F["di"]), origin="lower", extent=ext, cmap="magma")
ax[2].contour(np.linspace(0, 1, Gm), np.linspace(0, 1, Gm), reg_g,
              levels=np.arange(reg_g.max()+1)+0.5, colors="w", linewidths=0.6, alpha=0.6)
ax[2].set_title("(c) Dissimilarity Index surface\n(high = far from fitting data)")
fig.colorbar(im2, ax=ax[2], fraction=.046)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle("The spatial problem (severe-bias run): monitoring concentrates in accessible cells (a); "
             "models are evaluated only on the\nunmonitored majority (b); dissimilarity is high exactly "
             "where monitoring is sparse (c) — the signal DI-conditioning exploits", y=1.05)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig9_setup.png"), bbox_inches="tight"); plt.close(fig)

print("figures written:", sorted(os.listdir(FIG)))
