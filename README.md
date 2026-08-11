# Dissimilarity-Adaptive Conformal Prediction for Spatial Models under Uneven Monitoring

Code, raw experimental output, and full verification for a study of conformal
prediction under spatially uneven environmental monitoring networks.

**Manuscript status:** prepared for journal submission. A link to the published
or publicly available version will be added here when available. This repository
is the code and reproducibility record for that manuscript, released
independently of it.

## The problem

Environmental ML models (soil, water quality, flood risk) are trained on
monitoring networks that cluster in accessible locations, not sampled evenly
across space. Split conformal prediction is only *marginally* valid under
exchangeability, so when the calibration set is drawn from this accessible,
clustered subset while predictions are needed everywhere, coverage collapses
in the under-monitored regions that most need honest uncertainty — exactly
where a decision-maker has the least recourse if the interval is wrong.

## The method

The Area of Applicability (AOA) framework (Meyer & Pebesma, 2021) defines a
Dissimilarity Index (DI): a per-point distance to the nearest training-set
observation in a standardized, weighted feature space. It was designed as an
extrapolation *diagnostic*, not a calibration tool. This project asks whether
the DI can instead be used to *condition* conformal calibration — i.e. as the
grouping variable of a Mondrian conformal taxonomy, optionally combined with
conformalized quantile regression (CQR). Two variants are implemented and
compared:

- **DI-normalized** — a smooth multiplicative width normalizer `q·(κ+d(x))`.
- **DI-CQR / DI-Mondrian** — bin calibration points into `K` DI quantile bins
  (fixed a priori, not tuned on test outcomes) and compute a finite-sample
  conformal quantile within each bin.

Strict leakage controls are enforced throughout: DI is always computed
against the model-fitting split only; bin edges come from calibration DI, never
test data; small/empty bins and out-of-range queries fall back to / clip to the
global or top bin. See `code/aoa.py` and `code/conformal.py`.

**What is and isn't novel.** Conditioning conformal calibration on a
covariate/dissimilarity signal is not new (Mondrian conformal, localized
conformal, weighted conformal under covariate shift, and existing geospatial
conformal methods all address related problems — see the manuscript's related
work). The contribution here is narrower: using the specific, off-the-shelf AOA
DI as the conditioning signal, benchmarking it directly against these existing
approaches (including localized conformal and oracle/estimated weighted
conformal), and reporting an honest characterization of when it helps and when
it fails — rather than a claim of a new algorithm.

## Experiments

1. **Simulation** — a `56×56` ground-truth-everywhere synthetic landscape with
   accessibility-driven monitoring bias at four strengths (none / mild /
   moderate / severe), 30 seeds each, all methods benchmarked head-to-head.
2. **Meuse** — 155 real floodplain soil samples (zinc), semi-synthetic
   clustered-monitoring bias, 30 repeats.
3. **LUCAS 2015** — 21,687 EU topsoil sites, log organic carbon predicted from
   23 bioclimatic/terrain covariates, semi-synthetic clustered bias at
   EU scale, 20 repeats. (Raw LUCAS data is not redistributed here — see
   Reproducing below.)
4. **Robustness check** — the moderate/severe simulation benchmark repeated with
   histogram gradient boosting as the mean model instead of random forest,
   testing whether the main finding depends on the mean-model family.

All experiments report marginal coverage, worst-region coverage, coverage gap,
mean interval width, and Winkler/interval score against a nominal 90% target,
with paired *t*-test and Wilcoxon signed-rank significance where relevant.

## Key results (from `results/`, all reproducible from raw output)

**Simulation (nominal 0.90, 30 seeds):**

| Regime | Method | Marginal coverage | Worst-region coverage |
|---|---|---|---|
| No bias | split | 0.92 | 0.83 |
| No bias | DI-CQR | 0.94 | 0.88 |
| Moderate | split | 0.58 | 0.25 |
| Moderate | DI-CQR | 0.86 | 0.61 |
| Severe | split | 0.45 | 0.15 |
| Severe | DI-CQR | 0.77 | 0.45 |

Under no bias, split conformal is already near-nominal — confirming the
failure under bias is a calibration/test distribution mismatch, not a flaw in
the base estimator. DI-conditional methods give the best interval score among
practical (non-oracle) methods and consistently improve worst-region coverage,
without matching nominal coverage or beating oracle weighted conformal (which
needs 6–10× wider intervals to lead on coverage).

**Meuse (real data, nominal 0.90, 30 repeats):** split marginal 0.93 /
worst-region 0.84; DI-CQR marginal 0.97 / worst-region 0.93. The shift here is
milder than the severe simulation regime, so split does not collapse, but
DI-CQR still gives the best worst-region coverage.

**LUCAS 2015 (real data, EU scale, nominal 0.90, 20 repeats):** split marginal
0.77 / worst-region 0.57; DI-CQR marginal 0.91 / worst-region 0.79 at a
competitive interval score. **DI-normalized reaches even higher raw coverage
(0.97 marginal / 0.90 worst-region) but only through an unbounded width blow-up**
(mean width 10.8 vs DI-CQR's 3.5) — its multiplicative normalizer is unbounded
and explodes under the large DI values encountered in this 23-dimensional
setting, giving it the *worst* interval score of any conditioning method. This
is reported as an honest failure mode, not patched: **how** DI is incorporated
materially affects robustness, and DI-CQR's binning/clipping avoids the
pathology that DI-normalized exhibits at scale.

**Robustness (second mean-model family — histogram gradient boosting):** the
same pattern holds in the moderate/severe simulation benchmark when the mean
model is histogram gradient boosting rather than random forest (moderate: split
marginal 0.57/worst 0.23 vs. DI-CQR 0.85/0.61; severe: split 0.44/0.14 vs.
DI-CQR 0.77/0.45), indicating the finding is not an artifact of one mean-model
family.

**Mechanistic bridge:** under accessibility-driven selection, the inverse
selection probability `1/p_sel(x)` is the covariate-shift weight used by
weighted conformal. DI correlates with it (Spearman 0.68 / 0.91 / 0.94 for
mild / moderate / severe bias), which is why DI-conditioning captures much of
the same shift information without needing to know the selection mechanism —
and why its benefit grows with bias strength (`results/sim_bridge.csv`).

## Repository layout

```
code/       aoa.py            Dissimilarity Index computation (leakage-safe)
            landscape.py      synthetic landscape / monitoring-bias simulator
            metrics.py        coverage, width, interval-score metrics
            conformal.py      split / normalized / DI-normalized / Mondrian /
                              DI-CQR / localized / weighted conformal methods
            run_sim.py        main simulation benchmark (4 regimes x 30 seeds)
            aggregate.py      aggregates raw simulation output to summary CSVs
            run_meuse.py      Meuse real-data semi-synthetic experiment
            prep_lucas.py     builds the LUCAS analysis dataset (not included)
            run_lucas.py      LUCAS real-data semi-synthetic experiment
            run_robust.py     second-model-family (HistGBR) robustness check
            make_tables.py    generates headline_numbers.json from raw output
            figures.py        generates all figures from raw/aggregated output
            verify.py         asserts every reported number against raw output
results/    raw + aggregated CSV/JSON for every experiment above
figures/    publication figures generated from the results above
```

## Reproducing the results

Requires Python 3.10+ with scikit-learn, scipy, pandas, matplotlib, and
scikit-gstat (for the bundled Meuse dataset).

```bash
cd code
python run_sim.py            # repeat until "120/120" (resumable)
python aggregate.py          # -> results/sim_*.csv, sim_fields_severe.json
python run_meuse.py          # repeat until "30/30"
python run_meuse.py agg      # -> results/meuse_summary.csv
python run_robust.py         # HistGBR robustness check (resumable)
python make_tables.py        # -> results/headline_numbers.json
python figures.py            # -> figures/*.png
python verify.py             # asserts all reported numbers against raw output
```

**LUCAS 2015 is not reproducible out of the box** because the raw JRC/ESDAC
data cannot be redistributed here. To reproduce it: download the LUCAS 2015
topsoil CSV + shapefile and the ancillary covariates CSV from ESDAC, point
`code/prep_lucas.py` at them, run it to build `results/lucas_prepared.csv`,
then run `python run_lucas.py` (repeat until complete) and
`python run_lucas.py agg`. The aggregated LUCAS output already in `results/`
(`lucas_summary.csv`, `lucas_sig.json`, `lucas_raw.jsonl`) lets the reported
numbers be checked without re-downloading anything.

## Limitations (stated explicitly, not just in the paper)

- Meuse and LUCAS are **semi-synthetically** biased (real covariates and
  targets, but a simulated clustered-monitoring selection mechanism) — neither
  is a naturally uneven monitoring network.
- No reproduction of existing official geospatial-conformal systems; this
  project implements clearly-labelled reimplementations of related ideas
  (e.g. localized and geographic-Mondrian conformal) for benchmarking, not
  their original code.
- DI-conditional methods do not reach nominal per-region coverage under
  moderate/severe bias — consistent with known distribution-free
  conditional-coverage limits, not a calibration guarantee.
- All performance claims are scoped to the evaluated simulations and two real
  datasets, not asserted as general guarantees.

## Citation

A citation entry will be added here once the manuscript has a public
preprint or journal DOI. In the meantime, please cite this repository
directly (e.g. via GitHub's "Cite this repository" or by URL).

## License

MIT — see `LICENSE`.
